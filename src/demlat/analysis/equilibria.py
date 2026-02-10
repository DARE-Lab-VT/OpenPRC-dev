"""
Equilibrium Analyzer for DEMLAT
================================

Finds multiple equilibria of bar-hinge systems using deflated Newton's method
on the gradient of the potential energy. Classifies stability via Hessian eigenvalues.

Theory
------
For a conservative system with potential energy V(x; θ):
    - Equilibria satisfy ∇V(x) = 0
    - Stability is determined by the Hessian H = ∇²V(x):
        * All eigenvalues > 0  →  stable (local minimum)
        * Any eigenvalue < 0   →  unstable (saddle or maximum)
        * Any eigenvalue ≈ 0   →  marginally stable / mechanism mode

Deflation: After finding equilibrium x*, we solve instead:
    M(x; x*) · ∇V(x) = 0,  where M deflates known roots so Newton
    doesn't reconverge to them.

Usage
-----
    from demlat.analysis.equilibria import EquilibriumAnalyzer

    exp = demlat.Experiment("my_experiment/")
    analyzer = EquilibriumAnalyzer(exp)
    results = analyzer.find_equilibria(n_attempts=100)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import time
import json

import jax
import jax.numpy as jnp
from jax import jit, grad, jacfwd, jacrev

from enum import Enum


class StabilityType(Enum):
    STABLE = "stable"              # All eigenvalues > 0 (local minimum)
    UNSTABLE_SADDLE = "saddle"     # Mixed sign eigenvalues
    UNSTABLE_MAXIMUM = "maximum"   # All eigenvalues < 0
    MARGINAL = "marginal"          # Has near-zero eigenvalues (mechanism modes)


@dataclass
class Equilibrium:
    """A single equilibrium state of the system."""
    positions: np.ndarray            # (n_nodes, 3) equilibrium configuration
    energy: float                    # V(x*) potential energy at equilibrium
    eigenvalues: np.ndarray          # Eigenvalues of the Hessian (sorted)
    eigenvectors: np.ndarray         # Corresponding eigenvectors
    stability_type: StabilityType
    n_negative: int                  # Number of negative eigenvalues (Morse index)
    n_zero: int                      # Number of near-zero eigenvalues (mechanisms)
    converged: bool                  # Whether Newton actually converged
    residual: float                  # ||∇V(x*)|| at termination
    n_iterations: int                # Newton iterations used

    @property
    def is_stable(self) -> bool:
        return self.stability_type == StabilityType.STABLE

    @property
    def min_eigenvalue(self) -> float:
        return float(self.eigenvalues[0])

    @property
    def morse_index(self) -> int:
        """Number of unstable directions (negative eigenvalues)."""
        return self.n_negative

    def mechanism_modes(self, tol: float = 1e-6) -> np.ndarray:
        """Return eigenvectors corresponding to near-zero eigenvalues."""
        mask = np.abs(self.eigenvalues) < tol
        return self.eigenvectors[:, mask]


@dataclass
class EquilibriumSet:
    """Collection of equilibria found for a system."""
    equilibria: List[Equilibrium] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    timing: float = 0.0  # Total wall-clock time

    @property
    def stable(self) -> List[Equilibrium]:
        return [eq for eq in self.equilibria if eq.is_stable]

    @property
    def unstable(self) -> List[Equilibrium]:
        return [eq for eq in self.equilibria if not eq.is_stable]

    @property
    def saddles(self) -> List[Equilibrium]:
        return [eq for eq in self.equilibria
                if eq.stability_type == StabilityType.UNSTABLE_SADDLE]

    def summary(self) -> str:
        lines = [
            f"Equilibrium Analysis Results",
            f"============================",
            f"Total found: {len(self.equilibria)}",
            f"  Stable (minima):    {len(self.stable)}",
            f"  Saddle points:      {len(self.saddles)}",
            f"  Unstable (maxima):  {len([e for e in self.equilibria if e.stability_type == StabilityType.UNSTABLE_MAXIMUM])}",
            f"  Marginal:           {len([e for e in self.equilibria if e.stability_type == StabilityType.MARGINAL])}",
            f"Wall time: {self.timing:.2f}s",
            f"",
        ]
        for i, eq in enumerate(self.equilibria):
            lines.append(
                f"  [{i}] E={eq.energy:.6f}  type={eq.stability_type.value:>8s}  "
                f"morse_idx={eq.morse_index}  n_zero={eq.n_zero}  "
                f"min_eig={eq.min_eigenvalue:.2e}  residual={eq.residual:.2e}"
            )
        return "\n".join(lines)

    def save(self, path: Path):
        """Save results to JSON + NPZ."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Metadata
        meta = {
            "n_equilibria": len(self.equilibria),
            "n_stable": len(self.stable),
            "n_unstable": len(self.unstable),
            "timing": self.timing,
            "parameters": {k: str(v) for k, v in self.parameters.items()},
            "equilibria": []
        }
        for i, eq in enumerate(meta["equilibria"]):
            meta["equilibria"].append({
                "index": i,
                "energy": float(eq.energy),
                "stability_type": eq.stability_type.value,
                "morse_index": eq.morse_index,
                "n_zero": eq.n_zero,
                "min_eigenvalue": float(eq.min_eigenvalue),
                "residual": float(eq.residual),
                "converged": eq.converged,
            })

        with open(path / "equilibria_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Numerical data
        arrays = {}
        for i, eq in enumerate(self.equilibria):
            arrays[f"positions_{i}"] = eq.positions
            arrays[f"eigenvalues_{i}"] = eq.eigenvalues
            arrays[f"eigenvectors_{i}"] = eq.eigenvectors
        np.savez_compressed(path / "equilibria_data.npz", **arrays)


class EquilibriumAnalyzer:
    """
    Finds and classifies equilibria of a DEMLAT bar-hinge system.

    Builds a JAX potential energy function from experiment geometry,
    then uses deflated Newton iteration on ∇V = 0 with Hessian-based
    stability classification.
    """

    def __init__(self, experiment):
        """
        Parameters
        ----------
        experiment : demlat.Experiment or path-like
            A configured DEMLAT experiment (geometry + physics must exist).
        """
        from demlat.core.experiment import Experiment

        if not isinstance(experiment, Experiment):
            experiment = Experiment(experiment)

        self.experiment = experiment
        self.config = experiment.config
        self._load_geometry()
        self._build_energy_function()

    def _load_geometry(self):
        """Extract geometry and physics parameters from experiment files."""
        import h5py

        phys = self.config.get("global_physics", {})
        self.gravity = float(phys.get("gravity", -9.81))
        self.damping = float(phys.get("global_damping", 0.1))

        with self.experiment.get_geometry_reader() as f:
            self.x0 = np.array(f["nodes/positions"])
            self.masses = np.array(f["nodes/masses"])
            self.attrs = np.array(f["nodes/attributes"])
            self.n_nodes = self.x0.shape[0]

            # Bars
            self.n_bars = 0
            if "elements/bars/indices" in f and f["elements/bars/indices"].shape[0] > 0:
                bar_k = np.array(f["elements/bars/stiffness"])
                is_soft = bar_k >= 0
                if np.any(is_soft):
                    self.n_bars = int(np.sum(is_soft))
                    self.bar_indices = np.array(f["elements/bars/indices"])[is_soft]
                    self.bar_k = bar_k[is_soft]
                    self.bar_l0 = np.array(f["elements/bars/rest_length"])[is_soft]
                    if "elements/bars/prestress" in f:
                        self.bar_prestress = np.array(f["elements/bars/prestress"])[is_soft]
                    else:
                        self.bar_prestress = np.zeros(self.n_bars)

            # Hinges
            self.n_hinges = 0
            if "elements/hinges/indices" in f and f["elements/hinges/indices"].shape[0] > 0:
                hinge_k = np.array(f["elements/hinges/stiffness"])
                is_soft = hinge_k >= 0
                if np.any(is_soft):
                    self.n_hinges = int(np.sum(is_soft))
                    self.hinge_indices = np.array(f["elements/hinges/indices"])[is_soft]
                    self.hinge_k = hinge_k[is_soft]
                    self.hinge_phi0 = np.array(f["elements/hinges/angle"])[is_soft]

        # Fixed node mask
        self.fixed_mask = (self.attrs & 1).astype(bool)
        self.free_mask = ~self.fixed_mask
        self.n_free = int(np.sum(self.free_mask))
        self.free_indices = np.where(self.free_mask)[0]
        self.fixed_indices = np.where(self.fixed_mask)[0]

        # We work in reduced coordinates: only free node DOFs
        self.n_dof = self.n_free * 3

        print(f"[EquilibriumAnalyzer] {self.n_nodes} nodes ({self.n_free} free, "
              f"{len(self.fixed_indices)} fixed), {self.n_bars} bars, {self.n_hinges} hinges")
        print(f"[EquilibriumAnalyzer] DOF = {self.n_dof}")

    def _build_energy_function(self):
        """
        Build JAX potential energy V(q) where q is the reduced coordinate vector
        (free node positions only). Fixed nodes are baked in as constants.
        """
        # Convert to JAX arrays
        mass_j = jnp.array(self.masses, dtype=jnp.float32)
        x0_fixed = jnp.array(self.x0[self.fixed_mask], dtype=jnp.float32)
        free_idx = jnp.array(self.free_indices, dtype=jnp.int32)
        fixed_idx = jnp.array(self.fixed_indices, dtype=jnp.int32)

        n_nodes = self.n_nodes
        gravity = self.gravity

        # Bar data
        n_bars = self.n_bars
        if n_bars > 0:
            bar_idx = jnp.array(self.bar_indices, dtype=jnp.int32)
            bar_k = jnp.array(self.bar_k, dtype=jnp.float32)
            bar_l0 = jnp.array(self.bar_l0, dtype=jnp.float32)
            bar_prestress = jnp.array(self.bar_prestress, dtype=jnp.float32)

        # Hinge data
        n_hinges = self.n_hinges
        if n_hinges > 0:
            hinge_idx = jnp.array(self.hinge_indices, dtype=jnp.int32)
            hinge_k = jnp.array(self.hinge_k, dtype=jnp.float32)
            hinge_phi0 = jnp.array(self.hinge_phi0, dtype=jnp.float32)

        def _reconstruct_full(q_free):
            """Reconstruct full (n_nodes, 3) positions from free DOFs."""
            x_full = jnp.zeros((n_nodes, 3), dtype=jnp.float32)
            # Place free nodes
            q_reshaped = q_free.reshape(-1, 3)
            x_full = x_full.at[free_idx].set(q_reshaped)
            # Place fixed nodes
            if len(fixed_idx) > 0:
                x_full = x_full.at[fixed_idx].set(x0_fixed)
            return x_full

        def potential_energy(q_free):
            """Total potential energy as function of free DOFs only."""
            x = _reconstruct_full(q_free)
            pe = 0.0

            # Gravity
            pe += jnp.sum(mass_j * (-gravity) * x[:, 2])

            # Bars: V = 0.5 * k * (L - L_eff)^2
            if n_bars > 0:
                xi = x[bar_idx[:, 0]]
                xj = x[bar_idx[:, 1]]
                dx = xj - xi
                dist = jnp.linalg.norm(dx, axis=1) + 1e-12
                l_eff = bar_l0 * (1.0 + bar_prestress)
                pe += 0.5 * jnp.sum(bar_k * (dist - l_eff) ** 2)

            # Hinges: V = 0.5 * k * (phi - phi0)^2
            if n_hinges > 0:
                xj = x[hinge_idx[:, 0]]
                xk = x[hinge_idx[:, 1]]
                xi = x[hinge_idx[:, 2]]
                xl = x[hinge_idx[:, 3]]

                r_ij = xi - xj
                r_kj = xk - xj
                r_kl = xk - xl

                m = jnp.cross(r_ij, r_kj)
                n = jnp.cross(r_kj, r_kl)

                len_m = jnp.linalg.norm(m, axis=1) + 1e-12
                len_n = jnp.linalg.norm(n, axis=1) + 1e-12

                m_hat = m / len_m[:, None]
                n_hat = n / len_n[:, None]

                cos_phi = jnp.clip(jnp.sum(m_hat * n_hat, axis=1), -1.0, 1.0)

                len_rkj = jnp.linalg.norm(r_kj, axis=1) + 1e-12
                rkj_hat = r_kj / len_rkj[:, None]
                sin_phi = jnp.sum(jnp.cross(m_hat, n_hat) * rkj_hat, axis=1)

                phi = jnp.arctan2(sin_phi, cos_phi)
                delta = phi - hinge_phi0
                delta = jnp.arctan2(jnp.sin(delta), jnp.cos(delta))

                pe += 0.5 * jnp.sum(hinge_k * delta ** 2)

            return pe

        # Store functions
        self._potential_energy = potential_energy
        self._reconstruct_full = _reconstruct_full

        # Build gradient and Hessian via AD
        self._gradient = jit(grad(potential_energy))
        self._hessian = jit(jacfwd(grad(potential_energy)))

        # JIT the energy itself
        self._energy_jit = jit(potential_energy)

        # Warm up JIT
        q0 = jnp.array(self.x0[self.free_mask].flatten(), dtype=jnp.float32)
        _ = self._energy_jit(q0)
        _ = self._gradient(q0)
        print(f"[EquilibriumAnalyzer] Energy function built and JIT-compiled "
              f"(V(x0) = {float(self._energy_jit(q0)):.6f})")

    def _q_from_positions(self, x_full: np.ndarray) -> jnp.ndarray:
        """Extract free DOF vector from full position array."""
        return jnp.array(x_full[self.free_mask].flatten(), dtype=jnp.float32)

    def _positions_from_q(self, q: jnp.ndarray) -> np.ndarray:
        """Reconstruct full positions from free DOF vector."""
        return np.array(self._reconstruct_full(q))

    # =========================================================================
    # Newton Solver with Deflation
    # =========================================================================

    def _newton_solve(
            self,
            q0: jnp.ndarray,
            known_roots: List[jnp.ndarray],
            max_iter: int = 200,
            tol: float = 1e-8,
            deflation_power: float = 2.0,
            line_search: bool = True,
    ) -> Optional[Equilibrium]:
        """
        Deflated Newton iteration to find a root of ∇V(q) = 0.

        Deflation operator M(q; {q*_i}) = ∏_i ||q - q*_i||^p
        We solve: ∇V(q) / M(q) = 0 (same roots as ∇V except known ones).

        In practice, we modify the Newton direction rather than the residual:
            H * Δq = -g / M(q)
        where g = ∇V, H = ∇²V.

        Parameters
        ----------
        q0 : initial guess (free DOFs)
        known_roots : list of previously found equilibria (free DOFs)
        max_iter : maximum Newton iterations
        tol : convergence tolerance on ||∇V||
        deflation_power : exponent p in deflation operator
        line_search : whether to use backtracking line search
        """
        q = q0.copy()

        for iteration in range(max_iter):
            g = self._gradient(q)
            g_norm = float(jnp.linalg.norm(g))

            # Check convergence on the undeflated gradient
            if g_norm < tol:
                return self._classify_equilibrium(q, g_norm, iteration, converged=True)

            # Compute deflation factor
            deflation = 1.0
            for qstar in known_roots:
                dist_sq = float(jnp.sum((q - qstar) ** 2))
                if dist_sq < 1e-20:
                    # Too close to known root, bail out
                    return None
                deflation *= dist_sq ** (deflation_power / 2.0)

            # Hessian
            H = self._hessian(q)

            # Deflated RHS: solve H @ dq = -g / M
            rhs = -g / (deflation + 1e-30)

            # Regularized solve (Hessian may be singular near bifurcations)
            try:
                # Add small regularization for robustness
                reg = 1e-8 * jnp.eye(self.n_dof)
                dq = jnp.linalg.solve(H + reg, rhs)
            except Exception:
                return None

            # Check for NaN
            if jnp.any(jnp.isnan(dq)):
                return None

            # Backtracking line search on ||∇V||²
            if line_search:
                alpha = 1.0
                current_merit = g_norm ** 2
                for _ in range(20):
                    q_trial = q + alpha * dq
                    g_trial = self._gradient(q_trial)
                    trial_merit = float(jnp.sum(g_trial ** 2))
                    if trial_merit < current_merit * (1 - 1e-4 * alpha):
                        break
                    alpha *= 0.5
                else:
                    # Line search failed, take small step anyway
                    alpha = 0.01
                q = q + alpha * dq
            else:
                q = q + dq

            # Clamp to prevent explosion
            q = jnp.clip(q, -100.0, 100.0)

        # Did not converge within max_iter
        g_final = self._gradient(q)
        g_norm = float(jnp.linalg.norm(g_final))
        if g_norm < tol * 100:  # Near-converged
            return self._classify_equilibrium(q, g_norm, max_iter, converged=False)
        return None

    def _classify_equilibrium(
            self, q: jnp.ndarray, residual: float, n_iter: int, converged: bool
    ) -> Equilibrium:
        """Compute Hessian eigenvalues and classify stability."""
        H = self._hessian(q)
        H_np = np.array(H)

        # Symmetrize (numerical noise)
        H_np = 0.5 * (H_np + H_np.T)

        eigenvalues, eigenvectors = np.linalg.eigh(H_np)

        # Classification thresholds
        zero_tol = 1e-6 * max(1.0, abs(eigenvalues[-1]))  # relative to largest
        n_negative = int(np.sum(eigenvalues < -zero_tol))
        n_zero = int(np.sum(np.abs(eigenvalues) <= zero_tol))
        n_positive = int(np.sum(eigenvalues > zero_tol))

        if n_negative == 0 and n_zero == 0:
            stability = StabilityType.STABLE
        elif n_negative == 0 and n_zero > 0:
            stability = StabilityType.MARGINAL
        elif n_positive == 0 and n_zero == 0:
            stability = StabilityType.UNSTABLE_MAXIMUM
        else:
            stability = StabilityType.UNSTABLE_SADDLE

        return Equilibrium(
            positions=self._positions_from_q(q),
            energy=float(self._energy_jit(q)),
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            stability_type=stability,
            n_negative=n_negative,
            n_zero=n_zero,
            converged=converged,
            residual=residual,
            n_iterations=n_iter,
        )

    # =========================================================================
    # Initial Guess Strategies
    # =========================================================================

    def _generate_initial_guesses(
            self,
            n_attempts: int,
            strategies: List[str],
            perturbation_scale: float = 0.1,
            rng: np.random.Generator = None,
    ) -> List[jnp.ndarray]:
        """
        Generate diverse initial guesses for Newton solver.

        Strategies
        ----------
        'reference' : Start from the reference (undeformed) configuration.
        'random'    : Random perturbations around reference.
        'kinematic' : Perturbations along known kinematic modes (e.g., folding
                      direction for origami). Uses the softest eigenmodes of
                      the Hessian at the reference config.
        'compressed': Systematically compressed/extended configurations along z.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        q_ref = self._q_from_positions(self.x0)
        guesses = []

        n_per_strategy = max(1, n_attempts // len(strategies))

        for strategy in strategies:
            if strategy == "reference":
                guesses.append(q_ref)

            elif strategy == "random":
                for _ in range(n_per_strategy):
                    perturbation = rng.normal(0, perturbation_scale, size=self.n_dof)
                    guesses.append(q_ref + jnp.array(perturbation, dtype=jnp.float32))

            elif strategy == "kinematic":
                # Use soft modes of Hessian at reference as perturbation directions
                H0 = np.array(self._hessian(q_ref))
                H0 = 0.5 * (H0 + H0.T)
                eigvals, eigvecs = np.linalg.eigh(H0)

                # Take the softest modes (smallest |eigenvalue|)
                n_modes = min(6, self.n_dof)
                soft_modes = eigvecs[:, :n_modes]

                for _ in range(n_per_strategy):
                    # Random combination of soft modes
                    coeffs = rng.normal(0, perturbation_scale, size=n_modes)
                    perturbation = soft_modes @ coeffs
                    guesses.append(q_ref + jnp.array(perturbation, dtype=jnp.float32))

            elif strategy == "compressed":
                # Systematic compression along z-axis
                for frac in np.linspace(0.1, 0.9, n_per_strategy):
                    q_compressed = q_ref.copy()
                    # Scale z-coordinates of free nodes
                    q_reshaped = np.array(q_compressed).reshape(-1, 3)
                    z_range = q_reshaped[:, 2].max() - q_reshaped[:, 2].min()
                    if z_range > 1e-6:
                        z_mid = 0.5 * (q_reshaped[:, 2].max() + q_reshaped[:, 2].min())
                        q_reshaped[:, 2] = z_mid + (q_reshaped[:, 2] - z_mid) * frac
                    guesses.append(jnp.array(q_reshaped.flatten(), dtype=jnp.float32))

            else:
                raise ValueError(f"Unknown strategy: {strategy}")

        return guesses

    # =========================================================================
    # Deduplication
    # =========================================================================

    def _is_duplicate(
            self,
            q_new: jnp.ndarray,
            known: List[jnp.ndarray],
            tol: float = 1e-4,
    ) -> bool:
        """Check if q_new is close to any known root."""
        for qk in known:
            dist = float(jnp.linalg.norm(q_new - qk))
            if dist < tol:
                return True
        return False

    # =========================================================================
    # Main API
    # =========================================================================

    def find_equilibria(
            self,
            n_attempts: int = 100,
            strategies: Optional[List[str]] = None,
            perturbation_scale: float = 0.1,
            deflation: bool = True,
            tol: float = 1e-8,
            max_newton_iter: int = 200,
            dedup_tol: float = 1e-4,
            seed: int = 42,
            verbose: bool = True,
    ) -> EquilibriumSet:
        """
        Find multiple equilibria of the system.

        Parameters
        ----------
        n_attempts : int
            Number of initial guesses to try.
        strategies : list of str
            Initial guess strategies. Default: ['reference', 'random', 'kinematic', 'compressed']
        perturbation_scale : float
            Scale of random perturbations relative to system size.
        deflation : bool
            Use deflation to avoid reconverging to known roots.
        tol : float
            Convergence tolerance for ||∇V||.
        max_newton_iter : int
            Maximum Newton iterations per attempt.
        dedup_tol : float
            Distance tolerance for considering two equilibria identical.
        seed : int
            Random seed for reproducibility.
        verbose : bool
            Print progress.

        Returns
        -------
        EquilibriumSet
            Collection of found equilibria with metadata.
        """
        if strategies is None:
            strategies = ["reference", "random", "kinematic", "compressed"]

        rng = np.random.default_rng(seed)
        t0 = time.time()

        # Generate initial guesses
        guesses = self._generate_initial_guesses(
            n_attempts, strategies, perturbation_scale, rng
        )

        if verbose:
            print(f"\n[EquilibriumAnalyzer] Starting search with {len(guesses)} initial guesses")
            print(f"  Strategies: {strategies}")
            print(f"  Deflation: {deflation}")
            print(f"  Tolerance: {tol}")

        found_roots: List[jnp.ndarray] = []  # q vectors of found equilibria
        results: List[Equilibrium] = []

        for i, q0 in enumerate(guesses):
            known = found_roots if deflation else []

            eq = self._newton_solve(
                q0, known,
                max_iter=max_newton_iter,
                tol=tol,
                line_search=True,
            )

            if eq is not None and eq.converged:
                q_found = self._q_from_positions(eq.positions)

                if not self._is_duplicate(q_found, found_roots, tol=dedup_tol):
                    found_roots.append(q_found)
                    results.append(eq)

                    if verbose:
                        print(
                            f"  [{i+1}/{len(guesses)}] NEW equilibrium #{len(results)}: "
                            f"E={eq.energy:.6f}, type={eq.stability_type.value}, "
                            f"morse={eq.morse_index}"
                        )
            elif verbose and (i + 1) % max(1, len(guesses) // 10) == 0:
                print(f"  [{i+1}/{len(guesses)}] ... ({len(results)} found so far)")

        elapsed = time.time() - t0

        result_set = EquilibriumSet(
            equilibria=results,
            parameters={
                "n_attempts": n_attempts,
                "strategies": strategies,
                "perturbation_scale": perturbation_scale,
                "deflation": deflation,
                "tol": tol,
                "seed": seed,
            },
            timing=elapsed,
        )

        if verbose:
            print(f"\n{result_set.summary()}")

        return result_set

    # =========================================================================
    # Convenience: Energy Landscape
    # =========================================================================

    def energy_along_mode(
            self,
            mode_index: int = 0,
            amplitude_range: Tuple[float, float] = (-1.0, 1.0),
            n_points: int = 200,
            base_config: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute energy along a specific eigenmode of the Hessian.

        Useful for 1D energy landscape visualization and understanding
        the shape of the energy surface near an equilibrium.

        Parameters
        ----------
        mode_index : int
            Which eigenmode (0 = softest).
        amplitude_range : tuple
            (min, max) amplitude along the mode.
        n_points : int
            Number of sample points.
        base_config : np.ndarray, optional
            Base configuration (full positions). Default: reference config.

        Returns
        -------
        amplitudes : np.ndarray
            Parameter values along the mode.
        energies : np.ndarray
            V(x0 + alpha * mode) for each alpha.
        """
        if base_config is None:
            q_base = self._q_from_positions(self.x0)
        else:
            q_base = self._q_from_positions(base_config)

        # Get mode direction
        H = np.array(self._hessian(q_base))
        H = 0.5 * (H + H.T)
        eigvals, eigvecs = np.linalg.eigh(H)
        mode = eigvecs[:, mode_index]

        amplitudes = np.linspace(amplitude_range[0], amplitude_range[1], n_points)
        energies = np.zeros(n_points)

        for i, alpha in enumerate(amplitudes):
            q = q_base + alpha * jnp.array(mode, dtype=jnp.float32)
            energies[i] = float(self._energy_jit(q))

        return amplitudes, energies

    def energy_landscape_2d(
            self,
            mode_i: int = 0,
            mode_j: int = 1,
            amplitude_range: Tuple[float, float] = (-1.0, 1.0),
            n_points: int = 50,
            base_config: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute 2D energy landscape over two eigenmodes.

        Returns
        -------
        A_i, A_j : np.ndarray
            Meshgrid of amplitudes.
        E : np.ndarray
            Energy at each grid point.
        """
        if base_config is None:
            q_base = self._q_from_positions(self.x0)
        else:
            q_base = self._q_from_positions(base_config)

        H = np.array(self._hessian(q_base))
        H = 0.5 * (H + H.T)
        eigvals, eigvecs = np.linalg.eigh(H)
        vi = eigvecs[:, mode_i]
        vj = eigvecs[:, mode_j]

        a = np.linspace(amplitude_range[0], amplitude_range[1], n_points)
        A_i, A_j = np.meshgrid(a, a)
        E = np.zeros_like(A_i)

        for ii in range(n_points):
            for jj in range(n_points):
                q = q_base + A_i[ii, jj] * jnp.array(vi) + A_j[ii, jj] * jnp.array(vj)
                E[ii, jj] = float(self._energy_jit(q))

        return A_i, A_j, E