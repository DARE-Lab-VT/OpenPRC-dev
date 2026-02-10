"""
Equilibrium Finder for DEMLAT Bar-Hinge Models
================================================

Finds all fixed points of a conservative bar-hinge system by solving:

    ∇V(x) = 0

where V(x) is the total potential energy (gravity + elastic bars + elastic hinges).

Method: Deflated Newton-Raphson
    1. Find a root x*_1 via Newton
    2. Deflate: modify the system so Newton is repelled from x*_1
    3. Repeat to find x*_2, x*_3, ...

Reference: Farrell, Birkisson & Funke (2015)
    "Deflation Techniques for Finding Distinct Solutions of Nonlinear PDEs"

Usage:
    finder = EquilibriumFinder.from_experiment("experiments/yoshimura_test")
    results = finder.find_all()
    finder.save_results(results, "equilibria.h5")
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad, jacobian, hessian
from functools import partial
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import h5py
import json
import time


# ============================================================
# Data Structures
# ============================================================

@dataclass
class Equilibrium:
    """A single equilibrium state."""
    positions: np.ndarray  # (n_nodes, 3)
    energy: float  # V(x*)
    eigenvalues: np.ndarray  # of the Hessian (reduced space)
    stability: str  # 'stable', 'unstable', 'saddle'
    index: int  # number of negative eigenvalues (Morse index)
    residual: float  # ||∇V(x*)||
    n_iterations: int  # Newton iterations to converge
    source_guess: str  # label for the initial guess that found this


@dataclass
class FinderResults:
    """Collection of all found equilibria."""
    equilibria: List[Equilibrium] = field(default_factory=list)
    n_total_attempts: int = 0
    n_converged: int = 0
    wall_time: float = 0.0
    metadata: Dict = field(default_factory=dict)

    @property
    def n_stable(self):
        return sum(1 for eq in self.equilibria if eq.stability == 'stable')

    @property
    def n_unstable(self):
        return sum(1 for eq in self.equilibria if eq.stability != 'stable')

    def summary(self):
        print(f"\n{'=' * 60}")
        print(f"  Equilibrium Finder Results")
        print(f"{'=' * 60}")
        print(f"  Total attempts:     {self.n_total_attempts}")
        print(f"  Converged:          {self.n_converged}")
        print(f"  Unique equilibria:  {len(self.equilibria)}")
        print(f"  Stable:             {self.n_stable}")
        print(f"  Unstable:           {self.n_unstable}")
        print(f"  Wall time:          {self.wall_time:.2f}s")
        print(f"{'=' * 60}")
        for i, eq in enumerate(self.equilibria):
            neg = eq.index
            print(f"  [{i}] E={eq.energy:+.6f}  "
                  f"stability={eq.stability:<8s}  "
                  f"morse_index={neg}  "
                  f"residual={eq.residual:.2e}  "
                  f"from={eq.source_guess}")
        print()


# ============================================================
# Potential Energy Builder
# ============================================================

class PotentialEnergyBuilder:
    """
    Builds a JAX-differentiable potential energy function V(x)
    from DEMLAT geometry data.

    This is intentionally decoupled from JaxSolver so it can be
    used standalone for equilibrium analysis.
    """

    def __init__(self, geometry: Dict, physics: Dict):
        """
        Args:
            geometry: dict with keys 'nodes', 'masses', 'bars', 'hinges'
                      (matching DEMLAT HDF5 schema)
            physics:  dict with 'gravity' (float)
        """
        self.n_nodes = len(geometry['nodes'])
        self.ref_positions = np.array(geometry['nodes'], dtype=np.float32)
        self.mass = jnp.array(geometry.get('masses', np.ones(self.n_nodes) * 0.01),
                              dtype=jnp.float32)
        self.gravity = float(physics.get('gravity', 0.0))

        # --- Parse Bars ---
        self.n_bars = 0
        bars = geometry.get('bars', {})
        if bars and len(bars.get('indices', [])) > 0:
            idx = np.array(bars['indices'])
            k = np.array(bars['stiffness'])
            soft = k >= 0
            if np.any(soft):
                self.n_bars = int(np.sum(soft))
                self.bar_indices = jnp.array(idx[soft], dtype=jnp.int32)
                self.bar_k = jnp.array(k[soft], dtype=jnp.float32)
                self.bar_l0 = jnp.array(bars['rest_length'][soft], dtype=jnp.float32)

                if 'prestress' in bars and bars['prestress'] is not None:
                    self.bar_prestress = jnp.array(bars['prestress'][soft], dtype=jnp.float32)
                else:
                    self.bar_prestress = jnp.zeros(self.n_bars, dtype=jnp.float32)

        # --- Parse Hinges ---
        self.n_hinges = 0
        hinges = geometry.get('hinges', {})
        if hinges and len(hinges.get('indices', [])) > 0:
            idx = np.array(hinges['indices'])
            k = np.array(hinges['stiffness'])
            soft = k >= 0
            if np.any(soft):
                self.n_hinges = int(np.sum(soft))
                self.hinge_indices = jnp.array(idx[soft], dtype=jnp.int32)
                self.hinge_k = jnp.array(k[soft], dtype=jnp.float32)

                # Handle both key conventions: 'rest_angle' (internal) or 'angle' (HDF5)
                angle_data = hinges.get('rest_angle', hinges.get('angle'))
                if angle_data is None:
                    raise KeyError("Hinge data missing both 'rest_angle' and 'angle' keys")
                self.hinge_phi0 = jnp.array(np.array(angle_data)[soft], dtype=jnp.float32)

        print(f"[PotentialEnergyBuilder] {self.n_nodes} nodes, "
              f"{self.n_bars} bars, {self.n_hinges} hinges, "
              f"gravity={self.gravity}")

    def build(self):
        """
        Returns a pure JAX function: V(x_flat) -> scalar
        where x_flat is positions flattened to (n_nodes * 3,).

        This function is compatible with jax.grad, jax.hessian, etc.
        """
        n_nodes = self.n_nodes
        mass = self.mass
        gravity = self.gravity
        n_bars = self.n_bars
        n_hinges = self.n_hinges

        # Capture element data in closure
        if n_bars > 0:
            bar_idx = self.bar_indices
            bar_k = self.bar_k
            bar_l0 = self.bar_l0
            bar_ps = self.bar_prestress

        if n_hinges > 0:
            h_idx = self.hinge_indices
            h_k = self.hinge_k
            h_phi0 = self.hinge_phi0

        def potential_energy(x_flat):
            x = x_flat.reshape(n_nodes, 3)
            pe = 0.0

            # --- Gravity ---
            if gravity != 0.0:
                pe += jnp.sum(mass * (-gravity) * x[:, 2])

            # --- Bars: 0.5 * k * (L - L0_eff)^2 ---
            if n_bars > 0:
                xi = x[bar_idx[:, 0]]
                xj = x[bar_idx[:, 1]]
                dx = xj - xi
                dist = jnp.linalg.norm(dx, axis=1)
                l_eff = bar_l0 * (1.0 + bar_ps)
                pe += 0.5 * jnp.sum(bar_k * (dist - l_eff) ** 2)

            # --- Hinges: 0.5 * k * wrap(phi - phi0)^2 ---
            if n_hinges > 0:
                # Index convention from solver_jax.py:
                #   [j, k, i, l] -> edge j-k, faces (i,j,k) and (k,j,l)
                j_idx = h_idx[:, 0]
                k_idx = h_idx[:, 1]
                i_idx = h_idx[:, 2]
                l_idx = h_idx[:, 3]

                xj = x[j_idx]
                xk = x[k_idx]
                xi = x[i_idx]
                xl = x[l_idx]

                r_ij = xi - xj
                r_kj = xk - xj
                r_kl = xk - xl

                m_vec = jnp.cross(r_ij, r_kj)
                n_vec = jnp.cross(r_kj, r_kl)

                len_m = jnp.linalg.norm(m_vec, axis=1) + 1e-12
                len_n = jnp.linalg.norm(n_vec, axis=1) + 1e-12

                m_hat = m_vec / len_m[:, None]
                n_hat = n_vec / len_n[:, None]

                cos_phi = jnp.sum(m_hat * n_hat, axis=1)
                cos_phi = jnp.clip(cos_phi, -1.0, 1.0)

                len_rkj = jnp.linalg.norm(r_kj, axis=1) + 1e-12
                rkj_hat = r_kj / len_rkj[:, None]
                sin_phi = jnp.sum(jnp.cross(m_hat, n_hat) * rkj_hat, axis=1)

                phi = jnp.arctan2(sin_phi, cos_phi)
                delta = phi - h_phi0
                delta = jnp.arctan2(jnp.sin(delta), jnp.cos(delta))

                pe += 0.5 * jnp.sum(h_k * delta ** 2)

            return pe

        return potential_energy


# ============================================================
# Deflated Newton Solver
# ============================================================

class DeflatedNewtonSolver:
    """
    Finds multiple roots of ∇V(x) = 0 for stiff bar-hinge systems using a
    two-phase approach with rigid body mode projection and deflation.

    Phase 1: Energy minimization via projected gradient descent with momentum
             (handles the severe ill-conditioning from k_bar >> k_hinge)
    Phase 2: Newton-Raphson polishing to machine precision

    Rigid body modes are projected out throughout. Deflation (Farrell et al.
    2015) prevents re-convergence to known roots.
    """

    def __init__(self,
                 energy_fn,  # V(x) -> scalar
                 residual_fn,  # g(x) = ∇V(x) -> (n,)
                 n_nodes: int,
                 ref_positions,  # (n_nodes, 3) reference config for null space
                 deflation_power: float = 2.0,
                 tol: float = 1e-6,
                 max_iter: int = 2000,
                 uniqueness_tol: float = 1e-3):
        self.V = energy_fn
        self.g = residual_fn
        self.n_nodes = n_nodes
        self.dim = n_nodes * 3
        self.p = deflation_power
        self.tol = tol
        self.max_iter = max_iter
        self.uniqueness_tol = uniqueness_tol

        # JIT compile
        self._energy_fn = jit(energy_fn)
        self._res_fn = jit(residual_fn)
        self._jac_fn = jit(jacobian(residual_fn))

        # Build rigid body mode projector
        self._build_null_space_projector(ref_positions)

        # Known roots for deflation
        self.known_roots: List[jnp.ndarray] = []

    def _build_null_space_projector(self, ref_positions):
        """
        Build projector P = I - N @ N^T that removes rigid body modes.
        N is orthonormal basis for 3 translations + 3 rotations.
        """
        pos = np.array(ref_positions, dtype=np.float64).reshape(self.n_nodes, 3)
        centroid = pos.mean(axis=0)
        r = pos - centroid

        null_modes = []
        for axis in range(3):
            mode = np.zeros((self.n_nodes, 3), dtype=np.float64)
            mode[:, axis] = 1.0
            null_modes.append(mode.flatten())

        for axis in range(3):
            omega = np.zeros(3, dtype=np.float64)
            omega[axis] = 1.0
            mode = np.cross(omega, r)
            null_modes.append(mode.flatten())

        N = np.column_stack(null_modes)
        Q, _ = np.linalg.qr(N)

        self._null_basis = jnp.array(Q, dtype=jnp.float64)
        self._n_rigid = Q.shape[1]
        print(f"  [Null space] {self._n_rigid} rigid body modes projected out")

    def _project(self, v):
        """Remove rigid body components from vector v."""
        coeffs = self._null_basis.T @ v
        return v - self._null_basis @ coeffs

    def _deflation_weight(self, x):
        """
        Deflation penalty added to energy to repel from known roots.
        W(x) = sum_i  sigma / ||P(x - x*_i)||^p
        """
        if len(self.known_roots) == 0:
            return 0.0
        w = 0.0
        for x_star in self.known_roots:
            diff = self._project(x - x_star)
            dist = jnp.sqrt(jnp.sum(diff ** 2) + 1e-30)
            w += 1.0 / (dist ** self.p + 1e-12)
        return w

    def _is_duplicate(self, x_new):
        """Check if x_new matches any known root (in deformation space)."""
        x_proj = self._project(x_new)
        for x_star in self.known_roots:
            x_star_proj = self._project(x_star)
            if float(jnp.linalg.norm(x_proj - x_star_proj)) < self.uniqueness_tol:
                return True
        return False

    def solve_from(self, x0, label="unknown"):
        """
        Find equilibrium from initial guess x0 using two-phase approach.

        Phase 1: Projected gradient descent with adaptive step size.
                 Robust to ill-conditioning. Runs until residual is small.
        Phase 2: Projected Newton-Raphson for quadratic convergence.
                 Kicks in when close enough for Newton to work.

        Returns (result_dict, converged) or (None, False).
        """
        x = jnp.array(x0, dtype=jnp.float64)

        # ---- Phase 1: Gradient descent with momentum ----
        newton_threshold = self.tol * 1e3  # switch to Newton when residual < this
        lr = 1e-2  # initial learning rate
        momentum = 0.9
        v = jnp.zeros_like(x)
        best_res = float('inf')
        stall_count = 0

        phase1_iters = int(self.max_iter * 0.8)  # budget 80% for phase 1

        for iteration in range(phase1_iters):
            gx = self._res_fn(x)
            gx_proj = self._project(gx)
            res_norm = float(jnp.linalg.norm(gx_proj))

            if res_norm < self.tol:
                if self._is_duplicate(x):
                    return None, False
                return {
                    'x': np.array(x), 'residual': res_norm,
                    'n_iter': iteration, 'label': label
                }, True

            if res_norm < newton_threshold:
                break  # switch to phase 2

            # Add deflation gradient to repel from known roots
            if len(self.known_roots) > 0:
                deflation_grad = jnp.zeros_like(x)
                for x_star in self.known_roots:
                    diff = self._project(x - x_star)
                    dist_sq = jnp.sum(diff ** 2) + 1e-20
                    # Gradient of 1/||d||^p = -p * d / ||d||^{p+2}
                    deflation_grad += -self.p * diff / (dist_sq ** (self.p / 2.0 + 1.0))
                deflation_strength = min(0.1 * res_norm, 1.0)
                gx_proj = gx_proj + deflation_strength * self._project(deflation_grad)

            # Adaptive learning rate via line search on energy
            E_curr = float(self._energy_fn(x))

            # Momentum update
            v = momentum * v - lr * gx_proj
            v = self._project(v)  # keep in deformation subspace

            x_trial = x + v
            E_trial = float(self._energy_fn(x_trial))

            if E_trial < E_curr:
                x = x_trial
                lr = min(lr * 1.05, 1.0)  # speed up
            else:
                # Backtrack
                v = jnp.zeros_like(x)
                lr *= 0.5
                x = x - lr * gx_proj
                x = x  # keep current position, reduce lr

            # Track progress
            if res_norm < best_res * 0.999:
                best_res = res_norm
                stall_count = 0
            else:
                stall_count += 1

            if stall_count > 200:
                # Completely stalled — give up on this guess
                return None, False

            if lr < 1e-15:
                return None, False

        # ---- Phase 2: Newton-Raphson polishing ----
        phase2_iters = self.max_iter - phase1_iters
        mu = 1e-4  # LM regularization

        for iteration in range(phase2_iters):
            gx = self._res_fn(x)
            gx_proj = self._project(gx)
            res_norm = float(jnp.linalg.norm(gx_proj))

            if res_norm < self.tol:
                if self._is_duplicate(x):
                    return None, False
                total_iter = phase1_iters + iteration
                return {
                    'x': np.array(x), 'residual': res_norm,
                    'n_iter': total_iter, 'label': label
                }, True

            J = self._jac_fn(x)

            # Projected regularized Hessian
            N = self._null_basis
            null_penalty = 1e6
            J_reg = J + mu * jnp.eye(self.dim) + null_penalty * (N @ N.T)

            try:
                delta = jnp.linalg.solve(J_reg, gx_proj)
            except Exception:
                mu *= 10
                if mu > 1e8:
                    return None, False
                continue

            delta = self._project(delta)

            if not jnp.isfinite(delta).all():
                return None, False

            # Backtracking line search
            alpha = 1.0
            accepted = False
            for _ in range(30):
                x_trial = x - alpha * delta
                if not jnp.isfinite(x_trial).all():
                    alpha *= 0.5
                    continue

                gx_trial = self._res_fn(x_trial)
                trial_norm = float(jnp.linalg.norm(self._project(gx_trial)))

                if trial_norm < res_norm:
                    x = x_trial
                    mu = max(mu * 0.1, 1e-12)
                    accepted = True
                    break
                alpha *= 0.5

            if not accepted:
                mu = min(mu * 10, 1e6)

        return None, False

    def find_from_guesses(self, guesses, labels=None):
        """Run solver from multiple initial guesses with deflation."""
        if labels is None:
            labels = [f"guess_{i}" for i in range(len(guesses))]

        found = []
        for i, (x0, label) in enumerate(zip(guesses, labels)):
            result, converged = self.solve_from(x0, label=label)

            if converged and result is not None:
                x_root = jnp.array(result['x'], dtype=jnp.float64)
                self.known_roots.append(x_root)
                found.append(result)
                E = float(self._energy_fn(x_root))
                print(f"  [Root {len(found)}] from {label:25s} "
                      f"E={E:+.8f}  res={result['residual']:.2e}  "
                      f"iter={result['n_iter']}")

        return found


# ============================================================
# Main Finder Class
# ============================================================

class EquilibriumFinder:
    """
    Top-level interface for finding equilibria of a DEMLAT bar-hinge system.

    Usage:
        finder = EquilibriumFinder.from_experiment("experiments/yoshimura_test")
        results = finder.find_all(num_random=50)
        results.summary()
        finder.save_results(results, "equilibria.h5")
    """

    def __init__(self, geometry: Dict, physics: Dict):
        self.geometry = geometry
        self.physics = physics

        self.n_nodes = len(geometry['nodes'])
        self.dim = self.n_nodes * 3
        self.ref_positions = np.array(geometry['nodes'], dtype=np.float64)

        # Build potential energy (float64 for Newton precision)
        builder = PotentialEnergyBuilder(geometry, physics)
        V_f32 = builder.build()

        # Wrap to accept float64 and promote internally
        def V_f64(x_flat):
            return V_f32(x_flat.astype(jnp.float32)).astype(jnp.float64)

        # Actually, better: rebuild with float64 throughout
        # For now, just use the f32 version — JAX will handle promotion
        self.V = jit(V_f32)
        self.grad_V = jit(grad(V_f32))
        self.hess_V = jit(hessian(V_f32))

        # Verify at reference config
        x0 = jnp.array(self.ref_positions.flatten(), dtype=jnp.float32)
        v0 = self.V(x0)
        g0 = self.grad_V(x0)
        print(f"[EquilibriumFinder] Reference config: V={float(v0):.6f}, "
              f"||∇V||={float(jnp.linalg.norm(g0)):.6f}")

    @classmethod
    def from_experiment(cls, experiment_dir, config_overrides=None):
        """
        Load geometry and physics from a DEMLAT experiment directory.

        Reads:
            input/geometry.h5  — nodes, bars, hinges
            input/config.json  — physics params
        """
        root = Path(experiment_dir)
        geom_path = root / "input" / "geometry.h5"
        config_path = root / "input" / "config.json"

        if not geom_path.exists():
            raise FileNotFoundError(f"Geometry not found: {geom_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        # --- Load geometry ---
        geometry = {}
        with h5py.File(geom_path, 'r') as f:
            geometry['nodes'] = f['nodes/positions'][:]

            if 'nodes/masses' in f:
                geometry['masses'] = f['nodes/masses'][:]

            if 'elements/bars/indices' in f:
                geometry['bars'] = {
                    'indices': f['elements/bars/indices'][:].astype(np.int32),
                }
                for attr in ['stiffness', 'rest_length', 'prestress']:
                    key = f'elements/bars/{attr}'
                    if key in f:
                        geometry['bars'][attr] = f[key][:]

            if 'elements/hinges/indices' in f:
                geometry['hinges'] = {
                    'indices': f['elements/hinges/indices'][:].astype(np.int32),
                }
                for attr in ['stiffness', 'damping']:
                    key = f'elements/hinges/{attr}'
                    if key in f:
                        geometry['hinges'][attr] = f[key][:]

                # Rest angle: stored as 'angle' in HDF5
                if 'elements/hinges/angle' in f:
                    geometry['hinges']['rest_angle'] = f['elements/hinges/angle'][:]
                elif 'elements/hinges/rest_angle' in f:
                    geometry['hinges']['rest_angle'] = f['elements/hinges/rest_angle'][:]

        # --- Load physics ---
        with open(config_path, 'r') as f:
            config = json.load(f)

        physics = config.get('global_physics', config.get('physics', {}))

        if config_overrides:
            physics.update(config_overrides)

        return cls(geometry, physics)

    # ----------------------------------------------------------
    # Initial Guess Generation
    # ----------------------------------------------------------

    def _generate_guesses(self, num_random=50, perturbation_scale=0.1,
                          include_compressed=True, n_intermediate=10):
        """
        Generate physics-informed initial guesses.

        For origami structures, equilibria lie on or near the kinematic
        manifold (configurations with bars at rest length). We bias
        guesses toward this manifold by:
          1. Z-scaling (folding/unfolding the cylinder)
          2. Radial perturbation of mid-ring nodes (origami breathing mode)
          3. Small random perturbations (for nearby saddle points)
        """
        guesses = []
        labels = []
        x_ref = self.ref_positions.copy()  # (n_nodes, 3)
        n_nodes = self.n_nodes

        # 1. Reference config
        guesses.append(x_ref.flatten().copy())
        labels.append("reference")

        # 2. Z-scaled (folding/unfolding): scale z toward/away from midplane
        if include_compressed:
            z_min, z_max = x_ref[:, 2].min(), x_ref[:, 2].max()
            z_mid = (z_min + z_max) / 2.0

            for scale in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9,
                          1.1, 1.3, 1.5, 2.0, 3.0]:
                x_mod = x_ref.copy()
                x_mod[:, 2] = z_mid + (x_mod[:, 2] - z_mid) * scale
                guesses.append(x_mod.flatten())
                labels.append(f"z_scale_{scale:.1f}")

        # 3. Radial breathing: move mid-layer nodes inward/outward
        #    This is the primary origami deformation mode
        #    Detect mid-layer as nodes with z closest to z_midplane
        z_vals = x_ref[:, 2]
        z_sorted = np.sort(np.unique(np.round(z_vals, 6)))
        if len(z_sorted) >= 3:
            z_mid_layer = z_sorted[len(z_sorted) // 2]
            mid_mask = np.abs(z_vals - z_mid_layer) < 0.01

            for radial_scale in [-0.5, -0.3, -0.2, -0.1, -0.05,
                                 0.05, 0.1, 0.2, 0.3, 0.5]:
                x_mod = x_ref.copy()
                # Move mid nodes radially
                xy_norm = np.linalg.norm(x_mod[mid_mask, :2], axis=1, keepdims=True)
                xy_norm = np.maximum(xy_norm, 1e-12)
                direction = x_mod[mid_mask, :2] / xy_norm
                x_mod[mid_mask, :2] += radial_scale * direction
                guesses.append(x_mod.flatten())
                labels.append(f"radial_{radial_scale:+.2f}")

        # 4. Combined: z-scale + radial
        for z_sc in [0.3, 0.5, 0.7, 1.3]:
            for r_sc in [-0.2, -0.1, 0.1, 0.2]:
                x_mod = x_ref.copy()
                x_mod[:, 2] = z_mid + (x_mod[:, 2] - z_mid) * z_sc
                if len(z_sorted) >= 3:
                    xy_norm = np.linalg.norm(x_mod[mid_mask, :2], axis=1, keepdims=True)
                    xy_norm = np.maximum(xy_norm, 1e-12)
                    direction = x_mod[mid_mask, :2] / xy_norm
                    x_mod[mid_mask, :2] += r_sc * direction
                guesses.append(x_mod.flatten())
                labels.append(f"combined_z{z_sc:.1f}_r{r_sc:+.1f}")

        # 5. Random perturbations around reference (small scale)
        key = jax.random.PRNGKey(42)
        for i in range(num_random):
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, (self.dim,)) * perturbation_scale
            guesses.append(x_ref.flatten() + np.array(noise))
            labels.append(f"perturbed_{i}")

        return guesses, labels

    # ----------------------------------------------------------
    # Stability Classification
    # ----------------------------------------------------------

    def classify_equilibrium(self, x_flat):
        """
        Classify an equilibrium by computing eigenvalues of the Hessian.

        For a free-floating body (no fixed nodes, no gravity), 6 eigenvalues
        correspond to rigid body modes and should be near-zero. We exclude
        these from the stability classification.

        Returns:
            eigenvalues, stability_label, morse_index
        """
        H = np.array(self.hess_V(jnp.array(x_flat)))

        eigenvalues = np.linalg.eigvalsh(H)

        # Separate rigid body modes from deformation modes
        # Rigid body modes have |λ| << smallest elastic eigenvalue
        # Heuristic: anything below 1e-3 * median(|λ|) is a rigid body mode
        abs_eigs = np.abs(eigenvalues)
        median_eig = np.median(abs_eigs[abs_eigs > 1e-12]) if np.any(abs_eigs > 1e-12) else 1.0
        rigid_body_tol = min(1e-3 * median_eig, 1e-2)

        deformation_eigs = eigenvalues[abs_eigs > rigid_body_tol]
        n_rigid = np.sum(abs_eigs <= rigid_body_tol)

        # Count negative deformation eigenvalues (Morse index)
        negative = int(np.sum(deformation_eigs < -rigid_body_tol))

        if negative == 0:
            stability = 'stable'
        elif negative == len(deformation_eigs):
            stability = 'unstable'
        else:
            stability = 'saddle'

        return eigenvalues, stability, negative

    # ----------------------------------------------------------
    # Main Entry Point
    # ----------------------------------------------------------

    def find_all(self,
                 num_random: int = 50,
                 perturbation_scale: float = 0.1,
                 tol: float = 1e-6,
                 max_iter: int = 2000,
                 uniqueness_tol: float = 1e-3,
                 deflation_power: float = 2.0) -> FinderResults:
        """
        Find all equilibria of the system.

        Args:
            num_random:         number of random/perturbed initial guesses
            perturbation_scale: magnitude of perturbations around reference
            tol:                Newton convergence tolerance
            max_iter:           max Newton iterations per attempt
            uniqueness_tol:     distance below which two roots are identical
            deflation_power:    deflation exponent p

        Returns:
            FinderResults with all found equilibria
        """
        t_start = time.time()

        # Generate guesses
        guesses, labels = self._generate_guesses(
            num_random=num_random,
            perturbation_scale=perturbation_scale
        )

        print(f"\n[EquilibriumFinder] Searching with {len(guesses)} initial guesses...")
        print(f"  Tolerance: {tol:.1e}, Max iter: {max_iter}")
        print(f"  Deflation power: {deflation_power}")
        print()

        # Create solver
        solver = DeflatedNewtonSolver(
            energy_fn=self.V,
            residual_fn=self.grad_V,
            n_nodes=self.n_nodes,
            ref_positions=self.ref_positions,
            deflation_power=deflation_power,
            tol=tol,
            max_iter=max_iter,
            uniqueness_tol=uniqueness_tol
        )

        # Run deflated Newton
        raw_results = solver.find_from_guesses(guesses, labels)

        # Classify each found equilibrium
        results = FinderResults(
            n_total_attempts=len(guesses),
            metadata={
                'n_nodes': self.n_nodes,
                'tol': tol,
                'max_iter': max_iter,
                'uniqueness_tol': uniqueness_tol,
                'deflation_power': deflation_power,
                'num_random': num_random,
            }
        )

        for raw in raw_results:
            x_flat = jnp.array(raw['x'], dtype=jnp.float32)

            # Energy
            energy = float(self.V(x_flat))

            # Stability
            eigenvalues, stability, morse_index = self.classify_equilibrium(x_flat)

            eq = Equilibrium(
                positions=raw['x'].reshape(self.n_nodes, 3),
                energy=energy,
                eigenvalues=eigenvalues,
                stability=stability,
                index=morse_index,
                residual=raw['residual'],
                n_iterations=raw['n_iter'],
                source_guess=raw['label']
            )
            results.equilibria.append(eq)

        results.n_converged = len(raw_results)
        results.wall_time = time.time() - t_start

        return results

    # ----------------------------------------------------------
    # I/O: Save / Load
    # ----------------------------------------------------------

    def save_results(self, results: FinderResults, filepath: str):
        """
        Save equilibria to HDF5 in DEMLAT-compatible format.

        Structure:
            /metadata/              — finder settings
            /equilibria/
                /eq_000/
                    positions       (n_nodes, 3)
                    energy          scalar
                    eigenvalues     (dim,)
                    stability       string
                    morse_index     int
                    residual        float
                /eq_001/
                    ...
            /summary/
                n_found             int
                n_stable            int
                energies            (n_found,)
                stabilities         string array
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(filepath, 'w') as f:
            # --- Metadata ---
            meta = f.create_group('metadata')
            meta.attrs['n_nodes'] = self.n_nodes
            meta.attrs['n_equilibria'] = len(results.equilibria)
            meta.attrs['wall_time'] = results.wall_time
            meta.attrs['n_attempts'] = results.n_total_attempts
            meta.attrs['n_converged'] = results.n_converged
            for k, v in results.metadata.items():
                meta.attrs[k] = v

            # --- Reference geometry ---
            f.create_dataset('reference/positions', data=self.ref_positions)

            # --- Each equilibrium ---
            eq_group = f.create_group('equilibria')
            for i, eq in enumerate(results.equilibria):
                g = eq_group.create_group(f'eq_{i:03d}')
                g.create_dataset('positions', data=eq.positions)
                g.create_dataset('eigenvalues', data=eq.eigenvalues)
                g.attrs['energy'] = eq.energy
                g.attrs['stability'] = eq.stability
                g.attrs['morse_index'] = eq.index
                g.attrs['residual'] = eq.residual
                g.attrs['n_iterations'] = eq.n_iterations
                g.attrs['source_guess'] = eq.source_guess

            # --- Summary arrays (convenient for plotting) ---
            summary = f.create_group('summary')
            if results.equilibria:
                summary.create_dataset('energies',
                                       data=np.array([eq.energy for eq in results.equilibria]))
                summary.create_dataset('morse_indices',
                                       data=np.array([eq.index for eq in results.equilibria]))
                summary.create_dataset('residuals',
                                       data=np.array([eq.residual for eq in results.equilibria]))

                # String array for stability labels
                dt = h5py.string_dtype()
                stab = summary.create_dataset('stabilities',
                                              shape=(len(results.equilibria),), dtype=dt)
                for i, eq in enumerate(results.equilibria):
                    stab[i] = eq.stability

        print(f"[Saved] {len(results.equilibria)} equilibria -> {filepath}")

    @staticmethod
    def load_results(filepath: str) -> FinderResults:
        """Load equilibria from HDF5."""
        filepath = Path(filepath)
        results = FinderResults()

        with h5py.File(filepath, 'r') as f:
            results.n_total_attempts = int(f['metadata'].attrs.get('n_attempts', 0))
            results.n_converged = int(f['metadata'].attrs.get('n_converged', 0))
            results.wall_time = float(f['metadata'].attrs.get('wall_time', 0))

            eq_group = f['equilibria']
            for key in sorted(eq_group.keys()):
                g = eq_group[key]
                eq = Equilibrium(
                    positions=g['positions'][:],
                    energy=float(g.attrs['energy']),
                    eigenvalues=g['eigenvalues'][:],
                    stability=str(g.attrs['stability']),
                    index=int(g.attrs['morse_index']),
                    residual=float(g.attrs['residual']),
                    n_iterations=int(g.attrs['n_iterations']),
                    source_guess=str(g.attrs['source_guess'])
                )
                results.equilibria.append(eq)

        return results


# ============================================================
# Parameter Sweep Utility
# ============================================================

def sweep_stiffness(experiment_dir, k_facet_values, **finder_kwargs):
    """
    Sweep over facet stiffness and track how equilibria evolve.

    This is the starting point for bifurcation analysis.
    """
    root = Path(experiment_dir)
    geom_path = root / "input" / "geometry.h5"
    config_path = root / "input" / "config.json"

    with open(config_path, 'r') as f:
        config = json.load(f)
    physics = config.get('global_physics', config.get('physics', {}))

    all_results = {}

    for k_facet in k_facet_values:
        print(f"\n{'=' * 60}")
        print(f"  k_facet = {k_facet}")
        print(f"{'=' * 60}")

        # Load geometry and override hinge stiffness
        geometry = {}
        with h5py.File(geom_path, 'r') as f:
            geometry['nodes'] = f['nodes/positions'][:]
            if 'nodes/masses' in f:
                geometry['masses'] = f['nodes/masses'][:]
            if 'elements/bars/indices' in f:
                geometry['bars'] = {
                    'indices': f['elements/bars/indices'][:].astype(np.int32),
                }
                for attr in ['stiffness', 'rest_length', 'prestress']:
                    key = f'elements/bars/{attr}'
                    if key in f:
                        geometry['bars'][attr] = f[key][:]
            if 'elements/hinges/indices' in f:
                geometry['hinges'] = {
                    'indices': f['elements/hinges/indices'][:].astype(np.int32),
                }
                for attr in ['stiffness', 'damping']:
                    key = f'elements/hinges/{attr}'
                    if key in f:
                        geometry['hinges'][attr] = f[key][:]

                if 'elements/hinges/angle' in f:
                    geometry['hinges']['rest_angle'] = f['elements/hinges/angle'][:]
                elif 'elements/hinges/rest_angle' in f:
                    geometry['hinges']['rest_angle'] = f['elements/hinges/rest_angle'][:]

        # Override hinge stiffness
        if 'hinges' in geometry:
            geometry['hinges']['stiffness'][:] = k_facet

        finder = EquilibriumFinder(geometry, physics)
        results = finder.find_all(**finder_kwargs)
        results.summary()

        all_results[k_facet] = results

    return all_results


# ============================================================
# Quick Test (standalone)
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exp_dir = sys.argv[1]
    else:
        exp_dir = "experiments/yoshimura_test"

    print(f"Loading experiment from: {exp_dir}")
    finder = EquilibriumFinder.from_experiment(exp_dir)

    results = finder.find_all(
        num_random=100,
        perturbation_scale=0.5,
        tol=1e-8,
        max_iter=300,
        uniqueness_tol=1e-3
    )

    results.summary()

    # Save
    out_path = Path(exp_dir) / "output" / "equilibria.h5"
    finder.save_results(results, out_path)

    # Verify round-trip
    loaded = EquilibriumFinder.load_results(out_path)
    print(f"Loaded back {len(loaded.equilibria)} equilibria.")
