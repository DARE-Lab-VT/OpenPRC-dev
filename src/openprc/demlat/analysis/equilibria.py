"""
Equilibrium Finder for DEMLAT Bar-Hinge Models
================================================

Finds equilibria of the bar-hinge system by treating it as a black box:

    x_new, v_new = model.dynamics_step(x, v=0, dt)

At equilibrium, starting from rest: v_new ≈ 0 (no acceleration → no net force).

Method: Multi-start Dynamic Relaxation with Deflation
    1. Generate physics-informed initial guesses (origami kinematics)
    2. For each guess, run damped dynamics until velocity → 0
    3. Reject duplicates via deformation-space distance metric
    4. Classify stability via linearized dynamics (numerical Jacobian)

This approach automatically inherits all physics from the solver
(scaling, non-dimensionalization, bar/hinge forces) without
replicating the energy function.

Usage:
    finder = EquilibriumFinder.from_experiment("experiments/yoshimura_test")
    results = finder.find_all()
    results.summary()
    finder.save_results(results)
"""

import numpy as np
import h5py
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


# ============================================================
# Data Structures
# ============================================================

@dataclass
class Equilibrium:
    """A single equilibrium state."""
    positions: np.ndarray
    energy_proxy: float
    eigenvalues: Optional[np.ndarray]
    stability: str
    index: int
    residual: float
    n_steps: int
    source_guess: str


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
        print(f"\n{'=' * 70}")
        print(f"  Equilibrium Finder Results")
        print(f"{'=' * 70}")
        print(f"  Total attempts:     {self.n_total_attempts}")
        print(f"  Converged:          {self.n_converged}")
        print(f"  Unique equilibria:  {len(self.equilibria)}")
        print(f"  Stable:             {self.n_stable}")
        print(f"  Unstable:           {self.n_unstable}")
        print(f"  Wall time:          {self.wall_time:.2f}s")
        print(f"{'=' * 70}")
        for i, eq in enumerate(self.equilibria):
            z_range = eq.positions[:, 2].max() - eq.positions[:, 2].min()
            print(f"  [{i}] stability={eq.stability:<8s}  "
                  f"index={eq.index}  "
                  f"z_range={z_range:.4f}  "
                  f"residual={eq.residual:.2e}  "
                  f"steps={eq.n_steps}  "
                  f"from={eq.source_guess}")
        print()


# ============================================================
# Core: Dynamic Relaxation
# ============================================================

class DynamicRelaxationSolver:
    """
    Finds a single equilibrium via damped dynamics.

    Starting from an initial configuration, we run the physics engine
    with heavy artificial damping until kinetic energy → 0. This
    automatically follows the energy landscape downhill to the nearest
    local minimum, handling stiffness and conditioning naturally.
    """

    def __init__(self, model, dt=0.0005, damping=0.90, max_steps=50000,
                 tol=1e-6, adaptive_damping=True, verbose=False):
        self.model = model
        self.dt = dt
        self.damping_base = damping
        self.max_steps = max_steps
        self.tol = tol
        self.adaptive = adaptive_damping
        self.verbose = verbose
        self.n_nodes = model.n_nodes

    def relax(self, x_init, label="unknown"):
        """
        Run dynamic relaxation from x_init.

        Always returns the best configuration found (lowest max velocity).
        The info dict contains 'converged' (strict tol) and 'best_vel'
        so the caller can decide what to accept.

        Returns:
            (best_positions, info_dict)
        """
        x = x_init.copy().astype(np.float64)
        v = np.zeros_like(x)
        damping = self.damping_base

        prev_ke = float('inf')
        stall_count = 0

        # Track best state seen during relaxation
        best_x = x.copy()
        best_vel = float('inf')

        for step in range(self.max_steps):
            x_new, v_new = self.model.dynamics_step(x, v, dt=self.dt)

            # Artificial damping
            v_new *= damping

            nodal_speeds = np.linalg.norm(v_new, axis=1)
            max_vel = float(np.max(nodal_speeds))
            ke = float(0.5 * np.sum(nodal_speeds ** 2))

            # Track best
            if max_vel < best_vel:
                best_vel = max_vel
                best_x = x_new.copy()

            # Adaptive damping
            if self.adaptive:
                if ke < prev_ke:
                    damping = min(damping + 0.001, 0.98)
                else:
                    damping = max(damping - 0.01, 0.5)
                    if ke > prev_ke * 2.0:
                        v_new *= 0.1

            # Strict convergence
            if max_vel < self.tol:
                _, v_verify = self.model.dynamics_step(x_new, np.zeros_like(x_new), dt=self.dt)
                residual = float(np.max(np.linalg.norm(v_verify, axis=1))) / self.dt

                if self.verbose:
                    print(f"    [{label}] Converged step={step}, residual={residual:.2e}")
                return x_new, {
                    'converged': True, 'n_steps': step,
                    'residual': residual, 'best_vel': max_vel,
                    'label': label
                }

            # Stall detection
            if abs(ke - prev_ke) < 1e-15 * max(abs(ke), 1e-30):
                stall_count += 1
            else:
                stall_count = 0

            if stall_count > 500:
                if self.verbose:
                    print(f"    [{label}] Stalled step={step}, best_vel={best_vel:.2e}")
                break

            prev_ke = ke
            x = x_new
            v = v_new

            if self.verbose and step % 2000 == 0:
                print(f"    [{label}] step={step:6d} max_vel={max_vel:.2e} "
                      f"best={best_vel:.2e} damp={damping:.3f}")

        # Didn't strictly converge — verify and return best state anyway
        _, v_verify = self.model.dynamics_step(best_x, np.zeros_like(best_x), dt=self.dt)
        residual = float(np.max(np.linalg.norm(v_verify, axis=1))) / self.dt

        if self.verbose:
            print(f"    [{label}] Best vel={best_vel:.2e}, residual={residual:.2e}")

        return best_x, {
            'converged': False, 'n_steps': step if step < self.max_steps else self.max_steps,
            'residual': residual, 'best_vel': best_vel,
            'label': label
        }


# ============================================================
# Main Finder
# ============================================================

class EquilibriumFinder:
    """
    Multi-start dynamic relaxation equilibrium finder.

    Uses model.dynamics_step as a black box — automatically inherits
    all physics (scaling, forces, gravity, collisions, etc).
    """

    def __init__(self, model, ref_positions):
        self.model = model
        self.n_nodes = model.n_nodes
        self.ref_positions = ref_positions.copy()
        self.found_equilibria: List[np.ndarray] = []

    @classmethod
    def from_experiment(cls, experiment_dir, backend='jax'):
        """Create finder from a DEMLAT experiment directory."""
        import demlat
        from demlat.models.barhinge import BarHingeModel

        exp = demlat.Simulation(experiment_dir)
        model = BarHingeModel(exp, backend=backend)

        print(f"[EquilibriumFinder] {model.n_nodes} nodes, backend={model.backend_name}")
        return cls(model, model.x0.copy())

    # ----------------------------------------------------------
    # Duplicate Detection (rigid-body invariant)
    # ----------------------------------------------------------

    def _deformation_distance(self, x1, x2):
        """RMSD after Kabsch alignment (translation + rotation invariant)."""
        c1 = x1 - x1.mean(axis=0)
        c2 = x2 - x2.mean(axis=0)

        H = c1.T @ c2
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        R = Vt.T @ np.diag([1, 1, d]) @ U.T

        c2_aligned = (R @ c2.T).T
        return float(np.sqrt(np.mean(np.sum((c1 - c2_aligned) ** 2, axis=1))))

    def _is_duplicate(self, x_new, tol=1e-3):
        for x_found in self.found_equilibria:
            if self._deformation_distance(x_new, x_found) < tol:
                return True
        return False

    # ----------------------------------------------------------
    # Initial Guess Generation
    # ----------------------------------------------------------

    def _generate_guesses(self, num_random=30, noise_scale=0.05):
        """
        Generate initial guesses spanning the origami configuration space.

        Samples along: z-compression, radial breathing, twist, combined modes.
        """
        guesses = []
        labels = []
        x_ref = self.ref_positions.copy()
        n = self.n_nodes

        z_vals = x_ref[:, 2]
        z_unique = np.sort(np.unique(np.round(z_vals, 6)))
        z_min, z_max = z_vals.min(), z_vals.max()
        z_mid_val = (z_min + z_max) / 2.0

        # Mid-layer mask
        if len(z_unique) >= 3:
            z_mid_layer = z_unique[len(z_unique) // 2]
            mid_mask = np.abs(z_vals - z_mid_layer) < 0.01
        else:
            mid_mask = np.zeros(n, dtype=bool)
        has_mid = np.any(mid_mask)

        def add(x, lbl):
            guesses.append(x.copy())
            labels.append(lbl)

        # 1. Reference
        add(x_ref, "reference")

        # # 2. Z-compression/extension
        # for z_s in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
        #             1.1, 1.2, 1.5, 2.0, 2.5, 3.0]:
        #     x_m = x_ref.copy()
        #     x_m[:, 2] = z_mid_val + (x_m[:, 2] - z_mid_val) * z_s
        #     add(x_m, f"z_{z_s:.2f}")
        #
        # # 3. Radial breathing
        # if has_mid:
        #     for r_s in [-0.4, -0.3, -0.2, -0.15, -0.1, -0.05,
        #                 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]:
        #         x_m = x_ref.copy()
        #         xy = x_m[mid_mask, :2]
        #         nrm = np.maximum(np.linalg.norm(xy, axis=1, keepdims=True), 1e-12)
        #         x_m[mid_mask, :2] += r_s * (xy / nrm)
        #         add(x_m, f"radial_{r_s:+.2f}")
        #
        # # 4. Combined z + radial
        # if has_mid:
        #     for z_s in [0.1, 0.3, 0.5, 0.7, 1.3, 1.5, 2.0]:
        #         for r_s in [-0.3, -0.15, 0.0, 0.15, 0.3]:
        #             x_m = x_ref.copy()
        #             x_m[:, 2] = z_mid_val + (x_m[:, 2] - z_mid_val) * z_s
        #             xy = x_m[mid_mask, :2]
        #             nrm = np.maximum(np.linalg.norm(xy, axis=1, keepdims=True), 1e-12)
        #             x_m[mid_mask, :2] += r_s * (xy / nrm)
        #             add(x_m, f"comb_z{z_s:.1f}_r{r_s:+.2f}")
        #
        # # 5. Twist
        # for tw in [0.2, 0.5, 1.0, -0.2, -0.5]:
        #     x_m = x_ref.copy()
        #     hfrac = (x_m[:, 2] - z_min) / max(z_max - z_min, 1e-12)
        #     theta = tw * hfrac
        #     xn = x_m[:, 0] * np.cos(theta) - x_m[:, 1] * np.sin(theta)
        #     yn = x_m[:, 0] * np.sin(theta) + x_m[:, 1] * np.cos(theta)
        #     x_m[:, 0], x_m[:, 1] = xn, yn
        #     add(x_m, f"twist_{tw:+.1f}")
        #
        # # 6. Inverted
        # x_m = x_ref.copy()
        # x_m[:, 2] = z_max - (x_m[:, 2] - z_min)
        # add(x_m, "inverted")

        # 7. Random perturbations
        rng = np.random.RandomState(42)
        for scale_name, scale in [("sm", 0.1), ("md", 0.5), ("lg", 1.5)]:
            for i in range(max(num_random // 3, 1)):
                add(x_ref + rng.randn(n, 3) * scale, f"rnd_{scale_name}_{i}")

        # 8. Mid-layer shifts
        if has_mid:
            for sz in [-0.1, 0.1]:
                for sr in [-0.1, 0.1]:
                    x_m = x_ref.copy()
                    x_m[mid_mask, 2] += sz
                    xy = x_m[mid_mask, :2]
                    nrm = np.maximum(np.linalg.norm(xy, axis=1, keepdims=True), 1e-12)
                    x_m[mid_mask, :2] += sr * (xy / nrm)
                    add(x_m, f"midshift_z{sz:+.1f}_r{sr:+.1f}")

        return guesses, labels

    # ----------------------------------------------------------
    # Stability
    # ----------------------------------------------------------

    def _classify_stability(self, x_eq, dt_probe=1e-5, perturbation=1e-5):
        """
        Classify via numerical linearization of the force field.
        Probes random directions to build a reduced Jacobian.
        """
        n = self.n_nodes
        dim = n * 3

        _, v0 = self.model.dynamics_step(x_eq, np.zeros((n, 3)), dt=dt_probe)
        a0 = (v0 / dt_probe).flatten()

        n_probes = min(dim, 30)
        rng = np.random.RandomState(123)
        dirs = rng.randn(n_probes, dim)
        dirs, _ = np.linalg.qr(dirs.T)
        dirs = dirs.T[:n_probes]

        A = np.zeros((n_probes, n_probes))
        for j in range(n_probes):
            x_p = (x_eq.flatten() + perturbation * dirs[j]).reshape(n, 3)
            _, v_p = self.model.dynamics_step(x_p, np.zeros((n, 3)), dt=dt_probe)
            da = (v_p / dt_probe).flatten() - a0
            for i in range(n_probes):
                A[i, j] = np.dot(dirs[i], da) / perturbation

        A = 0.5 * (A + A.T)
        eigs = np.linalg.eigvalsh(A)

        tol_eig = max(np.max(np.abs(eigs)) * 1e-3, 1e-6)
        n_unstable = int(np.sum(eigs > tol_eig))

        if n_unstable == 0:
            stability = 'stable'
        else:
            stability = 'saddle' if n_unstable < n_probes else 'unstable'

        return stability, n_unstable, eigs

    # ----------------------------------------------------------
    # Main
    # ----------------------------------------------------------

    def find_all(self, num_random=30, noise_scale=0.05, dt=0.0005,
                 damping=0.90, max_steps=50000, tol=1e-6,
                 accept_tol=1e-3, uniqueness_tol=1e-3,
                 classify=True, verbose=True):
        """
        Find all equilibria via multi-start dynamic relaxation.

        Args:
            tol:            strict convergence tolerance (velocity)
            accept_tol:     soft acceptance tolerance — states with
                            best_vel < accept_tol are kept even if they
                            didn't hit strict tol. Set equal to tol for
                            strict-only behavior.
        """
        t_start = time.time()

        guesses, labels = self._generate_guesses(num_random, noise_scale)

        print(f"\n[EquilibriumFinder] {len(guesses)} initial guesses")
        print(f"  dt={dt}, damping={damping}, max_steps={max_steps}")
        print(f"  tol={tol} (strict), accept_tol={accept_tol} (soft)")
        print()

        solver = DynamicRelaxationSolver(
            self.model, dt=dt, damping=damping,
            max_steps=max_steps, tol=tol,
            adaptive_damping=True, verbose=verbose
        )

        results = FinderResults(
            n_total_attempts=len(guesses),
            metadata={'dt': dt, 'damping': damping, 'max_steps': max_steps,
                      'tol': tol, 'accept_tol': accept_tol,
                      'uniqueness_tol': uniqueness_tol,
                      'n_nodes': self.n_nodes}
        )

        for i, (x0, label) in enumerate(zip(guesses, labels)):
            x0 = np.array(x0).reshape(self.n_nodes, 3)
            x_eq, info = solver.relax(x0, label=label)

            # Accept if strictly converged OR below soft tolerance
            best_vel = info.get('best_vel', float('inf'))
            is_acceptable = info['converged'] or best_vel < accept_tol

            if not is_acceptable:
                if verbose and best_vel < accept_tol * 100:
                    print(f"    [{label}] Rejected (best_vel={best_vel:.2e})")
                continue

            results.n_converged += 1

            if self._is_duplicate(x_eq, tol=uniqueness_tol):
                if verbose:
                    print(f"    [{label}] → duplicate")
                continue

            self.found_equilibria.append(x_eq.copy())

            residual = info['residual']
            if classify:
                stability, n_unstable, eigs = self._classify_stability(x_eq)
            else:
                stability, n_unstable, eigs = 'unknown', 0, None

            converge_tag = "✓" if info['converged'] else "~"

            eq = Equilibrium(
                positions=x_eq.copy(),
                energy_proxy=float(np.sum((x_eq - self.ref_positions) ** 2)),
                eigenvalues=eigs, stability=stability,
                index=n_unstable, residual=residual,
                n_steps=info['n_steps'], source_guess=label
            )
            results.equilibria.append(eq)

            z_range = x_eq[:, 2].max() - x_eq[:, 2].min()
            print(f"  {converge_tag} Eq {len(results.equilibria)} from {label:30s} "
                  f"z={z_range:.4f} {stability} res={residual:.2e} "
                  f"vel={best_vel:.2e}")

        results.wall_time = time.time() - t_start
        return results

    # ----------------------------------------------------------
    # I/O
    # ----------------------------------------------------------

    def save_results(self, results, filepath=None):
        """Save equilibria to HDF5."""
        if filepath is None:
            filepath = "equilibria.h5"
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(filepath, 'w') as f:
            meta = f.create_group('metadata')
            meta.attrs['n_nodes'] = self.n_nodes
            meta.attrs['n_equilibria'] = len(results.equilibria)
            meta.attrs['wall_time'] = results.wall_time
            meta.attrs['n_attempts'] = results.n_total_attempts
            meta.attrs['n_converged'] = results.n_converged
            for k, v in results.metadata.items():
                try:
                    meta.attrs[k] = v
                except TypeError:
                    pass

            f.create_dataset('reference/positions', data=self.ref_positions)

            eq_grp = f.create_group('equilibria')
            for i, eq in enumerate(results.equilibria):
                g = eq_grp.create_group(f'eq_{i:03d}')
                g.create_dataset('positions', data=eq.positions)
                if eq.eigenvalues is not None:
                    g.create_dataset('eigenvalues', data=eq.eigenvalues)
                g.attrs['energy_proxy'] = eq.energy_proxy
                g.attrs['stability'] = eq.stability
                g.attrs['index'] = eq.index
                g.attrs['residual'] = eq.residual
                g.attrs['n_steps'] = eq.n_steps
                g.attrs['source_guess'] = eq.source_guess

            if results.equilibria:
                s = f.create_group('summary')
                s.create_dataset('residuals',
                                 data=np.array([eq.residual for eq in results.equilibria]))
                dt = h5py.string_dtype()
                stab = s.create_dataset('stabilities',
                                        shape=(len(results.equilibria),), dtype=dt)
                for i, eq in enumerate(results.equilibria):
                    stab[i] = eq.stability

        print(f"[Saved] {len(results.equilibria)} equilibria → {filepath}")

    @staticmethod
    def load_results(filepath):
        """Load equilibria from HDF5."""
        results = FinderResults()
        with h5py.File(filepath, 'r') as f:
            results.n_total_attempts = int(f['metadata'].attrs.get('n_attempts', 0))
            results.n_converged = int(f['metadata'].attrs.get('n_converged', 0))
            results.wall_time = float(f['metadata'].attrs.get('wall_time', 0))

            for key in sorted(f['equilibria'].keys()):
                g = f['equilibria'][key]
                eq = Equilibrium(
                    positions=g['positions'][:],
                    energy_proxy=float(g.attrs.get('energy_proxy', 0)),
                    eigenvalues=g['eigenvalues'][:] if 'eigenvalues' in g else None,
                    stability=str(g.attrs.get('stability', 'unknown')),
                    index=int(g.attrs.get('index', 0)),
                    residual=float(g.attrs.get('residual', 0)),
                    n_steps=int(g.attrs.get('n_steps', 0)),
                    source_guess=str(g.attrs.get('source_guess', 'loaded'))
                )
                results.equilibria.append(eq)
        return results


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import sys

    exp_dir = sys.argv[1] if len(sys.argv) > 1 else "experiments/yoshimura_equilibrium"
    backend = sys.argv[2] if len(sys.argv) > 2 else "jax"

    finder = EquilibriumFinder.from_experiment(exp_dir, backend=backend)
    results = finder.find_all(verbose=True)
    results.summary()

    out_path = Path(exp_dir) / "output" / "equilibria.h5"
    finder.save_results(results, out_path)
