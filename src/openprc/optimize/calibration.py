"""
Calibration Module
===================

Gradient-based calibration of DEMLAT bar-hinge model parameters
against a reference trajectory.

Usage
-----
::

    from openprc.optimize import Calibration
    from openprc.demlat.models.barhinge import BarHingeModel

    cal = Calibration(BarHingeModel, backend='jax')
    cal.load_geometry(experiment_dir)
    cal.load_reference(experiment_dir / "output" / "simulation.h5")

    cal.optimize_params(bar_stiffness=True, hinge_stiffness=True)
    cal.set_bounds(bar_stiffness=(0.1, 1000.0))

    result = cal.run(max_iterations=500, lr=0.01, cost='mse')
    cal.save(experiment_dir / "optimized_geometry.h5")
"""

import time
import json
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from jax import jit, grad, value_and_grad
from pathlib import Path

from .theta import (
    make_mask, apply_mask, make_bounds, clip_to_bounds,
    adam_state_init, adam_update, theta_to_numpy,
)
from .costs import get_cost
from .reference import ReferenceData

from openprc.schemas.logging import get_logger


class CalibrationResult:
    """Container for calibration results."""

    def __init__(self):
        self.theta_initial = None
        self.theta_optimized = None
        self.loss_history = []
        self.grad_norm_history = []
        self.iterations = 0
        self.wall_time = 0.0
        self.converged = False
        self.final_loss = float('inf')

    def summary(self):
        """Print a summary of the calibration."""
        print(f"\n{'=' * 50}")
        print(f"  Calibration Result")
        print(f"{'=' * 50}")
        print(f"  Iterations    : {self.iterations}")
        print(f"  Final loss    : {self.final_loss:.6e}")
        print(f"  Converged     : {self.converged}")
        print(f"  Wall time     : {self.wall_time:.1f}s")
        print(f"  Loss reduction: {self.loss_history[0]:.6e} -> {self.final_loss:.6e}")

        if self.loss_history[0] > 0:
            reduction = (1.0 - self.final_loss / self.loss_history[0]) * 100
            print(f"  Improvement   : {reduction:.1f}%")

        # Per-parameter changes
        print(f"\n  Parameter changes:")
        for key in self.theta_optimized:
            old = np.array(self.theta_initial[key])
            new = np.array(self.theta_optimized[key])
            if old.size <= 10:
                print(f"    {key}: {old} -> {new}")
            else:
                rel_change = np.mean(np.abs(new - old) / (np.abs(old) + 1e-12))
                print(f"    {key}: mean relative change = {rel_change:.4f}")

        print(f"{'=' * 50}\n")


class Calibration:
    """
    Gradient-based calibration of bar-hinge model parameters.

    Uses the JAX differentiable solver to compute ∂Loss/∂θ and
    iteratively update parameters to match a reference trajectory.
    """

    def __init__(self, model_cls, backend='jax'):
        """
        Parameters
        ----------
        model_cls : class
            DEMLAT model class (e.g. BarHingeModel).
        backend : str
            Must be 'jax'.
        """
        if backend != 'jax':
            raise ValueError(
                f"Calibration requires backend='jax', got '{backend}'. "
                f"CUDA and CPU backends are not differentiable."
            )

        self.logger = get_logger("optimize.calibration")
        self.model_cls = model_cls
        self.backend = backend

        # Will be populated by load_geometry / load_reference
        self.solver = None
        self.geometry_dir = None
        self.reference = None
        self.config = None

        # Optimization settings (populated by optimize_params / set_bounds)
        self._optimize_flags = {}
        self._bounds_config = {}

        # Cost
        self._cost_fn = None
        self._cost_kwargs = {}

    # ============================================================
    # Setup
    # ============================================================

    def load_geometry(self, experiment_dir):
        """
        Load geometry.h5 and config.json from a DEMLAT experiment directory.
        Initializes the JAX solver with the model's connectivity and initial params.
        """
        self.geometry_dir = Path(experiment_dir)
        config_path = self.geometry_dir / "input" / "config.json"
        geom_path = self.geometry_dir / "input" / "geometry.h5"

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        if not geom_path.exists():
            raise FileNotFoundError(f"Geometry not found: {geom_path}")

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Read geometry
        with h5py.File(geom_path, 'r') as f:
            positions = np.array(f['nodes/positions'])
            masses = np.array(f['nodes/masses'])
            attrs = np.array(f['nodes/attributes'], dtype=np.uint8)
            n_nodes = positions.shape[0]

            bars = None
            if 'elements/bars' in f:
                bars = {
                    'indices': np.array(f['elements/bars/indices']),
                    'stiffness': np.array(f['elements/bars/stiffness']),
                    'damping': np.array(f['elements/bars/damping']),
                    'rest_length': np.array(f['elements/bars/rest_length']),
                }
                if 'elements/bars/prestress' in f:
                    bars['prestress'] = np.array(f['elements/bars/prestress'])

            hinges = None
            if 'elements/hinges' in f:
                hinges = {
                    'indices': np.array(f['elements/hinges/indices']),
                    'stiffness': np.array(f['elements/hinges/stiffness']),
                    'damping': np.array(f['elements/hinges/damping']),
                    'angle': np.array(f['elements/hinges/angle']),
                }

        # Build solver
        options = self.config.get('global_physics', {})
        from openprc.demlat.models.barhinge.solver_jax import JaxSolver
        self.solver = JaxSolver(n_nodes, masses, attrs, bars, hinges, options)

        # Store initial positions for rollout
        self._x0 = jnp.array(positions, dtype=jnp.float32)
        self._v0 = jnp.zeros_like(self._x0)

        self.logger.info(f"Loaded geometry: {n_nodes} nodes from {geom_path}")

    def load_reference(self, path):
        """Load the target trajectory to calibrate against."""
        self.reference = ReferenceData(path)
        self.logger.info(f"Loaded reference: {self.reference}")

    def optimize_params(self, **kwargs):
        """
        Specify which parameters to optimize.

        Example::

            cal.optimize_params(
                bar_stiffness=True,
                bar_damping=False,
                hinge_stiffness=True,
            )
        """
        self._optimize_flags = kwargs

    def set_bounds(self, **kwargs):
        """
        Set optimization bounds for parameters.

        Example::

            cal.set_bounds(
                bar_stiffness=(0.1, 1000.0),
                hinge_stiffness=(0.001, 100.0),
            )
        """
        self._bounds_config = kwargs

    # ============================================================
    # Run Optimization
    # ============================================================

    def run(self, max_iterations=500, lr=1e-3, cost='mse',
            optimizer='adam', tol=1e-8, patience=50,
            save_interval=10, dt=None, substeps=None,
            verbose=True, cost_kwargs=None):
        """
        Run calibration.

        Parameters
        ----------
        max_iterations : int
            Maximum number of gradient steps.
        lr : float
            Learning rate.
        cost : str or callable
            Cost function name (from registry) or custom callable.
        optimizer : str
            'adam' or 'sgd'.
        tol : float
            Convergence tolerance on loss change.
        patience : int
            Stop if no improvement for this many iterations.
        save_interval : int
            Print progress every N iterations.
        dt : float or None
            Simulation timestep. None = read from config.
        substeps : int or None
            Number of simulation steps between each saved frame.
            None = infer from config (dt_sim / dt_save).
        verbose : bool
            Print progress.
        cost_kwargs : dict or None
            Extra kwargs passed to the cost function.

        Returns
        -------
        CalibrationResult
        """
        if self.solver is None:
            raise RuntimeError("Call load_geometry() first.")
        if self.reference is None:
            raise RuntimeError("Call load_reference() first.")

        # --- Resolve timing ---
        sim_dt = dt or float(self.config.get('simulation', {}).get('dt_base', 0.001))
        save_dt = self.reference.dt_save

        if substeps is None:
            substeps = max(1, int(round(save_dt / sim_dt)))

        n_save_frames = self.reference.n_frames
        total_steps = substeps * (n_save_frames - 1)   # -1 because frame 0 is initial

        self.logger.info(
            f"Calibration: {n_save_frames} frames, {substeps} substeps/frame, "
            f"dt={sim_dt}, total sim steps={total_steps}"
        )

        # --- Reference trajectory ---
        x_ref = self.reference.get_trajectory_jax(start_frame=1)  # skip frame 0 (= IC)

        # --- Initial state ---
        x0, v0 = self.reference.get_initial_state_jax()

        # --- Build actuation sequences ---
        n_nodes = self.solver.n_nodes
        f_ext_per_step, pos_target_per_step = self._build_actuation_sequences(
            total_steps, n_nodes, sim_dt
        )

        # --- Build differentiable rollout ---
        rollout_fn = self.solver.get_rollout_fn()

        # --- Cost function ---
        cost_kw = cost_kwargs or {}
        if isinstance(cost, str) and cost == 'strain_mse':
            cost_kw.setdefault('bar_indices', np.array(self.solver.bar_indices))
        cost_fn = get_cost(cost, **cost_kw)

        # --- Theta, mask, bounds ---
        theta = self.solver.get_theta()
        mask = make_mask(theta, self._optimize_flags)
        lo, hi = make_bounds(theta, self._bounds_config)

        # --- Build loss + grad function ---
        def loss_fn(theta_opt):
            # Run differentiable simulation
            x_traj_all, _, _ = rollout_fn(
                theta_opt, x0, v0, f_ext_per_step, pos_target_per_step, sim_dt
            )

            # Subsample to match reference save interval
            # x_traj_all is (total_steps, N, 3), we want every `substeps`-th frame
            x_traj_saved = x_traj_all[substeps - 1::substeps]  # (n_save_frames-1, N, 3)

            return cost_fn(x_traj_saved, x_ref)

        loss_and_grad = jit(value_and_grad(loss_fn))

        # --- Optimizer state ---
        if optimizer == 'adam':
            opt_state = adam_state_init(theta)
        elif optimizer == 'sgd':
            opt_state = None
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        # --- Result container ---
        result = CalibrationResult()
        result.theta_initial = jax.tree.map(lambda x: x.copy(), theta)

        # --- Optimization loop ---
        best_loss = float('inf')
        patience_counter = 0
        t_start = time.time()

        if verbose:
            print(f"\n{'Iter':>6}  {'Loss':>12}  {'|grad|':>12}  {'Time':>8}")
            print(f"{'-' * 44}")

        for iteration in range(max_iterations):
            # Forward + backward
            loss_val, grads = loss_and_grad(theta)
            loss_scalar = float(loss_val)

            # Gradient norm (masked params only)
            masked_grads = apply_mask(grads, mask)
            grad_norm = float(jnp.sqrt(
                sum(jnp.sum(g ** 2) for g in jax.tree.leaves(masked_grads))
            ))

            result.loss_history.append(loss_scalar)
            result.grad_norm_history.append(grad_norm)

            # Print progress
            if verbose and (iteration % save_interval == 0 or iteration == max_iterations - 1):
                elapsed = time.time() - t_start
                print(f"{iteration:>6}  {loss_scalar:>12.6e}  {grad_norm:>12.6e}  {elapsed:>7.1f}s")

            # Convergence check
            if loss_scalar < best_loss - tol:
                best_loss = loss_scalar
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    print(f"\nConverged (patience={patience} reached at iter {iteration})")
                result.converged = True
                break

            if grad_norm < tol:
                if verbose:
                    print(f"\nConverged (gradient norm < {tol} at iter {iteration})")
                result.converged = True
                break

            # Update
            if optimizer == 'adam':
                theta, opt_state = adam_update(theta, grads, mask, opt_state, lr=lr)
            else:
                from .theta import sgd_update
                theta = sgd_update(theta, grads, mask, lr)

            # Project onto bounds
            theta = clip_to_bounds(theta, lo, hi)

        # --- Finalize ---
        result.theta_optimized = theta
        result.iterations = iteration + 1
        result.wall_time = time.time() - t_start
        result.final_loss = float(result.loss_history[-1])

        # Update solver with optimized theta
        self.solver.set_theta(theta)

        if verbose:
            result.summary()

        return result

    # ============================================================
    # Actuation
    # ============================================================

    def _build_actuation_sequences(self, total_steps, n_nodes, sim_dt):
        """
        Build per-timestep actuation arrays from signals.h5 + config actuators.

        Returns
        -------
        f_ext_per_step : (total_steps, N, 3) – external force at each sim step
        pos_target_per_step : (total_steps, N, 3) – position targets at each sim step
            For non-driven nodes, values are the initial position (masked out in rollout).
        """
        f_ext = np.zeros((total_steps, n_nodes, 3), dtype=np.float32)
        # Initialize pos targets to initial position (won't affect non-driven nodes)
        pos_target = np.tile(
            np.array(self._x0)[None, :, :],
            (total_steps, 1, 1)
        ).astype(np.float32)

        actuators = self.config.get('actuators', [])
        if not actuators:
            return jnp.array(f_ext), jnp.array(pos_target)

        # Load signals
        signals_path = self.geometry_dir / "input" / "signals.h5"
        if not signals_path.exists():
            self.logger.warning(f"No signals.h5 found at {signals_path}, using zero actuation.")
            return jnp.array(f_ext), jnp.array(pos_target)

        signals = {}
        sig_dt = sim_dt
        with h5py.File(signals_path, 'r') as f:
            sig_dt = float(f.attrs.get('dt_base', sim_dt))
            for k in f.keys():
                if isinstance(f[k], h5py.Dataset):
                    signals[k] = np.array(f[k])

        # Resolve each actuator
        for act_def in actuators:
            node_idx = act_def.get('node_idx', act_def.get('node_id'))
            if node_idx is None:
                continue

            act_type = act_def.get('type', 'position')
            sig_name = act_def.get('signal_name', act_def.get('signal'))
            dof = np.array(act_def.get('dof', [1, 1, 1]), dtype=np.float32)

            if sig_name is None or sig_name not in signals:
                self.logger.warning(f"Signal '{sig_name}' not found for node {node_idx}")
                continue

            signal_data = signals[sig_name]  # (T_sig,) or (T_sig, 3)

            # Sample signal at each simulation timestep
            for step_idx in range(total_steps):
                t = step_idx * sim_dt
                sig_idx = min(int(round(t / sig_dt)), len(signal_data) - 1)
                value = signal_data[sig_idx]

                if act_type == 'position':
                    # value is either scalar or (3,)
                    if np.ndim(value) == 0:
                        # Scalar signal applied to all active DOFs
                        val_3d = np.array([value, value, value]) * dof
                    else:
                        val_3d = np.array(value[:3]) * dof

                    pos_target[step_idx, node_idx] = val_3d

                elif act_type == 'force':
                    if np.ndim(value) == 0:
                        val_3d = np.array([value, value, value]) * dof
                    else:
                        val_3d = np.array(value[:3]) * dof

                    f_ext[step_idx, node_idx] = val_3d

        n_pos_driven = sum(1 for a in actuators if a.get('type', 'position') == 'position')
        n_force_driven = sum(1 for a in actuators if a.get('type', 'position') == 'force')

        if n_pos_driven > 0:
            self.logger.info(f"Built position actuation for {n_pos_driven} actuator(s)")
        if n_force_driven > 0:
            self.logger.info(f"Built force actuation for {n_force_driven} actuator(s)")

        return jnp.array(f_ext), jnp.array(pos_target)

    # ============================================================
    # Save / Export
    # ============================================================

    def save(self, path):
        """
        Save optimized parameters as a new geometry.h5 file.

        The file has the same structure as the original geometry.h5
        but with updated parameter values. Can be loaded directly by
        DEMLAT for re-simulation.
        """
        path = Path(path)
        theta_np = theta_to_numpy(self.solver.get_theta())

        # Copy original geometry and overwrite parameters
        import shutil
        src = self.geometry_dir / "input" / "geometry.h5"
        shutil.copy2(src, path)

        with h5py.File(path, 'a') as f:
            # Update bar parameters
            if 'bar_stiffness' in theta_np and 'elements/bars' in f:
                # Need to handle soft-only indexing
                all_stiffness = np.array(f['elements/bars/stiffness'])
                is_soft = all_stiffness >= 0
                all_stiffness[is_soft] = theta_np['bar_stiffness']
                f['elements/bars/stiffness'][...] = all_stiffness

            if 'bar_damping' in theta_np and 'elements/bars/damping' in f:
                all_damping = np.array(f['elements/bars/damping'])
                is_soft = np.array(f['elements/bars/stiffness']) >= 0
                all_damping[is_soft] = theta_np['bar_damping']
                f['elements/bars/damping'][...] = all_damping

            if 'bar_rest_length' in theta_np and 'elements/bars/rest_length' in f:
                all_rl = np.array(f['elements/bars/rest_length'])
                is_soft = np.array(f['elements/bars/stiffness']) >= 0
                all_rl[is_soft] = theta_np['bar_rest_length']
                f['elements/bars/rest_length'][...] = all_rl

            # Update hinge parameters
            if 'hinge_stiffness' in theta_np and 'elements/hinges' in f:
                all_k = np.array(f['elements/hinges/stiffness'])
                is_soft = all_k >= 0
                all_k[is_soft] = theta_np['hinge_stiffness']
                f['elements/hinges/stiffness'][...] = all_k

            if 'hinge_rest_angle' in theta_np and 'elements/hinges/angle' in f:
                all_phi = np.array(f['elements/hinges/angle'])
                is_soft = np.array(f['elements/hinges/stiffness']) >= 0
                all_phi[is_soft] = theta_np['hinge_rest_angle']
                f['elements/hinges/angle'][...] = all_phi

            # Store calibration metadata
            if 'calibration' in f:
                del f['calibration']
            cal_grp = f.create_group('calibration')
            for key, val in theta_np.items():
                cal_grp.create_dataset(key, data=val)

        self.logger.info(f"Saved optimized geometry to {path}")