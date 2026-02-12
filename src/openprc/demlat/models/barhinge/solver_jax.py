"""
JAX Native Physics Solver
==========================

Uses JAX's automatic differentiation for robust and efficient physics.
Forces are derived automatically from Potential Energy and Rayleigh Dissipation functions.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad
from functools import partial

from ...utils.logging import get_logger


class JaxSolver:
    """
    JAX-native physics solver using Automatic Differentiation (AD).

    Instead of manually implementing force kernels, we define:
    1. Potential Energy Function V(x) -> Conservative Forces F = -grad(V)
    2. Rayleigh Dissipation Function R(v) -> Damping Forces F = -grad_v(R)

    This ensures mathematical consistency and leverages JAX's XLA compiler for optimization.
    """

    def __init__(self, n_nodes, mass, attributes, bars, hinges=None, options=None):
        """Initialize JAX solver."""
        self.logger = get_logger("demlat.model.jax")
        self.n_nodes = n_nodes
        self.options = options or {}

        # Convert to JAX arrays (immutable)
        self.mass = jnp.array(mass, dtype=jnp.float32)
        self.attrs = jnp.array(attributes, dtype=jnp.uint8)

        # Physics parameters
        self.gravity = float(self.options.get('gravity', -9.81))
        self.damping = float(self.options.get('global_damping', 0.1))

        # Parse bars (soft only)
        self.n_bars = 0
        if bars and len(bars.get('indices', [])) > 0:
            is_soft = np.array(bars['stiffness']) >= 0
            if np.any(is_soft):
                self.n_bars = int(np.sum(is_soft))
                self.bar_indices = jnp.array(bars['indices'][is_soft], dtype=jnp.int32)
                self.bar_k = jnp.array(bars['stiffness'][is_soft], dtype=jnp.float32)
                self.bar_l0 = jnp.array(bars['rest_length'][is_soft], dtype=jnp.float32)
                self.bar_c = jnp.array(bars['damping'][is_soft], dtype=jnp.float32)

                if 'prestress' in bars and bars['prestress'] is not None:
                    self.bar_prestress = jnp.array(bars['prestress'][is_soft], dtype=jnp.float32)
                else:
                    self.bar_prestress = jnp.zeros(self.n_bars, dtype=jnp.float32)

        # Parse hinges (soft only)
        self.n_hinges = 0
        if hinges and len(hinges.get('indices', [])) > 0:
            is_soft = np.array(hinges['stiffness']) >= 0
            if np.any(is_soft):
                self.n_hinges = int(np.sum(is_soft))
                self.hinge_indices = jnp.array(hinges['indices'][is_soft], dtype=jnp.int32)
                self.hinge_k = jnp.array(hinges['stiffness'][is_soft], dtype=jnp.float32)
                self.hinge_phi0 = jnp.array(hinges['angle'][is_soft], dtype=jnp.float32)
                self.hinge_c = jnp.array(hinges['damping'][is_soft], dtype=jnp.float32)

        # Actuation Indices
        self.pos_actuator_indices = jnp.where(self.attrs & 2)[0]
        self.force_actuator_indices = jnp.where(self.attrs & 4)[0]
        self.fixed_indices = jnp.where(self.attrs & 1)[0]

        # State: [positions, velocities] flattened
        self.state = jnp.zeros(2 * n_nodes * 3, dtype=jnp.float32)

        # Build JIT-compiled physics function
        self._build_physics_function()

        self.logger.info(f"JAX AD Solver: {n_nodes} nodes, {self.n_bars} bars, {self.n_hinges} hinges")

    def _build_physics_function(self):
        """Build JIT-compiled physics ODE system using Automatic Differentiation."""

        # --- 1. Potential Energy Function (V) ---
        def potential_energy(x):
            pe = 0.0

            # Gravity: V = -m * g * z (assuming g is along Z and negative)
            # If gravity = -9.81, Force = m*(-9.81).
            # F = -grad(V) => grad(V) = -F = -m*(-9.81) = m*9.81.
            # V = m * 9.81 * z = m * (-gravity) * z.
            pe += jnp.sum(self.mass * (-self.gravity) * x[:, 2])

            # Bars: V = 0.5 * k * (L - L0)^2
            if self.n_bars > 0:
                i = self.bar_indices[:, 0]
                j = self.bar_indices[:, 1]
                
                xi = x[i]
                xj = x[j]
                dx = xj - xi
                dist = jnp.linalg.norm(dx, axis=1)
                
                l_eff = self.bar_l0 * (1.0 + self.bar_prestress)
                
                # Energy
                pe += 0.5 * jnp.sum(self.bar_k * (dist - l_eff)**2)

            # Hinges: V = 0.5 * k * (phi - phi0)^2
            if self.n_hinges > 0:
                j_idx = self.hinge_indices[:, 0]
                k_idx = self.hinge_indices[:, 1]
                i_idx = self.hinge_indices[:, 2]
                l_idx = self.hinge_indices[:, 3]

                xj = x[j_idx]
                xk = x[k_idx]
                xi = x[i_idx]
                xl = x[l_idx]

                # Vectors
                r_ij = xi - xj
                r_kj = xk - xj
                r_kl = xk - xl

                # Normals
                m = jnp.cross(r_ij, r_kj)
                n = jnp.cross(r_kj, r_kl)
                
                # Lengths (add epsilon to avoid NaN in grad)
                len_m = jnp.linalg.norm(m, axis=1) + 1e-12
                len_n = jnp.linalg.norm(n, axis=1) + 1e-12
                
                # Unit normals
                m_hat = m / len_m[:, None]
                n_hat = n / len_n[:, None]
                
                # Cosine
                cos_phi = jnp.sum(m_hat * n_hat, axis=1)
                cos_phi = jnp.clip(cos_phi, -1.0, 1.0)
                
                # Sine (via triple product with edge vector)
                # Edge vector r_kj
                len_rkj = jnp.linalg.norm(r_kj, axis=1) + 1e-12
                rkj_hat = r_kj / len_rkj[:, None]
                
                sin_phi = jnp.sum(jnp.cross(m_hat, n_hat) * rkj_hat, axis=1)
                
                # Angle in (-pi, pi]
                phi = jnp.arctan2(sin_phi, cos_phi)
                
                # Difference
                delta = phi - self.hinge_phi0
                
                # Wrap delta to [-pi, pi]
                delta = jnp.arctan2(jnp.sin(delta), jnp.cos(delta))
                
                pe += 0.5 * jnp.sum(self.hinge_k * delta**2)

            return pe

        # --- 2. Rayleigh Dissipation Function (R) ---
        def dissipation_function(x, v):
            R = 0.0
            
            # Global Viscous Damping: F = -c * m * v
            # R = 0.5 * c * m * v^2
            R += 0.5 * self.damping * jnp.sum(self.mass[:, None] * v**2)
            
            # Bar Internal Damping
            if self.n_bars > 0:
                i = self.bar_indices[:, 0]
                j = self.bar_indices[:, 1]
                
                dx = x[j] - x[i]
                dv = v[j] - v[i]
                
                dist = jnp.linalg.norm(dx, axis=1) + 1e-12
                n_vec = dx / dist[:, None]
                
                # Relative velocity along the bar
                v_rel = jnp.sum(dv * n_vec, axis=1)
                
                # R = 0.5 * c * v_rel^2
                R += 0.5 * jnp.sum(self.bar_c * v_rel**2)
                
            return R

        # --- 3. Derivatives ---
        # Force = -grad(Potential)
        get_conservative_forces = jit(grad(potential_energy))
        
        # Damping Force = -grad_v(Rayleigh)
        get_dissipative_forces = jit(grad(dissipation_function, argnums=1))

        # --- 4. ODE System ---
        @jit
        def physics_ode(state, t, external_forces):
            n = self.n_nodes
            x = state[:n*3].reshape(n, 3)
            v = state[n*3:].reshape(n, 3)

            # Compute Forces via AD
            # Note: grad returns +gradient, so Force is -gradient
            f_cons = -get_conservative_forces(x)
            f_diss = -get_dissipative_forces(x, v)
            
            f_total = f_cons + f_diss + external_forces.reshape(n, 3)
            
            # Clamp forces for stability
            f_total = jnp.clip(f_total, -1e6, 1e6)

            # Acceleration
            a = f_total / self.mass[:, None]

            # Constraints
            # Fixed (1) or PosDriven (2)
            is_constrained = (self.attrs & 1) | (self.attrs & 2)
            active_mask = (is_constrained == 0)
            
            # Zero acceleration for constrained nodes
            a = jnp.where(active_mask[:, None], a, 0.0)
            
            # However, we must ensure FIXED nodes have v=0.
            is_fixed = (self.attrs & 1)
            fixed_mask = (is_fixed == 0)
            v_out = jnp.where(fixed_mask[:, None], v, 0.0)

            return jnp.concatenate([v_out.flatten(), a.flatten()])

        self.physics_ode = physics_ode

        # --- 5. Integrator ---
        @jit
        def rk4_step(state, t, dt, external_forces):
            k1 = physics_ode(state, t, external_forces)
            k2 = physics_ode(state + 0.5 * dt * k1, t + 0.5 * dt, external_forces)
            k3 = physics_ode(state + 0.5 * dt * k2, t + 0.5 * dt, external_forces)
            k4 = physics_ode(state + dt * k3, t + dt, external_forces)
            return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        self.rk4_step_jit = rk4_step

    def upload_state(self, x, v):
        x_flat = jnp.array(x, dtype=jnp.float32).flatten()
        v_flat = jnp.array(v, dtype=jnp.float32).flatten()
        self.state = jnp.concatenate([x_flat, v_flat])

    def download_state(self):
        n = self.n_nodes
        x = np.array(self.state[:n*3]).reshape(n, 3)
        v = np.array(self.state[n*3:]).reshape(n, 3)
        return x, v

    def step(self, t, dt, actuation_map):
        n = self.n_nodes
        f_ext = jnp.zeros(n * 3, dtype=jnp.float32)

        # Position Actuation: Set Velocity
        if len(self.pos_actuator_indices) > 0 and actuation_map:
            state_copy = self.state
            
            current_x = self.state[:n*3].reshape(n, 3)
            
            for node_idx in self.pos_actuator_indices:
                node_idx = int(node_idx)
                if node_idx in actuation_map and actuation_map[node_idx]['type'] == 'position':
                    target_pos = jnp.array(actuation_map[node_idx]['value'], dtype=jnp.float32)
                    current_pos = current_x[node_idx]
                    
                    # Calculate required velocity
                    if dt > 1e-9:
                        v_cmd = (target_pos - current_pos) / dt
                    else:
                        v_cmd = jnp.zeros(3)
                        
                    # Set velocity in state
                    state_copy = state_copy.at[n*3 + node_idx*3 : n*3 + (node_idx+1)*3].set(v_cmd)

            self.state = state_copy

        # Force Actuation
        if len(self.force_actuator_indices) > 0 and actuation_map:
            f_ext_array = np.array(f_ext).reshape(n, 3)
            for node_idx in self.force_actuator_indices:
                node_idx = int(node_idx)
                if node_idx in actuation_map and actuation_map[node_idx]['type'] == 'force':
                    f_ext_array[node_idx] = np.array(actuation_map[node_idx]['value'], dtype=np.float32)
            f_ext = jnp.array(f_ext_array.flatten(), dtype=jnp.float32)

        # Integration
        self.state = self.rk4_step_jit(self.state, t, dt, f_ext)

        # Post-step cleanup for fixed nodes
        if len(self.fixed_indices) > 0:
            for idx in self.fixed_indices:
                idx = int(idx)
                self.state = self.state.at[n*3 + idx*3 : n*3 + (idx+1)*3].set(0.0)
