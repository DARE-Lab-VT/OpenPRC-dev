"""
JAX Native Physics Solver (Differentiable)
============================================

Uses JAX's automatic differentiation for robust and efficient physics.
Forces are derived automatically from Potential Energy and Rayleigh Dissipation functions.

REFACTORED: All physics parameters (theta) are explicit function arguments,
enabling end-to-end differentiable simulation for optimization and calibration.
The public API remains fully backward-compatible with the CUDA solver interface.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad
from functools import partial

from openprc.schemas.logging import get_logger


# ============================================================
# Theta (Parameter) Utilities
# ============================================================

def make_theta(bars=None, hinges=None, options=None):
    """
    Construct a theta pytree from simulation config.

    Returns
    -------
    dict  –  JAX pytree with all differentiable physics parameters.
        Keys: bar_stiffness, bar_damping, bar_rest_length, bar_prestress,
              hinge_stiffness, hinge_damping, hinge_rest_angle,
              global_damping, gravity
    """
    options = options or {}
    theta = {
        'gravity': jnp.array(float(options.get('gravity', -9.81))),
        'global_damping': jnp.array(float(options.get('global_damping', 0.1))),
    }

    # Bars
    if bars and len(bars.get('indices', [])) > 0:
        is_soft = np.array(bars['stiffness']) >= 0
        if np.any(is_soft):
            theta['bar_stiffness'] = jnp.array(bars['stiffness'][is_soft], dtype=jnp.float32)
            theta['bar_damping'] = jnp.array(bars['damping'][is_soft], dtype=jnp.float32)
            theta['bar_rest_length'] = jnp.array(bars['rest_length'][is_soft], dtype=jnp.float32)
            prestress = bars.get('prestress')
            if prestress is not None:
                theta['bar_prestress'] = jnp.array(prestress[is_soft], dtype=jnp.float32)
            else:
                theta['bar_prestress'] = jnp.zeros(int(np.sum(is_soft)), dtype=jnp.float32)

    # Hinges
    if hinges and len(hinges.get('indices', [])) > 0:
        is_soft = np.array(hinges['stiffness']) >= 0
        if np.any(is_soft):
            theta['hinge_stiffness'] = jnp.array(hinges['stiffness'][is_soft], dtype=jnp.float32)
            theta['hinge_damping'] = jnp.array(hinges['damping'][is_soft], dtype=jnp.float32)
            theta['hinge_rest_angle'] = jnp.array(hinges['angle'][is_soft], dtype=jnp.float32)

    return theta


# ============================================================
# Pure-Functional Energy & Dissipation (theta-explicit)
# ============================================================

def _potential_energy(x, theta, mass, bar_indices, hinge_indices):
    """
    Total potential energy V(x; theta).

    All physics parameters come from `theta` so JAX can differentiate
    through them.  Topology (indices) and mass are static/traced.
    """
    pe = 0.0

    # --- Gravity ---
    gravity = theta['gravity']
    pe = pe + jnp.sum(mass * (-gravity) * x[:, 2])

    # --- Bars: V = 0.5 * k * (L - L_eff)^2 ---
    if bar_indices is not None:
        k = theta['bar_stiffness']
        l0 = theta['bar_rest_length']
        prestress = theta['bar_prestress']

        i = bar_indices[:, 0]
        j = bar_indices[:, 1]
        dx = x[j] - x[i]
        dist = jnp.linalg.norm(dx, axis=1)
        l_eff = l0 * (1.0 + prestress)
        pe = pe + 0.5 * jnp.sum(k * (dist - l_eff) ** 2)

    # --- Hinges: V = 0.5 * k * wrap(phi - phi0)^2 ---
    if hinge_indices is not None:
        k_h = theta['hinge_stiffness']
        phi0 = theta['hinge_rest_angle']

        j_idx = hinge_indices[:, 0]
        k_idx = hinge_indices[:, 1]
        i_idx = hinge_indices[:, 2]
        l_idx = hinge_indices[:, 3]

        xj, xk, xi, xl = x[j_idx], x[k_idx], x[i_idx], x[l_idx]

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
        delta = jnp.arctan2(jnp.sin(phi - phi0), jnp.cos(phi - phi0))

        pe = pe + 0.5 * jnp.sum(k_h * delta ** 2)

    return pe


def _dissipation_function(x, v, theta, mass, bar_indices):
    """
    Rayleigh dissipation R(x, v; theta).

    R is defined such that F_damping = -grad_v(R).
    """
    R = 0.0
    c_global = theta['global_damping']

    # Global viscous damping
    R = R + 0.5 * c_global * jnp.sum(mass[:, None] * v ** 2)

    # Bar internal damping
    if bar_indices is not None:
        c_bar = theta['bar_damping']
        i = bar_indices[:, 0]
        j = bar_indices[:, 1]

        dx = x[j] - x[i]
        dv = v[j] - v[i]
        dist = jnp.linalg.norm(dx, axis=1) + 1e-12
        n_vec = dx / dist[:, None]
        v_rel = jnp.sum(dv * n_vec, axis=1)

        R = R + 0.5 * jnp.sum(c_bar * v_rel ** 2)

    return R


# ============================================================
# Per-Element Energy Decomposition (for analysis / RC)
# ============================================================

def _bar_energies(x, theta, bar_indices):
    """Return per-bar potential energy array."""
    if bar_indices is None:
        return jnp.array([])
    k = theta['bar_stiffness']
    l0 = theta['bar_rest_length']
    prestress = theta['bar_prestress']
    i, j = bar_indices[:, 0], bar_indices[:, 1]
    dx = x[j] - x[i]
    dist = jnp.linalg.norm(dx, axis=1)
    l_eff = l0 * (1.0 + prestress)
    return 0.5 * k * (dist - l_eff) ** 2


def _hinge_energies(x, theta, hinge_indices):
    """Return per-hinge potential energy array."""
    if hinge_indices is None:
        return jnp.array([])
    k_h = theta['hinge_stiffness']
    phi0 = theta['hinge_rest_angle']

    j_idx = hinge_indices[:, 0]
    k_idx = hinge_indices[:, 1]
    i_idx = hinge_indices[:, 2]
    l_idx = hinge_indices[:, 3]

    xj, xk, xi, xl = x[j_idx], x[k_idx], x[i_idx], x[l_idx]
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
    delta = jnp.arctan2(jnp.sin(phi - phi0), jnp.cos(phi - phi0))

    return 0.5 * k_h * delta ** 2


# ============================================================
# Explicit Force Functions (no grad — for use inside rollout)
# ============================================================
# These compute forces directly so that the outer value_and_grad
# only needs single-level differentiation w.r.t. theta.

def _gravity_forces(mass, theta):
    """Return (N, 3) gravity force array."""
    g = theta['gravity']
    f = jnp.zeros((mass.shape[0], 3))
    return f.at[:, 2].set(mass * g)


def _bar_forces(x, theta, bar_indices):
    """Explicit bar spring forces. Returns (N, 3) accumulated forces."""
    if bar_indices is None:
        return jnp.zeros_like(x)

    k = theta['bar_stiffness']
    l0 = theta['bar_rest_length']
    prestress = theta['bar_prestress']

    i = bar_indices[:, 0]
    j = bar_indices[:, 1]

    dx = x[j] - x[i]
    dist = jnp.linalg.norm(dx, axis=1) + 1e-12
    l_eff = l0 * (1.0 + prestress)

    # Force magnitude: -k * (dist - l_eff), direction: along dx
    force_mag = k * (dist - l_eff) / dist  # scalar per bar
    f_vec = force_mag[:, None] * dx  # (M, 3) force on node i from bar

    # Scatter to nodes: f on i is +f_vec, f on j is -f_vec
    n = x.shape[0]
    forces = jnp.zeros((n, 3))
    forces = forces.at[i].add(f_vec)
    forces = forces.at[j].add(-f_vec)
    return forces


def _hinge_forces(x, theta, hinge_indices):
    """
    Explicit hinge torque forces via energy gradient.

    For hinges, the explicit force derivation is complex (involves
    derivatives of dihedral angle w.r.t. all 4 node positions).
    We use grad of the hinge energy only — this is a single-level grad
    w.r.t. x (not nested with the outer theta grad, since JAX handles
    grad-of-different-variables efficiently).
    """
    if hinge_indices is None:
        return jnp.zeros_like(x)

    def hinge_pe(positions):
        k_h = theta['hinge_stiffness']
        phi0 = theta['hinge_rest_angle']

        j_idx = hinge_indices[:, 0]
        k_idx = hinge_indices[:, 1]
        i_idx = hinge_indices[:, 2]
        l_idx = hinge_indices[:, 3]

        xj, xk, xi, xl = positions[j_idx], positions[k_idx], positions[i_idx], positions[l_idx]
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
        delta = jnp.arctan2(jnp.sin(phi - phi0), jnp.cos(phi - phi0))

        return 0.5 * jnp.sum(k_h * delta ** 2)

    return -grad(hinge_pe)(x)


def _global_damping_forces(v, mass, theta):
    """Explicit global viscous damping: F = -c * m * v."""
    c = theta['global_damping']
    return -c * mass[:, None] * v


def _bar_damping_forces(x, v, theta, bar_indices):
    """Explicit bar internal damping forces."""
    if bar_indices is None:
        return jnp.zeros_like(x)

    c_bar = theta['bar_damping']
    i = bar_indices[:, 0]
    j = bar_indices[:, 1]

    dx = x[j] - x[i]
    dv = v[j] - v[i]
    dist = jnp.linalg.norm(dx, axis=1) + 1e-12
    n_vec = dx / dist[:, None]

    # Relative velocity along bar axis
    v_rel = jnp.sum(dv * n_vec, axis=1)

    # Damping force along bar direction
    f_damp = c_bar * v_rel
    f_vec = f_damp[:, None] * n_vec  # (M, 3)

    # Force on i is +f_vec (opposes relative motion), on j is -f_vec
    n = x.shape[0]
    forces = jnp.zeros((n, 3))
    forces = forces.at[i].add(f_vec)
    forces = forces.at[j].add(-f_vec)
    return forces


def _compute_all_forces(x, v, theta, mass, bar_indices, hinge_indices):
    """Total force on all nodes (explicit, single-level differentiable)."""
    f = _gravity_forces(mass, theta)
    f = f + _bar_forces(x, theta, bar_indices)
    f = f + _hinge_forces(x, theta, hinge_indices)
    f = f + _global_damping_forces(v, mass, theta)
    f = f + _bar_damping_forces(x, v, theta, bar_indices)
    return f


# ============================================================
# Differentiable Rollout
# ============================================================

def build_rollout(n_nodes, mass, attrs, bar_indices, hinge_indices):
    """
    Build a pure-functional, JIT-compiled rollout function.

    Uses EXPLICIT force computation (not nested grad) so that the
    outer value_and_grad for optimization is single-level differentiation.

    Supports position actuation via velocity injection (matching JaxSolver.step).

    Returns
    -------
    rollout_fn : callable
        (theta, x0, v0, f_ext_sequence, pos_target_sequence, dt) -> trajectory
        Fully differentiable w.r.t. theta.
    """
    mass_jax = jnp.array(mass, dtype=jnp.float32)
    attrs_jax = jnp.array(attrs, dtype=jnp.uint8)

    bi = jnp.array(bar_indices, dtype=jnp.int32) if bar_indices is not None else None
    hi = jnp.array(hinge_indices, dtype=jnp.int32) if hinge_indices is not None else None

    is_constrained = (attrs_jax & 1) | (attrs_jax & 2)
    active_mask = (is_constrained == 0)
    fixed_mask = ((attrs_jax & 1) == 0)

    # Position-driven nodes: bit 1 (0x02)
    pos_driven_mask = ((attrs_jax & 2) != 0)  # (N,) bool
    has_pos_actuation = jnp.any(pos_driven_mask)

    # --- ODE right-hand side (explicit forces, no nested grad) ---
    def physics_ode(x, v, theta, f_ext):
        f_total = _compute_all_forces(x, v, theta, mass_jax, bi, hi) + f_ext
        f_total = jnp.clip(f_total, -1e6, 1e6)

        a = f_total / mass_jax[:, None]
        a = jnp.where(active_mask[:, None], a, 0.0)
        v_out = jnp.where(fixed_mask[:, None], v, 0.0)

        return v_out, a

    # --- Single RK4 step ---
    def rk4_step(x, v, theta, f_ext, dt):
        v1, a1 = physics_ode(x, v, theta, f_ext)
        v2, a2 = physics_ode(x + 0.5 * dt * v1, v + 0.5 * dt * a1, theta, f_ext)
        v3, a3 = physics_ode(x + 0.5 * dt * v2, v + 0.5 * dt * a2, theta, f_ext)
        v4, a4 = physics_ode(x + dt * v3, v + dt * a3, theta, f_ext)

        x_new = x + (dt / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4)
        v_new = v + (dt / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4)

        # Enforce constraints
        v_new = jnp.where(fixed_mask[:, None], v_new, 0.0)

        return x_new, v_new

    # --- Full rollout with checkpointing ---
    @jax.checkpoint
    def _single_step(carry, step_input):
        x, v, theta, dt = carry
        f_ext_t, pos_target_t = step_input

        # --- Position actuation: inject velocity ---
        # v_cmd = (target - current) / dt for driven nodes
        v_cmd = (pos_target_t - x) / jnp.maximum(dt, 1e-9)
        v = jnp.where(pos_driven_mask[:, None], v_cmd, v)

        x_new, v_new = rk4_step(x, v, theta, f_ext_t, dt)

        # Enforce position-driven nodes: set their position exactly
        x_new = jnp.where(pos_driven_mask[:, None], pos_target_t, x_new)

        return (x_new, v_new, theta, dt), x_new

    @jax.checkpoint
    def _single_step_no_actuation(carry, f_ext_t):
        x, v, theta, dt = carry
        x_new, v_new = rk4_step(x, v, theta, f_ext_t, dt)
        return (x_new, v_new, theta, dt), x_new

    def rollout(theta, x0, v0, f_ext_sequence, pos_target_sequence, dt):
        """
        Roll out simulation.

        Parameters
        ----------
        theta : dict             – physics parameters (differentiable)
        x0    : (N, 3)           – initial positions
        v0    : (N, 3)           – initial velocities
        f_ext_sequence : (T, N, 3) – external forces at each step
        pos_target_sequence : (T, N, 3) or None
            Position targets at each step for driven nodes.
            Non-driven node values are ignored (masked).
            Pass zeros array if no position actuation.
        dt    : float            – timestep

        Returns
        -------
        x_traj    : (T, N, 3) – positions at each step
        x_final   : (N, 3)   – final positions
        v_final   : (N, 3)   – final velocities
        """
        init_carry = (x0, v0, theta, jnp.array(dt))

        if has_pos_actuation:
            step_inputs = (f_ext_sequence, pos_target_sequence)
            (x_final, v_final, _, _), x_traj = jax.lax.scan(
                _single_step, init_carry, step_inputs
            )
        else:
            (x_final, v_final, _, _), x_traj = jax.lax.scan(
                _single_step_no_actuation, init_carry, f_ext_sequence
            )

        return x_traj, x_final, v_final

    return jit(rollout)


# ============================================================
# Main Solver Class (Backward-Compatible Public API)
# ============================================================

class JaxSolver:
    """
    JAX-native physics solver using Automatic Differentiation (AD).

    Defines:
      1. Potential Energy Function V(x, theta) -> Conservative Forces F = -grad_x(V)
      2. Rayleigh Dissipation Function R(v, theta) -> Damping Forces F = -grad_v(R)

    The solver is fully differentiable w.r.t. theta for use in optimization.
    The public API (upload_state, download_state, step) is identical to the
    CUDA solver, preserving backward compatibility.
    """

    def __init__(self, n_nodes, mass, attributes, bars, hinges=None, options=None):
        """Initialize JAX solver."""
        self.logger = get_logger("demlat.model.jax")
        self.n_nodes = n_nodes
        self.options = options or {}

        # Immutable topology / mass
        self.mass = jnp.array(mass, dtype=jnp.float32)
        self.attrs = jnp.array(attributes, dtype=jnp.uint8)

        # --- Build theta from config ---
        self.theta = make_theta(bars, hinges, options)

        # --- Store topology indices (static, not in theta) ---
        self.bar_indices = None
        self.n_bars = 0
        if bars and len(bars.get('indices', [])) > 0:
            is_soft = np.array(bars['stiffness']) >= 0
            if np.any(is_soft):
                self.n_bars = int(np.sum(is_soft))
                self.bar_indices = jnp.array(bars['indices'][is_soft], dtype=jnp.int32)

        self.hinge_indices = None
        self.n_hinges = 0
        if hinges and len(hinges.get('indices', [])) > 0:
            is_soft = np.array(hinges['stiffness']) >= 0
            if np.any(is_soft):
                self.n_hinges = int(np.sum(is_soft))
                self.hinge_indices = jnp.array(hinges['indices'][is_soft], dtype=jnp.int32)

        # Actuation / constraint indices
        self.pos_actuator_indices = jnp.where(self.attrs & 2)[0]
        self.force_actuator_indices = jnp.where(self.attrs & 4)[0]
        self.fixed_indices = jnp.where(self.attrs & 1)[0]

        # State
        self.state = jnp.zeros(2 * n_nodes * 3, dtype=jnp.float32)

        # Build JIT-compiled physics (uses self.theta internally)
        self._build_physics_function()

        self.logger.info(
            f"JAX AD Solver (differentiable): "
            f"{n_nodes} nodes, {self.n_bars} bars, {self.n_hinges} hinges"
        )

    # --------------------------------------------------------
    # Backward-compatible physics (reads theta from self)
    # --------------------------------------------------------

    def _build_physics_function(self):
        """Build JIT-compiled ODE using the functional energy definitions."""

        mass = self.mass
        bi = self.bar_indices
        hi = self.hinge_indices

        is_constrained = (self.attrs & 1) | (self.attrs & 2)
        active_mask = (is_constrained == 0)
        fixed_mask = ((self.attrs & 1) == 0)

        # We close over self.theta for the legacy API, but the underlying
        # functions are pure (theta-explicit) and reusable for optimization.

        def physics_ode(state, t, external_forces, theta):
            n = mass.shape[0]
            x = state[:n * 3].reshape(n, 3)
            v = state[n * 3:].reshape(n, 3)

            f_cons = -grad(lambda _x: _potential_energy(_x, theta, mass, bi, hi))(x)
            f_diss = -grad(lambda _v: _dissipation_function(x, _v, theta, mass, bi))(v)

            f_total = f_cons + f_diss + external_forces.reshape(n, 3)
            f_total = jnp.clip(f_total, -1e6, 1e6)

            a = f_total / mass[:, None]
            a = jnp.where(active_mask[:, None], a, 0.0)
            v_out = jnp.where(fixed_mask[:, None], v, 0.0)

            return jnp.concatenate([v_out.flatten(), a.flatten()])

        @jit
        def rk4_step(state, t, dt, external_forces, theta):
            k1 = physics_ode(state, t, external_forces, theta)
            k2 = physics_ode(state + 0.5 * dt * k1, t + 0.5 * dt, external_forces, theta)
            k3 = physics_ode(state + 0.5 * dt * k2, t + 0.5 * dt, external_forces, theta)
            k4 = physics_ode(state + dt * k3, t + dt, external_forces, theta)
            return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        self._physics_ode = physics_ode
        self._rk4_step_jit = rk4_step

    # --------------------------------------------------------
    # Public API (identical to CUDA solver)
    # --------------------------------------------------------

    def upload_state(self, x, v):
        x_flat = jnp.array(x, dtype=jnp.float32).flatten()
        v_flat = jnp.array(v, dtype=jnp.float32).flatten()
        self.state = jnp.concatenate([x_flat, v_flat])

    def download_state(self):
        n = self.n_nodes
        x = np.array(self.state[:n * 3]).reshape(n, 3)
        v = np.array(self.state[n * 3:]).reshape(n, 3)
        return x, v

    def step(self, t, dt, actuation_map):
        n = self.n_nodes
        f_ext = jnp.zeros(n * 3, dtype=jnp.float32)

        # Position actuation
        if len(self.pos_actuator_indices) > 0 and actuation_map:
            state_copy = self.state
            current_x = self.state[:n * 3].reshape(n, 3)

            for node_idx in self.pos_actuator_indices:
                node_idx = int(node_idx)
                if node_idx in actuation_map and actuation_map[node_idx]['type'] == 'position':
                    target_pos = jnp.array(actuation_map[node_idx]['value'], dtype=jnp.float32)
                    current_pos = current_x[node_idx]
                    if dt > 1e-9:
                        v_cmd = (target_pos - current_pos) / dt
                    else:
                        v_cmd = jnp.zeros(3)
                    state_copy = state_copy.at[n * 3 + node_idx * 3: n * 3 + (node_idx + 1) * 3].set(v_cmd)

            self.state = state_copy

        # Force actuation
        if len(self.force_actuator_indices) > 0 and actuation_map:
            f_ext_array = np.array(f_ext).reshape(n, 3)
            for node_idx in self.force_actuator_indices:
                node_idx = int(node_idx)
                if node_idx in actuation_map and actuation_map[node_idx]['type'] == 'force':
                    f_ext_array[node_idx] = np.array(actuation_map[node_idx]['value'], dtype=np.float32)
            f_ext = jnp.array(f_ext_array.flatten(), dtype=jnp.float32)

        # Integrate (pass theta explicitly to the JIT function)
        self.state = self._rk4_step_jit(self.state, t, dt, f_ext, self.theta)

        # Post-step: zero velocity of fixed nodes
        if len(self.fixed_indices) > 0:
            for idx in self.fixed_indices:
                idx = int(idx)
                self.state = self.state.at[n * 3 + idx * 3: n * 3 + (idx + 1) * 3].set(0.0)

    # --------------------------------------------------------
    # Differentiable API (new — used by openprc.optimize)
    # --------------------------------------------------------

    def get_theta(self):
        """Return current theta pytree (for inspection or as initial guess)."""
        return self.theta

    def set_theta(self, theta):
        """Update theta (e.g. after optimization)."""
        self.theta = theta

    def get_rollout_fn(self):
        """
        Return a pure-functional, JIT-compiled, differentiable rollout.

        Usage::

            rollout_fn = solver.get_rollout_fn()
            trajectory, x_final, v_final = rollout_fn(
                theta, x0, v0, f_ext_seq, pos_target_seq, dt
            )
            # pos_target_seq: (T, N, 3) position targets for driven nodes
            #   (zeros for non-actuated setups)
        """
        return build_rollout(
            self.n_nodes,
            np.array(self.mass),
            np.array(self.attrs),
            np.array(self.bar_indices) if self.bar_indices is not None else None,
            np.array(self.hinge_indices) if self.hinge_indices is not None else None,
        )

    def get_element_energies(self, x=None):
        """
        Per-element energy decomposition for the current (or given) state.

        Returns dict with 'bar_energies' and 'hinge_energies' arrays.
        """
        if x is None:
            x = jnp.array(self.state[:self.n_nodes * 3]).reshape(self.n_nodes, 3)

        return {
            'bar_energies': _bar_energies(x, self.theta, self.bar_indices),
            'hinge_energies': _hinge_energies(x, self.theta, self.hinge_indices),
        }

    def get_element_gradients(self, x=None):
        """
        Per-element gradient of total PE w.r.t. each parameter.

        Returns dict matching theta structure but with gradients.
        """
        if x is None:
            x = jnp.array(self.state[:self.n_nodes * 3]).reshape(self.n_nodes, 3)

        grad_fn = grad(lambda t: _potential_energy(x, t, self.mass, self.bar_indices, self.hinge_indices))
        return grad_fn(self.theta)