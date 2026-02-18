import numpy as np
import numba
from numba import njit, float64, int32, uint8
from ...utils.logging import get_logger

# --- Constants & Helpers ---
EPSILON = 1e-12
MIN_AREA = 1e-12
MAX_FORCE = 1e6

ATTR_FIXED = 1
ATTR_POS_DRIVEN = 2


@njit(fastmath=True)
def dot3(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@njit(fastmath=True)
def len3(v):
    return np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + EPSILON)


@njit(fastmath=True)
def cross3(a, b):
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ], dtype=float64)


@njit(fastmath=True)
def clamp(v, lo, hi):
    return max(lo, min(v, hi))


@njit(fastmath=True)
def is_active(idx, attrs):
    # Checks if particle is dynamic (not fixed, not position driven)
    return not ((attrs[idx] & ATTR_FIXED) or (attrs[idx] & ATTR_POS_DRIVEN))


# --- Physics Kernels (Numba) ---

@njit(parallel=False, fastmath=True)
def compute_bar_forces_cpu(n_bars, indices, params, x, v, attrs, f):
    for k in range(n_bars):
        i = indices[k, 0]
        j = indices[k, 1]

        stiffness = params[k, 0]
        if stiffness < 0: continue  # Handled by PBD

        l0 = params[k, 1]
        c = params[k, 2]
        prestress = params[k, 3]

        l_eff = l0 * (1.0 + prestress)

        # Vector i -> j
        dx = x[j] - x[i]
        dist = len3(dx)
        n = dx / dist

        # Spring Force
        f_spring = stiffness * (dist - l_eff)

        # Damping Force
        dv = v[j] - v[i]
        v_rel = dot3(dv, n)
        f_damp = c * v_rel

        f_total = clamp(f_spring + f_damp, -MAX_FORCE, MAX_FORCE)

        # Apply forces (Scatter)
        # Note: No atomic add needed if parallel=False.
        # If parallelizing, need temporary buffers or atomics.
        fx = f_total * n

        if is_active(i, attrs):
            f[i] += fx
        if is_active(j, attrs):
            f[j] -= fx


@njit(fastmath=True)
def compute_hinge_forces_cpu(n_hinges, indices, params, x, attrs, f):
    for h in range(n_hinges):
        j = indices[h, 0]
        k = indices[h, 1]
        i = indices[h, 2]
        l = indices[h, 3]

        k_tor = params[h, 0]
        phi0 = params[h, 1]

        # Vectors
        r_ij = x[i] - x[j]
        r_kj = x[k] - x[j]
        r_kl = x[k] - x[l]

        # Normals
        m = cross3(r_ij, r_kj)
        n = cross3(r_kj, r_kl)

        len_m = len3(m)
        len_n = len3(n)
        len_rkj = len3(r_kj)

        if len_m < MIN_AREA or len_n < MIN_AREA:
            continue

        # Angle
        cos_phi = clamp(dot3(m, n) / (len_m * len_n), -1.0, 1.0)
        phi = np.arccos(cos_phi)

        if dot3(m, r_kl) < 0:
            phi = -phi

        delta_phi = phi - phi0
        # Wrap
        if delta_phi > np.pi: delta_phi -= 2 * np.pi
        if delta_phi < -np.pi: delta_phi += 2 * np.pi

        f_torque = -k_tor * delta_phi

        # Gradients
        len_m_sq = len_m * len_m
        len_n_sq = len_n * len_n
        len_rkj_sq = len_rkj * len_rkj

        # Soften singularity
        min_sq_len = 1e-8
        len_m_sq = max(len_m_sq, min_sq_len)
        len_n_sq = max(len_n_sq, min_sq_len)

        # Coefficients
        coeff_i = len_rkj / len_m_sq
        dth_dxi = coeff_i * m

        coeff_l = -len_rkj / len_n_sq
        dth_dxl = coeff_l * n

        rij_dot_rkj = dot3(r_ij, r_kj)
        rkl_dot_rkj = dot3(r_kl, r_kj)

        c_ij = rij_dot_rkj / len_rkj_sq
        c_kl = rkl_dot_rkj / len_rkj_sq

        dth_dxj = (c_ij - 1.0) * dth_dxi - c_kl * dth_dxl
        dth_dxk = (c_kl - 1.0) * dth_dxl - c_ij * dth_dxi

        # Apply
        if is_active(i, attrs): f[i] += f_torque * dth_dxi
        if is_active(l, attrs): f[l] += f_torque * dth_dxl
        if is_active(j, attrs): f[j] += f_torque * dth_dxj
        if is_active(k, attrs): f[k] += f_torque * dth_dxk


@njit(fastmath=True)
def apply_global_forces_cpu(n_nodes, v, mass, attrs, damp, g, f):
    for i in range(n_nodes):
        if is_active(i, attrs):
            m = mass[i]
            # Damping
            f[i] -= damp * m * v[i]
            # Gravity (Z-axis)
            f[i, 2] += m * g


@njit(fastmath=True)
def rk4_integration_step(n, dt, x, v, f, mass, attrs, k_x, k_v, x_tmp, v_tmp, frac):
    for i in range(n):
        if not is_active(i, attrs):
            k_x[i] = 0.0
            k_v[i] = 0.0
            x_tmp[i] = x[i]
            v_tmp[i] = v[i]
            continue

        m = mass[i]
        minv = 1.0 / m if m > 0 else 0.0

        # k_v = a = F/m
        k_v[i] = f[i] * minv
        # k_x = v
        k_x[i] = v[i]

        # Prepare state for next stage
        dtf = dt * frac
        x_tmp[i] = x[i] + dtf * k_x[i]
        v_tmp[i] = v[i] + dtf * k_v[i]


@njit(fastmath=True)
def rk4_final_update_cpu(n, dt, x, v, k1x, k1v, k2x, k2v, k3x, k3v, k4x, k4v, attrs):
    dt6 = dt / 6.0
    for i in range(n):
        if is_active(i, attrs):
            x[i] += dt6 * (k1x[i] + 2 * k2x[i] + 2 * k3x[i] + k4x[i])
            v[i] += dt6 * (k1v[i] + 2 * k2v[i] + 2 * k3v[i] + k4v[i])


# --- PBD Kernels (Rigid Constraints) ---

@njit(fastmath=True)
def project_rigid_bars_cpu(n_rigid, indices, l0s, mass, attrs, x, omega):
    for k in range(n_rigid):
        i = indices[k, 0]
        j = indices[k, 1]
        l0 = l0s[k]

        dx = x[j] - x[i]
        dist = len3(dx)
        C = dist - l0

        if abs(C) < 1e-9: continue

        active_i = is_active(i, attrs)
        active_j = is_active(j, attrs)

        wi = (1.0 / mass[i]) if (active_i and mass[i] > 0) else 0.0
        wj = (1.0 / mass[j]) if (active_j and mass[j] > 0) else 0.0
        w_sum = wi + wj

        if w_sum < 1e-12: continue

        lambda_val = -omega * C / w_sum
        n = dx / dist
        corr = n * lambda_val

        if active_i: x[i] -= wi * corr
        if active_j: x[j] += wj * corr


@njit(fastmath=True)
def correct_rigid_velocity_cpu(n_rigid, indices, mass, attrs, x, v):
    for k in range(n_rigid):
        i = indices[k, 0]
        j = indices[k, 1]

        dx = x[j] - x[i]
        dist = len3(dx)
        n = dx / dist

        dv = v[j] - v[i]
        v_rel = dot3(dv, n)

        active_i = is_active(i, attrs)
        active_j = is_active(j, attrs)

        wi = (1.0 / mass[i]) if (active_i and mass[i] > 0) else 0.0
        wj = (1.0 / mass[j]) if (active_j and mass[j] > 0) else 0.0
        w_sum = wi + wj

        if w_sum < 1e-12: continue

        impulse = v_rel / w_sum

        if active_i: v[i] += wi * impulse * n
        if active_j: v[j] -= wj * impulse * n


@njit(fastmath=True)
def project_rigid_hinges_cpu(n_rigid, indices, phi0s, mass, attrs, x, omega):
    for h in range(n_rigid):
        j = indices[h, 0]
        k = indices[h, 1]
        i = indices[h, 2]
        l = indices[h, 3]
        phi_target = phi0s[h]

        # Fetch positions
        xi, xj, xk, xl = x[i], x[j], x[k], x[l]

        r_ij = xi - xj
        r_kj = xk - xj
        r_kl = xk - xl

        m = cross3(r_ij, r_kj)
        n = cross3(r_kj, r_kl)

        len_m = len3(m)
        len_n = len3(n)
        len_rkj = len3(r_kj)

        if len_m < MIN_AREA or len_n < MIN_AREA: continue

        cos_phi = clamp(dot3(m, n) / (len_m * len_n), -1.0, 1.0)
        phi = np.arccos(cos_phi)
        if dot3(m, r_kl) < 0: phi = -phi

        C = phi - phi_target
        # Normalize C to [-pi, pi]
        while C > np.pi: C -= 2 * np.pi
        while C < -np.pi: C += 2 * np.pi

        if abs(C) < 1e-4: continue

        # Gradients
        len_m_sq = len_m * len_m
        len_n_sq = len_n * len_n
        len_rkj_sq = len_rkj * len_rkj

        coeff_i = len_rkj / len_m_sq
        q_i = coeff_i * m

        coeff_l = -len_rkj / len_n_sq
        q_l = coeff_l * n

        rij_dot_rkj = dot3(r_ij, r_kj)
        rkl_dot_rkj = dot3(r_kl, r_kj)
        c_ij = rij_dot_rkj / len_rkj_sq
        c_kl = rkl_dot_rkj / len_rkj_sq

        q_j = (c_ij - 1.0) * q_i - c_kl * q_l
        q_k = (c_kl - 1.0) * q_l - c_ij * q_i

        # Weights
        active_i = is_active(i, attrs)
        active_j = is_active(j, attrs)
        active_k = is_active(k, attrs)
        active_l = is_active(l, attrs)

        wi = (1.0 / mass[i]) if (active_i and mass[i] > 0) else 0.0
        wj = (1.0 / mass[j]) if (active_j and mass[j] > 0) else 0.0
        wk = (1.0 / mass[k]) if (active_k and mass[k] > 0) else 0.0
        wl = (1.0 / mass[l]) if (active_l and mass[l] > 0) else 0.0

        sum_w_q2 = (wi * dot3(q_i, q_i) +
                    wj * dot3(q_j, q_j) +
                    wk * dot3(q_k, q_k) +
                    wl * dot3(q_l, q_l))

        if sum_w_q2 < 1e-12: continue

        lambda_val = -omega * C / sum_w_q2

        if active_i: x[i] += wi * lambda_val * q_i
        if active_j: x[j] += wj * lambda_val * q_j
        if active_k: x[k] += wk * lambda_val * q_k
        if active_l: x[l] += wl * lambda_val * q_l


class CpuSolver:
    """
    CPU-based fallback solver for the Bar-Hinge model.
    Mirrors CudaSolver API but uses Numba for execution.
    """

    def __init__(self, n_nodes, mass, attributes, bars, hinges=None, options=None):
        self.logger = get_logger("demlat.model.cpu")
        self.n_nodes = n_nodes
        self.options = options or {}

        # 1. State Allocation (Host RAM)
        self.x = np.zeros((n_nodes, 3), dtype=np.float64)
        self.v = np.zeros((n_nodes, 3), dtype=np.float64)
        self.f = np.zeros((n_nodes, 3), dtype=np.float64)

        # Physics Params
        self.mass = mass.astype(np.float64)
        self.attrs = attributes.astype(np.uint8)

        # 2. Bars (Soft vs Rigid)
        n_total_bars = len(bars['indices'])
        self.n_bars = 0
        self.n_rigid_bars = 0

        if n_total_bars > 0:
            is_rigid = bars['stiffness'] < 0

            # Soft
            soft_mask = ~is_rigid
            self.n_bars = np.sum(soft_mask)
            if self.n_bars > 0:
                self.bar_indices = bars['indices'][soft_mask].astype(np.int32)
                self.bar_params = np.zeros((self.n_bars, 4), dtype=np.float64)
                self.bar_params[:, 0] = bars['stiffness'][soft_mask]
                self.bar_params[:, 1] = bars['rest_length'][soft_mask]
                self.bar_params[:, 2] = bars['damping'][soft_mask]
                if 'prestress' in bars:
                    self.bar_params[:, 3] = bars['prestress'][soft_mask]

            # Rigid
            self.n_rigid_bars = np.sum(is_rigid)
            if self.n_rigid_bars > 0:
                self.rbar_indices = bars['indices'][is_rigid].astype(np.int32)
                self.rbar_l0 = bars['rest_length'][is_rigid].astype(np.float64)

        # 3. Hinges (Soft vs Rigid)
        self.n_hinges = 0
        self.n_rigid_hinges = 0

        if hinges and len(hinges['indices']) > 0:
            is_rigid_h = hinges['stiffness'] < 0

            # Soft
            soft_h_mask = ~is_rigid_h
            self.n_hinges = np.sum(soft_h_mask)
            if self.n_hinges > 0:
                self.hinge_indices = hinges['indices'][soft_h_mask].astype(np.int32)
                self.hinge_params = np.zeros((self.n_hinges, 4), dtype=np.float64)
                self.hinge_params[:, 0] = hinges['stiffness'][soft_h_mask]
                self.hinge_params[:, 1] = hinges['angle'][soft_h_mask]
                self.hinge_params[:, 2] = hinges['damping'][soft_h_mask]

            # Rigid
            self.n_rigid_hinges = np.sum(is_rigid_h)
            if self.n_rigid_hinges > 0:
                self.rhinge_indices = hinges['indices'][is_rigid_h].astype(np.int32)
                self.rhinge_phi = hinges['angle'][is_rigid_h].astype(np.float64)

        # 4. RK4 Buffers
        self.k1x = np.zeros_like(self.x);
        self.k1v = np.zeros_like(self.v)
        self.k2x = np.zeros_like(self.x);
        self.k2v = np.zeros_like(self.v)
        self.k3x = np.zeros_like(self.x);
        self.k3v = np.zeros_like(self.v)
        self.k4x = np.zeros_like(self.x);
        self.k4v = np.zeros_like(self.v)
        self.xt = np.zeros_like(self.x)
        self.vt = np.zeros_like(self.v)

        # 5. Actuation
        self.actuator_indices = np.where(self.attrs & 2)[0].astype(np.int32)

    def upload_state(self, x, v):
        self.x[:] = x.astype(np.float64)
        self.v[:] = v.astype(np.float64)

    def download_state(self):
        return self.x.astype(np.float32), self.v.astype(np.float32)

    def step(self, t, dt, actuation_map):
        # 1. Actuation
        if len(self.actuator_indices) > 0 and actuation_map:
            # We can do this with numpy mask since we are on CPU
            for idx in self.actuator_indices:
                if idx in actuation_map and actuation_map[idx]['type'] == 'position':
                    new_pos = actuation_map[idx]['value']
                    if dt > 1e-9:
                        self.v[idx] = (new_pos - self.x[idx]) / dt
                    self.x[idx] = new_pos

        # 2. Physics Params
        grav = self.options.get('gravity', -9.81)
        damp = self.options.get('global_damping', 0.1)

        def compute_forces(x_in, v_in, f_out):
            f_out.fill(0.0)
            if self.n_bars > 0:
                compute_bar_forces_cpu(self.n_bars, self.bar_indices, self.bar_params,
                                       x_in, v_in, self.attrs, f_out)
            if self.n_hinges > 0:
                compute_hinge_forces_cpu(self.n_hinges, self.hinge_indices, self.hinge_params,
                                         x_in, self.attrs, f_out)
            apply_global_forces_cpu(self.n_nodes, v_in, self.mass, self.attrs, damp, grav, f_out)

        # 3. RK4 Loop
        # Stage 1
        compute_forces(self.x, self.v, self.f)
        rk4_integration_step(self.n_nodes, dt, self.x, self.v, self.f, self.mass, self.attrs,
                             self.k1x, self.k1v, self.xt, self.vt, 0.5)
        # Stage 2
        compute_forces(self.xt, self.vt, self.f)
        rk4_integration_step(self.n_nodes, dt, self.x, self.v, self.f, self.mass, self.attrs,
                             self.k2x, self.k2v, self.xt, self.vt, 0.5)
        # Stage 3
        compute_forces(self.xt, self.vt, self.f)
        rk4_integration_step(self.n_nodes, dt, self.x, self.v, self.f, self.mass, self.attrs,
                             self.k3x, self.k3v, self.xt, self.vt, 1.0)
        # Stage 4
        compute_forces(self.xt, self.vt, self.f)
        rk4_integration_step(self.n_nodes, dt, self.x, self.v, self.f, self.mass, self.attrs,
                             self.k4x, self.k4v, self.xt, self.vt, 1.0)

        # Final Update
        rk4_final_update_cpu(self.n_nodes, dt, self.x, self.v,
                             self.k1x, self.k1v, self.k2x, self.k2v,
                             self.k3x, self.k3v, self.k4x, self.k4v, self.attrs)

        # 4. PBD Loop (Rigid)
        if self.n_rigid_bars > 0 or self.n_rigid_hinges > 0:
            for _ in range(5):
                if self.n_rigid_bars > 0:
                    project_rigid_bars_cpu(self.n_rigid_bars, self.rbar_indices, self.rbar_l0,
                                           self.mass, self.attrs, self.x, 0.8)
                if self.n_rigid_hinges > 0:
                    project_rigid_hinges_cpu(self.n_rigid_hinges, self.rhinge_indices, self.rhinge_phi,
                                             self.mass, self.attrs, self.x, 0.8)

            # Velocity Correction
            if self.n_rigid_bars > 0:
                correct_rigid_velocity_cpu(self.n_rigid_bars, self.rbar_indices,
                                           self.mass, self.attrs, self.x, self.v)
