/**
 * Bar-Hinge Physics Kernel
 * ========================
 *
 * This CUDA file implements the core physics calculations for a particle system
 * connected by bars (distance constraints) and hinges (angular constraints).
 *
 * Physics Model:
 * --------------
 * 1. Integration: Runge-Kutta 4 (RK4) for high-order accuracy in smooth regions.
 * 2. Constraints: Position Based Dynamics (PBD) for rigid (infinite stiffness) elements.
 *    - Note: Mixing RK4 and PBD is a hybrid approach. RK4 handles soft potentials,
 *      while PBD projects out errors for rigid constraints. This can degrade the
 *      theoretical order of accuracy of RK4 but provides stability for stiff links.
 *
 * Precision:
 * ----------
 * Uses double precision (64-bit) for all physics calculations to minimize
 * numerical drift, which is critical for scientific validity in chaotic systems.
 *
 * Thread Safety:
 * --------------
 * Uses atomicAdd for force accumulation and position updates to handle race conditions
 * where multiple elements affect the same node.
 * WARNING: The order of atomic operations is non-deterministic, meaning bit-exact
 * reproducibility between runs is not guaranteed.
 */

#include <math.h>

#define EPSILON 1e-12
#define MIN_AREA 1e-12
#define MAX_FORCE 1e6

// Attribute Bitmasks
#define ATTR_FIXED 1
#define ATTR_POS_DRIVEN 2
#define ATTR_FORCE_DRIVEN 4

// --- Helper Functions ---

__device__ inline double dot3(double ax, double ay, double az, double bx, double by, double bz) {
    return ax*bx + ay*by + az*bz;
}

__device__ inline void cross3(double ax, double ay, double az, double bx, double by, double bz, double* rx, double* ry, double* rz) {
    *rx = ay*bz - az*by; *ry = az*bx - ax*bz; *rz = ax*by - ay*bx;
}

__device__ inline double len3(double x, double y, double z) {
    return sqrt(x*x + y*y + z*z + EPSILON);
}

__device__ inline double clamp_d(double v, double lo, double hi) {
    return fmin(fmax(v, lo), hi);
}

__device__ inline bool is_physics_active(int i, const unsigned char* attrs) {
    unsigned char a = attrs[i];
    return !((a & ATTR_FIXED) || (a & ATTR_POS_DRIVEN));
}

/**
 * Atomic accumulation wrapper.
 * Checks if the target node is dynamic before adding values.
 */
__device__ inline void atomicAdd3(double* buf, int idx, double fx, double fy, double fz, const unsigned char* attrs) {
    if (is_physics_active(idx, attrs)) {
        atomicAdd(&buf[idx*3], fx);
        atomicAdd(&buf[idx*3+1], fy);
        atomicAdd(&buf[idx*3+2], fz);
    }
}

extern "C" {

// --- 1. Initialization & Actuation ---

__global__ void zero_forces(int n, double* force_buf) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        force_buf[i*3] = 0.0;
        force_buf[i*3+1] = 0.0;
        force_buf[i*3+2] = 0.0;
    }
}

__global__ void apply_position_actuation(int n_actuators, const int* act_indices, const double* act_values, double* x, double* v, double dt) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_actuators) return;

    int node_idx = act_indices[k];

    double old_x = x[node_idx*3];
    double old_y = x[node_idx*3 + 1];
    double old_z = x[node_idx*3 + 2];

    double new_x = act_values[k*3];
    double new_y = act_values[k*3 + 1];
    double new_z = act_values[k*3 + 2];

    x[node_idx*3]     = new_x;
    x[node_idx*3 + 1] = new_y;
    x[node_idx*3 + 2] = new_z;

    if (dt > 1e-9) {
        v[node_idx*3]     = (new_x - old_x) / dt;
        v[node_idx*3 + 1] = (new_y - old_y) / dt;
        v[node_idx*3 + 2] = (new_z - old_z) / dt;
    }
}

// --- 2. Force Computation (Bars) ---

/**
 * Computes spring-damper forces for soft bars.
 * F = k * (current_len - rest_len) + c * rel_velocity
 */
__global__ void compute_bar_forces(
    int n_bars,
    const int* bar_indices,   // [i, j]
    const double* bar_params, // [k, l0, c, prestress]
    const double* x,
    const double* v,
    const unsigned char* attrs,
    double* force_buf
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_bars) return;

    int i = bar_indices[k*2];
    int j = bar_indices[k*2 + 1];

    double stiffness = bar_params[k*4];
    if (stiffness < 0.0) return; // Rigid handled by PBD

    double l0        = bar_params[k*4 + 1];
    double c         = bar_params[k*4 + 2];
    double prestress = bar_params[k*4 + 3];

    double l_eff = l0 * (1.0 + prestress);

    double dx = x[j*3] - x[i*3];
    double dy = x[j*3+1] - x[i*3+1];
    double dz = x[j*3+2] - x[i*3+2];
    double dist = len3(dx, dy, dz);
    double nx = dx / dist; double ny = dy / dist; double nz = dz / dist;

    double f_spring = stiffness * (dist - l_eff);

    double dvx = v[j*3] - v[i*3];
    double dvy = v[j*3+1] - v[i*3+1];
    double dvz = v[j*3+2] - v[i*3+2];
    double v_rel = dot3(dvx, dvy, dvz, nx, ny, nz);
    double f_damp = c * v_rel;

    double f_total = clamp_d(f_spring + f_damp, -MAX_FORCE, MAX_FORCE);

    atomicAdd3(force_buf, i, f_total*nx, f_total*ny, f_total*nz, attrs);
    atomicAdd3(force_buf, j, -f_total*nx, -f_total*ny, -f_total*nz, attrs);
}

// --- 3. Force Computation (Hinges) ---

/**
 * Computes bending torque forces for soft hinges.
 * Based on the dihedral angle between two triangles (j-k-i) and (j-k-l).
 * The common edge is j-k.
 */
__global__ void compute_hinge_forces(
    int n_hinges,
    const int* indices,     // [j, k, i, l] stride 4
    const double* params,   // [k_tor, phi0, damping, _padding] stride 4
    const double* x,
    const double* v,        // Included for future damping implementation
    const unsigned char* attrs,
    double* force_buf
) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n_hinges) return;

    // Decode Indices (Spine: j-k, Wings: i, l)
    int j = indices[h*4];
    int k = indices[h*4+1];
    int i = indices[h*4+2];
    int l = indices[h*4+3];

    double k_tor = params[h*4];
    double phi0  = params[h*4+1];
    // Damping not yet implemented for angle velocity, placeholder

    // Positions
    double xj[3] = {x[j*3], x[j*3+1], x[j*3+2]};
    double xk[3] = {x[k*3], x[k*3+1], x[k*3+2]};
    double xi[3] = {x[i*3], x[i*3+1], x[i*3+2]};
    double xl[3] = {x[l*3], x[l*3+1], x[l*3+2]};

    // Edge Vectors
    double r_ij[3] = {xi[0]-xj[0], xi[1]-xj[1], xi[2]-xj[2]};
    double r_kj[3] = {xk[0]-xj[0], xk[1]-xj[1], xk[2]-xj[2]};
    double r_kl[3] = {xk[0]-xl[0], xk[1]-xl[1], xk[2]-xl[2]};

    // Normals (m and n)
    double m[3], n[3];
    cross3(r_ij[0], r_ij[1], r_ij[2], r_kj[0], r_kj[1], r_kj[2], &m[0], &m[1], &m[2]);
    cross3(r_kj[0], r_kj[1], r_kj[2], r_kl[0], r_kl[1], r_kl[2], &n[0], &n[1], &n[2]);

    double len_m = len3(m[0], m[1], m[2]);
    double len_n = len3(n[0], n[1], n[2]);
    double len_rkj = len3(r_kj[0], r_kj[1], r_kj[2]);

    if (len_m < MIN_AREA || len_n < MIN_AREA) return;

    // Angle Calculation
    double cos_phi = clamp_d(dot3(m[0],m[1],m[2], n[0],n[1],n[2]) / (len_m * len_n), -1.0, 1.0);
    double phi = acos(cos_phi);

    // Sign Check (Geometry Orientation)
    if (dot3(m[0],m[1],m[2], r_kl[0],r_kl[1],r_kl[2]) < 0) {
        phi = -phi;
    }

    // Wrap to canonical range [0, 2pi] usually not needed for delta, but good for consistency
    // Here we just compute delta directly
    double delta_phi = phi - phi0;

    // Shortest path wrap
    if (delta_phi > 3.1415926535) delta_phi -= 2.0 * 3.1415926535;
    if (delta_phi < -3.1415926535) delta_phi += 2.0 * 3.1415926535;

    // Torque magnitude
    double f_torque = -k_tor * delta_phi;

    // Forces (Gradients)
    double len_m_sq = len_m * len_m;
    double len_n_sq = len_n * len_n;
    double len_rkj_sq = len_rkj * len_rkj;

    // Soften singularity to prevent explosion at small areas
    double min_sq_len = 1e-8;
    len_m_sq = fmax(len_m_sq, min_sq_len);
    len_n_sq = fmax(len_n_sq, min_sq_len);

    // dTheta/dx_i = (len_rkj / len_m^2) * m
    double coeff_i = len_rkj / len_m_sq;
    double dth_dxi[3] = {coeff_i * m[0], coeff_i * m[1], coeff_i * m[2]};

    // dTheta/dx_l = -(len_rkj / len_n^2) * n
    double coeff_l = -len_rkj / len_n_sq;
    double dth_dxl[3] = {coeff_l * n[0], coeff_l * n[1], coeff_l * n[2]};

    // Cotan coefficients for j and k
    double rij_dot_rkj = dot3(r_ij[0],r_ij[1],r_ij[2], r_kj[0],r_kj[1],r_kj[2]);
    double rkl_dot_rkj = dot3(r_kl[0],r_kl[1],r_kl[2], r_kj[0],r_kj[1],r_kj[2]);

    double c_ij = rij_dot_rkj / len_rkj_sq;
    double c_kl = rkl_dot_rkj / len_rkj_sq;

    double dth_dxj[3];
    double dth_dxk[3];

    for(int d=0; d<3; d++) {
        dth_dxj[d] = (c_ij - 1.0) * dth_dxi[d] - c_kl * dth_dxl[d];
        dth_dxk[d] = (c_kl - 1.0) * dth_dxl[d] - c_ij * dth_dxi[d];
    }

    // Apply Forces
    atomicAdd3(force_buf, i, f_torque*dth_dxi[0], f_torque*dth_dxi[1], f_torque*dth_dxi[2], attrs);
    atomicAdd3(force_buf, l, f_torque*dth_dxl[0], f_torque*dth_dxl[1], f_torque*dth_dxl[2], attrs);
    atomicAdd3(force_buf, j, f_torque*dth_dxj[0], f_torque*dth_dxj[1], f_torque*dth_dxj[2], attrs);
    atomicAdd3(force_buf, k, f_torque*dth_dxk[0], f_torque*dth_dxk[1], f_torque*dth_dxk[2], attrs);
}

__global__ void apply_global_forces(
    int n_nodes,
    const double* v,
    const float* mass,
    const unsigned char* attrs,
    double damping,
    double g,
    double* force_buf
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes || !is_physics_active(i, attrs)) return;

    double m = (double)mass[i];
    force_buf[i*3]     -= damping * m * v[i*3];
    force_buf[i*3+1]   -= damping * m * v[i*3+1];
    force_buf[i*3+2]   -= damping * m * v[i*3+2];
    force_buf[i*3+2]   += m * g;
}

// --- 4. Integration (RK4) ---

__global__ void rk4_step_integrate(
    int n, double dt,
    const double* x, const double* v, const double* f,
    const float* mass, const unsigned char* attrs,
    double* k_x_out, double* k_v_out,
    double* x_temp, double* v_temp,
    double step_fraction
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (!is_physics_active(i, attrs)) {
        k_x_out[i*3]=0; k_x_out[i*3+1]=0; k_x_out[i*3+2]=0;
        k_v_out[i*3]=0; k_v_out[i*3+1]=0; k_v_out[i*3+2]=0;
        x_temp[i*3] = x[i*3]; x_temp[i*3+1] = x[i*3+1]; x_temp[i*3+2] = x[i*3+2];
        v_temp[i*3] = v[i*3]; v_temp[i*3+1] = v[i*3+1]; v_temp[i*3+2] = v[i*3+2];
        return;
    }

    double m = (double)mass[i];
    double minv = (m > 0) ? 1.0/m : 0.0;

    k_v_out[i*3] = f[i*3] * minv;
    k_v_out[i*3+1] = f[i*3+1] * minv;
    k_v_out[i*3+2] = f[i*3+2] * minv;

    k_x_out[i*3] = v[i*3];
    k_x_out[i*3+1] = v[i*3+1];
    k_x_out[i*3+2] = v[i*3+2];

    double dtf = dt * step_fraction;
    x_temp[i*3]   = x[i*3] + dtf * k_x_out[i*3];
    x_temp[i*3+1] = x[i*3+1] + dtf * k_x_out[i*3+1];
    x_temp[i*3+2] = x[i*3+2] + dtf * k_x_out[i*3+2];

    v_temp[i*3]   = v[i*3] + dtf * k_v_out[i*3];
    v_temp[i*3+1] = v[i*3+1] + dtf * k_v_out[i*3+1];
    v_temp[i*3+2] = v[i*3+2] + dtf * k_v_out[i*3+2];
}

__global__ void rk4_final_update(
    int n, double dt,
    double* x, double* v,
    const double* k1x, const double* k1v,
    const double* k2x, const double* k2v,
    const double* k3x, const double* k3v,
    const double* k4x, const double* k4v,
    const unsigned char* attrs
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !is_physics_active(i, attrs)) return;

    double dt6 = dt / 6.0;

    x[i*3]   += dt6 * (k1x[i*3]   + 2*k2x[i*3]   + 2*k3x[i*3]   + k4x[i*3]);
    x[i*3+1] += dt6 * (k1x[i*3+1] + 2*k2x[i*3+1] + 2*k3x[i*3+1] + k4x[i*3+1]);
    x[i*3+2] += dt6 * (k1x[i*3+2] + 2*k2x[i*3+2] + 2*k3x[i*3+2] + k4x[i*3+2]);

    v[i*3]   += dt6 * (k1v[i*3]   + 2*k2v[i*3]   + 2*k3v[i*3]   + k4v[i*3]);
    v[i*3+1] += dt6 * (k1v[i*3+1] + 2*k2v[i*3+1] + 2*k3v[i*3+1] + k4v[i*3+1]);
    v[i*3+2] += dt6 * (k1v[i*3+2] + 2*k2v[i*3+2] + 2*k3v[i*3+2] + k4v[i*3+2]);
}

// --- 5. Rigid Body Constraints (PBD) ---

__global__ void project_rigid_bars(
    int n_rigid,
    const int* indices,
    const double* l0s,
    const float* mass,
    const unsigned char* attrs,
    double* x,
    double omega
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_rigid) return;

    int i = indices[k*2];
    int j = indices[k*2 + 1];
    double l0 = l0s[k];

    double dx = x[j*3] - x[i*3];
    double dy = x[j*3+1] - x[i*3+1];
    double dz = x[j*3+2] - x[i*3+2];
    double dist = len3(dx, dy, dz);

    double C = dist - l0;
    if (fabs(C) < 1e-9) return;

    bool active_i = is_physics_active(i, attrs);
    bool active_j = is_physics_active(j, attrs);

    double wi = active_i ? (mass[i] > 0 ? 1.0/mass[i] : 0.0) : 0.0;
    double wj = active_j ? (mass[j] > 0 ? 1.0/mass[j] : 0.0) : 0.0;

    double w_sum = wi + wj;
    if (w_sum < 1e-12) return;

    double lambda = -omega * C / w_sum;
    double px = (dx / dist) * lambda;
    double py = (dy / dist) * lambda;
    double pz = (dz / dist) * lambda;

    if (active_i) {
        atomicAdd(&x[i*3], -wi * px);
        atomicAdd(&x[i*3+1], -wi * py);
        atomicAdd(&x[i*3+2], -wi * pz);
    }
    if (active_j) {
        atomicAdd(&x[j*3], wj * px);
        atomicAdd(&x[j*3+1], wj * py);
        atomicAdd(&x[j*3+2], wj * pz);
    }
}

__global__ void correct_rigid_velocity(
    int n_rigid,
    const int* indices,
    const float* mass,
    const unsigned char* attrs,
    const double* x,
    double* v
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_rigid) return;

    int i = indices[k*2];
    int j = indices[k*2 + 1];

    double dx = x[j*3] - x[i*3];
    double dy = x[j*3+1] - x[i*3+1];
    double dz = x[j*3+2] - x[i*3+2];
    double dist = len3(dx, dy, dz);
    double nx = dx/dist; double ny = dy/dist; double nz = dz/dist;

    double dvx = v[j*3] - v[i*3];
    double dvy = v[j*3+1] - v[i*3+1];
    double dvz = v[j*3+2] - v[i*3+2];
    double v_rel = dot3(dvx, dvy, dvz, nx, ny, nz);

    bool active_i = is_physics_active(i, attrs);
    bool active_j = is_physics_active(j, attrs);

    double wi = active_i ? (mass[i] > 0 ? 1.0/mass[i] : 0.0) : 0.0;
    double wj = active_j ? (mass[j] > 0 ? 1.0/mass[j] : 0.0) : 0.0;
    double w_sum = wi + wj;

    if (w_sum < 1e-12) return;

    double impulse = v_rel / w_sum;

    if (active_i) {
        atomicAdd(&v[i*3], wi * impulse * nx);
        atomicAdd(&v[i*3+1], wi * impulse * ny);
        atomicAdd(&v[i*3+2], wi * impulse * nz);
    }
    if (active_j) {
        atomicAdd(&v[j*3], -wj * impulse * nx);
        atomicAdd(&v[j*3+1], -wj * impulse * ny);
        atomicAdd(&v[j*3+2], -wj * impulse * nz);
    }
}


// --- 6. Rigid Hinge Constraints (PBD) ---

__global__ void project_rigid_hinges(
    int n_rigid,
    const int* indices,     // [j, k, i, l]
    const double* phi0s,    // Target angles
    const float* mass,
    const unsigned char* attrs,
    double* x,
    double omega
) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n_rigid) return;

    int j = indices[h*4];
    int k = indices[h*4+1];
    int i = indices[h*4+2];
    int l = indices[h*4+3];
    double phi_target = phi0s[h];

    // Fetch positions
    double xj[3] = {x[j*3], x[j*3+1], x[j*3+2]};
    double xk[3] = {x[k*3], x[k*3+1], x[k*3+2]};
    double xi[3] = {x[i*3], x[i*3+1], x[i*3+2]};
    double xl[3] = {x[l*3], x[l*3+1], x[l*3+2]};

    // Vectors
    double r_ij[3] = {xi[0]-xj[0], xi[1]-xj[1], xi[2]-xj[2]};
    double r_kj[3] = {xk[0]-xj[0], xk[1]-xj[1], xk[2]-xj[2]};
    double r_kl[3] = {xk[0]-xl[0], xk[1]-xl[1], xk[2]-xl[2]};

    // Normals
    double m[3], n[3];
    cross3(r_ij[0], r_ij[1], r_ij[2], r_kj[0], r_kj[1], r_kj[2], &m[0], &m[1], &m[2]);
    cross3(r_kj[0], r_kj[1], r_kj[2], r_kl[0], r_kl[1], r_kl[2], &n[0], &n[1], &n[2]);

    double len_m = len3(m[0], m[1], m[2]);
    double len_n = len3(n[0], n[1], n[2]);
    double len_rkj = len3(r_kj[0], r_kj[1], r_kj[2]);

    if (len_m < MIN_AREA || len_n < MIN_AREA) return;

    // Current Angle
    double cos_phi = clamp_d(dot3(m[0],m[1],m[2], n[0],n[1],n[2]) / (len_m * len_n), -1.0, 1.0);
    double phi = acos(cos_phi);
    if (dot3(m[0],m[1],m[2], r_kl[0],r_kl[1],r_kl[2]) < 0) phi = -phi;

    // Constraint C = phi - phi_target
    // Normalize delta to [-pi, pi]
    double C = phi - phi_target;
    while (C > 3.1415926535) C -= 2.0 * 3.1415926535;
    while (C < -3.1415926535) C += 2.0 * 3.1415926535;

    if (fabs(C) < 1e-4) return; // Tolerance

    // Gradients (Same as force formulation)
    double len_m_sq = len_m * len_m;
    double len_n_sq = len_n * len_n;
    double len_rkj_sq = len_rkj * len_rkj;

    // q_i = dC/dx_i
    double coeff_i = len_rkj / len_m_sq;
    double q_i[3] = {coeff_i * m[0], coeff_i * m[1], coeff_i * m[2]};

    // q_l = dC/dx_l
    double coeff_l = -len_rkj / len_n_sq;
    double q_l[3] = {coeff_l * n[0], coeff_l * n[1], coeff_l * n[2]};

    // q_j, q_k
    double rij_dot_rkj = dot3(r_ij[0],r_ij[1],r_ij[2], r_kj[0],r_kj[1],r_kj[2]);
    double rkl_dot_rkj = dot3(r_kl[0],r_kl[1],r_kl[2], r_kj[0],r_kj[1],r_kj[2]);
    double c_ij = rij_dot_rkj / len_rkj_sq;
    double c_kl = rkl_dot_rkj / len_rkj_sq;

    double q_j[3], q_k[3];
    for(int d=0; d<3; d++) {
        q_j[d] = (c_ij - 1.0) * q_i[d] - c_kl * q_l[d];
        q_k[d] = (c_kl - 1.0) * q_l[d] - c_ij * q_i[d];
    }

    // Inverse Masses (Weights)
    bool active_i = is_physics_active(i, attrs);
    bool active_j = is_physics_active(j, attrs);
    bool active_k = is_physics_active(k, attrs);
    bool active_l = is_physics_active(l, attrs);

    double wi = active_i ? (mass[i]>0 ? 1.0/mass[i] : 0.0) : 0.0;
    double wj = active_j ? (mass[j]>0 ? 1.0/mass[j] : 0.0) : 0.0;
    double wk = active_k ? (mass[k]>0 ? 1.0/mass[k] : 0.0) : 0.0;
    double wl = active_l ? (mass[l]>0 ? 1.0/mass[l] : 0.0) : 0.0;

    double sum_w_q2 = wi * dot3(q_i[0],q_i[1],q_i[2], q_i[0],q_i[1],q_i[2]) +
                      wj * dot3(q_j[0],q_j[1],q_j[2], q_j[0],q_j[1],q_j[2]) +
                      wk * dot3(q_k[0],q_k[1],q_k[2], q_k[0],q_k[1],q_k[2]) +
                      wl * dot3(q_l[0],q_l[1],q_l[2], q_l[0],q_l[1],q_l[2]);

    if (sum_w_q2 < 1e-12) return;

    // Lagrange Multiplier
    double lambda = -omega * C / sum_w_q2;

    // Updates
    if (active_i) atomicAdd3(x, i, wi*lambda*q_i[0], wi*lambda*q_i[1], wi*lambda*q_i[2], attrs);
    if (active_j) atomicAdd3(x, j, wj*lambda*q_j[0], wj*lambda*q_j[1], wj*lambda*q_j[2], attrs);
    if (active_k) atomicAdd3(x, k, wk*lambda*q_k[0], wk*lambda*q_k[1], wk*lambda*q_k[2], attrs);
    if (active_l) atomicAdd3(x, l, wl*lambda*q_l[0], wl*lambda*q_l[1], wl*lambda*q_l[2], attrs);
}

} // extern "C"