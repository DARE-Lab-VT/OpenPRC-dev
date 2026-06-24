import os
import csv
import json
import numpy as np
import taichi as ti
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.interpolate import CubicSpline
from scipy.stats import chi2
from numpy.lib.stride_tricks import sliding_window_view

# Initialize Taichi
ti.init(arch=ti.gpu, default_fp=ti.f64)
torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)
np.random.seed(42)

# --- Configuration ---
TRIAL_NAME = "Taichi_IID_Memory_Opt_30Hz_multi4_fast"
ROWS, COLS = 4, 4
EXPERIMENT_DIR = Path(__file__).parent.parent / "experiments" / TRIAL_NAME
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

# ==========================================
# Augmented Lagrangian & Constraints
# ==========================================
class PyTorchAL:
    def __init__(self, M_levels, device, tau=1.0):
        self.M = M_levels
        self.kappa = torch.zeros((M_levels, M_levels), device=device)
        self.tau = torch.ones((M_levels, M_levels), device=device) * tau
        self.prev_viol = torch.ones((M_levels, M_levels), device=device)
        
    def compute_loss(self, z_star):
        loss_reg = 0.0
        max_viol = 0.0
        viols = {}
        for i in range(self.M):
            for j in range(i+1, self.M):
                # Calculate Orthogonal Overlap exactly like the Taichi kernel
                numerator = z_star[:, i] * z_star[:, j]
                denominator = (z_star[:, i] + z_star[:, j]) / 2.0 + 1e-4
                viol = torch.mean(numerator / denominator)
                
                k = self.kappa[i, j]
                t = self.tau[i, j]
                
                loss_reg += k * viol + 0.5 * t * (viol ** 2)
                viols[(i, j)] = viol
                max_viol = max(max_viol, viol.item())
        return loss_reg, max_viol, viols
        
    def step_params(self, viols, verbose=False):
        # Optional but good practice: wrap parameter updates in no_grad
        with torch.no_grad():
            for (i, j), viol_tensor in viols.items():
                # 1. Extract the raw float to sever the autograd connection
                viol_val = viol_tensor.item() 
                
                # 2. Update parameters using the raw float
                self.kappa[i, j] += self.tau[i, j] * viol_val
                
                if viol_val > self.prev_viol[i, j]:
                    self.tau[i, j] *= 1.01
                elif viol_val < self.prev_viol[i, j]:
                    self.tau[i, j] *= 0.99
                    
                self.prev_viol[i, j] = viol_val
                
                if verbose:
                    print(f"AL Constraint ({i},{j}): Viol={viol_val:.6f}, Kappa={self.kappa[i, j].item():.4f}, Tau={self.tau[i, j].item():.4f}")
# ==========================================
# New: IID Signal Generator (Cubic Spline)
# ==========================================
def compute_effective_rank_torch(state_matrix):
    """Computes Shannon-entropy effective rank using PyTorch SVD."""
    if state_matrix.shape[0] < 2: return 1.0
    
    X_mean = state_matrix.mean(dim=0, keepdim=True)
    X_std = state_matrix.std(dim=0, keepdim=True) + 1e-12
    X_norm = (state_matrix - X_mean) / X_std
    
    _, s, _ = torch.linalg.svd(X_norm, full_matrices=False)
    s_norm = s / torch.sum(s)
    rank = torch.exp(-torch.sum(s_norm * torch.log(s_norm + 1e-12)))
    return rank.item()

# [UPDATE THIS FUNCTION]
def generate_iid_spline_input(duration, dt_sim, sample_hz=30.0, amp=0.02):
    t_sim = np.arange(0, duration, dt_sim)
    sample_interval = 1.0 / sample_hz  # Exactly 30Hz (0.0333...s)
    t_coarse = np.arange(0, duration + sample_interval, sample_interval)
    
    u_coarse = np.random.uniform(low=-1.0, high=1.0, size=len(t_coarse))
    cs = CubicSpline(t_coarse, u_coarse)
    u_fine = cs(t_sim) * amp
    v_fine = np.gradient(u_fine, dt_sim)
    
    return u_fine, v_fine, t_sim

def calculate_dambre_eps(effective_rank=1.5, test_duration=1000, p_value=1e-4):
    """Matches the calculation in run_memory_benchmark_pipeline.py"""
    t = chi2.isf(p_value, df=effective_rank)
    return (2.0 * t) / test_duration

# ==========================================
# Visualization Helpers
# ==========================================
def plot_material_distribution(z_star_np, K_vals, epoch, save_dir):
    """
    Plots a stacked continuous bar chart of the projected design vector z_star.
    Springs are sorted by the probability of the first material entry.
    """
    num_springs, M = z_star_np.shape
    
    # 1. Sort springs descending based on the 0-th material entry (e.g., k=0)
    sort_indices = np.argsort(z_star_np[:, 0])[::-1]
    sorted_z = z_star_np[sort_indices]
    
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    
    # 2. Setup a continuous colormap normalized to the stiffness ranges
    cmap = cm.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=min(K_vals), vmax=max(K_vals))
    
    x_positions = np.arange(num_springs)
    bottoms = np.zeros(num_springs)
    
    # 3. Stack the bars
    for m in range(M):
        # Assign color based on the actual stiffness value
        color = cmap(norm(K_vals[m]))
        
        ax.bar(
            x_positions, 
            sorted_z[:, m], 
            bottom=bottoms, 
            width=1.0,           # width=1.0 ensures absolutely no gaps
            color=color, 
            label=f'k = {K_vals[m]:.1f}',
            edgecolor='none'     # Removes inner borders for a clean histogram look
        )
        # Advance the bottom tracker for the next stack
        bottoms += sorted_z[:, m]
        
    # 4. Formatting
    ax.set_xlim(-0.5, num_springs - 0.5)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Spring ID (Sorted by Material 0)")
    ax.set_ylabel("Projected Probability ($z^*$)")
    ax.set_title(f"Material Distribution (Binarization Progress) - Epoch {epoch}")
    
    # Place legend safely outside the plot
    ax.legend(title="Stiffness", bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    
    # 5. Save and Close
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"material_dist_epoch_{epoch:03d}.png"), bbox_inches='tight')
        
    plt.close(fig)
def plot_epoch_topology(objects, springs, rho, masks, epoch, save_dir):
    """Plots the spring network geometry, using rho to determine line thickness."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for idx, s in enumerate(springs):
        node_a, node_b = int(s[0]), int(s[1])
        xa, ya = objects[node_a][0], objects[node_a][1]
        xb, yb = objects[node_b][0], objects[node_b][1]
        weight = float(rho[idx])
        if weight > 0.05: 
            ax.plot([xa, xb], [ya, yb], c='black', linewidth=weight * 8, alpha=1, zorder=1)

    ax.scatter(objects[:, 0], objects[:, 1], c='red', s=40, zorder=2, label='Active Node')
    input_idxs = masks['input_indices']
    if len(input_idxs) > 0:
        ax.scatter(objects[input_idxs, 0], objects[input_idxs, 1], 
                   c='green', s=80, edgecolors='black', zorder=3, label='Input Node')

    ax.set_title(f"Network Topology - Epoch {epoch}")
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"topology_epoch_{epoch:03d}.png"), dpi=150)
    plt.close(fig)
# ==========================================
# Geometry & Physics Setup
# ==========================================
def generate_openprc_grid_config(rows=4, cols=4):
    SPACING = 0.053     
    STIFFNESS = 100.0
    DAMPING = 0.32
    NODE_MASS = 0.01    
    PRETENSION = 0.10
    
    num_nodes = rows * cols
    objects = np.zeros((num_nodes, 3))
    node_indices = np.zeros((rows, cols), dtype=int)
    
    idx = 0
    for r in range(rows):
        for c in range(cols):
            objects[idx] = [c * SPACING, -r * SPACING, 0.0]
            node_indices[r, c] = idx
            idx += 1
            
    springs_list = []
    for r in range(rows):
        for c in range(cols - 1):
            springs_list.append([node_indices[r, c], node_indices[r, c + 1], SPACING * (1.0 - PRETENSION), STIFFNESS, 0.0, DAMPING, 0.0])
    for r in range(rows - 1):
        for c in range(cols):
            springs_list.append([node_indices[r, c], node_indices[r + 1, c], SPACING * (1.0 - PRETENSION), STIFFNESS, 0.0, DAMPING, 0.0])
    
    hinges_list = []
    HINGE_DAMPING = 0.001
    for r in range(rows - 1):
        for c in range(cols - 1):
            n_tl, n_tr, n_bl = node_indices[r, c], node_indices[r, c + 1], node_indices[r + 1, c]
            hinges_list.append([n_tr, n_tl, n_bl, HINGE_DAMPING, 0.0])
            
    fixed_mask = np.zeros(num_nodes, dtype=int)
    input_weights = np.zeros(num_nodes)
    act_indices = [node_indices[0, 0]]
    for i in act_indices: 
        input_weights[i] = 1.0
    
    fixed_indices = [
        node_indices[0, cols - 1], 
        node_indices[rows - 1, 0], 
        node_indices[rows - 1, cols - 1]
    ]
    for i in fixed_indices:
        fixed_mask[i] = 1  # 1 means rigidly fixed, 0 means free to move
        
    config = {
        'objects': objects,
        'springs': np.array(springs_list),
        'hinges': np.array(hinges_list),
        'masks': {'fixed': fixed_mask, 'active': ~fixed_mask.astype(bool), 'input_indices': act_indices},
        'weights': {'input_weights': input_weights}
    }
    return config, NODE_MASS

@ti.data_oriented
class DifferentiableReservoir:
    def __init__(self, config, dt, batch_steps, M_levels, K_vals, C_vals):
        self.dt = dt
        self.batch_steps = batch_steps 
        self.num_mass = len(config['objects'])
        self.num_springs = len(config['springs'])
        self.num_hinges = len(config['hinges'])
        
        self.M = M_levels
        self.z = ti.field(dtype=ti.f64, shape=(self.num_springs, self.M), needs_grad=True)
        self.z_star = ti.field(dtype=ti.f64, shape=(self.num_springs, self.M), needs_grad=True)
        self.discrete_k = ti.field(dtype=ti.f64, shape=self.M)
        self.discrete_c = ti.field(dtype=ti.f64, shape=self.M)

        self.x_rest = ti.Vector.field(3, dtype=ti.f64, shape=self.num_mass, needs_grad=True)
        self.spring_l0 = ti.field(dtype=ti.f64, shape=self.num_springs, needs_grad=True)
        
        self.x = ti.Vector.field(3, dtype=ti.f64, shape=(batch_steps, self.num_mass), needs_grad=True)
        self.v = ti.Vector.field(3, dtype=ti.f64, shape=(batch_steps, self.num_mass), needs_grad=True)
        self.force_buffer = ti.Vector.field(3, dtype=ti.f64, shape=self.num_mass, needs_grad=True)
        self.node_disp = ti.field(dtype=ti.f64, shape=(batch_steps, self.num_mass, 2), needs_grad=True)

        self.spring_a = ti.field(dtype=ti.i32, shape=self.num_springs)
        self.spring_b = ti.field(dtype=ti.i32, shape=self.num_springs)
        self.spring_k1 = ti.field(dtype=ti.f64, shape=self.num_springs)
        self.spring_d1 = ti.field(dtype=ti.f64, shape=self.num_springs)
        
        self.hinge_i = ti.field(dtype=ti.i32, shape=self.num_hinges)
        self.hinge_j = ti.field(dtype=ti.i32, shape=self.num_hinges)
        self.hinge_k = ti.field(dtype=ti.i32, shape=self.num_hinges)
        self.hinge_d_tor = ti.field(dtype=ti.f64, shape=self.num_hinges)

        self.input_weights = ti.field(dtype=ti.f64, shape=self.num_mass)
        self.fixed_mask = ti.field(dtype=ti.i32, shape=self.num_mass)
        self.loss = ti.field(dtype=ti.f64, shape=(), needs_grad=True)
        self._initialize_from_config(config, K_vals, C_vals)

    def _initialize_from_config(self, config, K_vals, C_vals):
        objects, springs, hinges = config['objects'], config['springs'], config['hinges']
        masks, weights = config['masks'], config['weights']['input_weights']
        for m in range(self.M):
            self.discrete_k[m] = K_vals[m]
            self.discrete_c[m] = C_vals[m]
        for i in range(self.num_mass):
            self.x_rest[i] = [objects[i, 0], objects[i, 1], objects[i, 2]]
            self.fixed_mask[i] = 1 if masks['fixed'][i] else 0
            self.input_weights[i] = weights[i]
        for s in range(self.num_springs):
            self.spring_a[s], self.spring_b[s] = int(springs[s, 0]), int(springs[s, 1])
            self.spring_l0[s] = springs[s, 2]

        for h in range(self.num_hinges):
            self.hinge_i[h], self.hinge_j[h], self.hinge_k[h] = int(hinges[h, 0]), int(hinges[h, 1]), int(hinges[h, 2])
            self.hinge_d_tor[h] = hinges[h, 3]

    # @ti.kernel
    # def project_z(self):
    #     """Stable Softmax projection inside Taichi with beta = 20"""
    #     for i in range(self.num_springs):
    #         max_z = -1e10
    #         for m in ti.static(range(self.M)):
    #             if self.z[i, m] > max_z:
    #                 max_z = self.z[i, m]
                    
    #         sum_exp = 0.0
    #         for m in ti.static(range(self.M)):
    #             sum_exp += ti.exp(20.0 * (self.z[i, m] - max_z))
                
    #         for m in ti.static(range(self.M)):
    #             self.z_star[i, m] = ti.exp(20.0 * (self.z[i, m] - max_z)) / sum_exp
    @ti.kernel
    def update_geometry_kernel(self):
        for s in range(self.num_springs):
            i, j = self.spring_a[s], self.spring_b[s]
            dist = ti.math.sqrt((self.x_rest[i] - self.x_rest[j]).norm_sqr() + 1e-12)
            self.spring_l0[s] = dist * 0.90
            
    @ti.kernel
    def init_simulation_state(self):
        for i in range(self.num_mass):
            self.x[0, i] = self.x_rest[i]
            self.v[0, i] = [0.0, 0.0, 0.0]

    @ti.kernel
    def compute_step(self, t: int, u_val: ti.f64, v_val: ti.f64):
        for i in range(self.num_mass):
            self.force_buffer[i] = ti.Vector([0.0, 0.0, 0.0])
            self.force_buffer[i] += -0.1 * self.v[t, i]

        for s in range(self.num_springs):
            i, j = self.spring_a[s], self.spring_b[s]
            disp = self.x[t, i] - self.x[t, j]
            dist = ti.math.sqrt(disp.norm_sqr() + 1e-12)
            dir_vec = disp / dist
            extension = dist - self.spring_l0[s]
            dot_v = (self.v[t, i] - self.v[t, j]).dot(dir_vec)

            k_eff = 0.0
            c_eff = 0.0
            for m in ti.static(range(self.M)):
                k_eff += self.z_star[s, m] * self.discrete_k[m]
                c_eff += self.z_star[s, m] * self.discrete_c[m]
                
            f_total = k_eff * extension + c_eff * dot_v
            force_vec = -f_total * dir_vec
            
            ti.atomic_add(self.force_buffer[i], force_vec)
            ti.atomic_add(self.force_buffer[j], -force_vec)
            
        for h in range(self.num_hinges):
            i, j, k = self.hinge_i[h], self.hinge_j[h], self.hinge_k[h]
            r_ji = self.x[t, i] - self.x[t, j]
            r_jk = self.x[t, k] - self.x[t, j]
            l_ji_sq, l_jk_sq = r_ji.norm_sqr() + 1e-12, r_jk.norm_sqr() + 1e-12
            v_rel_i, v_rel_k = self.v[t, i] - self.v[t, j], self.v[t, k] - self.v[t, j]
            omega_i, omega_k = r_ji.cross(v_rel_i) / l_ji_sq, r_jk.cross(v_rel_k) / l_jk_sq
            torque = -self.hinge_d_tor[h] * (omega_i - omega_k)
            F_i, F_k = torque.cross(r_ji) / l_ji_sq, -torque.cross(r_jk) / l_jk_sq
            ti.atomic_add(self.force_buffer[i], F_i)
            ti.atomic_add(self.force_buffer[k], F_k)
            ti.atomic_add(self.force_buffer[j], -F_i - F_k)

        for i in range(self.num_mass):
            if self.input_weights[i] > 0.0:
                self.x[t+1, i] = self.x_rest[i] + ti.Vector([self.input_weights[i] * u_val, 0.0, 0.0])
                self.v[t+1, i] = ti.Vector([self.input_weights[i] * v_val, 0.0, 0.0])
            else:
                acc = self.force_buffer[i] / 0.01
                self.v[t+1, i] = self.v[t, i] + acc * self.dt
                self.x[t+1, i] = self.x[t, i] + self.v[t+1, i] * self.dt

    @ti.kernel
    def compute_features(self):
        for t, i in ti.ndrange(self.batch_steps, self.num_mass):
            # [UPDATE MATH]: Capture both X and Y displacements relative to Node 0
            self.node_disp[t, i, 0] = self.x[t, i][0] - self.x[t, 0][0]
            self.node_disp[t, i, 1] = self.x[t, i][1] - self.x[t, 0][1]
    
    def run_batch_forward(self, u_array, v_array):
        self.update_geometry_kernel()
        self.init_simulation_state()
        for t in range(self.batch_steps - 1):
            self.compute_step(t, float(u_array[t]), float(v_array[t]))
        self.compute_features()

# ==========================================
# Targets & PyTorch Bridge
# ==========================================
def generate_benchmark_target_matrix(u_input, tau_d, k, n, tau_step=1):
    u = np.asarray(u_input, dtype=float)
    T, max_lag = u.shape[0], tau_d * k
    U_sub = sliding_window_view(u, max_lag + 1)[:, ::-1][:, [j * k for j in range(0, tau_d + 1, tau_step)]]
    Lvars = U_sub.shape[1]
    
    exps = []
    vec = np.zeros(Lvars, dtype=int)
    def rec(rem, idx):
        if idx == Lvars - 1:
            vec[idx] = rem
            exps.append(vec.copy())
        else:
            for v in range(rem + 1):
                vec[idx] = v; rec(rem - v, idx + 1)
                
    for d in range(1, n + 1): rec(d, 0)
    exps = np.array(exps)
    
    Y_valid = np.ones((U_sub.shape[0], exps.shape[0]), dtype=float)
    for j in range(Lvars):
        if np.any(exps[:, j]): Y_valid *= (U_sub[:, [j]] ** exps[:, j][None, :])
            
    Y_full = np.zeros((T, exps.shape[0]), dtype=float)
    Y_full[max_lag:, :] = Y_valid
    return Y_full

class ReservoirOptimizationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z_star_tensor, pos_tensor, res, u_sim, v_sim, downsample):
        ctx.res = res
        ctx.u_sim = u_sim
        ctx.v_sim = v_sim
        ctx.downsample = downsample

        res.x_rest.from_torch(pos_tensor.detach().contiguous())
        res.z_star.from_torch(z_star_tensor.detach().contiguous())
        res.x_rest.grad.fill(0)
        res.z_star.grad.fill(0)

        res.update_geometry_kernel()
        res.run_batch_forward(u_sim, v_sim)

        disp_30Hz = res.node_disp.to_torch(device=pos_tensor.device)[::downsample, :, :] 
        return disp_30Hz.reshape(disp_30Hz.shape[0], -1) 

    @staticmethod
    def backward(ctx, grad_outputs_30Hz):
        res = ctx.res
        downsample = ctx.downsample
        
        grad_outputs_clean = grad_outputs_30Hz.detach()
        target_device = grad_outputs_clean.device

        grad_sim = torch.zeros((res.batch_steps, res.num_mass * 2), dtype=torch.float64, device=target_device)
        grad_sim[::downsample, :] = grad_outputs_clean
        grad_sim_reshaped = grad_sim.reshape(res.batch_steps, res.num_mass, 2).contiguous()

        res.x_rest.grad.fill(0)
        res.z_star.grad.fill(0)
        res.node_disp.grad.from_torch(grad_sim_reshaped.detach())

        with ti.ad.Tape(loss=res.loss, validation=False):
            res.update_geometry_kernel()
            res.run_batch_forward(ctx.u_sim, ctx.v_sim)
            compute_dummy_loss(res, grad_sim_reshaped)

        grad_pos = res.x_rest.grad.to_torch(device=target_device)
        fixed_mask = torch.tensor(res.fixed_mask.to_numpy(), device=target_device)
        grad_pos = grad_pos * (1.0 - fixed_mask.unsqueeze(1))
        
        grad_z_star = res.z_star.grad.to_torch(device=target_device)
        return grad_z_star, grad_pos, None, None, None, None

@ti.kernel
def compute_dummy_loss(res: ti.template(), grad_tensor: ti.types.ndarray()):
    for t, i in ti.ndrange(res.batch_steps, res.num_mass):
        res.loss[None] += res.node_disp[t, i, 0] * grad_tensor[t, i, 0]
        res.loss[None] += res.node_disp[t, i, 1] * grad_tensor[t, i, 1]

@ti.kernel
def add_reg_loss(res: ti.template(), reg: ti.template()):
    res.loss[None] += reg.get_loss()
# ==========================================
# Main Optimization Pipeline
# ==========================================
if __name__ == "__main__":
    # 1. Setup Data & Folders
    topo_epoch_dir = EXPERIMENT_DIR / "topology_visualization"
    topo_curve_dir = EXPERIMENT_DIR / "material_distribution"
    os.makedirs(topo_epoch_dir, exist_ok=True)
    os.makedirs(topo_curve_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch utilizing device: {device}")

    PHYSICS_HZ = 3000
    TARGET_HZ = 30
    DOWNSAMPLE = PHYSICS_HZ // TARGET_HZ  # Mathematically locked to 100

    # 2. Generate the New IID Input
    T_DURATION = 25.0
    dt_sim = 1.0 / 3000.0
    u_sim, v_sim, _ = generate_iid_spline_input(T_DURATION, dt_sim, sample_hz=TARGET_HZ)
    u_30Hz = u_sim[::DOWNSAMPLE]

    # 3. Setup Physics & Params
    M_LEVELS = 4
    K_VALS = np.linspace(0.0, 100.0, M_LEVELS)
    C_VALS = np.linspace(0.0, 0.8, M_LEVELS)

    config, _ = generate_openprc_grid_config(ROWS, COLS) 
    
    z_param = nn.Parameter(torch.zeros((len(config['springs']), M_LEVELS), dtype=torch.float64, device=device))
    pos_param = nn.Parameter(torch.tensor(config['objects'], dtype=torch.float64, device=device))
    optimizer = torch.optim.Adam([z_param], lr=0.01)

    # 4. Generate Targets & [CRITICAL FIX 2]: Standardize Targets
    # Using tau_d=50, k=10 to safely simulate 500 frames of lag memory without exceeding duration.
    y_target_raw = generate_benchmark_target_matrix(u_30Hz, tau_d=30, k=1, n=4) 
    y_target_mean = y_target_raw.mean(axis=0)
    y_target_std = y_target_raw.std(axis=0) + 1e-12
    y_target_norm = (y_target_raw - y_target_mean) / y_target_std
    y_target_torch = torch.tensor(y_target_norm, dtype=torch.float64, device=device)
    
    washout_frames = 5 * TARGET_HZ   
    train_frames = 10 * TARGET_HZ 
    test_frames = 10 * TARGET_HZ       
    
    train_stop = washout_frames + train_frames
    test_stop = train_stop + test_frames
    
    y_train_torch = y_target_torch[washout_frames:train_stop]
    y_test_torch = y_target_torch[train_stop:test_stop]
    
    print(f"Data Splits -> Washout: {washout_frames}, Train: {train_frames}, Test: {test_frames}")
    res_opt = DifferentiableReservoir(config, dt_sim, len(u_sim), M_LEVELS, K_VALS, C_VALS)
    
    active_mask_2d = torch.ones(res_opt.num_mass * 2, dtype=torch.bool, device=pos_param.device)
    active_mask_2d[0:2] = False
    
    al_solver = PyTorchAL(M_LEVELS, device)

    # 5. Optimization Loop
    EPOCHS = 100
    loss_history, ipc_history, max_mse_history = [], [], []
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        beta = 20.0
        z_star_tensor = F.softmax(beta * z_param, dim=-1)

        # Physics Forward (100Hz output)
        X_30Hz = ReservoirOptimizationFunction.apply(z_star_tensor, pos_param, res_opt, u_sim, v_sim, DOWNSAMPLE)
        # Feature Prep (Strip Reference Node, Normalize)
        # Feature Prep (Strip Reference Node, Normalize)
        X_active = X_30Hz[:, active_mask_2d]
        X_norm = (X_active - X_active.mean(0)) / (X_active.std(0) + 1e-6)
        
        # Add bias column to the full simulated sequence
        X_full = torch.cat([torch.ones(len(X_norm), 1, device=X_norm.device), X_norm], dim=1)
        
        # Split into Train and Test
        X_train = X_full[washout_frames:train_stop]
        X_test = X_full[train_stop:test_stop]
        
        # Dynamic Effective Rank (Fix 3)
        current_rank = compute_effective_rank_torch(X_norm)
        dynamic_dambre_eps = calculate_dambre_eps(effective_rank=current_rank, test_duration=test_frames)
        
        # Solve Ridge on TRAIN
        I = torch.eye(X_train.shape[1], device=X_train.device) * 1e-6
        W_out = torch.linalg.solve(X_train.T @ X_train + I, X_train.T @ y_train_torch)
        
        # Evaluate MSE on TEST (Fix 1)
        mse_per_task = torch.mean((X_test @ W_out - y_test_torch)**2, dim=0)
        r2_per_task = 1.0 - mse_per_task
        
        # Capacity Calculations for Logging
        expected_noise_r2 = (X_test.shape[1] - 1) / X_test.shape[0]
        adjusted_eps = dynamic_dambre_eps + expected_noise_r2
        
        valid_capacity = torch.where(r2_per_task > adjusted_eps, r2_per_task, torch.zeros_like(r2_per_task))
        total_capacity = torch.sum(valid_capacity).item()
        
        # Loss Function (Fix 2)
        # We minimize the sum of MSE to maximize overall R2. 
        # Note: Do NOT use the `torch.where` threshold tensor for the loss itself, 
        # because a hard cutoff at 0.0 will destroy gradients for tasks currently below the threshold.
        loss = torch.sum(mse_per_task) / X_test.shape[0]  # Average MSE across all tasks
        loss_reg, max_viol, viols = al_solver.compute_loss(z_star_tensor)

        total_loss = loss + loss_reg
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_([z_param], max_norm=1.0)
        optimizer.step()
        al_solver.step_params(viols, verbose=(epoch % 10 == 0))
        with torch.no_grad():
            z_star_np = z_star_tensor.cpu().numpy()
            k_eff = np.sum(z_star_np * np.array(K_VALS), axis=1)
            c_eff = np.sum(z_star_np * np.array(C_VALS), axis=1)
            
        print(f"Epoch {epoch+1:03d} | Summed MSE: {loss.item():.5f} | IPC Capacity: {total_capacity:.2f} | Max MSE: {mse_per_task.max().item():.5f}")
        
        loss_history.append(loss.item())
        ipc_history.append(total_capacity)
        max_mse_history.append(mse_per_task.max().item())
        current_pos = pos_param.detach().cpu().numpy()

        # Plotting
        with torch.no_grad():
            z_star_np = z_star_tensor.cpu().numpy()
        plot_material_distribution(
            z_star_np, 
            K_VALS, 
            epoch, 
            topo_curve_dir
        )
        plot_epoch_topology(current_pos, config['springs'], k_eff / 100.0, config['masks'], epoch, topo_epoch_dir)
        #plot_topology_curve(current_rho, 0.05, epoch, topo_curve_dir)
    
    csv_path = EXPERIMENT_DIR / "loss_log.csv"
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Softmax_MSE_Loss", "IPC_Capacity", "Max_MSE"])
        for e_idx in range(EPOCHS):
            writer.writerow([
                e_idx + 1, 
                loss_history[e_idx], 
                ipc_history[e_idx], 
                max_mse_history[e_idx]
            ])
                
    # 6. Save Results
    plt.figure()
    plt.plot(loss_history)
    plt.title("Optimization Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Summed MSE Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig(EXPERIMENT_DIR / "loss_curve.png", dpi=150)
    plt.close()

    k_mat_opt, c_mat_opt = np.zeros((ROWS*COLS, ROWS*COLS)), np.zeros((ROWS*COLS, ROWS*COLS))
    for s in range(len(config['springs'])):
        i, j = int(config['springs'][s, 0]), int(config['springs'][s, 1])
        if k_eff[s] > 1.0: # Filter out absolute zero-springs
            k_mat_opt[i, j] = k_mat_opt[j, i] = float(k_eff[s])
            c_mat_opt[i, j] = c_mat_opt[j, i] = float(c_eff[s])
            
    with open(EXPERIMENT_DIR / "experiment.json", "w") as f:
        json.dump({"k_mat_opt": k_mat_opt.tolist(), "c_mat_opt": c_mat_opt.tolist()}, f, indent=4)
        
    print(f"\n[Success] Optimized for IID Spline Input and Dambre-Standardized Memory Tasks.")