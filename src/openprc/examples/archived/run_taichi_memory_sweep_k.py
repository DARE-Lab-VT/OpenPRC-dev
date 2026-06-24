import os
import csv
import json
import numpy as np
import taichi as ti
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import CubicSpline
from scipy.stats import chi2
from numpy.lib.stride_tricks import sliding_window_view

# Initialize Taichi
ti.init(arch=ti.gpu, default_fp=ti.f64)
torch.set_default_dtype(torch.float64)

# --- Configuration ---
TRIAL_NAME = "Taichi_IID_Memory_Opt_Low_k"
ROWS, COLS = 4, 4
EXPERIMENT_DIR = Path(__file__).parent.parent / "experiments" / TRIAL_NAME
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

# ==========================================
# New: IID Signal Generator (Cubic Spline)
# ==========================================
def generate_iid_spline_input(duration, dt_sim, sample_interval=0.1, amp=0.02):
    t_sim = np.arange(0, duration, dt_sim)
    t_coarse = np.arange(0, duration + sample_interval, sample_interval)
    
    # Generate random uniform points [-1, 1]
    u_coarse = np.random.uniform(low=-1.0, high=1.0, size=len(t_coarse))
    
    # Fit Cubic Spline for smooth but unpredictable motion
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

def plot_topology_curve(rho_np, yita, epoch, out_dir):
    """Plots the sorted topology variables (rho) to track binarization."""
    vals = sorted(rho_np)
    xs = np.arange(len(vals))
    plt.figure(figsize=(9, 2.5), dpi=150)
    plt.plot(xs, vals, linewidth=1.5)
    plt.axhline(y=yita, color='r', linestyle=':', label=r'$\eta$ (Threshold)')
    plt.ylim(-0.05, 1.05)
    plt.xlim(-0.5, len(vals) - 0.5)
    plt.xlabel("Spring ID (Sorted)")
    plt.ylabel("Rho (Density)")
    plt.title(f"Topology Variables - Epoch {epoch}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"topology_curve_{epoch:03d}.png"), bbox_inches="tight")
    plt.close()

# ==========================================
# Geometry & Physics Setup
# ==========================================
def generate_openprc_grid_config(rows=4, cols=4, stiffness=100.0):
    SPACING = 0.053     
    STIFFNESS = stiffness 
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
    act_indices = [node_indices[0, 0], node_indices[0, cols - 1], node_indices[rows - 1, 0], node_indices[rows - 1, cols - 1]]
    for i in act_indices: input_weights[i] = 1.0 
        
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
    def __init__(self, config, dt, batch_steps):
        self.dt = dt
        self.batch_steps = batch_steps 
        self.num_mass = len(config['objects'])
        self.num_springs = len(config['springs'])
        self.num_hinges = len(config['hinges'])
        
        self.x_rest = ti.Vector.field(3, dtype=ti.f64, shape=self.num_mass, needs_grad=True)
        self.spring_l0 = ti.field(dtype=ti.f64, shape=self.num_springs, needs_grad=True)
        self.rho = ti.field(dtype=ti.f64, shape=self.num_springs, needs_grad=True)
        
        self.x = ti.Vector.field(3, dtype=ti.f64, shape=(batch_steps, self.num_mass), needs_grad=True)
        self.v = ti.Vector.field(3, dtype=ti.f64, shape=(batch_steps, self.num_mass), needs_grad=True)
        self.force_buffer = ti.Vector.field(3, dtype=ti.f64, shape=self.num_mass, needs_grad=True)
        self.node_disp = ti.field(dtype=ti.f64, shape=(batch_steps, self.num_mass), needs_grad=True)
        
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
        self._initialize_from_config(config)

    def _initialize_from_config(self, config):
        objects, springs, hinges = config['objects'], config['springs'], config['hinges']
        masks, weights = config['masks'], config['weights']['input_weights']
        for i in range(self.num_mass):
            self.x_rest[i] = [objects[i, 0], objects[i, 1], objects[i, 2]]
            self.fixed_mask[i] = 1 if masks['fixed'][i] else 0
            self.input_weights[i] = weights[i]
        for s in range(self.num_springs):
            self.spring_a[s], self.spring_b[s] = int(springs[s, 0]), int(springs[s, 1])
            self.spring_l0[s], self.spring_k1[s], self.spring_d1[s] = springs[s, 2], springs[s, 3], springs[s, 5]
            self.rho[s] = 0.5
        for h in range(self.num_hinges):
            self.hinge_i[h], self.hinge_j[h], self.hinge_k[h] = int(hinges[h, 0]), int(hinges[h, 1]), int(hinges[h, 2])
            self.hinge_d_tor[h] = hinges[h, 3]

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
            f_total = self.spring_k1[s] * extension + self.spring_d1[s] * dot_v
            force_vec = -f_total * dir_vec * self.rho[s]
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
            # [CRITICAL FIX 1]: Relative Displacements (Reference = Node 0)
            # This perfectly mirrors NodeDisplacements(reference_node=0) in openprc
            # and prevents the readout from cheating by reading the global translation.
            self.node_disp[t, i] = self.x[t, i][0] - self.x[t, 0][0]

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
    def forward(ctx, pos_tensor, rho_tensor, res, u_1000, v_1000):
        ctx.res, ctx.u_1000, ctx.v_1000 = res, u_1000, v_1000
        res.x_rest.from_torch(pos_tensor)
        res.rho.from_torch(rho_tensor)
        res.x_rest.grad.fill(0); res.rho.grad.fill(0)
        
        res.run_batch_forward(u_1000, v_1000)
        return res.node_disp.to_torch()[::10, :] 

    @staticmethod
    def backward(ctx, grad_outputs_100Hz):
        res = ctx.res
        grad_1000Hz = torch.zeros((res.batch_steps, res.num_mass), dtype=torch.float64, device=grad_outputs_100Hz.device)
        grad_1000Hz[::10, :] = grad_outputs_100Hz

        res.x_rest.grad.fill(0); res.rho.grad.fill(0)
        res.node_disp.grad.from_torch(grad_1000Hz)

        with ti.ad.Tape(loss=res.loss, validation=False):
            res.update_geometry_kernel()
            res.run_batch_forward(ctx.u_1000, ctx.v_1000)
            compute_dummy_loss(res, grad_1000Hz)

        grad_pos = res.x_rest.grad.to_torch() * (1.0 - torch.tensor(res.fixed_mask.to_numpy(), device=grad_outputs_100Hz.device).unsqueeze(1))
        return grad_pos, res.rho.grad.to_torch(), None, None, None

@ti.kernel
def compute_dummy_loss(res: ti.template(), grad_tensor: ti.types.ndarray()):
    for t, i in ti.ndrange(res.batch_steps, res.num_mass):
        res.loss[None] += res.node_disp[t, i] * grad_tensor[t, i]

# ==========================================
# Main Optimization Pipeline
# ==========================================
if __name__ == "__main__":
    np.random.seed(42)
    # --- Sweep Configuration ---
    TRIAL_NAME = "Taichi_Stiffness_Sweep_Damping"
    MAIN_DIR = Path(__file__).parent.parent / "experiments" / TRIAL_NAME
    os.makedirs(MAIN_DIR, exist_ok=True)
    
    # [NEW]: Define your parameter sweep list here
    STIFFNESS_VALUES = np.linspace(25,225,9)
    
    # 1. Generate Input ONCE (Fair apples-to-apples comparison across all stiffnesses)
    T_DURATION = 25.0
    dt_sim = 0.001
    u_1000Hz, v_1000Hz, _ = generate_iid_spline_input(T_DURATION, dt_sim)
    u_100Hz = u_1000Hz[::10]

    # Generate Targets & Standardize
    y_target_raw = generate_benchmark_target_matrix(u_100Hz, tau_d=50, k=10, n=2) 
    y_target_mean = y_target_raw.mean(axis=0)
    y_target_std = y_target_raw.std(axis=0) + 1e-12
    y_target_norm = (y_target_raw - y_target_mean) / y_target_std
    
    dambre_eps = calculate_dambre_eps(effective_rank=1.5, test_duration=1000)
    print(f"Dambre Noise Floor Threshold (Eps): {dambre_eps:.6f}")

    washout_frames, train_frames = 500, 1000
    train_stop = washout_frames + train_frames
    
    # ==========================================
    # 2. START PARAMETER SWEEP
    # ==========================================
    for current_stiffness in STIFFNESS_VALUES:
        print(f"\n{'='*60}")
        print(f"=== Starting Optimization for Stiffness: {current_stiffness} ===")
        print(f"{'='*60}")
        
        EXPERIMENT_DIR = MAIN_DIR / f"stiffness_{current_stiffness}"
        topo_epoch_dir = EXPERIMENT_DIR / "topology_epoch"
        topo_curve_dir = EXPERIMENT_DIR / "topology_curve"
        os.makedirs(topo_epoch_dir, exist_ok=True)
        os.makedirs(topo_curve_dir, exist_ok=True)
        
        # Setup Physics & Params for CURRENT stiffness
        config, _ = generate_openprc_grid_config(ROWS, COLS, stiffness=current_stiffness) 
        res_opt = DifferentiableReservoir(config, dt_sim, len(u_1000Hz))
        
        rho_param = nn.Parameter(torch.ones(len(config['springs']), dtype=torch.float64) * 0.5)
        pos_param = nn.Parameter(torch.tensor(config['objects'], dtype=torch.float64))
        optimizer = torch.optim.Adam([rho_param], lr=0.05)
        
        y_target_torch = torch.tensor(y_target_norm, dtype=torch.float64).to(pos_param.device)
        y_train_torch = y_target_torch[washout_frames:train_stop]
        
        active_mask = torch.ones(res_opt.num_mass, dtype=torch.bool, device=pos_param.device)
        active_mask[0] = False # Discard Node 0 

        EPOCHS = 100
        loss_history = []
        ipc_history = []
        max_mse_history = []
        
        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            
            X_100Hz = ReservoirOptimizationFunction.apply(pos_param, rho_param, res_opt, u_1000Hz, v_1000Hz)
            
            X_active = X_100Hz[:, active_mask]
            X_norm = (X_active - X_active.mean(0)) / (X_active.std(0) + 1e-6)
            X_train = torch.cat([torch.ones(len(X_norm), 1, device=X_norm.device), X_norm], dim=1)[washout_frames:train_stop]
            
            I = torch.eye(X_train.shape[1], device=X_train.device) * 1e-6
            W_out = torch.linalg.solve(X_train.T @ X_train + I, X_train.T @ y_train_torch)
            
            mse_per_task = torch.mean((X_train @ W_out - y_train_torch)**2, dim=0)
            r2_per_task = 1.0 - mse_per_task
            
            valid_capacity = torch.where(r2_per_task > dambre_eps, r2_per_task, 0.0)
            total_capacity = torch.sum(valid_capacity).item()
            
            weights = F.softmax(mse_per_task / 0.05, dim=0)
            loss = torch.sum(weights * mse_per_task)

            loss.backward()
            torch.nn.utils.clip_grad_norm_([rho_param], max_norm=1.0)
            optimizer.step()
            
            with torch.no_grad():
                rho_param.clamp_(0.01, 1.0)
                
            loss_val = loss.item()
            max_mse_val = mse_per_task.max().item()
                
            print(f"Epoch {epoch+1:03d} | Softmax MSE: {loss_val:.5f} | IPC: {total_capacity:.2f} | Max MSE: {max_mse_val:.5f}")
            
            # [MODIFIED]: Append all three to histories
            loss_history.append(loss_val)
            ipc_history.append(total_capacity)
            max_mse_history.append(max_mse_val)

            loss_history.append(loss.item())
            current_rho, current_pos = rho_param.detach().cpu().numpy(), pos_param.detach().cpu().numpy()
            plot_epoch_topology(current_pos, config['springs'], current_rho, config['masks'], epoch, topo_epoch_dir)
            plot_topology_curve(current_rho, 0.05, epoch, topo_curve_dir)

        # Save Sweep Results
        plt.figure()
        plt.plot(loss_history)
        plt.title(f"Optimization Loss Curve (k={current_stiffness})")
        plt.xlabel("Epoch")
        plt.ylabel("Softmax MSE Loss")
        plt.grid(True, alpha=0.3)
        plt.savefig(EXPERIMENT_DIR / "loss_curve.png", dpi=150)
        plt.close()

        
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

        k_mat_opt, c_mat_opt = np.zeros((ROWS*COLS, ROWS*COLS)), np.zeros((ROWS*COLS, ROWS*COLS))
        for s in range(len(config['springs'])):
            i, j = int(config['springs'][s, 0]), int(config['springs'][s, 1])
            r = current_rho[s]
            if r > 0.05:
                k_mat_opt[i, j] = k_mat_opt[j, i] = config['springs'][s, 3] * r
                c_mat_opt[i, j] = c_mat_opt[j, i] = config['springs'][s, 5] * r
                
        with open(EXPERIMENT_DIR / "ga_results.json", "w") as f:
            json.dump({"k_mat_opt": k_mat_opt.tolist(), "c_mat_opt": c_mat_opt.tolist()}, f, indent=4)
            
        print(f"[Success] Saved stiffness {current_stiffness} results to {EXPERIMENT_DIR}")