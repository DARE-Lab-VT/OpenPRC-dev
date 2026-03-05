import os
import json
import time
import numpy as np
import pandas as pd
import taichi as ti
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path

from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# Initialize Taichi (GPU preferred, f64 required for stability)
ti.init(arch=ti.gpu, default_fp=ti.f64)
torch.set_default_dtype(torch.float64)
torch.cuda.set_device(0)

# --- Configuration & Paths ---
ROWS, COLS = 4, 4
EXPERIMENT_DIR = Path(__file__).parent.parent / "experiments" / f"spring_mass_{ROWS}x{COLS}_test"
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

def generate_openprc_grid_config(rows=4, cols=4):
    """Generates a fully connected graph to proxy linear springs + angular hinges."""
    SPACING = 0.053     # meters
    STIFFNESS = 222.15  # N/m
    DAMPING = 0.8
    NODE_MASS = 0.01    # kg
    
    num_nodes = rows * cols
    objects = np.zeros((num_nodes, 3))
    node_indices = np.zeros((rows, cols), dtype=int)
    
    # 1. Generate Nodes
    idx = 0
    for r in range(rows):
        for c in range(cols):
            objects[idx] = [c * SPACING, -r * SPACING, 0.0]
            node_indices[r, c] = idx
            idx += 1
            
    # 2. Generate Fully Connected Springs (Allows discovery of cross-bracing)
    springs_list = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            l0 = np.linalg.norm(objects[i] - objects[j])
            # Array Format: [node_a, node_b, l0, k1, k3, d1, d3]
            springs_list.append([i, j, l0, STIFFNESS, 0.0, DAMPING, 0.0])
            
    springs = np.array(springs_list)
    
    # 3. Masks and Actuation Weights
    fixed_mask = np.zeros(num_nodes, dtype=int)
    input_weights = np.zeros(num_nodes)
    
    # Actuate the four corners
    act_indices = [
        node_indices[0, 0], 
        node_indices[0, cols - 1], 
        node_indices[rows - 1, 0], 
        node_indices[rows - 1, cols - 1]
    ]
    
    for i in act_indices:
        input_weights[i] = 1.0 
        
    config = {
        'objects': objects,
        'springs': springs,
        'masks': {
            'fixed': fixed_mask,
            'active': ~fixed_mask.astype(bool),
            'input_indices': act_indices
        },
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
        
        # Geometry Fields
        self.x_rest = ti.Vector.field(3, dtype=ti.f64, shape=self.num_mass, needs_grad=True)
        self.spring_l0 = ti.field(dtype=ti.f64, shape=self.num_springs, needs_grad=True)
        self.rho = ti.field(dtype=ti.f64, shape=self.num_springs, needs_grad=True)
        
        # Simulation Fields
        self.x = ti.Vector.field(3, dtype=ti.f64, shape=(batch_steps, self.num_mass), needs_grad=True)
        self.v = ti.Vector.field(3, dtype=ti.f64, shape=(batch_steps, self.num_mass), needs_grad=True)
        self.force_buffer = ti.Vector.field(3, dtype=ti.f64, shape=self.num_mass, needs_grad=True)
        
        # Target Feature: X-axis Node Displacements
        self.node_disp = ti.field(dtype=ti.f64, shape=(batch_steps, self.num_mass), needs_grad=True)
        
        # Static Properties
        self.spring_a = ti.field(dtype=ti.i32, shape=self.num_springs)
        self.spring_b = ti.field(dtype=ti.i32, shape=self.num_springs)
        self.spring_k1 = ti.field(dtype=ti.f64, shape=self.num_springs)
        self.spring_d1 = ti.field(dtype=ti.f64, shape=self.num_springs)
        self.input_weights = ti.field(dtype=ti.f64, shape=self.num_mass)
        self.fixed_mask = ti.field(dtype=ti.i32, shape=self.num_mass)
        
        self.loss = ti.field(dtype=ti.f64, shape=(), needs_grad=True)
        self._initialize_from_config(config)

    def _initialize_from_config(self, config):
        objects = config['objects']
        springs = config['springs']
        masks = config['masks']
        weights = config['weights']['input_weights']
        
        for i in range(self.num_mass):
            self.x_rest[i] = [objects[i, 0], objects[i, 1], objects[i, 2]]
            self.fixed_mask[i] = 1 if masks['fixed'][i] else 0
            self.input_weights[i] = weights[i]
            
        for s in range(self.num_springs):
            self.spring_a[s] = int(springs[s, 0])
            self.spring_b[s] = int(springs[s, 1])
            self.spring_k1[s] = springs[s, 3]
            self.spring_d1[s] = springs[s, 5]
            self.spring_l0[s] = springs[s, 2]
            self.rho[s] = 0.5

    @ti.kernel
    def update_geometry_kernel(self):
        for s in range(self.num_springs):
            i, j = self.spring_a[s], self.spring_b[s]
            self.spring_l0[s] = (self.x_rest[i] - self.x_rest[j]).norm()
            
    @ti.kernel
    def init_simulation_state(self):
        for i in range(self.num_mass):
            self.x[0, i] = self.x_rest[i]
            self.v[0, i] = [0.0, 0.0, 0.0]

    @ti.kernel
    def compute_step(self, t: int, u_val: ti.f64, v_val: ti.f64):
        # 1. Base Forces: Global Damping only (Gravity acts in Z-axis, which is out-of-plane)
        for i in range(self.num_mass):
            self.force_buffer[i] = ti.Vector([0.0, 0.0, 0.0])
            self.force_buffer[i] += -0.1 * self.v[t, i]          # Global Damping

        # 2. Spring Forces (Weighted by Rho)
        for s in range(self.num_springs):
            i, j = self.spring_a[s], self.spring_b[s]
            
            disp = self.x[t, i] - self.x[t, j]
            dist = disp.norm() + 1e-12
            dir_vec = disp / dist
            
            extension = dist - self.spring_l0[s]
            rel_vel = self.v[t, i] - self.v[t, j]
            dot_v = rel_vel.dot(dir_vec)
            
            rho_s = self.rho[s]
            f_spring = self.spring_k1[s] * extension
            f_damp = self.spring_d1[s] * dot_v
            
            force_vec = -(f_spring + f_damp) * dir_vec * rho_s
            ti.atomic_add(self.force_buffer[i], force_vec)
            ti.atomic_add(self.force_buffer[j], -force_vec)

        # 3. Input Overrides & Integration
        for i in range(self.num_mass):
            if self.input_weights[i] > 0.0:
                # Actuate strictly in X-direction
                disp_x = self.input_weights[i] * u_val
                vel_x = self.input_weights[i] * v_val
                self.x[t+1, i] = self.x_rest[i] + ti.Vector([disp_x, 0.0, 0.0])
                self.v[t+1, i] = ti.Vector([vel_x, 0.0, 0.0])
            else:
                acc = self.force_buffer[i] / 0.01
                self.v[t+1, i] = self.v[t, i] + acc * self.dt
                self.x[t+1, i] = self.x[t, i] + self.v[t+1, i] * self.dt

    @ti.kernel
    def compute_features(self):
        """Extracts X-axis node displacements for the entire history"""
        for t, i in ti.ndrange(self.batch_steps, self.num_mass):
            self.node_disp[t, i] = self.x[t, i][0] - self.x_rest[i][0]

    def run_batch_forward(self, u_input_array, v_input_array):
        self.update_geometry_kernel()
        self.init_simulation_state()
        for t in range(self.batch_steps - 1):
            self.compute_step(t, float(u_input_array[t]), float(v_input_array[t]))
        self.compute_features()

# ==========================================
# Target Generator & PyTorch Bridge
# ==========================================
def generate_benchmark_target_matrix(u_input, tau_d, k, n, tau_step=1):
    u = np.asarray(u_input, dtype=float)
    T = u.shape[0]
    max_lag = tau_d * k
    
    U_full = sliding_window_view(u, max_lag + 1)[:, ::-1]
    lag_indices = [j * k for j in range(0, tau_d + 1, tau_step)]
    U_sub = U_full[:, lag_indices] 
    Lvars = U_sub.shape[1]
    
    exps = []
    vec = np.zeros(Lvars, dtype=int)
    
    def rec(rem, i):
        if i == Lvars - 1:
            vec[i] = rem
            exps.append(vec.copy())
        else:
            for v in range(rem + 1):
                vec[i] = v
                rec(rem - v, i + 1)
    
    for d in range(1, n + 1): rec(d, 0)
    exps = np.array(exps)
    K = exps.shape[0]
    
    Y_valid = np.ones((U_sub.shape[0], K), dtype=float)
    for j in range(Lvars):
        ej = exps[:, j]
        if np.any(ej):
            Y_valid *= (U_sub[:, [j]] ** ej[None, :])
            
    Y_full = np.zeros((T, K), dtype=float)
    Y_full[max_lag:, :] = Y_valid
    return Y_full, exps, lag_indices

class ReservoirOptimizationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pos_tensor, rho_tensor, reservoir, u_1000Hz, v_1000Hz):
        ctx.reservoir = reservoir
        ctx.u_1000Hz = u_1000Hz
        ctx.v_1000Hz = v_1000Hz

        reservoir.x_rest.from_torch(pos_tensor)
        reservoir.rho.from_torch(rho_tensor)
        reservoir.x_rest.grad.fill(0)
        reservoir.rho.grad.fill(0)

        reservoir.update_geometry_kernel()
        reservoir.run_batch_forward(u_1000Hz, v_1000Hz)

        # Retrieve 1000Hz features, downsample to 100Hz (Step = 10)
        disp_1000Hz = reservoir.node_disp.to_torch()
        disp_100Hz = disp_1000Hz[::10, :] 
        return disp_100Hz

    @staticmethod
    def backward(ctx, grad_outputs_100Hz):
        reservoir = ctx.reservoir
        
        # Upsample Gradients from 100Hz back to 1000Hz
        grad_1000Hz = torch.zeros((reservoir.batch_steps, reservoir.num_mass), 
                                  dtype=torch.float64, device=grad_outputs_100Hz.device)
        grad_1000Hz[::10, :] = grad_outputs_100Hz

        reservoir.x_rest.grad.fill(0)
        reservoir.rho.grad.fill(0)
        reservoir.node_disp.grad.from_torch(grad_1000Hz)

        with ti.ad.Tape(loss=reservoir.loss, validation=False):
            reservoir.update_geometry_kernel()
            reservoir.run_batch_forward(ctx.u_1000Hz, ctx.v_1000Hz)
            compute_dummy_loss(reservoir, grad_1000Hz)

        grad_pos = reservoir.x_rest.grad.to_torch()
        grad_rho = reservoir.rho.grad.to_torch()
        
        fixed_mask = torch.tensor(reservoir.fixed_mask.to_numpy(), device=grad_pos.device)
        grad_pos = grad_pos * (1.0 - fixed_mask.unsqueeze(1))
        return grad_pos, grad_rho, None, None, None

@ti.kernel
def compute_dummy_loss(reservoir: ti.template(), grad_tensor: ti.types.ndarray()):
    for t, i in ti.ndrange(reservoir.batch_steps, reservoir.num_mass):
        reservoir.loss[None] += reservoir.node_disp[t, i] * grad_tensor[t, i]

def plot_epoch_topology(objects, springs, rho, masks, epoch, save_dir):
    """Plots the spring network geometry, using rho to determine line thickness."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot springs with thickness based on rho
    for idx, s in enumerate(springs):
        node_a, node_b = int(s[0]), int(s[1])
        xa, ya = objects[node_a][0], objects[node_a][1]  # Using index 0 and 1 for X, Y in 3D
        xb, yb = objects[node_b][0], objects[node_b][1]
        
        weight = float(rho[idx])
        
        # Only draw springs with a meaningful density/stiffness
        if weight > 0.05: 
            ax.plot([xa, xb], [ya, yb], c='black', linewidth=weight * 10, alpha=1, zorder=1)

    # Plot Nodes
    ax.scatter(objects[:, 0], objects[:, 1], c='red', s=40, zorder=2, label='Active Node')
    
    # Highlight Input Nodes
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
# Main Optimization Pipeline
# ==========================================
if __name__ == "__main__":
    print(f"=== OpenPRC Compatible Taichi Topology Optimization ===")
    
    # Physics & Simulation Time (25 Seconds Total)
    dt_sim = 0.001       # 1000 Hz for Physics
    dt_save = 0.01       # 100 Hz for OpenPRC Readout
    T_DURATION = 25.0    
    
    t_eval_1000Hz = np.arange(0, T_DURATION, dt_sim)
    total_steps_1000Hz = len(t_eval_1000Hz)
    total_steps_100Hz = int(T_DURATION / dt_save)
    
    # Input Generation
    amp, f1, f2, f3 = 0.02, 2.11, 3.73, 4.33
    u_1000Hz = amp * np.sin(2 * np.pi * f1 * t_eval_1000Hz) * \
                     np.sin(2 * np.pi * f2 * t_eval_1000Hz) * \
                     np.sin(2 * np.pi * f3 * t_eval_1000Hz)
    v_1000Hz = np.gradient(u_1000Hz, dt_sim)
    
    # 100 Hz Downsampled input for Target Matrix
    u_100Hz = u_1000Hz[::10] 

    # Config
    config, _ = generate_openprc_grid_config(ROWS, COLS)
    num_springs = len(config['springs'])
    
    res_opt = DifferentiableReservoir(config, dt_sim, total_steps_1000Hz)
    
    # PyTorch Parameters
    pos_param = nn.Parameter(torch.tensor(config['objects'], dtype=torch.float64))
    rho_param = nn.Parameter(torch.ones(num_springs, dtype=torch.float64) * 0.5)
    optimizer = torch.optim.Adam([rho_param], lr=0.05)
    
    # --- Generate OpenPRC Benchmark Target Matrix ---
    # Memory benchmark: N=2, TAU_D=5, k_delay=100 (100 frames = 1.0 seconds at 100Hz)
    tau, k_delay, n_poly = 5, 100, 2 
    y_target_100Hz, exps, lags = generate_benchmark_target_matrix(u_100Hz, tau, k_delay, n_poly)
    
    # Normalize Targets
    y_target_100Hz = (y_target_100Hz - y_target_100Hz.mean(axis=0)) / (y_target_100Hz.std(axis=0) + 1e-12)
    y_target_torch = torch.tensor(y_target_100Hz, dtype=torch.float64).to(pos_param.device)
    
    # Define OpenPRC Timings (in 100Hz Frames)
    washout_frames = 500   # 5.0 seconds
    train_frames = 1000    # 10.0 seconds
    test_frames = 1000     # 10.0 seconds
    train_stop = washout_frames + train_frames
    
    y_train_torch = y_target_torch[washout_frames:train_stop]

    # --- Optimization Loop ---
    EPOCHS = 100

    vis_dir = EXPERIMENT_DIR / "topology_evolution"
    os.makedirs(vis_dir, exist_ok=True)
    loss_history = []

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        # 1. Forward Physics (Returns 100Hz displacements)
        X_100Hz = ReservoirOptimizationFunction.apply(pos_param, rho_param, res_opt, u_1000Hz, v_1000Hz)
        
        # 2. Standardize Features
        X_mean = X_100Hz.mean(dim=0, keepdim=True)
        X_std = X_100Hz.std(dim=0, keepdim=True) + 1e-6
        X_norm = (X_100Hz - X_mean) / X_std
        
        # 3. Add bias and Slice Train segment
        X_padded = torch.cat([torch.ones(total_steps_100Hz, 1, device=X_norm.device), X_norm], dim=1)
        X_train = X_padded[washout_frames:train_stop]
        
        # 4. Ridge Regression (lambda = 1e-6)
        lambda_reg = 1e-6
        I = torch.eye(X_train.shape[1], device=X_train.device) * lambda_reg
        W_out = torch.linalg.solve(X_train.T @ X_train + I, X_train.T @ y_train_torch)
        
        # 5. SoftMax Loss on Training MSE
        y_pred = X_train @ W_out
        mse_per_task = torch.mean((y_pred - y_train_torch)**2, dim=0)
        
        weights = F.softmax(mse_per_task / 0.05, dim=0)
        loss = torch.sum(weights * mse_per_task)

        print(f"Epoch {epoch+1}/{EPOCHS} | Softmax Loss: {loss.item():.5f} | Max MSE: {torch.max(mse_per_task).item():.5f}")
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            rho_param.clamp_(0.01, 1.0) # Prevent completely breaking connections

        with torch.no_grad():
            rho_param.clamp_(0.01, 1.0) # Prevent completely breaking connections
            
        # NEW: Track loss and plot epoch visualizations
        loss_history.append(loss.item())
        current_rho = rho_param.detach().cpu().numpy()
        current_pos = pos_param.detach().cpu().numpy()
        
        plot_epoch_topology(current_pos, config['springs'], current_rho, config['masks'], epoch, vis_dir)
        plot_topology_curve(current_rho, 0.05, epoch, vis_dir) # 0.05 matches our cutoff threshold

    # --- Save OpenPRC-Compatible Output ---
    print("\n=== Formatting Matrices for OpenPRC ===")
    final_rho = rho_param.detach().cpu().numpy()
    
    # Reconstruct Adjacency/Stiffness Matrices
    k_mat_opt = np.zeros((ROWS*COLS, ROWS*COLS))
    c_mat_opt = np.zeros((ROWS*COLS, ROWS*COLS))
    
    for s in range(num_springs):
        i, j = int(config['springs'][s, 0]), int(config['springs'][s, 1])
        r = final_rho[s]
        if r > 0.05: # Clean up weak/dead springs
            k_mat_opt[i, j] = k_mat_opt[j, i] = config['springs'][s, 3] * r
            c_mat_opt[i, j] = c_mat_opt[j, i] = config['springs'][s, 5] * r
            
    # Save as ga_results.json
    results_payload = {
        "best_solution": {
            "fitness": float(1.0 - torch.mean(mse_per_task).item()), # Proxy fitness
            "genes": [] # Empty, as we bypassed Fourier
        },
        "history": {"best": [], "mean": []},
        "k_mat_opt": k_mat_opt.tolist(),
        "c_mat_opt": c_mat_opt.tolist()
    }
    
    json_path = EXPERIMENT_DIR / "ga_results.json"
    with open(json_path, "w") as f:
        json.dump(results_payload, f, indent=4)
        
    print(f"\n[Success] Direct topological optimization complete!")
    print(f"Saved optimized matrices to: {json_path}")

    # NEW: Plot and save overall optimization loss curve
    plt.figure()
    plt.plot(loss_history)
    plt.title("Optimization Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Softmax MSE Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig(EXPERIMENT_DIR / "loss_curve.png", dpi=150)
    plt.close()
    
    print(f"Saved visual diagnostics to: {EXPERIMENT_DIR}")
    