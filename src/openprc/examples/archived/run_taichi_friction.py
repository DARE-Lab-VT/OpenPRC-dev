import os
import json
import time
import numpy as np
import taichi as ti
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path

from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler

# Initialize Taichi (GPU preferred, f64 required for stability)
ti.init(arch=ti.gpu, default_fp=ti.f64)
torch.set_default_dtype(torch.float64)
torch.cuda.set_device(0)

# --- Configuration & Paths ---
TRIAL_NAME = "Taichi_Friction"
ROWS, COLS = 4, 4
EXPERIMENT_DIR = Path(__file__).parent.parent / "experiments" / TRIAL_NAME
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

# ==========================================
# Visualization Helpers
# ==========================================
def plot_epoch_topology(objects, springs, rho, masks, epoch, save_dir):
    """Plots the spring network geometry, using rho to determine line thickness."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot springs
    for idx, s in enumerate(springs):
        node_a, node_b = int(s[0]), int(s[1])
        xa, ya = objects[node_a][0], objects[node_a][1]
        xb, yb = objects[node_b][0], objects[node_b][1]
        weight = float(rho[idx])
        
        if weight > 0.05: 
            ax.plot([xa, xb], [ya, yb], c='black', linewidth=weight * 8, alpha=1, zorder=1)

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
# Geometry & Physics Setup
# ==========================================
def generate_openprc_grid_config(rows=4, cols=4):
    SPACING = 0.053     
    STIFFNESS = 222.15  
    DAMPING = 0.8
    NODE_MASS = 0.01    
    PRETENSION = 0.10   # 10% Pretension
    
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
    # Horizontal & Vertical Springs with shortened rest lengths (Pretension)
    for r in range(rows):
        for c in range(cols - 1):
            # Rest length L0 = Actual Distance * (1 - 0.1)
            springs_list.append([node_indices[r, c], node_indices[r, c + 1], SPACING * (1.0 - PRETENSION), STIFFNESS, 0.0, DAMPING, 0.0])
    for r in range(rows - 1):
        for c in range(cols):
            springs_list.append([node_indices[r, c], node_indices[r + 1, c], SPACING * (1.0 - PRETENSION), STIFFNESS, 0.0, DAMPING, 0.0])
    
    hinges_list = []
    # Rotational Frictional Damping Coefficient
    HINGE_DAMPING = 0.001
    for r in range(rows - 1):
        for c in range(cols - 1):
            n_tl, n_tr, n_bl = node_indices[r, c], node_indices[r, c + 1], node_indices[r + 1, c]
            # Hinge structure: [Node_i, Vertex_j, Node_k, Damping_Coeff, Dummy_Rest_Angle]
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
        
        # Geometry Fields (3D)
        self.x_rest = ti.Vector.field(3, dtype=ti.f64, shape=self.num_mass, needs_grad=True)
        self.spring_l0 = ti.field(dtype=ti.f64, shape=self.num_springs, needs_grad=True)
        self.rho = ti.field(dtype=ti.f64, shape=self.num_springs, needs_grad=True)
        
        # Simulation Fields (3D)
        self.x = ti.Vector.field(3, dtype=ti.f64, shape=(batch_steps, self.num_mass), needs_grad=True)
        self.v = ti.Vector.field(3, dtype=ti.f64, shape=(batch_steps, self.num_mass), needs_grad=True)
        self.force_buffer = ti.Vector.field(3, dtype=ti.f64, shape=self.num_mass, needs_grad=True)
        self.node_disp = ti.field(dtype=ti.f64, shape=(batch_steps, self.num_mass), needs_grad=True)
        
        # Static Properties
        self.spring_a = ti.field(dtype=ti.i32, shape=self.num_springs)
        self.spring_b = ti.field(dtype=ti.i32, shape=self.num_springs)
        self.spring_k1 = ti.field(dtype=ti.f64, shape=self.num_springs)
        self.spring_d1 = ti.field(dtype=ti.f64, shape=self.num_springs)
        
        # Hinge Properties
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
        # 1. Global Damping
        for i in range(self.num_mass):
            self.force_buffer[i] = ti.Vector([0.0, 0.0, 0.0])
            self.force_buffer[i] += -0.1 * self.v[t, i]

        # 2. Linear Spring Forces
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
            
        # 3. Rotational Frictional Damping
        for h in range(self.num_hinges):
            i, j, k = self.hinge_i[h], self.hinge_j[h], self.hinge_k[h]
            r_ji = self.x[t, i] - self.x[t, j]
            r_jk = self.x[t, k] - self.x[t, j]
            l_ji_sq, l_jk_sq = r_ji.norm_sqr() + 1e-12, r_jk.norm_sqr() + 1e-12
            
            # Calculate angular velocity of each arm relative to vertex
            # Omega = (r x v) / r^2
            v_rel_i = self.v[t, i] - self.v[t, j]
            v_rel_k = self.v[t, k] - self.v[t, j]
            omega_i = r_ji.cross(v_rel_i) / l_ji_sq
            omega_k = r_jk.cross(v_rel_k) / l_jk_sq
            
            # Damping Torque resists relative angular velocity
            rel_omega = omega_i - omega_k
            torque = -self.hinge_d_tor[h] * rel_omega
            
            # Convert torque back to linear forces
            F_i = torque.cross(r_ji) / l_ji_sq
            F_k = -torque.cross(r_jk) / l_jk_sq
            ti.atomic_add(self.force_buffer[i], F_i)
            ti.atomic_add(self.force_buffer[k], F_k)
            ti.atomic_add(self.force_buffer[j], -F_i - F_k)

        # 4. Input Integration
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
            self.node_disp[t, i] = self.x[t, i][0] - self.x_rest[i][0]

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
    print(f"=== OpenPRC Compatible Taichi Topology Optimization ===\nOutput Dir: {EXPERIMENT_DIR}")
    
    # Setup Dedicated Subfolders
    topo_epoch_dir = EXPERIMENT_DIR / "topology_epoch"
    topo_curve_dir = EXPERIMENT_DIR / "topology_curve"
    os.makedirs(topo_epoch_dir, exist_ok=True)
    os.makedirs(topo_curve_dir, exist_ok=True)
    
    dt_sim, dt_save, T_DURATION = 0.001, 0.01, 25.0    
    t_eval_1000Hz = np.arange(0, T_DURATION, dt_sim)
    
    amp, f1, f2, f3 = 0.02, 2.11, 3.73, 4.33
    u_1000Hz = amp * np.sin(2 * np.pi * f1 * t_eval_1000Hz) * np.sin(2 * np.pi * f2 * t_eval_1000Hz) * np.sin(2 * np.pi * f3 * t_eval_1000Hz)
    v_1000Hz = np.gradient(u_1000Hz, dt_sim)
    
    config, _ = generate_openprc_grid_config(ROWS, COLS)
    res_opt = DifferentiableReservoir(config, dt_sim, len(t_eval_1000Hz))
    
    pos_param = nn.Parameter(torch.tensor(config['objects'], dtype=torch.float64))
    rho_param = nn.Parameter(torch.ones(len(config['springs']), dtype=torch.float64) * 0.5)
    optimizer = torch.optim.Adam([rho_param], lr=0.05)
    
    # Generate Targets
    y_target_100Hz = generate_benchmark_target_matrix(u_1000Hz[::10], tau_d=5, k=100, n=2)
    y_target_torch = torch.tensor((y_target_100Hz - y_target_100Hz.mean(axis=0)) / (y_target_100Hz.std(axis=0) + 1e-12), dtype=torch.float64).to(pos_param.device)
    
    washout_frames, train_frames = 500, 1000
    train_stop = washout_frames + train_frames
    y_train_torch = y_target_torch[washout_frames:train_stop]

    EPOCHS = 100
    loss_history = []
    
    # FIX 2: Create a boolean mask for ONLY the active (non-input) nodes
    active_mask = torch.tensor(config['masks']['active'], dtype=torch.bool, device=pos_param.device)

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        X_100Hz = ReservoirOptimizationFunction.apply(pos_param, rho_param, res_opt, u_1000Hz, v_1000Hz)
        
        # FIX 2: Extract ONLY the active nodes before normalizing.
        # This prevents singular matrix inversions caused by identical columns!
        X_active = X_100Hz[:, active_mask]
        X_norm = (X_active - X_active.mean(dim=0, keepdim=True)) / (X_active.std(dim=0, keepdim=True) + 1e-6)
        
        X_train = torch.cat([torch.ones(len(X_norm), 1, device=X_norm.device), X_norm], dim=1)[washout_frames:train_stop]
        
        # FIX 3: Increased Ridge penalty (1e-6 to 1e-3) to buffer matrix inversion gradients
        I = torch.eye(X_train.shape[1], device=X_train.device) * 1e-3
        W_out = torch.linalg.solve(X_train.T @ X_train + I, X_train.T @ y_train_torch)
        
        mse_per_task = torch.mean((X_train @ W_out - y_train_torch)**2, dim=0)
        loss = torch.sum(F.softmax(mse_per_task / 0.05, dim=0) * mse_per_task)

        print(f"Epoch {epoch+1:03d}/{EPOCHS} | Softmax Loss: {loss.item():.5f} | Max MSE: {torch.max(mse_per_task).item():.5f}")
        
        loss.backward()
        
        # FIX 4: Safety clip to guarantee `Adam` is never ruined by an anomaly
        torch.nn.utils.clip_grad_norm_([rho_param], max_norm=1.0)
        
        optimizer.step()
        
        with torch.no_grad():
            rho_param.clamp_(0.01, 1.0)
            
        # Logging & Plots to specific subfolders
        loss_history.append(loss.item())
        current_rho, current_pos = rho_param.detach().cpu().numpy(), pos_param.detach().cpu().numpy()
        plot_epoch_topology(current_pos, config['springs'], current_rho, config['masks'], epoch, topo_epoch_dir)
        plot_topology_curve(current_rho, 0.05, epoch, topo_curve_dir)

    # --- Save OpenPRC-Compatible Output ---
    plt.figure()
    plt.plot(loss_history)
    plt.title("Optimization Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Softmax MSE Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig(EXPERIMENT_DIR / "loss_curve.png", dpi=150)
    plt.close()

    k_mat_opt, c_mat_opt = np.zeros((ROWS*COLS, ROWS*COLS)), np.zeros((ROWS*COLS, ROWS*COLS))
    for s in range(len(config['springs'])):
        i, j = int(config['springs'][s, 0]), int(config['springs'][s, 1])
        r = current_rho[s]
        if r > 0.05:
            k_mat_opt[i, j] = k_mat_opt[j, i] = config['springs'][s, 3] * r
            c_mat_opt[i, j] = c_mat_opt[j, i] = config['springs'][s, 5] * r
            
    with open(EXPERIMENT_DIR / "ga_results.json", "w") as f:
        json.dump({"k_mat_opt": k_mat_opt.tolist(), "c_mat_opt": c_mat_opt.tolist()}, f, indent=4)
        
    print(f"\n[Success] Direct topological optimization complete!")