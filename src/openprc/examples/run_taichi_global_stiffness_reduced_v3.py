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
from scipy.interpolate import CubicSpline, interp1d
from scipy.stats import chi2
from numpy.lib.stride_tricks import sliding_window_view
import h5py
import random
import sys

# openprc-faithful per-epoch heatmap helpers (same examples/ directory)
sys.path.insert(0, str(Path(__file__).parent))
from iteration_heatmap import (
    build_legendre_targets, build_features, evaluate, plot_iteration_heatmap)

def global_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

global_seed(42)
ti.init(arch=ti.gpu, default_fp=ti.f64, random_seed=42)
torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)


# --- Configuration ---
# Optimization objective:
#   "linear_memory"  → L[C ] (Eq. 13): 30 degree-1 targets at delays 0–29; mean MSE loss
#   "nonlinearity"   → NL[C] (Eq. 14): all ~46k targets; degree-weighted MSE loss

# --- Real Input Path ---
REAL_INPUT_DATA = Path(__file__).parent.parent / "experiments" / "sim2real_plate" / "input_data" / "experiment.h5"

# --- Sub-Spring Physics Parameters ---
NUM_SUB_SPRINGS = 6          # Each logical spring subdivided into 3 physical segments
                              # 16 main + 24*2 sub = 64 total nodes (memory-efficient for AD)
SPACING = 0.053              # meters
NODE_MASS = 0.01             # kg per main node
# PRETENSION = 0.10            # 10% pretension on rest length
GLOBAL_DAMPING = 0.00001     # N*s/m global linear drag

# Macro corner hinge parameters (at main-node grid intersections)
HINGE_DAMPING = 0.0001
HINGE_STIFFNESS = 0.1
HINGE_STIFFNESS_OOP = 0.0001 #was 0.0001

# Micro sub-hinge parameters (along each spring chain)
SUB_HINGE_DAMPING = 0.0001
SUB_HINGE_STIFFNESS = 0.01
SUB_HINGE_STIFFNESS_OOP = 0.001

GRAVITY_Z = -9.81

# Damping-stiffness relationship: c scales linearly with k
K_MAX = 222.15          # maximum physical stiffness (N/m)
C_MAX = 0.80            # damping at K_MAX
C_DAMPING_SCALE = C_MAX / K_MAX   # c = k * C_DAMPING_SCALE

# Fixed stiffness levels: {level_index: fixed_value}
# Level 0 is always void (k=0). Add more entries to pin additional levels.

topologies_dict = {
    "topo_0": [], # Baseline: Fully connected
    "topo_1": [(5, 6), (6, 7), (8, 9)], # Example: Removed springs between 0-1 and 2-3
    "topo_2": [(2, 6), (4, 5), (8, 9), (10, 11)], # Example: Removed three internal springs
    "topo_3": [(1, 5), (2, 6), (9, 13), (10, 14)],
    "topo_4": [(1, 5), (2, 6), (5, 9), (6, 10), (9, 13), (10, 14)],
    "topo_5": [(4, 5), (5, 6), (6, 7), (8, 9), (9, 10), (10, 11)]
}

REAL_AMP_SCALE = 1.5
PRETENSION = 0.10            
TOPOLOGY_NO = None
LOSS_MODE = "linear_memory"
M_LEVELS_CONFIG = 4
K_FIXED_LEVELS = {0: 0.0}
# Minimum stiffness (N/m) enforced on every non-zero (learnable) level after each
# Adam step. Prevents a learnable level from collapsing toward 0 and becoming a
# second "void" material the topology constraints (which key void off level 0) miss.
K_MIN_LEVEL = 50.0

RUN = f"M{M_LEVELS_CONFIG}_{LOSS_MODE}_1"
TRIAL_NAME = f"Taichi_run+heatmap/{RUN}"
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
        with torch.no_grad():
            for (i, j), viol_tensor in viols.items():
                viol_val = viol_tensor.item()
                self.kappa[i, j] += self.tau[i, j] * viol_val
                if viol_val > self.prev_viol[i, j]:
                    self.tau[i, j] *= 1.01
                # elif viol_val < self.prev_viol[i, j] * 0.9:
                #     self.tau[i, j] *= 0.99
                self.prev_viol[i, j] = viol_val
                if verbose:
                    print(f"AL Constraint ({i},{j}): Viol={viol_val:.6f}, Kappa={self.kappa[i, j].item():.4f}, Tau={self.tau[i, j].item():.4f}")


class CombinedTopologyAL(PyTorchAL):
    def __init__(self, config, rows, cols, M_levels, device, tau=1.0):
        super().__init__(M_levels + 1, device, tau)
        self.M_real = M_levels
        self.hinge_idx = M_levels
        self.device = device
        self.rows = rows
        self.cols = cols
        self.num_springs = len(config['springs'])

        self.map_T = torch.full((rows * cols,), self.num_springs, dtype=torch.long, device=device)
        self.map_B = torch.full((rows * cols,), self.num_springs, dtype=torch.long, device=device)
        self.map_L = torch.full((rows * cols,), self.num_springs, dtype=torch.long, device=device)
        self.map_R = torch.full((rows * cols,), self.num_springs, dtype=torch.long, device=device)

        for s_idx, s in enumerate(config['springs']):
            n1, n2 = int(s[0]), int(s[1])
            if n2 == n1 + 1:
                self.map_R[n1] = s_idx
                self.map_L[n2] = s_idx
            elif n2 == n1 + cols:
                self.map_B[n1] = s_idx
                self.map_T[n2] = s_idx

        self.valid_nodes_mask = torch.ones(rows * cols, dtype=torch.bool, device=device)
        corners = [0, cols - 1, (rows - 1) * cols, rows * cols - 1]
        self.valid_nodes_mask[corners] = False

    def compute_loss(self, z_star):
        loss_reg = 0.0
        max_viol = 0.0
        viols = {}

        for i in range(self.M_real):
            for j in range(i + 1, self.M_real):
                numerator = z_star[:, i] * z_star[:, j]
                denominator = (z_star[:, i] + z_star[:, j]) / 2.0 + 1e-4
                viol = torch.mean(numerator / denominator)
                k = self.kappa[i, j]
                t = self.tau[i, j]
                loss_reg += k * viol + 0.5 * t * (viol ** 2)
                viols[(i, j)] = viol
                max_viol = max(max_viol, viol.item())

        p_spring = 1.0 - z_star[:, 0]
        p_padded = torch.cat([p_spring, torch.tensor([0.0], device=self.device)])
        p_T = p_padded[self.map_T]
        p_B = p_padded[self.map_B]
        p_L = p_padded[self.map_L]
        p_R = p_padded[self.map_R]
        H_i = p_L * p_R
        V_i = p_T * p_B
        penalties = (1.0 - H_i) * (1.0 - V_i)
        active_penalties = penalties[self.valid_nodes_mask]
        hinge_viol = torch.mean(active_penalties)
        hk = self.kappa[self.hinge_idx, self.hinge_idx]
        ht = self.tau[self.hinge_idx, self.hinge_idx]
        loss_reg += hk * hinge_viol + 0.5 * ht * (hinge_viol ** 2)
        viols[(self.hinge_idx, self.hinge_idx)] = hinge_viol
        max_viol = max(max_viol, hinge_viol.item())

        return loss_reg, max_viol, viols

# ==========================================
# Real Input Loader
# ==========================================
def load_real_input(filepath, duration, dt_sim, target_hz=30.0):
    """Load and resample real experimental waveform from HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        real_pos = f['time_series/nodes/positions'][:, 0, 0]
        real_data = real_pos - real_pos[0]
        real_max = np.max(np.abs(real_data))
        if real_max > 0:
            real_data = real_data / 10000.0 * REAL_AMP_SCALE

    t_real = np.arange(len(real_data)) / target_hz
    interp_func = interp1d(t_real, real_data, kind='cubic', fill_value="extrapolate")
    t_sim = np.arange(0, duration, dt_sim)
    u_fine = interp_func(t_sim)
    v_fine = np.gradient(u_fine, dt_sim)
    return u_fine, v_fine, t_sim

# ==========================================
# Signal Utilities
# ==========================================
def compute_effective_rank_torch(state_matrix):
    """Computes Shannon-entropy effective rank using PyTorch SVD."""
    if state_matrix.shape[0] < 2:
        return 1.0
    X_mean = state_matrix.mean(dim=0, keepdim=True)
    X_std = state_matrix.std(dim=0, keepdim=True) + 1e-12
    X_norm = (state_matrix - X_mean) / X_std
    _, s, _ = torch.linalg.svd(X_norm, full_matrices=False)
    s_norm = s / torch.sum(s)
    rank = torch.exp(-torch.sum(s_norm * torch.log(s_norm + 1e-12)))
    return rank.item()

def calculate_dambre_eps(effective_rank=1.5, test_duration=1000, p_value=1e-4):
    t = chi2.isf(p_value, df=effective_rank)
    return (2.0 * t) / test_duration

# ==========================================
# Visualization Helpers
# ==========================================
def plot_material_distribution(z_star_np, K_vals, epoch, save_dir):
    num_springs, M = z_star_np.shape
    dominant_class = np.argmax(z_star_np, axis=1)
    dominant_prob = np.max(z_star_np, axis=1)
    sort_indices = np.lexsort((dominant_prob, dominant_class))
    sorted_z = z_star_np[sort_indices]

    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    cmap = cm.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=min(K_vals), vmax=max(K_vals))
    x_positions = np.arange(num_springs)
    bottoms = np.zeros(num_springs)

    for m in range(M):
        color = cmap(norm(K_vals[m]))
        ax.bar(x_positions, sorted_z[:, m], bottom=bottoms, width=1.0,
               color=color, label=f'k = {K_vals[m]:.1f}', edgecolor='none')
        bottoms += sorted_z[:, m]

    sorted_classes = dominant_class[sort_indices]
    change_points = np.where(sorted_classes[:-1] != sorted_classes[1:])[0]
    for cp in change_points:
        ax.axvline(x=cp + 0.5, color='white', linestyle='--', linewidth=0.8, alpha=0.5)

    ax.set_xlim(-0.5, num_springs - 0.5)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Spring ID (Grouped by Material, Sorted by Confidence)")
    ax.set_ylabel("Selection Probability ($z^*$)")
    ax.set_title(f"Material Distribution (Epoch {epoch})")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Stiffness", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"material_dist_epoch_{epoch:03d}.png"), bbox_inches='tight')
    plt.close(fig)


def plot_epoch_topology(objects, springs, rho, masks, epoch, save_dir):
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
# Sub-Spring Geometry & Physics Setup
# ==========================================
def generate_subspring_config(rows, cols, num_sub_springs, removed_springs=None):
    """
    Build the full sub-spring topology from a rows×cols main-node grid.

    Each logical spring (main-node pair) is subdivided into `num_sub_springs`
    physical segments with intermediate nodes inserted along the chord.
    Returns everything the DifferentiableReservoir and AL solver need.
    """
    main_nodes_count = rows * cols
    objects = []          # [x, y, z] for every node (main + sub)
    node_indices = np.zeros((rows, cols), dtype=np.int32)

    # 1. Create main nodes
    idx = 0
    for r in range(rows):
        for c in range(cols):
            objects.append([c * SPACING, -r * SPACING, 0.0])
            node_indices[r, c] = idx
            idx += 1

    # 2. Build logical spring list (main-node pairs), respecting removed springs
    removed_set = set()
    if removed_springs:
        for (a, b) in removed_springs:
            removed_set.add((min(a, b), max(a, b)))

    logical_springs = []
    for r in range(rows):
        for c in range(cols - 1):
            na, nb = node_indices[r, c], node_indices[r, c + 1]
            if (min(na, nb), max(na, nb)) not in removed_set:
                logical_springs.append([na, nb])
    for r in range(rows - 1):
        for c in range(cols):
            na, nb = node_indices[r, c], node_indices[r + 1, c]
            if (min(na, nb), max(na, nb)) not in removed_set:
                logical_springs.append([na, nb])

    # 3. Subdivide each logical spring → physical springs + sub-nodes
    physical_springs = []
    sub_parent_map = []      # physical spring → logical spring index
    hinges_list = []
    hinge_dampings = []
    hinge_k_inplane = []
    hinge_k_oop = []
    first_sub_from = {}      # (a, b) → first sub-node index from a's end

    for parent_idx, (node_a, node_b) in enumerate(logical_springs):
        pos_a = np.array(objects[node_a])
        pos_b = np.array(objects[node_b])

        chain_nodes = [node_a]
        for step in range(1, num_sub_springs):
            fraction = step / num_sub_springs
            pos_inter = pos_a + fraction * (pos_b - pos_a)
            objects.append(pos_inter.tolist())
            chain_nodes.append(len(objects) - 1)
        chain_nodes.append(node_b)

        first_sub_from[(node_a, node_b)] = chain_nodes[1]
        first_sub_from[(node_b, node_a)] = chain_nodes[-2]

        # Physical sub-springs
        for i in range(num_sub_springs):
            physical_springs.append([chain_nodes[i], chain_nodes[i + 1]])
            sub_parent_map.append(parent_idx)

        # Micro sub-hinges (prevent string collapse)
        for i in range(1, num_sub_springs):
            hinges_list.append([chain_nodes[i - 1], chain_nodes[i], chain_nodes[i + 1]])
            hinge_dampings.append(SUB_HINGE_DAMPING)
            hinge_k_inplane.append(SUB_HINGE_STIFFNESS)
            hinge_k_oop.append(SUB_HINGE_STIFFNESS_OOP)

    # 4. Macro grid corner hinges (rigid-joint condition at main-node intersections)
    for r in range(rows - 1):
        for c in range(cols - 1):
            n_tl = node_indices[r, c]
            n_tr = node_indices[r, c + 1]
            n_bl = node_indices[r + 1, c]
            if (n_tl, n_tr) in first_sub_from and (n_tl, n_bl) in first_sub_from:
                s_right = first_sub_from[(n_tl, n_tr)]
                s_down  = first_sub_from[(n_tl, n_bl)]
                hinges_list.append([s_right, n_tl, s_down])
                hinge_dampings.append(HINGE_DAMPING)
                hinge_k_inplane.append(HINGE_STIFFNESS)
                hinge_k_oop.append(HINGE_STIFFNESS_OOP)

    # 5. Fixed constraints
    total_nodes = len(objects)
    fixed_mask = np.zeros(total_nodes, dtype=np.int32)
    fixed_indices = [
        node_indices[0, cols - 1],
        node_indices[rows - 1, 0],
        node_indices[rows - 1, cols - 1],
    ]
    for i in fixed_indices:
        fixed_mask[i] = 1

    # 6. Conserve total system mass across all nodes
    total_system_mass = main_nodes_count * NODE_MASS
    mass_per_node = total_system_mass / total_nodes
    node_masses = np.full(total_nodes, mass_per_node, dtype=np.float64)

    # Input weights: only node 0 is actuated
    input_weights = np.zeros(total_nodes)
    input_weights[0] = 1.0

    config = {
        'objects': np.array(objects),
        'springs': np.array(physical_springs),
        'hinges': np.array(hinges_list) if hinges_list else np.zeros((0, 3), dtype=np.int32),
        'sub_parent_map': np.array(sub_parent_map, dtype=np.int32),
        'hinge_dampings': np.array(hinge_dampings),
        'hinge_k_inplane': np.array(hinge_k_inplane),
        'hinge_k_oop': np.array(hinge_k_oop),
        'node_masses': node_masses,
        'masks': {
            'fixed': fixed_mask,
            'active': ~fixed_mask.astype(bool),
            'input_indices': [0],
        },
        'weights': {'input_weights': input_weights},
        'num_logical_springs': len(logical_springs),
        'main_nodes_count': main_nodes_count,
        'node_indices': node_indices,   # (rows, cols) grid → main node index
    }
    # Also return the logical_springs array so the AL solver can use it
    logical_springs_np = np.array(logical_springs, dtype=np.int32)
    return config, logical_springs_np

# ==========================================
# DifferentiableReservoir
# ==========================================
@ti.data_oriented
class DifferentiableReservoir:
    def __init__(self, config, dt, batch_steps, M_levels, num_sub_springs):
        self.dt = dt
        self.batch_steps = batch_steps
        self.num_mass = len(config['objects'])
        self.num_springs = len(config['springs'])
        self.num_hinges = len(config['hinges']) if len(config['hinges']) > 0 else 0
        self.num_logical_springs = config['num_logical_springs']
        self.main_nodes_count = config['main_nodes_count']
        self.num_sub_springs = num_sub_springs

        self.M = M_levels
        # z_star is indexed by LOGICAL spring (not physical)
        self.z_star = ti.field(dtype=ti.f64, shape=(self.num_logical_springs, self.M), needs_grad=True)
        # discrete_k is now globally optimizable; discrete_c is derived via C_DAMPING_SCALE
        self.discrete_k = ti.field(dtype=ti.f64, shape=self.M, needs_grad=True)

        self.x_rest = ti.Vector.field(3, dtype=ti.f64, shape=self.num_mass, needs_grad=True)
        self.spring_l0 = ti.field(dtype=ti.f64, shape=self.num_springs, needs_grad=True)

        self.x = ti.Vector.field(3, dtype=ti.f64, shape=(batch_steps, self.num_mass), needs_grad=True)
        self.v = ti.Vector.field(3, dtype=ti.f64, shape=(batch_steps, self.num_mass), needs_grad=True)
        self.force_buffer = ti.Vector.field(3, dtype=ti.f64, shape=self.num_mass, needs_grad=True)
        # Only store displacements for MAIN nodes to reduce memory
        self.node_disp = ti.field(dtype=ti.f64,
                                  shape=(batch_steps, self.main_nodes_count, 2), needs_grad=True)

        self.spring_a = ti.field(dtype=ti.i32, shape=self.num_springs)
        self.spring_b = ti.field(dtype=ti.i32, shape=self.num_springs)
        self.sub_parent_map = ti.field(dtype=ti.i32, shape=self.num_springs)

        # Always allocate hinge fields (shape ≥ 1) so compute_step compiles regardless of hinge count
        _h_shape = max(1, self.num_hinges)
        self.hinge_i = ti.field(dtype=ti.i32, shape=_h_shape)
        self.hinge_j = ti.field(dtype=ti.i32, shape=_h_shape)
        self.hinge_k = ti.field(dtype=ti.i32, shape=_h_shape)
        self.hinge_d_tor = ti.field(dtype=ti.f64, shape=_h_shape)
        self.hinge_k_inplane = ti.field(dtype=ti.f64, shape=_h_shape)
        self.hinge_k_oop = ti.field(dtype=ti.f64, shape=_h_shape)
        self.hinge_theta_0 = ti.field(dtype=ti.f64, shape=_h_shape)

        self.node_mass_field = ti.field(dtype=ti.f64, shape=self.num_mass)
        self.input_weights = ti.field(dtype=ti.f64, shape=self.num_mass)
        self.fixed_mask = ti.field(dtype=ti.i32, shape=self.num_mass)
        self.loss = ti.field(dtype=ti.f64, shape=(), needs_grad=True)

        self._initialize_from_config(config)

    def _initialize_from_config(self, config):
        objects = config['objects']
        springs = config['springs']
        hinges = config['hinges']
        masks = config['masks']
        weights = config['weights']['input_weights']

        # discrete_k is populated from PyTorch (globally optimizable) before every
        # forward pass via ReservoirOptimizationFunction, so no init value is needed here.

        for i in range(self.num_mass):
            self.x_rest[i] = [objects[i, 0], objects[i, 1], objects[i, 2]]
            self.fixed_mask[i] = 1 if masks['fixed'][i] else 0
            self.input_weights[i] = weights[i]
            self.node_mass_field[i] = config['node_masses'][i]

        for s in range(self.num_springs):
            self.spring_a[s] = int(springs[s, 0])
            self.spring_b[s] = int(springs[s, 1])
            self.sub_parent_map[s] = config['sub_parent_map'][s]

        for h in range(self.num_hinges):
            self.hinge_i[h] = int(hinges[h, 0])
            self.hinge_j[h] = int(hinges[h, 1])
            self.hinge_k[h] = int(hinges[h, 2])
            self.hinge_d_tor[h] = config['hinge_dampings'][h]
            self.hinge_k_inplane[h] = config['hinge_k_inplane'][h]
            self.hinge_k_oop[h] = config['hinge_k_oop'][h]

        if self.num_hinges > 0:
            self.init_rest_angles()

    @ti.kernel
    def init_rest_angles(self):
        """Capture geometric rest angle for each hinge at initialization."""
        for h in range(self.num_hinges):
            i, j, k = self.hinge_i[h], self.hinge_j[h], self.hinge_k[h]
            r_ji = self.x_rest[i] - self.x_rest[j]
            r_jk = self.x_rest[k] - self.x_rest[j]
            l_ji = ti.math.sqrt(r_ji.norm_sqr() + 1e-12)
            l_jk = ti.math.sqrt(r_jk.norm_sqr() + 1e-12)
            self.hinge_theta_0[h] = ti.math.acos(
                ti.math.clamp(r_ji.dot(r_jk) / (l_ji * l_jk), -1.0, 1.0))

    @ti.kernel
    def update_geometry_kernel(self):
        for s in range(self.num_springs):
            i, j = self.spring_a[s], self.spring_b[s]
            dist = ti.math.sqrt((self.x_rest[i] - self.x_rest[j]).norm_sqr() + 1e-12)
            self.spring_l0[s] = dist * (1.0 - PRETENSION)

    @ti.kernel
    def init_simulation_state(self):
        for i in range(self.num_mass):
            self.x[0, i] = self.x_rest[i]
            self.v[0, i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def compute_step(self, t: int, u_val: ti.f64, v_val: ti.f64):
        # --- Force initialization: gravity + global damping ---
        for i in range(self.num_mass):
            self.force_buffer[i] = ti.Vector([0.0, 0.0, GRAVITY_Z * self.node_mass_field[i]])
            self.force_buffer[i] += -GLOBAL_DAMPING * self.v[t, i]

        # --- Spring forces (sub-spring scaled for series impedance conservation) ---
        for s in range(self.num_springs):
            i, j = self.spring_a[s], self.spring_b[s]
            disp = self.x[t, i] - self.x[t, j]
            dist = ti.math.sqrt(disp.norm_sqr() + 1e-12)
            dir_vec = disp / dist
            extension = dist - self.spring_l0[s]
            dot_v = (self.v[t, i] - self.v[t, j]).dot(dir_vec)

            k_eff = 0.0
            p_idx = self.sub_parent_map[s]
            # Multiply by num_sub_springs to preserve global series impedance
            for m in ti.static(range(self.M)):
                k_eff += self.z_star[p_idx, m] * self.discrete_k[m] * float(self.num_sub_springs)
            # Damping derives from stiffness via a fixed linear ratio (C_DAMPING_SCALE)
            c_eff = k_eff * C_DAMPING_SCALE

            f_total = k_eff * extension + c_eff * dot_v
            force_vec = -f_total * dir_vec
            ti.atomic_add(self.force_buffer[i],  force_vec)
            ti.atomic_add(self.force_buffer[j], -force_vec)

        # --- Hinge forces: damping torque + elastic angular stiffness (in/out-of-plane) ---
        for h in range(self.num_hinges):
            i, j, k = self.hinge_i[h], self.hinge_j[h], self.hinge_k[h]
            r_ji = self.x[t, i] - self.x[t, j]
            r_jk = self.x[t, k] - self.x[t, j]
            l_ji_sq = r_ji.norm_sqr() + 1e-12
            l_jk_sq = r_jk.norm_sqr() + 1e-12
            l_ji = ti.math.sqrt(l_ji_sq)
            l_jk = ti.math.sqrt(l_jk_sq)

            v_rel_i = self.v[t, i] - self.v[t, j]
            v_rel_k = self.v[t, k] - self.v[t, j]
            omega_i = r_ji.cross(v_rel_i) / l_ji_sq
            omega_k = r_jk.cross(v_rel_k) / l_jk_sq

            # Velocity-based damping torque
            torque = -self.hinge_d_tor[h] * (omega_i - omega_k)

            # Angle-based elastic stiffness torque with in/out-of-plane decomposition
            theta = ti.math.acos(
                ti.math.clamp(r_ji.dot(r_jk) / (l_ji * l_jk), -1.0, 1.0))
            diff_theta = theta - self.hinge_theta_0[h]

            axis = r_ji.cross(r_jk)
            axis_norm = ti.math.sqrt(axis.norm_sqr() + 1e-12)
            n_vec = ti.Vector([0.0, 0.0, 1.0])   # fallback: pure in-plane
            if axis_norm > 1e-6:
                n_vec = axis / axis_norm

            # In-plane (about Z) component
            torque += self.hinge_k_inplane[h] * diff_theta * n_vec[2] * ti.Vector([0.0, 0.0, 1.0])
            # Out-of-plane (about XY) component
            torque += self.hinge_k_oop[h] * diff_theta * ti.Vector([n_vec[0], n_vec[1], 0.0])

            F_i =  torque.cross(r_ji) / l_ji_sq
            F_k = -torque.cross(r_jk) / l_jk_sq
            ti.atomic_add(self.force_buffer[i],  F_i)
            ti.atomic_add(self.force_buffer[k],  F_k)
            ti.atomic_add(self.force_buffer[j], -F_i - F_k)

        # --- Integration: actuated, fixed, or free nodes ---
        for i in range(self.num_mass):
            if self.input_weights[i] > 0.0:
                # Actuator: directly prescribed position & velocity
                self.x[t + 1, i] = self.x_rest[i] + ti.Vector([self.input_weights[i] * u_val, 0.0, 0.0])
                self.v[t + 1, i] = ti.Vector([self.input_weights[i] * v_val, 0.0, 0.0])
            elif self.fixed_mask[i] == 1:
                # Fixed corner: pinned to rest position
                self.x[t + 1, i] = self.x_rest[i]
                self.v[t + 1, i] = ti.Vector([0.0, 0.0, 0.0])
            else:
                # Free node: explicit Euler integration with per-node mass
                acc = self.force_buffer[i] / self.node_mass_field[i]
                self.v[t + 1, i] = self.v[t, i] + acc * self.dt
                self.x[t + 1, i] = self.x[t, i] + self.v[t + 1, i] * self.dt

    @ti.kernel
    def compute_features(self):
        """Record XY displacements of main nodes only (relative to actuator node 0)."""
        for t, i in ti.ndrange(self.batch_steps, self.main_nodes_count):
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
def generate_benchmark_target_matrix(u_input, tau_d, k, n, tau_step=1, return_exps=False):
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
        if np.any(exps[:, j]):
            Y_valid *= (U_sub[:, [j]] ** exps[:, j][None, :])

    Y_full = np.zeros((T, exps.shape[0]), dtype=float)
    Y_full[max_lag:, :] = Y_valid
    if return_exps:
        return Y_full, exps
    return Y_full


class ReservoirOptimizationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z_star_tensor, pos_tensor, k_levels_tensor, res, u_sim, v_sim, downsample):
        ctx.res = res
        ctx.u_sim = u_sim
        ctx.v_sim = v_sim
        ctx.downsample = downsample

        res.x_rest.from_torch(pos_tensor.detach().contiguous())
        res.z_star.from_torch(z_star_tensor.detach().contiguous())
        res.discrete_k.from_torch(k_levels_tensor.detach().contiguous())
        res.x_rest.grad.fill(0)
        res.z_star.grad.fill(0)
        res.discrete_k.grad.fill(0)

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

        grad_sim = torch.zeros(
            (res.batch_steps, res.main_nodes_count * 2),
            dtype=torch.float64, device=target_device)
        grad_sim[::downsample, :] = grad_outputs_clean
        grad_sim_reshaped = grad_sim.reshape(res.batch_steps, res.main_nodes_count, 2).contiguous()

        res.x_rest.grad.fill(0)
        res.z_star.grad.fill(0)
        res.discrete_k.grad.fill(0)
        res.node_disp.grad.from_torch(grad_sim_reshaped.detach())

        with ti.ad.Tape(loss=res.loss, validation=False):
            res.update_geometry_kernel()
            res.run_batch_forward(ctx.u_sim, ctx.v_sim)
            compute_dummy_loss(res, grad_sim_reshaped)

        grad_pos = res.x_rest.grad.to_torch(device=target_device)
        fixed_mask = torch.tensor(res.fixed_mask.to_numpy(), device=target_device)
        grad_pos = grad_pos * (1.0 - fixed_mask.unsqueeze(1))
        grad_z_star = res.z_star.grad.to_torch(device=target_device)
        grad_k_levels = res.discrete_k.grad.to_torch(device=target_device)
        return grad_z_star, grad_pos, grad_k_levels, None, None, None, None


@ti.kernel
def compute_dummy_loss(res: ti.template(), grad_tensor: ti.types.ndarray()):
    for t, i in ti.ndrange(res.batch_steps, res.main_nodes_count):
        res.loss[None] += res.node_disp[t, i, 0] * grad_tensor[t, i, 0]
        res.loss[None] += res.node_disp[t, i, 1] * grad_tensor[t, i, 1]

# ==========================================
# Main Optimization Pipeline
# ==========================================
if __name__ == "__main__":
    input(f'Running {TRIAL_NAME}. Proceed?')

    # 1. Setup folders & device
    topo_epoch_dir = EXPERIMENT_DIR / "topology_visualization"
    topo_curve_dir = EXPERIMENT_DIR / "material_distribution"
    heatmap_iter_dir = EXPERIMENT_DIR / "heatmap_iterations"
    os.makedirs(topo_epoch_dir, exist_ok=True)
    os.makedirs(topo_curve_dir, exist_ok=True)
    os.makedirs(heatmap_iter_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch utilizing device: {device}")

    PHYSICS_HZ = 12000 * 3  # dt = 8.33e-5s; smallest integer multiple of 30 above 10kHz
    TARGET_HZ = 30
    DOWNSAMPLE = PHYSICS_HZ // TARGET_HZ   # = 400, exact integer → exact 30Hz output

    # 2. Load real experimental input waveform
    T_DURATION = 25.0
    dt_sim = 1.0 / PHYSICS_HZ
    print(f"Loading real-life input waveform from: {REAL_INPUT_DATA}")
    u_sim, v_sim, _ = load_real_input(str(REAL_INPUT_DATA), T_DURATION, dt_sim, target_hz=30.0)
    u_30Hz = u_sim[::DOWNSAMPLE]
    print(f"Input loaded: {len(u_sim)} physics steps → {len(u_30Hz)} frames at 30 Hz")

    # 3. Build sub-spring topology.
    #    If TOPOLOGY_NO is set, delete the logical springs listed for that topology from
    #    the INITIAL topology (mirrors the k_mat=0 disconnection in
    #    plot_heatmap_sim2real_bending [frozen].py). The deleted springs are physically
    #    omitted, so the optimizer never sees them.
    if TOPOLOGY_NO is not None:
        init_removed_springs = topologies_dict[f"topo_{TOPOLOGY_NO}"]
        print(f"[Init topology topo_{TOPOLOGY_NO}] Removing {len(init_removed_springs)} "
              f"logical springs: {init_removed_springs}")
    else:
        init_removed_springs = None
    config, logical_springs_np = generate_subspring_config(
        ROWS, COLS, NUM_SUB_SPRINGS, removed_springs=init_removed_springs)
    total_nodes = len(config['objects'])
    num_logical = config['num_logical_springs']
    MAIN_NODES_COUNT = config['main_nodes_count']
    node_indices_flat = config['node_indices']   # (ROWS, COLS) array of main node IDs
    print(f"Sub-spring nodes: {total_nodes} total ({MAIN_NODES_COUNT} main + {total_nodes - MAIN_NODES_COUNT} sub)")
    print(f"Physical springs: {len(config['springs'])} | Logical springs: {num_logical} | Hinges: {len(config['hinges'])}")

    # 4. Material parameters: M_LEVELS discrete stiffness "membership" classes per
    #    logical spring. Levels listed in K_FIXED_LEVELS are pinned (e.g. level 0 is
    #    always void, k=0); all other levels are globally learnable stiffness values
    #    shared across every spring. Damping derives from stiffness via C_DAMPING_SCALE,
    #    so K_MAX=222.15 <-> C_MAX=0.80 (set C_MAX=0.32 above for the alternative ratio).
    M_LEVELS = M_LEVELS_CONFIG
    # M_LEVELS = 2   # void-or-solid: level 0 fixed at k=0, level 1 freely optimized

    # 5. Initialize learnable parameters.
    #    Deleted springs are already removed from the topology, so every remaining logical
    #    spring is a solid edge: start each at the solid level (level 1). Under
    #    softmax(beta*z) with beta=20, a +10 logit ⇒ solid probability ~1.0, i.e. full
    #    density (1 - z_star[:,0]) and full stiffness (k_eff = discrete_k[1] * num_sub_springs).
    z_init = torch.zeros((num_logical, M_LEVELS), dtype=torch.float64, device=device)
    #z_init[:, 1] += 10.0   # full solid initialization for the remaining edges
    z_param = nn.Parameter(z_init)

    pos_param = nn.Parameter(
        torch.tensor(config['objects'], dtype=torch.float64, device=device))

    # Globally learnable stiffness levels (raw values; softplus keeps them >= 0).
    # Levels not present in K_FIXED_LEVELS are optimized; init values mirror the
    # old preset linspace(0, K_MAX, M_LEVELS) so training starts from a similar point.
    k_preset = np.linspace(0.0, K_MAX, M_LEVELS)
    learnable_level_idxs = [m for m in range(M_LEVELS) if m not in K_FIXED_LEVELS]
    k_learnable_init = torch.tensor(
        [k_preset[m] for m in learnable_level_idxs], dtype=torch.float64, device=device)
    k_learnable_raw = nn.Parameter(k_learnable_init.clone())

    def build_k_levels():
        """Assemble the full (M_LEVELS,) stiffness array from fixed + learnable entries."""
        parts = []
        learn_i = 0
        for m in range(M_LEVELS):
            if m in K_FIXED_LEVELS:
                parts.append(torch.tensor([K_FIXED_LEVELS[m]], dtype=torch.float64, device=device))
            else:
                parts.append(F.softplus(k_learnable_raw[learn_i:learn_i + 1]))
                learn_i += 1
        return torch.cat(parts)

    optimizer = torch.optim.Adam([
        {'params': [z_param], 'lr': 0.005},
        {'params': [k_learnable_raw], 'lr': 20},
    ])

    # 6. Generate benchmark targets at 30 Hz (openprc-exact normalized-Legendre basis).
    #    Always build the FULL n=4 basis so the per-epoch heatmap is complete; the loss
    #    uses a mode-specific subset of these same targets, so the values feeding the loss
    #    are a strict subset of the capacities displayed on the heatmap.
    #    L[C] (Eq. 13, Dambre 2012): uniform mean MSE over the 30 degree-1 targets.
    #    NL[C] (Eq. 14, Dambre 2012): degree-weighted MSE over all ~46k targets.
    HEATMAP_N = 4
    HEATMAP_TAU_D = 29
    HEATMAP_N_LIST = list(range(1, HEATMAP_N + 1))      # degrees 1..4
    HEATMAP_TAU_LIST = list(range(HEATMAP_TAU_D + 1))   # delays 0..29

    y_target_raw, task_exps = build_legendre_targets(
        u_30Hz, tau_d=HEATMAP_TAU_D, k=1, n=HEATMAP_N)
    task_degrees_np = task_exps.sum(axis=1).astype(np.float64)

    # Per-target loss weights over the full target set (sum to 1).
    if LOSS_MODE == "linear_memory":
        w = np.where(task_degrees_np == 1, 1.0, 0.0)
    elif LOSS_MODE == "nonlinearity":
        w = task_degrees_np.copy()
    else:
        raise ValueError(f"Unknown LOSS_MODE: {LOSS_MODE!r}. Use 'linear_memory' or 'nonlinearity'.")
    task_loss_weights = torch.tensor(w / w.sum(), dtype=torch.float64, device=device)
    print(f"[{LOSS_MODE}] Full Legendre target matrix shape: {y_target_raw.shape}")

    # R2 (1 - SSE/SST) is invariant to this affine target normalization (the readout has a
    # bias term), so normalizing for loss conditioning leaves the displayed capacities exact.
    y_target_mean = y_target_raw.mean(axis=0)
    y_target_std  = y_target_raw.std(axis=0) + 1e-12
    y_target_norm = (y_target_raw - y_target_mean) / y_target_std
    y_target_torch = torch.tensor(y_target_norm, dtype=torch.float64, device=device)

    washout_frames = 5  * TARGET_HZ
    train_frames   = 10 * TARGET_HZ
    test_frames    = 10 * TARGET_HZ
    train_stop = washout_frames + train_frames
    test_stop  = train_stop + test_frames

    # (train/test target slices are taken inside evaluate(); no need to materialize them)
    print(f"Data splits → Washout: {washout_frames}, Train: {train_frames}, Test: {test_frames}")

    # 7. Create differentiable reservoir
    res_opt = DifferentiableReservoir(
        config, dt_sim, len(u_sim), M_LEVELS, NUM_SUB_SPRINGS)

    # Active readout mask: all main nodes except actuator (node 0)
    active_mask_2d = torch.ones(MAIN_NODES_COUNT * 2, dtype=torch.bool, device=device)
    active_mask_2d[0:2] = False   # exclude actuator XY

    # 8. AL solver operates on LOGICAL springs (not physical sub-springs)
    logical_config = {'springs': logical_springs_np}
    al_solver = CombinedTopologyAL(logical_config, ROWS, COLS, M_LEVELS, device)

    # Force all springs adjacent to any corner node to stay active.
    # Corners are the only nodes whose connectivity cannot be optimised away —
    # they must always have at least one solid spring to satisfy the AL constraint.
    corner_indices = [
        node_indices_flat[0, 0],          # top-left  (actuator)
        node_indices_flat[0, COLS - 1],   # top-right (fixed)
        node_indices_flat[ROWS - 1, 0],   # bottom-left (fixed)
        node_indices_flat[ROWS - 1, COLS - 1],  # bottom-right (fixed)
    ]
    corner_spring_indices = []
    for idx, s in enumerate(logical_springs_np):
        if int(s[0]) in corner_indices or int(s[1]) in corner_indices:
            corner_spring_indices.append(idx)
    corner_spring_indices = torch.tensor(corner_spring_indices, device=device)

    # 9. Optimization loop
    EPOCHS = 100
    loss_history, ipc_history, max_mse_history = [], [], []

    # Stream the loss log to disk every epoch (rather than only at the end) so
    # progress survives an interrupted run, and record the global stiffness
    # levels alongside each epoch since they now evolve during optimization.
    csv_path = EXPERIMENT_DIR / "loss_log.csv"
    csv_file = open(csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["Epoch", f"{LOSS_MODE}_Loss", "IPC_Capacity", "Max_MSE"]
        + [f"K_level_{m}" for m in range(M_LEVELS)])
    csv_file.flush()

    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        # Hard-force all corner-adjacent springs to active material
        beta = 20.0
        modified_z = z_param.clone()
        modified_z[corner_spring_indices, 0] -= 10.0
        z_star_tensor = F.softmax(beta * modified_z, dim=-1)

        # Assemble globally optimizable stiffness levels (fixed + learnable)
        k_levels_tensor = build_k_levels()

        # Physics forward pass (30 Hz output, main nodes only)
        X_30Hz = ReservoirOptimizationFunction.apply(
            z_star_tensor, pos_param, k_levels_tensor, res_opt, u_sim, v_sim, DOWNSAMPLE)

        # Feature preparation: openprc-exact NodeDisplacements (reference-subtract the
        # actuator node 0, drop it, StandardScaler) + leading bias column. [build_features]
        X_full = build_features(X_30Hz, MAIN_NODES_COUNT)

        # Dynamic effective rank & Dambre epsilon (on features, excluding the bias column)
        current_rank    = compute_effective_rank_torch(X_full[:, 1:])
        dynamic_dambre_eps = calculate_dambre_eps(effective_rank=current_rank, test_duration=test_frames)

        # openprc-exact ridge readout + per-target R2 (capacity) + heatmap assembly.
        # mse_per_task is differentiable (drives the loss); capacities/heatmap are detached.
        mse_per_task, capacities, heatmap = evaluate(
            X_full, y_target_torch, task_exps,
            washout=washout_frames, train_stop=train_stop, test_stop=test_stop,
            n_list=HEATMAP_N_LIST, tau_d_list=HEATMAP_TAU_LIST, k_delay=1,
            ridge=1e-6, eps=dynamic_dambre_eps)
        total_capacity = float(np.nansum(capacities))

        loss = torch.sum(task_loss_weights * mse_per_task)
        loss_reg, max_viol, viols = al_solver.compute_loss(z_star_tensor)
        total_loss   = loss + loss_reg
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_([z_param, k_learnable_raw], max_norm=1.0)
        optimizer.step()
        al_solver.step_params(viols, verbose=(epoch % 10 == 0))
        with torch.no_grad():
            z_param.clamp_(-10.0, 10.0)
            # Floor every learnable (non-zero) stiffness level at K_MIN_LEVEL. Since
            # k = softplus(k_learnable_raw), clamp the raw param at softplus^{-1}(K_MIN_LEVEL)
            # = K_MIN_LEVEL + log(1 - exp(-K_MIN_LEVEL)) (numerically stable form).
            raw_floor = float(K_MIN_LEVEL + np.log(-np.expm1(-K_MIN_LEVEL)))
            k_learnable_raw.clamp_(min=raw_floor)
            z_star_np = z_star_tensor.cpu().numpy()
            k_levels_np = k_levels_tensor.detach().cpu().numpy()
            k_eff = np.sum(z_star_np * k_levels_np, axis=1)
            c_eff = k_eff * C_DAMPING_SCALE

        print(f"Epoch {epoch+1:03d} | {LOSS_MODE} loss: {loss.item():.5f} | IPC: {total_capacity:.2f} | "
              f"Max MSE: {mse_per_task.max().item():.5f} | K levels: {np.round(k_levels_np, 2)}")
        loss_history.append(loss.item())
        ipc_history.append(total_capacity)
        max_mse_history.append(mse_per_task.max().item())

        csv_writer.writerow(
            [epoch + 1, loss.item(), total_capacity, mse_per_task.max().item()]
            + k_levels_np.tolist())
        csv_file.flush()

        current_pos = pos_param.detach().cpu().numpy()
        with torch.no_grad():
            z_star_np = z_star_tensor.cpu().numpy()
        plot_material_distribution(z_star_np, k_levels_np, epoch, topo_curve_dir)

        # Topology plot uses main nodes and logical spring list for visual clarity
        main_objects = config['objects'][:MAIN_NODES_COUNT]
        plot_epoch_topology(
            main_objects, logical_springs_np,
            k_eff / max(k_levels_np.max(), 1e-6), config['masks'], epoch, topo_epoch_dir)

        # openprc-identical R2 memory-capacity heatmap from this epoch's own waveforms
        plot_iteration_heatmap(
            heatmap, HEATMAP_N_LIST, HEATMAP_TAU_LIST, k_delay=1,
            amp=1, n_mass=ROWS * COLS,
            title_prefix=f"{TRIAL_NAME} [{LOSS_MODE}] epoch {epoch + 1}",
            epoch=epoch, save_dir=heatmap_iter_dir)

    # 10. Loss log was streamed to disk every epoch during the loop above
    csv_file.close()

    # 11. Save loss curve
    plt.figure()
    plt.plot(loss_history[1:])
    plt.title("Optimization Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Summed MSE Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig(EXPERIMENT_DIR / "loss_curve.png", dpi=150)
    plt.close()

    # 12. Reconstruct optimized k/c matrices (logical spring indexing)
    k_mat_opt = np.zeros((ROWS * COLS, ROWS * COLS))
    c_mat_opt = np.zeros((ROWS * COLS, ROWS * COLS))
    for s in range(num_logical):
        i, j = int(logical_springs_np[s, 0]), int(logical_springs_np[s, 1])
        if k_eff[s] > 10.0:  # clamp, cut at 10. Less than 10 is void. 
            k_mat_opt[i, j] = k_mat_opt[j, i] = float(k_eff[s])
            c_mat_opt[i, j] = c_mat_opt[j, i] = float(c_eff[s])

    with open(EXPERIMENT_DIR / "experiment.json", "w") as f:
        json.dump({"k_mat_opt": k_mat_opt.tolist(), "c_mat_opt": c_mat_opt.tolist()}, f, indent=4)

    # 13. Post-optimization heatmap benchmark via plate_v3 pipeline
    print("\n[Step 4] Running post-optimization heatmap benchmark...")
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
        from plot_heatmap_sim2real_bending import run_heatmap_pipeline_for_topology, plot_heatmap

        heatmap_result, exp_path, n_list, tau_list, k_val = run_heatmap_pipeline_for_topology(
            ROWS, COLS, k_mat_opt, c_mat_opt, run_suffix=f"{TRIAL_NAME}_opt"
        )
        if heatmap_result is not None:
            plot_heatmap(
                heatmap_result, n_list, tau_list, k_delay=k_val,
                amp=1, n_mass=ROWS * COLS,
                title_prefix=f"Optimized [{TRIAL_NAME}] (sub-spring physics)",
                save_dir=EXPERIMENT_DIR,
                save_name="heatmap_optimized",
                show=False, save_png=True, save_svg=True
            )
            print(f"[Saved] Heatmap → {EXPERIMENT_DIR / 'heatmap_optimized.png'}")
    except Exception as e:
        print(f"[Warning] Heatmap benchmark skipped: {e}")

    print(f"\n[Success] Optimized with sub-spring physics and real input waveform.")
    print(f"Results in: {EXPERIMENT_DIR}")
