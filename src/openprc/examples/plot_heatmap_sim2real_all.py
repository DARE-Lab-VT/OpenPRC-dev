import shutil
import sys
import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
original_rmtree = shutil.rmtree

def safe_rmtree(path, ignore_errors=False, onerror=None):
    try:
        original_rmtree(path, ignore_errors=ignore_errors, onerror=onerror)
    except PermissionError as e:
        if "WinError 32" in str(e) or "used by another process" in str(e):
            print(f"[Warning] Ignored Windows file lock on: {path}")
        else:
            raise

shutil.rmtree = safe_rmtree
# --- Path Setup ---
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

# --- Core Library Imports ---
from openprc.analysis.benchmarks.memory_benchmark import MemoryBenchmark
from openprc.reservoir.io.state_loader import StateLoader
from openprc.reservoir.features.node_features import NodeDisplacements # [MODIFIED]
from openprc.reservoir.training.trainer import Trainer
from openprc.reservoir.readout.ridge import Ridge
from openprc.demlat.utils.animator import ShowSimulation
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2 # [NEW]

# --- Optimization Imports ---
from openprc.optimization.search_spaces.fourier_series_2D import FourierSeries2D
from openprc.examples.spring_mass_2D import run_pipeline

def calculate_dambre_epsilon(effective_rank: int, test_duration: int, p_value: float = 1e-4) -> float:
    """[NEW] Matches the calculation in run_memory_benchmark_pipeline.py"""
    t = chi2.isf(p_value, df=effective_rank)
    return (2.0 * t) / test_duration

# --- [ADD THESE TWO NEW FUNCTIONS] ---
def compute_effective_rank(loader, features) -> float:
    """Calculates the true number of independent state variables using SVD entropy."""
    state_matrix = features.transform(loader)
    if state_matrix.shape[0] < 2: return 1.0

    state_matrix = StandardScaler().fit_transform(state_matrix)
    _, s, _      = np.linalg.svd(state_matrix, full_matrices=False)
    s_norm       = s / np.sum(s)
    return float(np.exp(-np.sum(s_norm * np.log(s_norm + 1e-12))))

def compute_test_frames(loader, test_duration_s: float = 10.0) -> int:
    """Extracts the exact number of test frames based on the simulation save_interval."""
    import h5py
    with h5py.File(loader.sim_path, 'r') as f:
        fps = float(f.attrs.get('fps', 29.97))
    return max(1, int(test_duration_s * fps))


def plot_heatmap(
    heatmap, n_list, tau_d_list, k_delay, amp, n_mass, title_prefix,
    vmin=None, vmax=None,
    save_dir=None,
    save_name=None,
    save_svg=True,
    save_png=False,
    dpi=300,
    show=True
):
    """Plots and saves the memory capacity heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    heatmap = heatmap.T

    if heatmap is not None and n_list is not None and tau_d_list is not None:
        full_title = (f"{title_prefix}\n" + rf"$R^2$ (upper)", rf"num_mass={n_mass}" + "\n" +
                      rf"k={k_delay}, A={amp}")

        im = ax.imshow(
            heatmap, aspect='auto', origin='lower',
            cmap='RdYlBu_r', vmin=vmin or 0.0, vmax=vmax or 1.0
        )

        n_rows, n_cols = heatmap.shape
        for y in range(n_rows):
            for x in range(n_cols):
                r2_val = heatmap[y, x]
                if not np.isnan(r2_val):
                    ax.text(x, y, f'{r2_val:.2f}',
                            ha='center', va='center', color='black', fontsize=8)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('$R^2$ Mean')
        ax.set_xlabel(r'$n$ (monomial degree)')
        ax.set_ylabel(r'$\tau$ (time delay)')
        ax.set_title(full_title, fontsize=8)

        ax.set_xticks(np.arange(len(n_list)))
        ax.set_yticks(np.arange(len(tau_d_list)))
        ax.set_xticklabels(n_list, fontsize=6)
        ax.set_yticklabels((np.array(tau_d_list) * k_delay), fontsize=6)

    fig.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        if save_svg:
            svg_path = save_dir / f"{save_name}.svg"
            fig.savefig(svg_path, format="svg", bbox_inches="tight")
            print(f"[Saved] Heatmap SVG -> {svg_path}", flush=True)
        if save_png:
            png_path = save_dir / f"{save_name}.png"
            fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
            print(f"[Saved] Heatmap PNG -> {png_path}", flush=True)

    if show:
        plt.show()
    else:
        plt.close(fig)


def run_heatmap_pipeline_for_topology(rows, cols, k_mat, c_mat, run_suffix):
    print(f"\n--- Running Full Pipeline for Topology: {run_suffix} ---")
    
    print(f"-> Running simulation...")
    _, experiment_path = run_pipeline(
        rows=rows, cols=cols, k_mat=k_mat, c_mat=c_mat, 
        ga_generation=run_suffix, amplitude=0.02, target_hz=30.0
    )
    h5_path = experiment_path / "output" / "simulation.h5"
    if not h5_path.exists():
        print(f"[Error] Simulation did not produce h5 file at: {h5_path}")
        return None, None, None, None, None
    print(f"-> Simulation complete. State data at {h5_path}")

    loader = StateLoader(h5_path)
    
    # [CRITICAL FIX 1]: Use BOTH X and Y displacements!
    features = NodeDisplacements(reference_node=0, dims=[0, 1]) 
    u_input = loader.get_actuation_signal(actuator_idx=0, dof=0)
    
    # [CRITICAL FIX 2]: Dynamically calculate exact Effective Rank (N) and Frames (T)
    N = compute_effective_rank(loader, features)
    T = compute_test_frames(loader, test_duration_s=10.0)
    dambre_eps = calculate_dambre_epsilon(effective_rank=N, test_duration=T)
    print(f"  Effective rank (N): {N:.4f}   Test frames (T): {T}   Epsilon: {dambre_eps:.6f}")

    # [CRITICAL FIX 3]: Match the run_plot_heatmap tau and k_delay exactly
    k_delay_val = 1 
    n_list = list(range(1, 5))     # Degrees 1 to 4
    tau_d_list = list(range(30))   # Delays 0 to 29
    
    heatmap = np.empty((len(n_list), len(tau_d_list)), dtype=float)

    print(f"-> Running benchmark sweep...")
    idx_pairs = list(product(range(len(n_list)), range(len(tau_d_list))))
    for (i, j) in tqdm(idx_pairs, total=len(idx_pairs), desc=f"Benchmarking {run_suffix}"):
        n_s, tau_s = n_list[i], tau_d_list[j]
        
        benchmark = MemoryBenchmark(group_name=f"mem_bench_n{n_s}_tau{tau_s}")
        benchmark_args = {"tau_s": tau_s, "n_s": n_s, "k_delay": k_delay_val, "ridge": 1e-6, "eps": dambre_eps}

        trainer = Trainer(
            loader=loader, features=features, readout=Ridge(benchmark_args["ridge"]),
            experiment_dir=experiment_path, washout=5.0, train_duration=10.0, test_duration=10.0
        )
        
        score = benchmark.run(trainer, u_input, **benchmark_args)
        score.save()

        capacities = score.metrics.get('capacities')
        heatmap[i, j] = np.nanmean(capacities) if capacities is not None and len(capacities) > 0 else np.nan

    print("--- Pipeline complete. ---")
    
    # [CRITICAL FIX 4]: Return the shape parameters so main() knows how to plot them!
    return heatmap, experiment_path, n_list, tau_d_list, k_delay_val


def main():
    """
    Standalone Pipeline to evaluate specific topologies for Sim2Real calibration.
    """
    # --- Configuration ---
    TRIAL_DIR = "sim2real_all"
    ROWS, COLS = 4, 4
    REMARK = "hinge_damping_0.5_amplitude_1.5"
    
    # --- Control Flags ---
    SWEEP_ALL = False  # Set to True to loop all, False to run only TOPOLOGY_NO
    TOPOLOGY_NO = 4

    TARGET_STIFFNESS = 222.15  
    TARGET_DAMPING = 0.8
    
    fourier = FourierSeries2D(ROWS, COLS)

    # ==========================================
    # --- Topology Dictionary ---
    # ==========================================
    topologies_dict = {
        "topo_1": [], # Baseline: Fully connected
        "topo_2": [(5, 6), (6, 7), (8, 9)], # Example: Removed springs between 0-1 and 2-3
        "topo_3": [(2, 6), (4, 5), (8, 9), (10, 11)], # Example: Removed three internal springs
        "topo_4": [(1, 5), (2, 6), (9, 13), (10, 14)],
        "topo_5": [(1, 5), (2, 6), (5, 9), (6, 10), (9, 13), (10, 14)]
    }

    # --- Filtering Logic ---
    if not SWEEP_ALL:
        target_key = f"topo_{TOPOLOGY_NO}"
        if target_key in topologies_dict:
            # Overwrite the dictionary with just the single target entry
            topologies_dict = {target_key: topologies_dict[target_key]}
        else:
            print(f"[Error] {target_key} not found in the dictionary.")
            return

    print("\n" + "="*50)
    print(f"Evaluating {len(topologies_dict)} Topologies for Sim2Real")
    print("="*50)

    # The loop handles both a single item or the full sweep seamlessly
    for topo_name, removed_springs in topologies_dict.items():
        print(f"\n>>> Starting evaluation for: {topo_name}")
        
        # Create a dedicated directory for THIS specific topology
        EXPERIMENT_DIR = src_dir / "experiments" / TRIAL_DIR / topo_name / REMARK
        os.makedirs(EXPERIMENT_DIR, exist_ok=True) 
        
        # 1. Generate the fully connected baseline matrices
        c_mat, k_mat = fourier.build_full_neighbor_topology(ROWS, COLS, rigid_outer_frame=False)
        
        # 2. Overwrite the default 1.0s with the physical target values
        k_mat = np.where(k_mat > 0, TARGET_STIFFNESS, 0.0)
        c_mat = np.where(c_mat > 0, TARGET_DAMPING, 0.0)
        
        # 3. Disconnect the removed springs for this specific topology
        for (node_i, node_j) in removed_springs:
            k_mat[node_i, node_j] = 0.0
            k_mat[node_j, node_i] = 0.0
            c_mat[node_i, node_j] = 0.0
            c_mat[node_j, node_i] = 0.0
            
        print(f"Removed {len(removed_springs)} springs. Proceeding to pipeline...")
        
        # 4. Run the benchmark pipeline
        heatmap_result, exp_path, n_list, tau_list, k_val = run_heatmap_pipeline_for_topology(
            ROWS, COLS, k_mat, c_mat, f"{topo_name}_sim"
        )
        
        # 5. Save results specific to this topology
        if heatmap_result is not None:
            plot_heatmap(
                heatmap_result, n_list, tau_list, k_delay=k_val, amp=1, n_mass=ROWS*COLS,
                title_prefix=f"Sim2Real [{topo_name}] (k={TARGET_STIFFNESS}, c={TARGET_DAMPING})",
                save_dir=EXPERIMENT_DIR, 
                save_name=f"heatmap_{topo_name}", 
                show=False, save_png=True, save_svg=True
            )
            
            print(f"[INFO] Launching GUI visualizer for {topo_name}...")
            ShowSimulation(str(exp_path))

    print(f"\n[Done] All evaluations finished. Base Directory: {src_dir / 'experiments' / TRIAL_DIR}")

if __name__ == "__main__":
    main()