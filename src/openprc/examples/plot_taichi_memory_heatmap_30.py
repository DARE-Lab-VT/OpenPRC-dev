import sys
import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product

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
    Unified Pipeline to visualize Before vs. After based on TRIAL_NAME.
    """
    # --- Configuration (Must match 1_grid_opt.py) ---
    TRIAL_NAME = "Taichi_IID_Memory_Opt_30Hz_multi_fast"
    ROWS, COLS = 4, 4
    
    EXPERIMENT_DIR = src_dir / "experiments" / TRIAL_NAME
    GA_RESULTS_PATH = EXPERIMENT_DIR / "experiment.json"
    
    # --- Load Optimized Data ---
    if not GA_RESULTS_PATH.exists():
        print(f"[Error] Results file not found: {GA_RESULTS_PATH}")
        print("Please run 1_grid_opt.py first to generate the optimized topology.")
        return

    print(f"-> Loading optimization results from: {GA_RESULTS_PATH}")
    with open(GA_RESULTS_PATH, 'r') as f:
        results_data = json.load(f)
    
    # Initialize Helper (Used for "Before" generation)
    fourier = FourierSeries2D(ROWS, COLS)

    # --- 1. "BEFORE" ANALYSIS (Uniform Grid) ---
    print("\n" + "="*50)
    print("STEP 1: Analyzing Pre-Optimization (Uniform) Topology")
    print("="*50)
    # Generate original uniform grid matrices
    c_mat_orig, k_mat_orig = fourier.build_full_neighbor_topology(ROWS, COLS, rigid_outer_frame=False)
    
    TARGET_STIFFNESS = 222.15  # Must match the physics from your Taichi script
    TARGET_DAMPING = 0.8

    # Overwrite the default 222.15 values wherever a spring exists
    k_mat_orig = np.where(k_mat_orig > 0, TARGET_STIFFNESS, 0.0)
    c_mat_orig = np.where(c_mat_orig > 0, TARGET_DAMPING, 0.0)
    
    heatmap_before, _, n_list, tau_list, k_val = run_heatmap_pipeline_for_topology(
        ROWS, COLS, k_mat_orig, c_mat_orig * 0.4, "uniform_grid"
    )
    
    if heatmap_before is not None:
        plot_heatmap(
            heatmap_before, n_list, tau_list, k_delay=k_val, amp=1, n_mass=ROWS*COLS,
            title_prefix="Memory Heatmap (Uniform Grid)",
            save_dir=EXPERIMENT_DIR, 
            save_name="heatmap_before_optimization", 
            show=False, save_png=True, save_svg=True
        )

    # --- 2. "AFTER" ANALYSIS (Direct Matrix Optimization) ---
    print("\n" + "="*50)
    print("STEP 2: Analyzing Post-Optimization Topology")
    print("="*50)
    
    # Check if we have direct matrix results (from Taichi)
    if "k_mat_opt" in results_data:
        k_mat_opt = np.array(results_data["k_mat_opt"])
        c_mat_opt = np.array(results_data["c_mat_opt"])
        
        heatmap_after, after_exp_path, n_list, tau_list, k_val = run_heatmap_pipeline_for_topology(
            ROWS, COLS, k_mat_opt, c_mat_opt, "optimized_topology"
        )
        
        if heatmap_after is not None:
            plot_heatmap(
                heatmap_after, n_list, tau_list, k_delay=k_val, amp=1, n_mass=ROWS*COLS,
                title_prefix="Memory Heatmap (After Taichi Optimization)",
                save_dir=EXPERIMENT_DIR, 
                save_name="heatmap_after_optimization", 
                show=False, save_png=True, save_svg=True
            )
            
            # Launch interactive visualizer for the final optimized run
            print(f"\n[INFO] Launching visualizer player for optimized run...")
            ShowSimulation(str(after_exp_path))
    else:
        print("[Error] No optimized matrices ('k_mat_opt') found in the JSON file.")

    print(f"\n[Done] All heatmaps saved to: {EXPERIMENT_DIR}")

if __name__ == "__main__":
    main()