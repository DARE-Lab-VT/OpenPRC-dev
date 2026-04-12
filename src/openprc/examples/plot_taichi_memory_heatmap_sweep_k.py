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
from scipy.stats import chi2 # [NEW]

# --- Optimization Imports ---
from openprc.optimization.search_spaces.fourier_series_2D import FourierSeries2D
from openprc.examples.spring_mass_2D import run_pipeline

def calculate_dambre_epsilon(effective_rank: int, test_duration: int, p_value: float = 1e-4) -> float:
    """[NEW] Matches the calculation in run_memory_benchmark_pipeline.py"""
    t = chi2.isf(p_value, df=effective_rank)
    return (2.0 * t) / test_duration

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
    """Runs simulation and benchmark sweep for a given topology."""
    print(f"\n--- Running Full Pipeline for Topology: {run_suffix} ---")
    
    # 1. Run simulation to get state data
    print(f"-> Running simulation...")
    _, experiment_path = run_pipeline(
        rows=rows, cols=cols, k_mat=k_mat, c_mat=c_mat, ga_generation=run_suffix
    )
    h5_path = experiment_path / "output" / "simulation.h5"
    if not h5_path.exists():
        print(f"[Error] Simulation did not produce h5 file at: {h5_path}")
        return None, None
    print(f"-> Simulation complete. State data at {h5_path}")

    # 2. Setup benchmark sweep
    loader = StateLoader(h5_path)
    
    # [MODIFIED]: Relative Displacements (Reference = Node 0, X-axis only)
    features = NodeDisplacements(reference_node=0, dims=[0]) 
    u_input = loader.get_actuation_signal(actuator_idx=0, dof=0)
    
    # [MODIFIED]: Match the optimizer's lag scale and Dambre threshold
    k_delay_val = 10 
    n_list = list(range(1, 5)) # IID memory degrades fast; N=1..4 is a realistic heatmap range
    tau_d_list = list(range(0, 50, 5)) # Sample the 500-step lag horizon
    
    # 10.0 seconds of test_duration at dt=0.01 is 1000 steps
    dambre_eps = calculate_dambre_epsilon(effective_rank=1.5, test_duration=1000)
    
    heatmap = np.empty((len(n_list), len(tau_d_list)), dtype=float)

    print(f"-> Running benchmark sweep...")
    idx_pairs = list(product(range(len(n_list)), range(len(tau_d_list))))
    for (i, j) in tqdm(idx_pairs, total=len(idx_pairs), desc=f"Benchmarking {run_suffix}"):
        n_s, tau_s = n_list[i], tau_d_list[j]
        
        benchmark = MemoryBenchmark(group_name=f"mem_bench_n{n_s}_tau{tau_s}")
        
        # [MODIFIED]: Use dambre_eps instead of 1e-9
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
    return heatmap, experiment_path


def main():
    """
    Unified Pipeline to visualize Before vs. After across a Stiffness Sweep.
    """
    TRIAL_NAME = "Taichi_Stiffness_Sweep_Damping"
    ROWS, COLS = 4, 4
    
    MAIN_DIR = src_dir / "experiments" / TRIAL_NAME
    if not MAIN_DIR.exists():
        print(f"[Error] Main sweep directory not found: {MAIN_DIR}")
        print("Please run the sweep optimization script first.")
        return
        
    fourier = FourierSeries2D(ROWS, COLS)
    
    # 1. Find all stiffness subfolders
    sweep_folders = [d for d in MAIN_DIR.iterdir() if d.is_dir() and d.name.startswith("stiffness_")]
    
    # 2. Iterate through each subfolder
    for exp_dir in sorted(sweep_folders, key=lambda x: float(x.name.split('_')[1])):
        print(f"\n{'#'*60}")
        print(f"Processing Sweep Directory: {exp_dir.name}")
        print(f"{'#'*60}")
        
        # Extract target stiffness dynamically from folder name
        target_stiffness = float(exp_dir.name.split('_')[1])
        
        GA_RESULTS_PATH = exp_dir / "ga_results.json"
        if not GA_RESULTS_PATH.exists():
            print(f"[Warning] Skipping {exp_dir.name}: No ga_results.json found.")
            continue

        print(f"-> Loading optimization results from: {GA_RESULTS_PATH}")
        with open(GA_RESULTS_PATH, 'r') as f:
            results_data = json.load(f)
            
        n_plot_list = list(range(1, 5))
        tau_plot_list = list(range(0, 50, 5))

        # --- STEP 1: "BEFORE" ANALYSIS ---
        print("\nSTEP 1: Analyzing Pre-Optimization (Uniform) Topology")
        c_mat_orig, k_mat_orig = fourier.build_full_neighbor_topology(ROWS, COLS, rigid_outer_frame=False)
        
        TARGET_DAMPING = 0.8
        # Ensure the baseline matrix uses the current sweep's stiffness!
        k_mat_orig = np.where(k_mat_orig > 0, target_stiffness, 0.0)
        c_mat_orig = np.where(c_mat_orig > 0, TARGET_DAMPING, 0.0)
        
        heatmap_before, _ = run_heatmap_pipeline_for_topology(
            ROWS, COLS, k_mat_orig, c_mat_orig * 0.4, f"{exp_dir.name}_uniform"
        )
        
        if heatmap_before is not None:
            plot_heatmap(
                heatmap_before, n_plot_list, tau_plot_list, k_delay=10, amp=1, n_mass=ROWS*COLS,
                title_prefix=f"Memory Heatmap (Uniform Grid, k={target_stiffness})",
                save_dir=exp_dir, # Save directly into the specific sweep subfolder
                save_name="heatmap_before_optimization", 
                show=False, save_png=True, save_svg=True
            )

        # --- STEP 2: "AFTER" ANALYSIS ---
        print("\nSTEP 2: Analyzing Post-Optimization Topology")
        if "k_mat_opt" in results_data:
            k_mat_opt = np.array(results_data["k_mat_opt"])
            c_mat_opt = np.array(results_data["c_mat_opt"])
            
            heatmap_after, after_exp_path = run_heatmap_pipeline_for_topology(
                ROWS, COLS, k_mat_opt, c_mat_opt, f"{exp_dir.name}_optimized"
            )
            
            if heatmap_after is not None:
                plot_heatmap(
                    heatmap_after, n_plot_list, tau_plot_list, k_delay=10, amp=1, n_mass=ROWS*COLS,
                    title_prefix=f"Memory Heatmap (Optimized, k={target_stiffness})",
                    save_dir=exp_dir, # Save directly into the specific sweep subfolder
                    save_name="heatmap_after_optimization", 
                    show=False, save_png=True, save_svg=True
                )
                
                # Disabled for automated sweeping so it doesn't block the loop
                # print(f"\n[INFO] Launching visualizer player for optimized run...")
                # ShowSimulation(str(after_exp_path)) 
        else:
            print(f"[Error] No optimized matrices ('k_mat_opt') found for {exp_dir.name}.")

    print(f"\n[Done] All sweep heatmaps generated and saved to subfolders in: {MAIN_DIR}")

if __name__ == "__main__":
    main()