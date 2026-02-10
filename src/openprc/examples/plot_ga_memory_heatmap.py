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
from openprc.reservoir.features.node_features import NodePositions
from openprc.reservoir.training.trainer import Trainer
from openprc.reservoir.readout.ridge import Ridge
from openprc.demlat.utils.viz_player import visualize_experiment

# --- GA-specific Imports ---
from openprc.optimization.search_spaces.fourier_series_2D import FourierSeries2D
from openprc.examples.spring_mass_2D import run_pipeline


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

def plot_ga_convergence(history, save_dir):
    """Plots the best and mean fitness from GA history."""
    print("-> Plotting GA convergence...")
    best_fitness = history.get('best', [])
    mean_fitness = history.get('mean', [])
    
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness, label="Best Fitness")
    plt.plot(mean_fitness, label="Mean Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.title("GA Fitness Convergence")
    plt.grid(True)
    
    if save_dir:
        path = save_dir / "ga_convergence_plot.png"
        plt.savefig(path, dpi=300)
        print(f"[Saved] GA convergence plot -> {path}")
    plt.show()

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
    features = NodePositions()
    u_input = loader.get_actuation_signal(actuator_idx=0, dof=0)
    
    k_delay_val = 30
    n_list = list(range(1, 9))
    tau_d_list = list(range(6))
    heatmap = np.empty((len(n_list), len(tau_d_list)), dtype=float)

    print(f"-> Running benchmark sweep...")
    idx_pairs = list(product(range(len(n_list)), range(len(tau_d_list))))
    for (i, j) in tqdm(idx_pairs, total=len(idx_pairs), desc=f"Benchmarking {run_suffix}"):
        n_s, tau_s = n_list[i], tau_d_list[j]
        
        benchmark = MemoryBenchmark(group_name=f"mem_bench_n{n_s}_tau{tau_s}")
        benchmark_args = {"tau_s": tau_s, "n_s": n_s, "k_delay": k_delay_val, "ridge": 1e-6}

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
    Main pipeline for 'before vs. after' GA analysis.
    """
    # --- Shared Configuration ---
    ROWS, COLS = 4, 4
    GA_EXPERIMENT_DIR = src_dir / "experiments" / f"spring_mass_{ROWS}x{COLS}_test"
    GA_RESULTS_PATH = GA_EXPERIMENT_DIR / "ga_results.json"
    
    # --- Load GA Data ---
    if not GA_RESULTS_PATH.exists():
        print(f"[Error] GA results file not found: {GA_RESULTS_PATH}"); return

    print(f"-> Loading GA results from: {GA_RESULTS_PATH}")
    with open(GA_RESULTS_PATH, 'r') as f:
        ga_results = json.load(f)
    best_genes = ga_results.get("best_solution", {}).get("genes")
    ga_history = ga_results.get("history", {})
    
    # --- Initialize Fourier Series Helper ---
    fourier = FourierSeries2D(ROWS, COLS)

    # --- 1. "BEFORE" ANALYSIS (Uniform Grid) ---
    print("\n" + "="*50)
    print("STEP 1: Analyzing Pre-Optimization (Uniform) Topology")
    print("="*50)
    c_mat_orig, k_mat_orig = fourier.build_full_neighbor_topology(ROWS, COLS, rigid_outer_frame=False)
    heatmap_before, _ = run_heatmap_pipeline_for_topology(
        ROWS, COLS, k_mat_orig, c_mat_orig * 0.4, "uniform_grid"
    )
    if heatmap_before is not None:
        plot_heatmap(
            heatmap_before, list(range(1, 9)), list(range(6)), k_delay=30, amp=1, n_mass=ROWS*COLS,
            title_prefix="Memory Heatmap (Before Optimization)",
            save_dir=GA_EXPERIMENT_DIR, save_name="heatmap_before_optimization", show=True
        )

    # --- 2. "AFTER" ANALYSIS (GA Optimized) ---
    if not best_genes:
        print("[Skipping] No 'best_solution' found in GA results for 'After' analysis."); return
        
    print("\n" + "="*50)
    print("STEP 2: Analyzing Post-Optimization (GA) Topology")
    print("="*50)
    F_cont = fourier.generate_fourier_series(best_genes)
    F_norm = fourier.discretize_fourier_series(F_cont)
    c_mat_opt, k_mat_opt = fourier.build_adjacency_matrix(F_norm, ROWS, COLS, rigid_outer_frame=False)
    
    heatmap_after, after_exp_path = run_heatmap_pipeline_for_topology(
        ROWS, COLS, k_mat_opt, c_mat_opt * 0.4, "optimized_ga"
    )
    if heatmap_after is not None:
        plot_heatmap(
            heatmap_after, list(range(1, 9)), list(range(6)), k_delay=30, amp=1, n_mass=ROWS*COLS,
            title_prefix="Memory Heatmap (After Optimization)",
            save_dir=GA_EXPERIMENT_DIR, save_name="heatmap_after_optimization", show=True
        )

    # --- 3. Additional GA Visualizations ---
    print("\n" + "="*50)
    print("STEP 3: Plotting Additional GA Visualizations")
    print("="*50)
    
    # Plot GA fitness convergence
    if ga_history:
        plot_ga_convergence(ga_history, GA_EXPERIMENT_DIR)

    # Plot Fourier visualization for the best solution
    fourier.plot_fourier_viz(
        F=F_cont, F_discr_norm=F_norm, A_new=c_mat_opt, K_new=k_mat_opt,
        A_orig=c_mat_orig, K_orig=k_mat_orig, rows=ROWS, cols=COLS
    )

    # Launch interactive visualizer for the 'after' simulation
    if after_exp_path and after_exp_path.exists():
        print(f"\n[INFO] Opening interactive visualizer for the optimized run...")
        visualize_experiment(str(after_exp_path))
    else:
        print("[Warning] Could not find experiment path for the 'after' run to visualize.")

if __name__ == "__main__":
    main()
