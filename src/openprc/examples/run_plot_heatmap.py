import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler

# --- Path Setup ---
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

# --- Core Library Imports ---
from openprc.analysis.benchmarks.memory_benchmark import MemoryBenchmark
from openprc.reservoir.io.state_loader import StateLoader
from openprc.reservoir.features.node_features import NodePositions, NodeDisplacements
from openprc.reservoir.features.bar_features import BarExtensions, BarLengths
from openprc.reservoir.training.trainer import Trainer
from openprc.reservoir.readout.ridge import Ridge
from openprc.analysis.visualization.time_series import TimeSeriesComparison


def calculate_dambre_epsilon(effective_rank: int, test_duration: int, p_value: float = 1e-4) -> float:
    """
    Calculates the exact theoretical threshold (epsilon) for IPC 
    based on Dambre et al.'s chi-squared method.
    
    Parameters:
    - effective_rank (N): The number of independent state variables (e.g., 9).
    - test_duration (T): The number of samples in your test set.
    - p_value (p): The acceptable probability of a false positive (default 10^-4).
    
    Returns:
    - epsilon: The strict cutoff value to use in the Heaviside step function.
    """
    # 1. Find the threshold 't' using the Inverse Survival Function (ISF) 
    # of the chi-squared distribution with N degrees of freedom.
    # This finds 't' such that P(chi^2(N) >= t) = p
    t = chi2.isf(p_value, df=effective_rank)
    
    # 2. Calculate the final epsilon: 2t / T
    # The factor of 2 is the intentional doubling to account for 
    # non-independent variables in real dynamical systems.
    epsilon = (2.0 * t) / test_duration
    
    return epsilon


def compute_effective_rank(loader, features) -> float:
    """
    Entropy-based effective rank (standard in reservoir computing).

    Uses the Shannon entropy of normalised singular values:
        s_norm         = s / sum(s)
        effective_rank = exp( -sum(s_norm * log(s_norm)) )

    Computed on the full state matrix with no washout stripping,
    matching state_matrix_analysis_logic() exactly.

    Parameters
    ----------
    loader   : StateLoader
    features : feature extractor (e.g. NodeDisplacements)

    Returns
    -------
    effective_rank : float
    """
    state_matrix = features.transform(loader)

    if state_matrix.shape[0] < 2:
        return 1.0

    state_matrix = StandardScaler().fit_transform(state_matrix)
    _, s, _      = np.linalg.svd(state_matrix, full_matrices=False)
    s_norm       = s / np.sum(s)
    return float(np.exp(-np.sum(s_norm * np.log(s_norm + 1e-12))))


def compute_test_frames(loader, test_duration_s: float = 10.0) -> int:
    """
    Derive T (number of test frames) from fps stored in the H5 file.

    Parameters
    ----------
    loader         : StateLoader
    test_duration_s: same value passed as test_duration to Trainer

    Returns
    -------
    T : int
    """
    import h5py
    with h5py.File(loader.sim_path, 'r') as f:
        fps = float(f.attrs.get('fps', 29.97))
    return max(1, int(test_duration_s * fps))


def plot_heatmap(
    heatmap, n_list, tau_d_list, k_delay, amp, n,
    vmin=None, vmax=None,
    save_dir=None,
    save_name=None,
    save_svg=True,
    save_png=False,
    dpi=300,
    show=True
):
    fig, ax = plt.subplots(figsize=(10, 8))
    heatmap = heatmap.T

    if heatmap is not None and n_list is not None and tau_d_list is not None:
        title = (rf"$R^2$ (upper)", rf"num_mass={n}" + "\n" +
                 rf"k={k_delay}, A={amp}")

        if vmin is None:
            vmin = 0.0
        if vmax is None:
            vmax = 1.0

        im = ax.imshow(
            heatmap, aspect='auto', origin='lower',
            cmap='RdYlBu_r', vmin=vmin, vmax=vmax
        )

        n_rows, n_cols = heatmap.shape
        for y in range(n_rows):
            for x in range(n_cols):
                r2_val = heatmap[y, x]

                # Upper: R^2
                if not np.isnan(r2_val):
                    ax.text(x, y, f'{r2_val:.2f}',
                            ha='center', va='center', color='black', fontsize=8)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('$R^2$ Mean')
        ax.set_xlabel(r'$n$ (monomial degree)')
        ax.set_ylabel(r'$\tau$ (time delay)')
        ax.set_title(title, fontsize=8)

        ax.set_xticks(np.arange(len(n_list)))
        ax.set_yticks(np.arange(len(tau_d_list)))
        ax.set_xticklabels(n_list, fontsize=6)
        ax.set_yticklabels((np.array(tau_d_list) * k_delay), fontsize=6)

    fig.tight_layout()

    # ---- Save here (before show) ----
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        if save_name is None:
            # Default filename
            save_name = f"heatmap_R2_n{n}_A{amp}_k{k_delay}"

        if save_svg:
            svg_path = os.path.join(save_dir, f"{save_name}.svg")
            fig.savefig(svg_path, format="svg", bbox_inches="tight")
            print(f"[Saved] Heatmap SVG -> {svg_path}", flush=True)

        if save_png:
            png_path = os.path.join(save_dir, f"{save_name}.png")
            fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
            print(f"[Saved] Heatmap PNG -> {png_path}", flush=True)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax

def main():
    """
    A pipeline to run the memory benchmark on a given experiment.
    This script will first run the benchmark to calculate all memory capacities,
    then prompt the user to select which readout to train and save permanently.
    """
    
    # 1. Define the Experiment Path
    TOPOLOGY = "topology_9_prestress"
    AMPLITUDE = "amp=2.5"
    SAMPLE = "sample_0"
    
    data_root = src_dir.parent / "openprc" / "experiments" / "spring_mass_4x4"
    experiment_dir = data_root / TOPOLOGY / AMPLITUDE / SAMPLE
    h5_path = experiment_dir / "experiment.h5"
    save_path = experiment_dir / "plots"

    if not h5_path.exists():
        print(f"[Error] Experiment file not found: {h5_path}")
        return

    print(f"-> Loading Experiment: {AMPLITUDE} / {SAMPLE}")
    
    # 2. Shared Setup
    loader = StateLoader(h5_path)
    features = NodeDisplacements(reference_node=0, dims=[0, 1])

    print(f"Loaded {loader.total_frames} frames from {h5_path.name}")

    # Per-sample effective rank — computed before get_actuation_signal
    # to match the call order in run_state_matrix_analysis.py
    N = compute_effective_rank(loader, features)
    u_input = loader.get_actuation_signal(actuator_idx=0, dof=0)
    T = compute_test_frames(loader, test_duration_s=10.0)
    eps = calculate_dambre_epsilon(effective_rank=N, test_duration=T)
    print(f"  Effective rank (N): {N:.4f}   Test frames (T): {T}   Epsilon: {eps:.6f}")
    
    # 3. Define Benchmark and its arguments
    n_list = list(range(1, 5))
    tau_d_list = list(range(30))
    k_delay = 1

    heatmap = np.empty((len(n_list), len(tau_d_list)), dtype=float)

    idx_pairs = list(product(range(len(n_list)), range(len(tau_d_list))))
    for (i, j) in tqdm(idx_pairs, total=len(idx_pairs), leave=True):
        n_s = n_list[i]
        tau_s = tau_d_list[j]
        benchmark = MemoryBenchmark(group_name="memory_benchmark")
        benchmark_args = {
            "tau_s": tau_s,
            "n_s": n_s,
            "k_delay": k_delay,
            "eps": eps,
            "ridge": 1e-6
        }

        trainer = Trainer(
            loader=loader,
            features=features,
            readout=Ridge(benchmark_args.get("ridge")),
            experiment_dir=experiment_dir,
            washout=5.0,
            train_duration=10.0,
            test_duration=10.0,
        )
        
        # 4. First Run: Calculate all capacities
        # print(f"\n--- Running Initial Benchmark to Calculate All Capacities ---")
        score = benchmark.run(trainer, u_input, **benchmark_args)
        score.save()
        # print("--- Initial run complete. ---")

        # 5. Print key metrics and prepare for interactive selection
        if not score.metrics:
            print("Benchmark did not produce any metrics. Exiting.")
            return

        # print("\n[Benchmark Results]")
        # print(f"  >> Total Capacity: {score.metrics.get('total_capacity', 0):.4f}")
        # print(f"  >> Linear Memory Capacity: {score.metrics.get('linear_memory_capacity', 0):.4f}")
        # print(f"  >> Nonlinear Memory Capacity: {score.metrics.get('nonlinear_memory_capacity', 0):.4f}")

        capacities = score.metrics.get('capacities')
        basis_names_bytes = score.metrics.get('basis_names', [])
        basis_names = [name.decode('utf-8') for name in basis_names_bytes]

        if capacities is None or not basis_names:
            print("No capacities or basis names found in metrics. Exiting.")
            return

        # 6. Interactive Readout Selection
        heatmap[i, j] = np.nanmean(capacities)

    plot_heatmap(heatmap, n_list, tau_d_list, k_delay=k_delay, amp=1, n=16, save_dir=save_path, save_svg=True)
    

if __name__ == "__main__":
    main()