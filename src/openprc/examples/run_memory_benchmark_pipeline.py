import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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
from openprc.reservoir.features.bar_features import BarExtensions
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

    Computed before get_actuation_signal to avoid any loader side effects.
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
    """
    import h5py
    with h5py.File(loader.sim_path, 'r') as f:
        fps = float(f.attrs.get('fps', 29.97))
    return max(1, int(test_duration_s * fps))


def main():
    """
    A pipeline to run the memory benchmark on a given experiment.
    This script will first run the benchmark to calculate all memory capacities,
    then prompt the user to select which readout to train and save permanently.
    """
    
    # 1. Define the Experiment Path
    TOPOLOGY = "topology_9"
    AMPLITUDE = "amp=2.5"
    SAMPLE = "sample_0"
    
    data_root = src_dir.parent / "openprc" / "experiments" / "spring_mass_4x4"
    experiment_dir = data_root / TOPOLOGY / AMPLITUDE / SAMPLE
    h5_path = experiment_dir / "experiment.h5"

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
    T = compute_test_frames(loader, test_duration_s=10.0)
    u_input = loader.get_actuation_signal(actuator_idx=0, dof=0)
    # stride = 3
    # u_input = u_input[::stride]

    eps = calculate_dambre_epsilon(effective_rank=N, test_duration=T)
    print(f"  Effective rank (N): {N:.4f}   Test frames (T): {T}   Epsilon: {eps:.6f}")
    
    # 3. Define Benchmark and its arguments
    benchmark = MemoryBenchmark(group_name="memory_benchmark")
    benchmark_args = {
        "tau_s": 90,     # Increased to look further back since k_delay is 1
        "n_s": 1,
        "k_delay": 1,    # Changed to 1
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
    print(f"\n--- Running Initial Benchmark to Calculate All Capacities ---")
    score = benchmark.run(trainer, u_input, **benchmark_args)
    score.save()
    print("--- Initial run complete. ---")

    # 5. Print key metrics
    if not score.metrics:
        print("Benchmark did not produce any metrics. Exiting.")
        return

    capacities = score.metrics.get('capacities')
    basis_names_bytes = score.metrics.get('basis_names', [])
    basis_names = [name.decode('utf-8') for name in basis_names_bytes]

    if capacities is None or not basis_names:
        print("No capacities or basis names found in metrics. Exiting.")
        return

    # 6. Interactive Readout Selection
    valid_indices = ~np.isnan(capacities)
    sorted_indices = np.argsort(capacities[valid_indices])[::-1]
    sorted_capacities = capacities[valid_indices][sorted_indices]
    sorted_names = [basis_names[i] for i in np.where(valid_indices)[0][sorted_indices]]

    print("\n--- Interactive Readout Selection ---")
    for i, (name, cap) in enumerate(zip(sorted_names, sorted_capacities)):
        if cap >= 0.00: 
            print(f"  Index {i}: Capacity = {cap:.4f}, Basis = '{name}'")

    try:
        user_input = input("\nEnter the index of the readout to save (or press Enter to skip): ")
        if user_input.strip() != "":
            selected_index = int(user_input)
            if 0 <= selected_index < len(sorted_names):
                selected_basis = sorted_names[selected_index]
                print(f"\nYou selected index {selected_index}: '{selected_basis}'")
                
                benchmark_args['save_readouts_for'] = [selected_basis]
                score = benchmark.run(trainer, u_input, **benchmark_args)
                
                visualizer = TimeSeriesComparison()
                if score.readout_path and visualizer:
                    plot_path = visualizer.plot(score.readout_path, start_frame=0, end_frame=500).save()
                    print(f" >> Plot saved to: {plot_path}")
    except (ValueError, EOFError):
        print("Skipping readout saving.")

    print("\n[Benchmark Results]")
    print(f"  >> Total Capacity: {score.metrics.get('total_capacity', 0):.4f}")
    print(f"  >> Linear Memory Capacity: {score.metrics.get('linear_memory_capacity', 0):.4f}")
    print(f"  >> Nonlinear Memory Capacity: {score.metrics.get('nonlinear_memory_capacity', 0):.4f}")
    print(f"Your strict theoretical threshold is: {eps:.6f}")

    # 7. Visualize the individual capacities
    plot_dir = experiment_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    if capacities is not None and basis_names is not None:
        threshold = 0.00
        filtered_indices = [i for i, c in enumerate(capacities) if c > threshold]
        
        if filtered_indices:
            filtered_scores = capacities[filtered_indices]
            filtered_names = [basis_names[i] for i in filtered_indices]

            sort_idx = np.argsort(filtered_scores)[::-1]
            sorted_scores = filtered_scores[sort_idx]
            sorted_names = [filtered_names[i] for i in sort_idx]

            top_n = 200  
            if len(sorted_scores) > top_n:
                sorted_scores = sorted_scores[:top_n]
                sorted_names = sorted_names[:top_n]

            dynamic_width = max(10, len(sorted_scores) * 0.3)
            
            plt.figure(figsize=(dynamic_width, 8))
            plt.bar(range(len(sorted_scores)), sorted_scores, align='center', width=0.8)
            plt.xticks(range(len(sorted_scores)), sorted_names, rotation=90, fontsize=9, ha='center')
            plt.ylabel("Capacity ($R^2$ Score)")
            plt.title(f"Information Processing Capacity (Top {len(sorted_scores)} Tasks)")
            plt.ylim(0, 1.05)
            plt.xlim(-1, len(sorted_scores))
            plt.tight_layout()
            plt.savefig(plot_dir / "information_processing_capacity.svg")
            plt.close()

    # --- 8. Visualize Cumulative Linear Memory Capacity vs Delay (Tau) ---
    print("\n  >> Generating Cumulative Linear Capacity vs Tau plot...")
    
    exponents = score.metrics.get('exponents')
    tau_s = 120
    
    if capacities is not None and exponents is not None:
        # Initialize storage
        linear_cap_per_tau = np.zeros(tau_s + 1)
        nonlinear_cap_per_tau = np.zeros(tau_s + 1)
        
        for i, exp in enumerate(exponents):
            if np.isnan(capacities[i]) or capacities[i] <= 0:
                continue
            
            # Identify exact max delay index for this basis
            active_lags = np.where(exp > 0)[0]
            if len(active_lags) == 0: continue
            max_tau_idx = np.max(active_lags)
            
            if np.sum(exp) == 1: # Linear
                linear_cap_per_tau[max_tau_idx] = capacities[i]
            elif np.sum(exp) > 1: # Nonlinear
                nonlinear_cap_per_tau[max_tau_idx] += capacities[i]
                
        # Cumulative totals
        cum_linear = np.cumsum(linear_cap_per_tau)
        cum_nonlinear = np.cumsum(nonlinear_cap_per_tau)
        tau_steps = np.arange(tau_s + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
        
        # --- Subplot 1: Linear Memory ---
        ax1.bar(tau_steps, cum_linear, color='steelblue', alpha=0.7, label='Cumulative Linear')
        ax1.plot(tau_steps, linear_cap_per_tau, color='red', marker='.', label='Indiv. Linear')
        ax1.set_title("Linear Memory Capacity ($n=1$)")
        ax1.set_ylabel("Cumulative Capacity")
        ax1.set_ylim(0, max(1.1, cum_linear[-1] * 1.2))
        ax1.legend()

        # --- Subplot 2: Nonlinear Memory ---
        ax2.bar(tau_steps, cum_nonlinear, color='forestgreen', alpha=0.7, label='Cumulative Nonlinear')
        ax2.plot(tau_steps, nonlinear_cap_per_tau, color='orange', marker='.', label='Indiv. Nonlinear')
        ax2.set_title(f"Nonlinear Capacity ($1 < n \leq {benchmark_args['n_s']}$)")
        ax2.set_ylabel("Cumulative Capacity")
        ax2.set_ylim(0, max(1.1, cum_nonlinear[-1] * 1.2))
        ax2.legend()

        for ax in [ax1, ax2]:
            ax.set_xlabel("Delay Step ($\\tau$)")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
            ax.grid(axis='y', linestyle='--', alpha=0.6)

        plt.suptitle(f"IPC Analysis: {TOPOLOGY} | $N_{{eff}}={N:.2f}$ | $\\epsilon={eps:.4f}$", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        save_path = plot_dir / "unified_ipc_analysis.svg"
        plt.savefig(save_path)
        plt.show()
        print(f" >> Unified IPC plot saved to: {save_path}")


if __name__ == "__main__":
    main()