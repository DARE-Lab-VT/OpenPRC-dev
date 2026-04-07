"""
run_memory_benchmark_miura.py
==============================
Memory benchmark for Miura-ori reservoir.
Loads an existing simulation.h5 — does NOT re-run the simulation.
"""

import sys
import h5py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import chi2
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import StandardScaler

current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

from openprc.analysis.benchmarks.memory_benchmark import MemoryBenchmark
from openprc.reservoir.io.state_loader import StateLoader
from openprc.reservoir.features.node_features import NodePositions, NodeDisplacements
from openprc.reservoir.features.bar_features import BarExtensions
from openprc.reservoir.training.trainer import Trainer
from openprc.reservoir.readout.ridge import Ridge
from openprc.analysis.visualization.time_series import TimeSeriesComparison


# ── VT color palette ──────────────────────────────────────────────────────────
VT_MAROON       = '#861F41'
VT_ORANGE       = '#E5751F'
VT_STONE        = '#75787B'
VT_MAROON_LIGHT = '#C4627A'
VT_ORANGE_LIGHT = '#F0A875'


# ── helpers ───────────────────────────────────────────────────────────────────

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
    with h5py.File(loader.sim_path, 'r') as f:
        fps = float(f.attrs.get('fps', 100.0))  # save_interval=0.01 → 100 fps
    return max(1, int(test_duration_s * fps))


# ── main ──────────────────────────────────────────────────────────────────────

def main():

    # ── 1. Paths ──────────────────────────────────────────────────────────────
    XN, YN     = 4, 8
    GENERATION = 3

    experiment_dir = (
        src_dir / "experiments"
        / f"miura_ori_{XN}x{YN}"
        / f"generation_{GENERATION}"
    )
    sim_path = experiment_dir / "output" / "simulation.h5"

    if not sim_path.exists():
        print(f"[Error] Simulation not found: {sim_path}")
        return

    print(f"-> Loading: {sim_path}")

    # ── 2. Loader + features ──────────────────────────────────────────────────
    loader   = StateLoader(sim_path)
    features = NodeDisplacements(reference_node=0, dims=[2])
    # alternatives:
    # features = NodePositions(dims=[2])
    # features = NodeDisplacements(reference_node=0, dims=[0, 1, 2])
    # features = BarExtensions()

    print(f"Loaded {loader.total_frames} frames from {sim_path.name}")

    # ── 3. Load and normalise u_input ─────────────────────────────────────────
    u_raw   = loader.get_actuation_signal(actuator_idx=0, dof=2)
    u_dyn   = u_raw - u_raw[0]
    u_input = 0.5 * (u_dyn - u_dyn.min()) / (u_dyn.max() - u_dyn.min())
    u_input = u_input[: loader.total_frames]
    print(f"u_input shape: {u_input.shape}, "
          f"range: [{u_input.min():.3f}, {u_input.max():.3f}], "
          f"std: {u_input.std():.4f}")

    # ── 4. Rank + epsilon ─────────────────────────────────────────────────────
    N           = compute_effective_rank(loader, features)
    test_frames = int(10.0 / loader.dt)
    eps         = calculate_dambre_epsilon(effective_rank=N, test_duration=test_frames)
    print(f"Rank (N): {N}   Test frames (T): {test_frames}   Epsilon: {eps:.6f}")

    # ── 5. Trainer + benchmark args ───────────────────────────────────────────
    benchmark_args = {
        "tau_s"   : 30,
        "n_s"     : 2,
        "k_delay" : 1,
        "eps"     : eps,
        "ridge"   : 1e-6,
    }

    trainer = Trainer(
        loader=loader,
        features=features,
        readout=Ridge(benchmark_args["ridge"]),
        experiment_dir=experiment_dir,
        washout=5.0,
        train_duration=10.0,
        test_duration=10.0,
    )

    # ── 6. Run benchmark ──────────────────────────────────────────────────────
    print("\n--- Running Memory Benchmark ---")
    benchmark = MemoryBenchmark(group_name="memory_benchmark")
    score     = benchmark.run(trainer, u_input, **benchmark_args)
    score.save()
    print("--- Benchmark complete ---")

    if not score.metrics:
        print("No metrics returned. Exiting.")
        return

    # ── 7. Interactive readout selection ──────────────────────────────────────
    capacities        = score.metrics.get('capacities')
    basis_names_bytes = score.metrics.get('basis_names', [])
    basis_names       = [n.decode('utf-8') for n in basis_names_bytes]

    if capacities is None or not basis_names:
        print("No capacities found. Exiting.")
        return

    valid_mask        = ~np.isnan(capacities)
    sorted_indices    = np.argsort(capacities[valid_mask])[::-1]
    sorted_capacities = capacities[valid_mask][sorted_indices]
    sorted_names      = [basis_names[i] for i in np.where(valid_mask)[0][sorted_indices]]

    print("\n--- Capacities ---")
    for i, (name, cap) in enumerate(zip(sorted_names, sorted_capacities)):
        if cap >= 0.0:
            print(f"  {i:3d}: {cap:.4f}  {name}")

    try:
        user_input = input("\nEnter index to save readout (or Enter to skip): ")
        if user_input.strip():
            idx = int(user_input)
            if 0 <= idx < len(sorted_names):
                selected = sorted_names[idx]
                print(f"Saving readout for: '{selected}'")
                benchmark_args['save_readouts_for'] = [selected]
                score = benchmark.run(trainer, u_input, **benchmark_args)
                visualizer = TimeSeriesComparison()
                if score.readout_path:
                    plot_path = visualizer.plot(
                        score.readout_path, start_frame=0, end_frame=500
                    ).save()
                    print(f"Plot saved: {plot_path}")
    except (ValueError, EOFError):
        print("Skipping.")

    print(f"\n[Results]")
    print(f"  Total capacity         : {score.metrics.get('total_capacity', 0):.4f}")
    print(f"  Linear memory capacity : {score.metrics.get('linear_memory_capacity', 0):.4f}")
    print(f"  Nonlinear capacity     : {score.metrics.get('nonlinear_memory_capacity', 0):.4f}")
    print(f"  Epsilon threshold      : {eps:.6f}")

    # ── 8. Plots ──────────────────────────────────────────────────────────────
    plot_dir = experiment_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Bar chart of top capacities
    filtered_idx = [i for i, c in enumerate(capacities) if c > 0.0]
    if filtered_idx:
        f_scores = capacities[filtered_idx]
        f_names  = [basis_names[i] for i in filtered_idx]
        sort_idx = np.argsort(f_scores)[::-1][:200]
        f_scores = f_scores[sort_idx]
        f_names  = [f_names[i] for i in sort_idx]

        plt.figure(figsize=(max(10, len(f_scores) * 0.3), 8))
        plt.bar(range(len(f_scores)), f_scores, color=VT_MAROON, width=0.8)
        plt.xticks(range(len(f_scores)), f_names, rotation=90, fontsize=9)
        plt.ylabel("Capacity ($R^2$)")
        plt.title(
            f"IPC — Miura-ori {XN}x{YN} gen{GENERATION}",
            color='black', fontweight='bold'
        )
        plt.ylim(0, 1.05)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(plot_dir / "information_processing_capacity.svg")
        plt.close()

    # Cumulative linear vs nonlinear per tau
    exponents = score.metrics.get('exponents')
    tau_s     = benchmark_args['tau_s']

    if exponents is not None:
        lin_per_tau    = np.zeros(tau_s + 1)
        nonlin_per_tau = np.zeros(tau_s + 1)

        for i, exp in enumerate(exponents):
            if np.isnan(capacities[i]) or capacities[i] <= 0:
                continue
            active = np.where(exp > 0)[0]
            if len(active) == 0:
                continue
            max_tau = np.max(active)
            if max_tau > tau_s:
                continue
            if np.sum(exp) == 1:
                lin_per_tau[max_tau] = capacities[i]
            else:
                nonlin_per_tau[max_tau] += capacities[i]

        cum_lin    = np.cumsum(lin_per_tau)
        cum_nonlin = np.cumsum(nonlin_per_tau)
        tau_steps  = np.arange(tau_s + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

        # ── Linear Memory ─────────────────────────────────────────────────
        ax1.bar(tau_steps, cum_lin,
                color=VT_MAROON_LIGHT, alpha=0.85, label='Cumulative')
        ax1.plot(tau_steps, lin_per_tau,
                 color=VT_MAROON, marker='.', linewidth=1.8, label='Individual')
        ax1.set_title("Linear Memory Capacity", fontsize=13)
        ax1.set_ylabel("Capacity")
        ax1.set_ylim(0, max(1.1, cum_lin[-1] * 1.2))
        ax1.legend()

        # ── Nonlinear Memory ──────────────────────────────────────────────
        ax2.bar(tau_steps, cum_nonlin,
                color=VT_ORANGE_LIGHT, alpha=0.85, label='Cumulative')
        ax2.plot(tau_steps, nonlin_per_tau,
                 color=VT_ORANGE, marker='.', linewidth=1.8, label='Individual')
        ax2.set_title(f"Nonlinear Capacity (n≤{benchmark_args['n_s']})", fontsize=13)
        ax2.set_ylabel("Capacity")
        ax2.set_ylim(0, max(1.1, cum_nonlin[-1] * 1.2))
        ax2.legend()

        for ax in [ax1, ax2]:
            ax.set_xlabel("Delay τ")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
            ax.grid(axis='y', linestyle='--', alpha=0.4, color=VT_STONE)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.suptitle(
            f"IPC — Miura-ori {XN}x{YN} | $N_{{rank}}={round(N, 2)}$ | $\\epsilon={eps:.4f}$",
            fontsize=14, color='black', fontweight='bold'
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(plot_dir / "unified_ipc_analysis.svg")
        plt.show()
        print(f"IPC plot saved to: {plot_dir / 'unified_ipc_analysis.svg'}")


if __name__ == "__main__":
    main()