import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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


def main():
    """
    A pipeline to run the memory benchmark on a given experiment.
    This script will first run the benchmark to calculate all memory capacities,
    then prompt the user to select which readout to train and save permanently.
    """
    
    # 1. Define the Experiment Path
    experiment_subpath = "spring_mass_3x3_test/generation_0"
    experiments_dir = src_dir / "experiments"
    experiment_dir = experiments_dir / experiment_subpath
    sim_path = experiment_dir / "output" / "simulation.h5"

    if not sim_path.exists():
        print(f"Simulation missing at {sim_path}!")
        print("Please run a simulation for the experiment first.")
        print("For example, you can run: python src/examples/simple_user_api.py")
        return
    
    # 2. Shared Setup
    loader = StateLoader(sim_path)
    features = NodePositions()
    u_input = loader.get_actuation_signal(actuator_idx=0, dof=0)
    
    print(f"Loaded {loader.total_frames} frames from {sim_path.name}")
    
    # 3. Define Benchmark and its arguments
    benchmark = MemoryBenchmark(group_name="memory_benchmark")
    benchmark_args = {
        "tau_s": 5,
        "n_s": 2,
        "k_delay": 100,
        "ridge": 1e-6
    }

    trainer = Trainer(
        loader=loader,
        features=features,
        readout=Ridge(benchmark_args.get("ridge")),
        experiment_dir=experiment_dir,
        washout=5.0,
        train_duration=20.0,
        test_duration=5.0,
    )
    
    # 4. First Run: Calculate all capacities
    print(f"\n--- Running Initial Benchmark to Calculate All Capacities ---")
    score = benchmark.run(trainer, u_input, **benchmark_args)
    score.save()
    print("--- Initial run complete. ---")

    # 5. Print key metrics and prepare for interactive selection
    if not score.metrics:
        print("Benchmark did not produce any metrics. Exiting.")
        return

    print("\n[Benchmark Results]")
    print(f"  >> Total Capacity: {score.metrics.get('total_capacity', 0):.4f}")
    print(f"  >> Linear Memory Capacity: {score.metrics.get('linear_memory_capacity', 0):.4f}")
    print(f"  >> Nonlinear Memory Capacity: {score.metrics.get('nonlinear_memory_capacity', 0):.4f}")

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
    print("The following memory tasks are available to be saved:")
    for i, (name, cap) in enumerate(zip(sorted_names, sorted_capacities)):
        if cap > 1e-9: # Only show tasks with non-trivial capacity
            print(f"  Index {i}: Capacity = {cap:.4f}, Basis = '{name}'")

    try:
        user_input = input("\nEnter the index of the readout to save (or press Enter to skip): ")
        if user_input.strip() == "":
            print("Skipping readout saving.")
        else:
            try:
                selected_index = int(user_input)
                if 0 <= selected_index < len(sorted_names):
                    selected_basis = sorted_names[selected_index]
                    print(f"\nYou selected index {selected_index}: '{selected_basis}'")
                    
                    # Add the selected basis to benchmark_args and re-run
                    benchmark_args['save_readouts_for'] = [selected_basis]
                    
                    print(f"\n--- Re-running benchmark to train and save readout for '{selected_basis}' ---")
                    benchmark.run(trainer, u_input, **benchmark_args)
                    print("--- Readout saved successfully. ---")
                    
                else:
                    print("Invalid index. No readout will be saved.")
            except ValueError:
                print("Invalid input. Please enter a number. No readout will be saved.")
    except EOFError:
        print("\nNon-interactive mode detected (e.g., CI/CD). Skipping readout saving.")

    # 7. Visualize the capacities
    print("\n[Visualizing Benchmark Results]")
    # ... (Visualization code remains the same, but let's uncomment it)
    if capacities is not None and basis_names is not None:
        threshold = 1e-9
        filtered_indices = [i for i, c in enumerate(capacities) if c > threshold]
        
        if not filtered_indices:
            print("No capacities above threshold to plot.")
            return

        filtered_scores = capacities[filtered_indices]
        filtered_names = [basis_names[i] for i in filtered_indices]

        # Sort for better visualization
        vis_sorted_indices = np.argsort(filtered_scores)[::-1]
        vis_sorted_scores = filtered_scores[vis_sorted_indices]
        vis_sorted_names = [filtered_names[i] for i in vis_sorted_indices]

        plt.figure(figsize=(12, 8))
        plt.bar(range(len(vis_sorted_scores)), vis_sorted_scores, tick_label=vis_sorted_names)
        plt.xticks(rotation=90)
        plt.ylabel("Capacity (R^2 Score)")
        plt.title("Information Processing Capacity - Individual Tasks")
        plt.ylim(0, 1)
        plt.tight_layout()

        plot_dir = experiment_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        plot_path = plot_dir / "information_processing_capacity.svg"
        plt.savefig(plot_path)
        plt.close()
        print(f"  >> Plot saved to: {plot_path}")

if __name__ == "__main__":
    main()
