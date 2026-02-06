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
    """
    
    # 1. Define the Experiment Path
    # This example uses a standard test experiment.
    # Make sure you have run a simulation for this experiment first.
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
    # The group_name determines where the results are saved in the metrics file.
    benchmark = MemoryBenchmark(group_name="memory_benchmark")
    
    # These arguments control the memory task calculation.
    benchmark_args = {
        "tau_s": 5,         # Maximum lag to test for memory
        "n_s": 2,            # Polynomial degree of inputs (1 for linear memory)
        "k_delay": 100,        # Step size for lags
        "ridge": 1e-6        # Regularization for the regression
    }

    # The memory benchmark uses parameters from the trainer for washout, train and test durations
    trainer = Trainer(
        loader=loader,
        features=features,
        readout=Ridge(benchmark_args.get("ridge")), # Readout is not used by memory benchmark, but trainer needs it
        experiment_dir=experiment_dir,
        washout=5.0,
        train_duration=20.0,
        test_duration=5.0,
    )
    
    # 4. Run the benchmark
    print(f"Running benchmark: {benchmark.__class__.__name__}")
    print(f"Experiment: {experiment_dir.name}")
    score = benchmark.run(trainer, u_input, **benchmark_args)
    score.save()
    
    print(f"--- Benchmark complete for: {experiment_dir.name} ---")

    # 5. Print the key metrics
    if score.metrics:
        print("\n[Benchmark Results]")
        
        total_cap = score.metrics.get('total_capacity')
        linear_cap = score.metrics.get('linear_memory_capacity')
        nonlinear_cap = score.metrics.get('nonlinear_memory_capacity')

        if total_cap is not None:
            print(f"  >> Total Capacity: {total_cap:.4f}")
        if linear_cap is not None:
            print(f"  >> Linear Memory Capacity: {linear_cap:.4f}")
        if nonlinear_cap is not None:
            print(f"  >> Nonlinear Memory Capacity: {nonlinear_cap:.4f}")

        print("\nDetailed metrics saved to HDF5 file in the experiment's 'metrics' directory.")

    # 6. Visualize the capacities
    if score.metrics:
        print("\n[Visualizing Benchmark Results]")
        
        capacities = score.metrics.get('capacities')
        # basis_names are saved as bytes, need to decode
        basis_names = [name.decode('utf-8') for name in score.metrics.get('basis_names')]
        
        if capacities is not None and basis_names is not None:
            
            # Filter for scores above a threshold to make the plot readable
            threshold = 0.00
            
            filtered_indices = [i for i, score in enumerate(capacities) if score > threshold]
            
            if not filtered_indices:
                print("No capacities above threshold to plot.")
                return

            filtered_scores = capacities[filtered_indices]
            filtered_names = [basis_names[i] for i in filtered_indices]

            # Sort for better visualization
            sorted_indices = np.argsort(filtered_scores)[::-1]
            sorted_scores = filtered_scores[sorted_indices]
            sorted_names = [filtered_names[i] for i in sorted_indices]

            # plt.figure(figsize=(12, 6))
            # plt.bar(range(len(sorted_scores)), sorted_scores, tick_label=sorted_names)
            # plt.xticks(rotation=90)
            # plt.ylabel("Capacity")
            # plt.title("Information Processing Capacity - Individual Capacities of Basis Functions")
            # plt.ylim(0, 1)
            # plt.tight_layout()

            # # Save the plot
            # plot_dir = experiment_dir / "plots"
            # plot_dir.mkdir(exist_ok=True)
            # plot_path = plot_dir / "information_processing_capacity.svg"
            # plt.savefig(plot_path)
            # plt.close()
            # print(f"  >> Plot saved to: {plot_path}")
        
        print(f"Min Capacity = {np.nanmin(capacities):.4f}")


if __name__ == "__main__":
    main()
