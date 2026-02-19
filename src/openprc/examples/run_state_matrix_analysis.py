import sys
import numpy as np
from pathlib import Path

# --- Path Setup ---
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

# --- Core Library Imports ---
from openprc.reservoir.io.state_loader import StateLoader
from openprc.reservoir.features.node_features import NodeDisplacements
from openprc.reservoir.readout.ridge import Ridge
from openprc.reservoir.training.trainer import Trainer
from openprc.analysis.benchmarks.custom_benchmark import CustomBenchmark

def state_matrix_analysis_logic(benchmark, trainer, u_input, **kwargs):
    """
    Computes the effective rank and conditioning number of the reservoir states matrix X_full.
    """
    # 1. Get X_full from trainer
    X_full = trainer.features.transform(trainer.loader)
    
    # 2. Compute singular values
    singular_values = np.linalg.svd(X_full, compute_uv=False)
    
    # 3. Compute effective rank
    normalized_singular_values = singular_values / np.sum(singular_values)
    entropy = -np.sum(normalized_singular_values * np.log(normalized_singular_values))
    effective_rank = np.exp(entropy)
    
    # 4. Compute condition number
    condition_number = singular_values[0] / singular_values[-1] if singular_values[-1] > 1e-9 else np.inf

    # 5. Define the metrics and metadata dictionaries to be returned
    metrics = {
        'effective_rank': effective_rank,
        'condition_number': condition_number,
        'num_singular_values': len(singular_values)
    }
    metadata = {
        'feature_type': trainer.features.__class__.__name__,
        'benchmark_class': benchmark.__class__.__name__
    }
    
    return metrics, metadata

def main():
    """
    An example pipeline that runs a custom benchmark to analyze the reservoir state matrix.
    """
    
    # 1. Define the Experiment Path
    experiment_subpath = "spring_mass_4x4_test/generation_0"
    experiments_dir = src_dir / "experiments"
    experiment_dir = experiments_dir / experiment_subpath
    sim_path = experiment_dir / "output" / "simulation.h5"

    if not sim_path.exists():
        print(f"Simulation missing at {sim_path}! Please run a simulation first.")
        return
    
    # 2. Shared Setup
    loader = StateLoader(sim_path)
    features = NodeDisplacements(reference_node=0, dims=[0])
    u_input = loader.get_actuation_signal(actuator_idx=0, dof=0)
    
    print(f"Loaded {loader.total_frames} frames from {sim_path.name}")

    # 3. Define the Trainer
    # Note: The trainer is required by the benchmark, but for this specific analysis,
    # we don't need to run the training process itself. We only need the trainer
    # to get access to the features and other experiment settings.
    trainer = Trainer(
        loader=loader,
        features=features,
        readout=Ridge(1e-5), # Dummy readout, not used in this benchmark
        experiment_dir=experiment_dir
    )

    # 4. Instantiate and run the CustomBenchmark
    benchmark = CustomBenchmark(
        group_name="state_matrix_analysis",
        benchmark_logic=state_matrix_analysis_logic
    )
    
    print(f"Running benchmark: {benchmark.__class__.__name__}")
    score = benchmark.run(trainer, u_input)
    score.save(filename="state_matrix_analysis.h5")
    
    print(f"--- Benchmark complete for: {experiment_dir.name} ---")

    # 5. Print Metrics
    if score.metrics:
        print("\n[Processing] Printing Benchmark Results:")
        def print_metrics(metrics_dict, indent=""):
            for key, value in metrics_dict.items():
                print_prefix = f"{indent}>> {key}: "
                if isinstance(value, dict):
                    print(f"{indent}>> {key}:")
                    print_metrics(value, indent + "  ")
                elif isinstance(value, (int, float, np.number)):
                    print(f"{print_prefix}{value:.5f}")
                else:
                    print(f"{print_prefix}{value}")
        print_metrics(score.metrics, indent="  ")


if __name__ == "__main__":
    main()
