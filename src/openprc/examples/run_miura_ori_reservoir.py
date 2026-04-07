import sys
import numpy as np
from pathlib import Path

# --- Path Setup ---
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

# --- Core Library Imports ---
from openprc.reservoir.io.state_loader import StateLoader
from openprc.reservoir.features.node_features import NodePositions, NodeDisplacements
from openprc.reservoir.features.bar_features import BarExtensions, BarLengths
from openprc.reservoir.readout.ridge import Ridge
from openprc.reservoir.training.trainer import Trainer
from openprc.analysis.visualization.time_series import TimeSeriesComparison
from openprc.analysis.tasks.imitation import NARMA_task
from openprc.analysis.benchmarks.narma_benchmark import NARMABenchmark


def main():
    """
    Miura-ori reservoir pipeline:
    1. Load an existing simulation.h5
    2. Select reservoir features
    3. Run NARMA2 benchmark
    4. Visualize results
    """

    # -----------------------------------------------------------------
    # 1. Define the experiment path
    # -----------------------------------------------------------------
    # Change this to your actual Miura simulation directory
    experiment_subpath = "miura_ori_4x8/generation_2"

    experiments_dir = src_dir / "experiments"
    experiment_dir = experiments_dir / experiment_subpath
    sim_path = experiment_dir / "output" / "simulation.h5"

    if not sim_path.exists():
        print(f"Simulation missing at {sim_path}!")
        return

    # -----------------------------------------------------------------
    # 2. Shared setup
    # -----------------------------------------------------------------
    loader = StateLoader(sim_path)
    print(f"Loaded {loader.total_frames} frames from {sim_path.name}")

    # -----------------------------------------------------------------
    # 3. Feature selection
    # -----------------------------------------------------------------
    # Recommended first try for Miura:
    # use node displacements relative to one base node, and keep x-z motion.
    features = NodeDisplacements(reference_node=0, dims=[2])

    # Alternative options you may want to try:
    # features = NodePositions(dims=[0, 2])
    # features = NodeDisplacements(reference_node=0, dims=[2])
    # features = BarExtensions()
    # features = BarLengths()

    # -----------------------------------------------------------------
    # 4. Input signal extraction
    # -----------------------------------------------------------------
    # Your Miura model used base excitation applied in z direction,
    # so the actuation signal is likely stored in dof=2, not dof=0.
    try:
        u_input = loader.get_actuation_signal(actuator_idx=0, dof=2)
        print("Loaded actuation signal from actuator 0, dof 2.")
    except Exception:
        print("Could not load dof=2 actuation. Falling back to dof=0.")
        u_input = loader.get_actuation_signal(actuator_idx=0, dof=0)

    # -----------------------------------------------------------------
    # 5. Define trainer
    # -----------------------------------------------------------------
    trainer = Trainer(
        loader=loader,
        features=features,
        readout=Ridge(1e-5),
        experiment_dir=experiment_dir,
        washout=5.0,
        train_duration=10.0,
        test_duration=10.0,
    )

    # -----------------------------------------------------------------
    # 6. Run NARMA2 benchmark
    # -----------------------------------------------------------------
    print("\n[Workflow: Benchmark]")

    benchmark = NARMABenchmark(group_name="narma_benchmark")
    benchmark_args = {"order": 2}

    print(f"Running benchmark: {benchmark.__class__.__name__}")
    score = benchmark.run(trainer, u_input, **benchmark_args)
    score.save()

    print(f"--- Benchmark complete for: {experiment_dir.name} ---")

    # -----------------------------------------------------------------
    # 7. Visualize
    # -----------------------------------------------------------------
    visualizer = TimeSeriesComparison()
    if score.readout_path and visualizer:
        print("\n[Processing] Visualizing results")
        plot_path = visualizer.plot(score.readout_path, start_frame=0, end_frame=500).save()
        print(f" >> Plot saved to: {plot_path}")

    # -----------------------------------------------------------------
    # 8. Print metrics
    # -----------------------------------------------------------------
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