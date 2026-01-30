import sys
import numpy as np
from pathlib import Path

# --- Path Setup ---
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

# --- Core Library Imports ---
from reservoir.io.state_loader import StateLoader
from reservoir.features.node_features import NodePositions
from reservoir.readout.ridge import Ridge
from reservoir.utils.trainer import Trainer
from analysis.visualization.time_series import TimeSeriesComparison

# --- User-configurable Imports for Task and Benchmark ---
# A user can import any task or benchmark that follows the base interfaces
from reservoir.tasks.memory.narma import NARMA
from analysis.benchmarks.memory_benchmark import NARMABenchmark


def run_pipeline(experiment_subpath, task, benchmark, benchmark_args, visualizer=None):
    """
    A generic pipeline that trains a readout, runs a benchmark, and visualizes.
    It is agnostic to the specific task or benchmark being used.
    """
    experiments_dir = src_dir / "experiments"
    experiment_dir = experiments_dir / experiment_subpath
    sim_path = experiment_dir / "output" / "simulation.h5"

    if not sim_path.exists():
        print(f"Simulation missing at {sim_path}!"); return

    print(f"--- Pipeline for: {experiment_dir.name} ---")
    print(f"Task: {task.__class__.__name__} | Benchmark: {benchmark.__class__.__name__}")
    
    # 1. Load Data
    loader = StateLoader(sim_path)
    print(f"Loaded {loader.total_frames} frames.")

    # 2. Initialize Trainer
    trainer = Trainer(
        features=NodePositions(),
        readout=Ridge(1e-5),
        experiment_dir=experiment_dir,
        washout=500,
        train_len=2000,
        test_len=500,
        actuator_idx=0  # Task is driven by the first actuator signal
    )
    
    # 3. Train
    print(f"\n[Processing] Training for {task.__class__.__name__}")
    result = trainer.fit(loader, task) 
    readout_path = result.save()
    
    # 4. Analyze
    print(f"\n[Processing] Running benchmark: {benchmark.__class__.__name__}")
    score = benchmark.run(experiment_dir, **benchmark_args)
    score.save()
    
    # The pipeline doesn't need to know the result type, only that it has a `__dict__`
    for key, value in score.__dict__.items():
        if isinstance(value, (int, float, np.number)):
             print(f" >> {key}: {value:.5f}")
    
    # 5. Visualize (if provided)
    if visualizer:
        print("\n[Processing] Visualizing results")
        # The result object can find the readout path for the visualizer
        plot_path = visualizer.plot(score.readout_path, start_frame=0, end_frame=500).save()
        print(f" >> Plot saved to: {plot_path}")

if __name__ == "__main__":
    # --- This block is the main configuration area for the user ---
    
    # 1. Define the Experiment Path
    exp_path = "spring_mass_3x3_test/generation_0"

    # 2. Define the Task
    task = NARMA(2)
    
    # 3. Define the Benchmark and its specific arguments
    # --- Option A: Standard NARMA Benchmark ---
    benchmark = NARMABenchmark()
    benchmark_args = {"order": 2}
    
    # --- Option B: User-Defined Custom Benchmark ---
    # To create your own benchmark
    
    # 4. Define the Visualizer
    visualizer = TimeSeriesComparison()
    
    # 5. Run the pipeline with the chosen components
    run_pipeline(exp_path, task, benchmark, benchmark_args, visualizer)