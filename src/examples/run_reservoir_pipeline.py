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
from reservoir.tasks.memory.narma import NARMA
from analysis.benchmarks.memory_benchmark import NARMABenchmark


def run_pipeline(experiment_subpath, task, benchmark, benchmark_args=None, visualizer=None):
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
        washout=5.0,
        train_duration=20.0,
        test_duration=5.0,
        actuator_idx=0,  # Task is driven by the first actuator signal
        dof=0   # The desired degree of freedom --> 0: 'x', 1: 'y', 2: 'z', None: return 3D np.ndarray
    )
    
    # 3. Train
    print(f"\n[Processing] Training for {task.__class__.__name__}")
    result = trainer.fit(loader, task) 
    result.save()
    
    # 4. Analyze
    print(f"\n[Processing] Running benchmark: {benchmark.__class__.__name__}")
    score = benchmark.run(experiment_dir, **benchmark_args)
    score.save()
    
    # [OPTIONAL] Print metrics as you desired, feel free to modify to suit your need
    print("\n[Processing] Printing Benchmark Results:")
    def print_metrics(metrics_dict, indent=""):
        for key, value in metrics_dict.items():
            print_prefix = f"{indent}>> {key}: "
            if isinstance(value, dict):
                print(f"{indent}>> {key}:")
                print_metrics(value, indent + "  ")
            elif isinstance(value, (int, float, np.number)):
                print(f"{print_prefix}{value:.5f}")
            elif isinstance(value, np.ndarray):
                print(f"{print_prefix}<np.ndarray, shape: {value.shape}>")
            elif isinstance(value, list):
                print(f"{print_prefix}<list, len: {len(value)}>")
            else:
                # Fallback for other types
                print(f"{print_prefix}{value}")

    print_metrics(score.metrics, indent="  ")
    
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
    # --- Option A: Standard NARMA Task ---
    task = NARMA(2)

    # --- Option B: User-Defined Custom Task
    # task = YOUR_TASK: np.ndarray
    
    # 3. Define the Benchmark and its specific arguments
    # --- Option A: Standard NARMA Benchmark ---
    benchmark = NARMABenchmark(group_name="narma_benchmark")
    benchmark_args = {"order": 2}
    
    # --- Option B: User-Defined Custom Benchmark via Function ---
    # To define a desired custom benchmark, you can pass a function directly to the `CustomBenchmark` class.
    # from analysis.benchmarks.custom_benchmark import CustomBenchmark
    # import h5py

    # def nrmse_logic_with_metadata(benchmark_instance, **kwargs):
    #     """
    #     A custom function that calculates NRMSE and returns metrics and metadata.
    #     This example demonstrates returning various data types, including nested
    #     dictionaries, which the `save` method can now handle.
    #     """
    #     readout_path = benchmark_instance.readout_path
        
    #     #================ Define your own logic here #================#
    #     with h5py.File(readout_path, 'r') as f:
    #         group = 'validation' if 'validation' in f else 'training'
    #         target = f[f'{group}/target'][:]
    #         prediction = f[f'{group}/prediction'][:]

    #     rmse = np.sqrt(np.mean((target - prediction)**2))
    #     std_dev = np.std(target)
    #     nrmse = rmse / (std_dev if std_dev > 1e-9 else 1.0)
    #     #=============================================================#
        
    #     metrics = {
    #         'nrmse': nrmse,
    #         'components': {
    #             'rmse': rmse,
    #             'std_dev': std_dev
    #         },
    #         'some_list': [1, 2, 3],
    #         'a_numpy_array': np.array([
    #             [1, 2, 3],
    #             [4, 5, 6]
    #         ])
    #     }
    #     metadata = {
    #         'source_readout': str(readout_path),
    #         'calculation_type': 'NRMSE_with_components',
    #         'random_example_metadata': 'hello_world'
    #     }
        
    #     return metrics, metadata

    # benchmark = CustomBenchmark(group_name="custom_nrmse_benchmark", benchmark_logic=nrmse_logic_with_metadata)
    # benchmark_args = {}     # Define benchmark arguments if you have any, otherwise you may delete this line or leave it blank
    
    # 4. Define the Visualizer
    visualizer = TimeSeriesComparison()
    
    # 5. Run the pipeline with the chosen components
    run_pipeline(exp_path, task, benchmark, benchmark_args, visualizer)