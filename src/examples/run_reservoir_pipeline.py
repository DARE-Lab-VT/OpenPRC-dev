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
from reservoir.features.bar_features import BarExtensions, BarLengths
from reservoir.readout.ridge import Ridge
from reservoir.training.trainer import Trainer
from analysis.visualization.time_series import TimeSeriesComparison
from analysis.tasks.memory import NARMA
from analysis.benchmarks.memory_benchmark import NARMABenchmark
from analysis.benchmarks.custom_benchmark import CustomBenchmark


def main():
    """
    A generic pipeline that supports two optional workflows:
    1. 'train': Directly trains a readout for a specific task.
    2. 'benchmark': Runs a benchmark which handles its own training.
    
    Please comment out the workflow you do not wish to run.
    """
    
    # 1. Define the Experiment Path
    experiment_subpath = "spring_mass_3x3_test/generation_0"
    experiments_dir = src_dir / "experiments"
    experiment_dir = experiments_dir / experiment_subpath
    sim_path = experiment_dir / "output" / "simulation.h5"

    if not sim_path.exists():
        print(f"Simulation missing at {sim_path}!"); return
    
    # 2. Shared Setup
    loader = StateLoader(sim_path)
    features = NodePositions()
    u_input = loader.get_actuation_signal(actuator_idx=0, dof=0)
    
    print(f"Loaded {loader.total_frames} frames from {sim_path.name}")

    # --- Workflow 1: Train readout weights only ---
    # Comment out this block to run Workflow 2
    if True:
        print("\n[Workflow: Train]")
        
        # 3. Define the Task
        task_order = 2
        task_name = f"NARMA{task_order}"
        y_full = NARMA(u_input, order=task_order)

        # 4. Define Trainer and Train
        trainer = Trainer(
            loader=loader, 
            features=features,
            readout=Ridge(1e-5),
            experiment_dir=experiment_dir,
            washout=5.0,
            train_duration=20.0,
            test_duration=5.0,
        )
        
        result = trainer.train(y_full, task_name=task_name) 
        readout_path = result.save()
        
        print(f"--- Training complete for: {experiment_dir.name} ---")

        # 5. Visualize
        visualizer = TimeSeriesComparison()
        if readout_path and visualizer:
            print("\n[Processing] Visualizing results")
            plot_path = visualizer.plot(readout_path, start_frame=0, end_frame=500).save()
            print(f" >> Plot saved to: {plot_path}")


    # --- Workflow 2: Run Benchmark (handles training automatically) ---
    # Uncomment this block to run this workflow
    if True:
        print("\n[Workflow: Benchmark]")
        
        # 3. Define Trainer and Benchmark
        trainer = Trainer(
            loader=loader,
            features=features,
            readout=Ridge(1e-5),
            experiment_dir=experiment_dir,
            washout=5.0,
            train_duration=20.0,
            test_duration=5.0,
        )
        benchmark = NARMABenchmark(group_name="narma_benchmark")
        benchmark_args = {"order": 2}
        
        # 4. Run benchmark
        print(f"Running benchmark: {benchmark.__class__.__name__}")
        score = benchmark.run(trainer, u_input, **benchmark_args)
        score.save()
        
        print(f"--- Benchmark complete for: {experiment_dir.name} ---")

        # 5. Visualize
        visualizer = TimeSeriesComparison()
        if score.readout_path and visualizer:
            print("\n[Processing] Visualizing results")
            plot_path = visualizer.plot(score.readout_path, start_frame=0, end_frame=500).save()
            print(f" >> Plot saved to: {plot_path}")

        # 6. Print Metrics
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


    # --- Workflow 3: Run a Custom Benchmark (illustrates usage) ---
    # This workflow shows how to use the CustomBenchmark by defining the logic
    # directly within the pipeline. It replicates the NARMA benchmark for demonstration.
    # To run, uncomment this block and comment out other workflows.
    if True:
        print("\n[Workflow: Custom Benchmark]")

        # 3. Define the benchmark logic function
        def custom_narma_logic(benchmark, trainer, u_input, **kwargs):
            """This function contains the full logic for the custom benchmark."""
            # Get task-specific parameters from keyword arguments
            order = kwargs.get('order', 2)

            # Generate the NARMA task data, same as in the standard benchmark
            y_full = NARMA(u_input, order=order)

            # Use the provided trainer to run the training process
            task_name = f"CustomNARMA{order}"
            training_result = trainer.train(y_full, task_name=task_name)
            training_result.save()

            # Calculate the NRMSE metric from the test results
            _, target, prediction = training_result.cache['test']
            rmse = np.sqrt(np.mean((target - prediction)**2))
            std_dev = np.std(target)
            nrmse = rmse / (std_dev if std_dev > 1e-9 else 1.0)
            
            # Define the metrics and metadata dictionaries to be returned
            metrics = {f'custom_narma{order}_nrmse': nrmse}
            metadata = {
                'source_readout': str(benchmark.readout_path),
                'narma_order': order,
                'training_feature_config': training_result.feature_config,
                'benchmark_class': benchmark.__class__.__name__
            }
            
            return metrics, metadata

        # 4. Define the Trainer, which is shared context for the benchmark
        trainer = Trainer(
            loader=loader,
            features=features,
            readout=Ridge(1e-5),
            experiment_dir=experiment_dir,
            washout=5.0,
            train_duration=20.0,
            test_duration=5.0,
        )

        # 5. Instantiate the CustomBenchmark, passing the logic function
        benchmark = CustomBenchmark(
            group_name="custom_narma_benchmark",
            benchmark_logic=custom_narma_logic
        )
        benchmark_args = {"order": 2}

        # 6. Run the benchmark
        print(f"Running benchmark: {benchmark.__class__.__name__}")
        score = benchmark.run(trainer, u_input, **benchmark_args)
        score.save()
        
        print(f"--- Benchmark complete for: {experiment_dir.name} ---")

        # 7. Print Metrics
        if score.metrics:
            print("\n[Processing] Printing Benchmark Results:")
            # This helper function is defined locally for clarity
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