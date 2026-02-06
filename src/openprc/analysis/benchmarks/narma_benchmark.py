import numpy as np
import h5py
from pathlib import Path
from openprc.analysis.benchmarks.base import BaseBenchmark
from openprc.analysis.tasks.imitation import NARMA_task
from openprc.reservoir.training.trainer import Trainer


class NARMABenchmark(BaseBenchmark):
    """
    Calculates the Normalized Root Mean Square Error (NRMSE) for a NARMA task.
    This benchmark now handles the training process internally.
    """
    def __init__(self, group_name: str = "narma_benchmark"):
        super().__init__(group_name)

    @staticmethod
    def _generate_narma_task(u_input, order=2, a=0.3, b=0.05, c=1.5, d=0.1):
        """Generates the NARMA target signal."""
        return NARMA_task(u_input, order, a, b, c, d)

    def run(self, trainer: Trainer, u_input: np.ndarray, **benchmark_args) -> 'NARMABenchmark':
        """
        Runs the NARMA benchmark, including training and evaluation.
        
        Args:
            trainer (Trainer): The trainer object, pre-configured with a loader.
            u_input (np.ndarray): The input signal for the NARMA task.
            benchmark_args (dict): Keyword arguments for the benchmark.
                                   For NARMA, this should contain 'order'.
        
        Returns:
            The benchmark instance with populated metrics.
        """
        order = benchmark_args.get('order', 10)

        # Step 1: Initialize paths and setup from the trainer.
        self._setup(trainer.experiment_dir)
        
        # Step 2: Generate the task using the provided u_input.
        y_full = self._generate_narma_task(u_input, order=order)
        
        # Step 3: Train the readout.
        task_name = f"NARMA{order}"
        training_result = trainer.train(y_full, task_name=task_name)
        training_result.save()
        
        # Step 4: Calculate NRMSE from the training result's test cache.
        _, target, prediction = training_result.cache['test']

        rmse = np.sqrt(np.mean((target - prediction)**2))
        std_dev = np.std(target)
        nrmse = rmse / (std_dev if std_dev > 1e-9 else 1.0)

        # Step 5: Populate metrics and metadata.
        self.metrics = {
            f'narma{order}_nrmse': nrmse
        }
        self.metadata = {
            'source_readout': str(self.readout_path),
            'narma_order': order,
            'training_feature_config': training_result.feature_config
        }
        
        # Step 6: Return self.
        return self