import h5py
import numpy as np
from pathlib import Path

from .base import BaseBenchmark, BaseBenchmarkResult

class NARMABenchmarkResult(BaseBenchmarkResult):
    """
    Holds the NARMA score and saves it to the standard metrics file.
    """
    def __init__(self, experiment_dir: Path, nrmse: float, order: int):
        super().__init__(experiment_dir)
        self.nrmse = nrmse
        self.order = order

    def save(self) -> Path:
        """Saves the narma{order}_nrmse metric to the metrics.h5 file."""
        metrics_path = self.get_metrics_path()
        
        print(f"Updating Metrics Report: {metrics_path}")
        
        # USE APPEND MODE ('a') to allow multiple benchmarks/tasks in one file
        with h5py.File(metrics_path, 'a') as f:
            # Update attributes
            f.attrs['source_readout'] = str(self.readout_path)
            
            # Create benchmark_suite group if it doesn't exist
            suite_grp = f.require_group('benchmark_suite')
            
            # Save specific task metric (e.g., 'narma10_nrmse')
            metric_key = f'narma{self.order}_nrmse'
            
            # Overwrite if exists, otherwise create
            if metric_key in suite_grp:
                suite_grp[metric_key][...] = self.nrmse
            else:
                suite_grp.create_dataset(metric_key, data=self.nrmse)
            
        return metrics_path


class NARMABenchmark(BaseBenchmark):
    """Benchmark for calculating the Normalized Root Mean Square Error (NRMSE) for a NARMA task."""
    
    def run(self, experiment_dir: Path, order: int = 10) -> NARMABenchmarkResult:
        """
        Calculates the NRMSE from a readout artifact in the experiment directory.

        Args:
            experiment_dir (Path): Path to the experiment directory.
            order (int): The order of the NARMA task being evaluated.

        Returns:
            NARMABenchmarkResult: A result object containing the NRMSE score.
        """
        # The result object has a helper property to find the readout file
        temp_result_for_path = NARMABenchmarkResult(experiment_dir, 0.0, 0)
        readout_path = temp_result_for_path.readout_path
        
        with h5py.File(readout_path, 'r') as f:
            # Prefer to score on validation data if it exists
            group = 'validation' if 'validation' in f else 'training'
            target = f[f'{group}/target'][:]
            prediction = f[f'{group}/prediction'][:]

        rmse = np.sqrt(np.mean((target - prediction)**2))
        std_dev = np.std(target)
        
        # Avoid division by zero if the target signal is flat
        nrmse = rmse / (std_dev if std_dev > 1e-9 else 1.0)

        return NARMABenchmarkResult(experiment_dir, nrmse, order)