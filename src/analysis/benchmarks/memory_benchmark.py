import numpy as np
import h5py
from pathlib import Path
from .base import BaseBenchmark

class NARMABenchmark(BaseBenchmark):
    """
    Calculates the Normalized Root Mean Square Error (NRMSE) for a NARMA task.
    """
    def __init__(self):
        super().__init__()

    def run(self, experiment_dir: Path, order: int = 10) -> 'NARMABenchmark':
        """
        Runs the NARMA benchmark calculation.
        
        Args:
            experiment_dir (Path): The path to the experiment directory.
            order (int): The order of the NARMA task.
        
        Returns:
            The benchmark instance with populated metrics.
        """
        # Step 1: Initialize paths.
        self._setup(experiment_dir)
        
        # Step 2: Get data.
        readout_path = self.readout_path
        
        # Step 3: Perform calculation.
        with h5py.File(readout_path, 'r') as f:
            group = 'validation' if 'validation' in f else 'training'
            target = f[f'{group}/target'][:]
            prediction = f[f'{group}/prediction'][:]

        rmse = np.sqrt(np.mean((target - prediction)**2))
        std_dev = np.std(target)
        nrmse = rmse / (std_dev if std_dev > 1e-9 else 1.0)

        # Step 4: Populate metrics.
        self.metrics = {
            f'narma{order}_nrmse': nrmse
        }
        self.metadata = {
            'source_readout': str(self.readout_path),
            'narma_order': order
        }
        
        # Step 5: Return self.
        return self