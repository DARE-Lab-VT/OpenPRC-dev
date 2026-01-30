import h5py
import numpy as np
from pathlib import Path

class NARMABenchmarkResult:
    """
    Holds the score and saves to 'metrics/metrics.h5'.
    Schema Compliance:
    """
    def __init__(self, readout_path, nrmse, order):
        self.readout_path = Path(readout_path)
        self.nrmse = nrmse
        self.order = order

    def save(self):
        # Locate metrics directory parallel to readout directory
        experiment_root = self.readout_path.parent.parent
        metrics_dir = experiment_root / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_path = metrics_dir / "metrics.h5"
        
        print(f"Updating Metrics Report: {metrics_path}")
        
        # USE APPEND MODE ('a') to allow multiple tasks in one file
        with h5py.File(metrics_path, 'a') as f:
            # Update attributes
            f.attrs['source_readout'] = str(self.readout_path)
            
            # Create benchmark_suite group if it doesn't exist
            if 'benchmark_suite' not in f:
                suite_grp = f.create_group('benchmark_suite')
            else:
                suite_grp = f['benchmark_suite']
            
            # Save specific task metric (e.g., 'narma10_nrmse')
            metric_key = f'narma{self.order}_nrmse'
            
            # Overwrite if exists, otherwise create
            if metric_key in suite_grp:
                suite_grp[metric_key][...] = self.nrmse
            else:
                suite_grp.create_dataset(metric_key, data=self.nrmse)
            
        return metrics_path

class NARMABenchmark:
    def run(self, readout_path, order=10):
        readout_path = Path(readout_path)
        with h5py.File(readout_path, 'r') as f:
            group = 'validation' if 'validation' in f else 'training'
            target = f[f'{group}/target'][:]
            prediction = f[f'{group}/prediction'][:]

        rmse = np.sqrt(np.mean((target - prediction)**2))
        std_dev = np.std(target)
        nrmse = rmse / (std_dev if std_dev > 1e-9 else 1.0)

        return NARMABenchmarkResult(readout_path, nrmse, order)