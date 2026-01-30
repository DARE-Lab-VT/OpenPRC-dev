import numpy as np
import h5py
from pathlib import Path
from .base import BaseBenchmark, BaseBenchmarkResult

class NARMABenchmarkResult(BaseBenchmarkResult):
    """
    ### --- USER DEFINED SECTION --- ###
    Here is where you define WHAT you are saving.
    If you create a new benchmark (e.g. Memory Capacity), you change:
    1. The __init__ arguments (your specific scores)
    2. The save() dictionary mapping
    """
    
    # 1. DEFINE YOUR DATA VARIABLES
    # 'nrmse' and 'order' are specific to NARMA. 
    # If this was 'MemoryBenchmark', you might have 'memory_score' and 'lag' here.
    def __init__(self, experiment_dir: Path, nrmse: float = None, order: int = None):
        super().__init__(experiment_dir)
        self.nrmse = nrmse      # User-defined
        self.order = order      # User-defined

    # 2. MAP YOUR DATA TO THE SAVE HELPER
    def save(self) -> Path:
        return self.save_metric_to_suite(
            metrics={
                f'narma{self.order}_nrmse': self.nrmse  # User can add more metrics by appending this dictionary
            },
            metadata={
                'source_readout': self.readout_path     # User can add more metadata by appending this dictionary
            }
        )


class NARMABenchmark(BaseBenchmark):
    """
    ### --- USER DEFINED LOGIC --- ###
    """
    def run(self, experiment_dir: Path, order: int = 10) -> NARMABenchmarkResult:
        # A. Context: Create the result object just to use the FIXED path helpers
        context = NARMABenchmarkResult(experiment_dir)
        
        # B. Get Data: Use the fixed helper properties
        # There are options like: geometry_path, signal_path, readout_path, and simulation_path. User can call it as follows:
        readout_path = context.readout_path
        
        # C. Do Math: This is your custom logic
        with h5py.File(readout_path, 'r') as f:
            group = 'validation' if 'validation' in f else 'training'
            target = f[f'{group}/target'][:]
            prediction = f[f'{group}/prediction'][:]

        rmse = np.sqrt(np.mean((target - prediction)**2))
        std_dev = np.std(target)
        nrmse = rmse / (std_dev if std_dev > 1e-9 else 1.0)

        # D. Return Result: Populate the USER-DEFINED variables
        return NARMABenchmarkResult(
            experiment_dir=experiment_dir, 
            nrmse=nrmse,  # <--- Placing your calculated data into your container
            order=order
        )