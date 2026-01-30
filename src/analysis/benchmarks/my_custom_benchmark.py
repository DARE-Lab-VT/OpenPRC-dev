import numpy as np
import h5py
from pathlib import Path
from .base import BaseBenchmark, BaseBenchmarkResult

class MyBenchmarkResult(BaseBenchmarkResult):
    """
    ### --- USER DEFINED SECTION --- ###
    Here is where you define WHAT you are saving.
    If you create a new benchmark (e.g. Memory Capacity), you change:
    1. The __init__ arguments (your specific scores)
    2. The save() dictionary mapping
    """
    
    # 1. DEFINE YOUR DATA VARIABLES
    # 'my_score' is specific to this benchmark. 
    # If this was 'MemoryBenchmark', you might have 'memory_score' and 'lag' here.
    def __init__(self, experiment_dir: Path, my_score: float = None):
        super().__init__(experiment_dir)
        self.my_score = my_score      # User-defined

    # 2. MAP YOUR DATA TO THE SAVE HELPER
    def save(self) -> Path:
        return self.save_metric_to_suite(
            metrics={
                'my_benchmark_score': self.my_score  # User can add more metrics by appending this dictionary
            },
            metadata={
                'source_readout': self.readout_path     # User can add more metadata by appending this dictionary
            }
        )


class MyBenchmark(BaseBenchmark):
    """
    ### --- USER DEFINED LOGIC --- ###
    """
    def run(self, experiment_dir: Path, **kwargs) -> MyBenchmarkResult:
        # A. Context: Create the result object just to use the FIXED path helpers
        context = MyBenchmarkResult(experiment_dir)
        
        # B. Get Data: Use the fixed helper properties
        # There are options like: geometry_path, signal_path, readout_path, and simulation_path. User can call it as follows:
        readout_path = context.readout_path
        
        # C. Do Math: This is your custom logic
        calculated_score = 0.0 # Default value
        
        if readout_path.exists():
            with h5py.File(readout_path, 'r') as f:
                group = 'validation' if 'validation' in f else 'training'
                
                # Example of loading data:
                # target = f[f'{group}/target'][:]
                # prediction = f[f'{group}/prediction'][:]
                
                # Perform your calculation here...
                calculated_score = 0.99 # Placeholder result

        # D. Return Result: Populate the USER-DEFINED variables
        return MyBenchmarkResult(
            experiment_dir=experiment_dir, 
            my_score=calculated_score  # <--- Placing your calculated data into your container
        )
