import numpy as np
from pathlib import Path
from typing import Callable, Dict, Any, Tuple
from .base import BaseBenchmark
from reservoir.utils.trainer import Trainer


class CustomBenchmark(BaseBenchmark):
    """
    ### --- User-Defined Custom Benchmark --- ###

    A flexible benchmark that executes a user-provided logic function, while
    maintaining a `run` method signature consistent with other benchmarks.

    The user is expected to define a function that implements the entire
    benchmark logic. This function is passed to the constructor and will be
    called by the `run` method, receiving the necessary context (like the
    trainer and input signal) to execute the benchmark workflow.
    """
    def __init__(self, group_name: str, benchmark_logic: Callable[..., Tuple[Dict[str, Any], Dict[str, Any]]]):
        super().__init__(group_name)
        if not callable(benchmark_logic):
            raise TypeError("The 'benchmark_logic' provided must be a callable function.")
        self.benchmark_logic = benchmark_logic

    def run(self, trainer: Trainer, u_input: np.ndarray, **kwargs) -> 'CustomBenchmark':
        """
        Executes the user-defined benchmark logic.

        Args:
            trainer (Trainer): The trainer object, which will be passed to the logic function.
            u_input (np.ndarray): The input signal, which will be passed to the logic function.
            **kwargs: Custom arguments that will be passed to the `benchmark_logic` function.
        """
        # Step 1: Initialize paths from the trainer, aligning with other benchmarks.
        self._setup(trainer.experiment_dir)

        # Step 2: Execute the user-provided logic function, passing all available context.
        result = self.benchmark_logic(self, trainer, u_input, **kwargs)

        # Step 3: Validate the return type from the user's function.
        if not (isinstance(result, tuple) and len(result) == 2 and
                isinstance(result[0], dict) and isinstance(result[1], dict)):
            raise TypeError("The benchmark logic function must return a tuple of (metrics, metadata), "
                            "where both are dictionaries.")
        
        metrics, metadata = result

        # Step 4: Populate the instance with the results from the user's function.
        self.metrics = metrics
        self.metadata = metadata
        
        # Step 5: Return the instance itself.
        return self