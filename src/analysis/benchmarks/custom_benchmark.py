from pathlib import Path
from typing import Callable, Dict, Any, Tuple
from .base import BaseBenchmark

class CustomBenchmark(BaseBenchmark):
    """
    ### --- User-Defined Custom Benchmark --- ###

    This class allows for the creation of a benchmark by providing a single
    function during initialization, instead of requiring class inheritance.

    **Instructions:**
    1.  Define a function that contains your benchmark logic.
        - This function must accept the benchmark instance as its first argument.
        - It must return a tuple containing two dictionaries: `(metrics, metadata)`.
    2.  Instantiate `CustomBenchmark`, passing your logic function to the constructor.

    **Example Pipeline Workflow:**
    ```python
    def my_logic(benchmark, **kwargs):
        # ... perform calculations ...
        metrics = {"my_metric": 0.99}
        metadata = {"source_readout": str(benchmark.readout_path)}
        return metrics, metadata

    benchmark = CustomBenchmark(group_name="my_custom_benchmark", benchmark_logic=my_logic)
    results = benchmark.run(Path("/path/to/experiment"))
    results.save()
    print(results.metrics)
    ```
    """
    def __init__(self, group_name: str, benchmark_logic: Callable[..., Tuple[Dict[str, Any], Dict[str, Any]]]):
        super().__init__(group_name)
        if not callable(benchmark_logic):
            raise TypeError("The 'benchmark_logic' provided must be a callable function.")
        self.benchmark_logic = benchmark_logic

    def run(self, experiment_dir: Path, **kwargs) -> 'CustomBenchmark':
        """
        This method implements the full benchmark logic by calling the user-defined function.
        """
        # Step 1: Initialize paths and validate the directory.
        self._setup(experiment_dir)

        # Step 2: Execute the user-provided logic function.
        result = self.benchmark_logic(self, **kwargs)

        # Step 3: Validate the return type and unpack.
        if not (isinstance(result, tuple) and len(result) == 2):
            raise TypeError("The benchmark logic function must return a tuple of (metrics, metadata).")
        
        metrics, metadata = result
        
        if not isinstance(metrics, dict):
            raise TypeError("The first element returned from the logic function must be a metrics dictionary.")
        if not isinstance(metadata, dict):
            raise TypeError("The second element returned from the logic function must be a metadata dictionary.")

        # Step 4: Populate the instance with the results.
        self.metrics = metrics
        self.metadata = metadata
        
        # Step 5: Return the instance itself.
        return self