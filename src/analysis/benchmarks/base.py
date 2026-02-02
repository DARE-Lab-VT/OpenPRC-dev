import h5py
import warnings
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseBenchmark(ABC):
    """
    ### --- FIXED INFRASTRUCTURE --- ###
    This base class provides the core utilities for creating benchmarks.
    
    A concrete benchmark should inherit from this class and implement the
    abstract `run` method. This base class provides:
    - A standard constructor.
    - A `_setup` method to be called inside `run` to initialize paths.
    - File and directory path properties.
    - A `save` method to persist metrics.
    """
    def __init__(self):
        self.experiment_dir: Path | None = None
        self.metrics: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}

    def _setup(self, experiment_dir: Path):
        """Initializes the experiment directory and validates it."""
        self.experiment_dir = Path(experiment_dir)
        if not self.experiment_dir.is_dir():
            raise FileNotFoundError(f"Experiment directory not found: {self.experiment_dir}")

    # --- 1. DIRECTORY PROPERTIES (Dynamically available after `_setup` is called) ---
    @property
    def input_dir(self) -> Path: 
        if self.experiment_dir is None: raise RuntimeError("Accessing 'input_dir' before 'run' has called '_setup'.")
        return self.experiment_dir / "input"
    @property
    def output_dir(self) -> Path: 
        if self.experiment_dir is None: raise RuntimeError("Accessing 'output_dir' before 'run' has called '_setup'.")
        return self.experiment_dir / "output"
    @property
    def readout_dir(self) -> Path: 
        if self.experiment_dir is None: raise RuntimeError("Accessing 'readout_dir' before 'run' has called '_setup'.")
        return self.experiment_dir / "readout"
    @property
    def metrics_dir(self) -> Path: 
        if self.experiment_dir is None: raise RuntimeError("Accessing 'metrics_dir' before 'run' has called '_setup'.")
        return self.experiment_dir / "metrics"
    
    # --- 2. FILE ACCESSORS (Dynamically available after `_setup` is called) ---
    @property
    def geometry_path(self) -> Path:
        path = self.input_dir / "geometry.h5"
        if not path.exists(): raise FileNotFoundError(f"geometry.h5 not found in {self.input_dir}")
        return path
    
    @property
    def signal_path(self) -> Path:
        path = self.input_dir / "signals.h5"
        if not path.exists(): raise FileNotFoundError(f"signals.h5 not found in {self.input_dir}")
        return path

    @property
    def simulation_path(self) -> Path:
        path = self.output_dir / "simulation.h5"
        if not path.exists(): raise FileNotFoundError(f"simulation.h5 not found in {self.output_dir}")
        return path

    @property
    def readout_path(self) -> Path:
        files = list(self.readout_dir.glob("readout_*.h5"))
        if not files: raise FileNotFoundError(f"No readout file found in {self.readout_dir}")
        if len(files) > 1:
            warnings.warn(f"Multiple readouts found: {[f.name for f in files]}. Defaulting to {files[0].name}")
        return files[0]

    def get_metrics_path(self, filename: str = "metrics.h5") -> Path:
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        return self.metrics_dir / filename

    # --- 3. THE GENERIC SAVER (FIXED) ---
    def save(self, filename: str = "metrics.h5") -> Path:
        if self.experiment_dir is None:
            raise RuntimeError("Cannot save before running the benchmark, as `experiment_dir` is not set.")
        
        path = self.get_metrics_path(filename)
        
        with h5py.File(path, 'a') as f:
            if self.metadata:
                for key, value in self.metadata.items():
                    if isinstance(value, (list, tuple)):
                        f.attrs[key] = ", ".join(str(v) for v in value)
                    else:
                        f.attrs[key] = str(value)
            
            results_grp = f.require_group('benchmark_results')
            
            def _recursive_save(group, data_dict):
                for key, value in data_dict.items():
                    if key in group:
                        del group[key]
                    
                    if isinstance(value, dict):
                        new_group = group.create_group(key)
                        _recursive_save(new_group, value)
                    else:
                        try:
                            group.create_dataset(key, data=value)
                        except TypeError:
                            # As a fallback for simple non-numpy types, save as string
                            group.create_dataset(key, data=str(value))

            _recursive_save(results_grp, self.metrics)
                
        return path

    @abstractmethod
    def run(self, experiment_dir: Path, **kwargs) -> 'BaseBenchmark':
        """
        User must implement this method to set up the experiment,
        run the benchmark calculation, populate self.metrics and self.metadata,
        and return the instance.
        """
        pass