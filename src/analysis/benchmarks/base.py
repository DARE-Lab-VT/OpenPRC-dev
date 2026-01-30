import h5py
import warnings
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseBenchmarkResult(ABC):
    """
    ### --- FIXED INFRASTRUCTURE --- ###
    This class handles the "Boring Stuff": 
    1. finding files
    2. validating directories
    3. writing generic dictionaries to disk
    """
    def __init__(self, experiment_dir: Path):
        self.experiment_dir = Path(experiment_dir)
        if not self.experiment_dir.is_dir():
            raise FileNotFoundError(f"Experiment directory not found: {self.experiment_dir}")

    # --- 1. DIRECTORY PROPERTIES (FIXED) ---
    @property
    def input_dir(self) -> Path: return self.experiment_dir / "input"
    @property
    def output_dir(self) -> Path: return self.experiment_dir / "output"
    @property
    def readout_dir(self) -> Path: return self.experiment_dir / "readout"
    @property
    def metrics_dir(self) -> Path: return self.experiment_dir / "metrics"
    
    # --- 2. FILE ACCESSORS (FIXED) ---
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
    def save_metric_to_suite(self, metrics: Dict[str, Any], metadata: Dict[str, Any] = None) -> Path:
        path = self.get_metrics_path()
        
        with h5py.File(path, 'a') as f:
            # --- INTELLIGENT METADATA HANDLING ---
            if metadata:
                for key, value in metadata.items():
                    # 1. If it's a list/tuple (Multiple Files), join them cleanly
                    if isinstance(value, (list, tuple)):
                        # Converts [Path('a'), Path('b')] -> "a, b"
                        f.attrs[key] = ", ".join(str(v) for v in value)
                    # 2. Otherwise, just save as string
                    else:
                        f.attrs[key] = str(value)
            
            # --- METRIC SAVING ---
            suite_grp = f.require_group('benchmark_suite')
            for key, value in metrics.items():
                if key in suite_grp: del suite_grp[key]
                suite_grp.create_dataset(key, data=value)
                
        return path

    @abstractmethod
    def save(self) -> Path:
        """
        User must implement this to map their specific variables (like nrmse)
        to the generic 'metrics' dictionary.
        """
        pass

class BaseBenchmark(ABC):
    @abstractmethod
    def run(self, experiment_dir: Path, **kwargs) -> BaseBenchmarkResult:
        pass