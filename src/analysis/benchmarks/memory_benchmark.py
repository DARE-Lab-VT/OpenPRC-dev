import numpy as np
import h5py
from pathlib import Path

from .base import BaseBenchmark
from analysis.tasks.imitation import memory_task
from reservoir.training.trainer import Trainer

class MemoryBenchmark(BaseBenchmark):
    """
    Computes the memory capacity and R2 score of predicting nonlinear combinations of past inputs.
    """
    def __init__(self, group_name: str = "memory_benchmark"):
        super().__init__(group_name)

    def run(self, trainer: Trainer, u_input: np.ndarray, **benchmark_args) -> 'MemoryBenchmark':
        """
        Runs the memory benchmark.

        Args:
            trainer (Trainer): The trainer object, pre-configured with a loader and features.
            u_input (np.ndarray): The input signal for the memory task.
            benchmark_args (dict): Keyword arguments for the benchmark.
                Required:
                    - tau_s: max lag for inputs
                    - n_s: max degree of polynomial
                    - k_delay: delay step
                Optional:
                    - ridge (float): Ridge regression regularization. Default: 1e-6.

        Returns:
            The benchmark instance with populated metrics.
        """
        self._setup(trainer.experiment_dir)

        # 1. Get X_full from trainer
        X_full = trainer.features.transform(trainer.loader)

        # 2. Get params for memory_task from trainer and benchmark_args
        dt = trainer.loader.dt
        washout_duration = trainer.washout
        train_duration = trainer.train_duration
        test_duration = trainer.test_duration

        washout_frames = int(washout_duration / dt)
        train_frames = int(train_duration / dt)
        test_frames = int(test_duration / dt)
        
        train_stop = washout_frames + train_frames
        
        required_len = washout_frames + train_frames + test_frames
        if len(X_full) < required_len:
            raise ValueError(
                f"Simulation too short! Need {required_len} frames "
                f"({washout_duration + train_duration + test_duration:.2f}s), "
                f"but simulation only has {len(X_full)} frames ({len(X_full) * dt:.2f}s)."
            )

        # 3. Run memory task
        results = memory_task(
            X=X_full,
            u_input=u_input,
            washout=washout_frames,
            train_stop=train_stop,
            test_duration=test_frames,
            tau_s=benchmark_args['tau_s'],
            n_s=benchmark_args['n_s'],
            k_delay=benchmark_args['k_delay'],
            ridge=benchmark_args.get('ridge', 1e-6)
        )
        
        # 4. Populate metrics and metadata
        self.metrics = {
            'linear_memory_capacity': results['linear_memory_capacity'],
            'nonlinear_memory_capacity': results['nonlinear_memory_capacity'],
            'total_capacity': results['total_capacity'],
            'capacities': results['capacities'],
            'basis_names': np.array(results['basis_names'], dtype='S'), # HDF5 compatible
            'exponents': results['exponents'],
            'degrees': results['degrees']
        }
        
        self.metadata = {
            'tau_s': benchmark_args['tau_s'],
            'n_s': benchmark_args['n_s'],
            'k_delay': benchmark_args['k_delay'],
            'ridge': benchmark_args.get('ridge', 1e-6),
            'washout': washout_duration,
            'train_duration': train_duration,
            'test_duration': test_duration,
            'feature_type': trainer.features.__class__.__name__
        }

        return self
