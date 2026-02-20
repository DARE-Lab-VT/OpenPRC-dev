import numpy as np
import h5py
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler

from openprc.analysis.benchmarks.base import BaseBenchmark
from openprc.analysis.tasks.imitation import memory_task
from openprc.reservoir.training.trainer import Trainer

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
                    - save_readouts_for (list[str]): A list of basis function names for which to
                                                     train and save the readout.
        Returns:
            The benchmark instance with populated metrics.
        """
        self._setup(trainer.experiment_dir)

        # 1. Get X_full from trainer
        X_full = trainer.features.transform(trainer.loader)
        scaler_X = StandardScaler()
        X_std = scaler_X.fit_transform(X_full)

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
        if len(X_std) < required_len:
            raise ValueError(
                f"Simulation too short! Need {required_len} frames "
                f"({washout_duration + train_duration + test_duration:.2f}s), "
                f"but simulation only has {len(X_std)} frames ({len(X_std) * dt:.2f}s)."
            )

        # 3. Run memory task
        results = memory_task(
            X=X_std,
            u_input=u_input,
            washout=washout_frames,
            train_stop=train_stop,
            test_duration=test_frames,
            tau_s=benchmark_args['tau_s'],
            n_s=benchmark_args['n_s'],
            k_delay=benchmark_args['k_delay'],
            eps=benchmark_args['eps']
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

        # 5. Train and save readouts if requested
        save_readouts_for = benchmark_args.get('save_readouts_for')
        if save_readouts_for:
            basis_names = results['basis_names']  # This is already a list of strings
            exponents = results['exponents']
            basis_to_exp = {name: exp for name, exp in zip(basis_names, exponents)}

            u = np.asarray(u_input, dtype=np.float32)
            tau_s = benchmark_args['tau_s']
            k_delay = benchmark_args['k_delay']
            max_lag = tau_s * k_delay
            
            U_full = sliding_window_view(u.flatten(), max_lag + 1)[:, ::-1]
            lag_indices = [j * k_delay for j in range(tau_s + 1)]
            U_sub = U_full[:, lag_indices]
            
            for basis_name in save_readouts_for:
                if basis_name not in basis_to_exp:
                    print(f"Warning: Basis function '{basis_name}' not found. Skipping.")
                    continue

                exp_vector = basis_to_exp[basis_name]
                y_target_full = np.prod(np.power(U_sub, exp_vector), axis=1)
                
                padding_size = len(u_input) - len(y_target_full)
                y_padded = np.pad(y_target_full, (padding_size, 0), 'constant', constant_values=0)
                y_reshaped = y_padded.reshape(-1, 1)

                print(f"Training and saving readout for: {basis_name}")
                safe_name = basis_name.replace(' ', '_').replace('^', 'p').replace('(', '').replace(')', '').replace('-', 'm')
                task_name = f"memory_{safe_name}"
                
                training_result = trainer.train(y_reshaped, task_name=task_name)
                training_result.save()

        return self
