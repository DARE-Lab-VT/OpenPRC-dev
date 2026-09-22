import numpy as np
import h5py
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler

from openprc.analysis.benchmarks.base import BaseBenchmark
from openprc.analysis.tasks.imitation import memory_task
from openprc.analysis.utils.training_utils import scale_iid_input, legendre_target
from openprc.reservoir.training.trainer import Trainer

class MemoryBenchmark(BaseBenchmark):
    """
    Computes the memory capacity and R2 score of predicting nonlinear combinations of past inputs.
    """
    def __init__(self, group_name: str = "memory_benchmark"):
        super().__init__(group_name)

    def run(self, trainer: Trainer, u_input: np.ndarray, *, prepared_states=None, **benchmark_args) -> 'MemoryBenchmark':
        """
        Runs the memory benchmark.

        Args:
            trainer (Trainer): The trainer object, pre-configured with a loader and features.
            u_input (np.ndarray): Original uniform IID symbols, one per feature row. Never pass
                an interpolated command or measured actuator waveform.
            prepared_states (np.ndarray, optional): Unstandardized feature matrix
                after example-level preprocessing, e.g. initial-position subtraction
                and multiplexing. If omitted, extract features from the trainer.
            benchmark_args (dict): Keyword arguments for the benchmark.
                Required:
                    - tau_s: max lag for inputs
                    - n_s: max degree of polynomial
                    - k_delay: delay step
                Optional:
                    - sample_dt (float): Seconds per transformed feature row.
                      Defaults to loader.dt; specify the IID interval after multiplexing.
                    - ridge (float): Ridge regression regularization. Default: 1e-6.
                    - save_readouts_for (list[str]): A list of basis function names for which to
                                                     train and save the readout.
        Returns:
            The benchmark instance with populated metrics.
        """
        allowed = {"tau_s", "n_s", "k_delay", "ridge", "input_bounds", "save_readouts_for", "sample_dt"}
        unexpected = set(benchmark_args) - allowed
        if unexpected:
            raise TypeError(f"Unsupported memory benchmark arguments: {sorted(unexpected)}")
        self._setup(trainer.experiment_dir)

        # 1. Use preprocessed features when supplied; otherwise extract normally.
        # Scaling and IPC training remain the same in both paths.
        if prepared_states is not None and benchmark_args.get('save_readouts_for'):
            raise ValueError(
                'Saved-readout retraining extracts trainer.features again; it cannot '
                'reproduce externally prepared_states. Use a matching feature pipeline.'
            )
        X_full = (trainer.features.transform(trainer.loader) if prepared_states is None
                  else np.asarray(prepared_states))

        # 2. Get params for memory_task from trainer and benchmark_args
        dt = benchmark_args.get('sample_dt', trainer.loader.dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('sample_dt must be positive and finite.')
        if benchmark_args.get('save_readouts_for') and not np.isclose(
            dt, trainer.loader.dt, rtol=1e-12, atol=0.0
        ):
            raise ValueError(
                'Saving readouts through Trainer requires its loader clock to match '
                'the transformed feature clock; sample_dt differs from loader.dt.'
            )
        washout_duration = trainer.washout
        train_duration = trainer.train_duration
        test_duration = trainer.test_duration

        def duration_frames(duration):
            if not np.isfinite(duration) or duration < 0:
                raise ValueError('Durations must be finite and nonnegative.')
            samples = duration / dt
            nearest = round(samples)
            # Do not lose a row when an integer duration ratio rounds just below
            # its true value (e.g. symbol counts converted to seconds and back).
            if abs(samples - nearest) <= 8 * np.finfo(float).eps * max(1., abs(samples)):
                return nearest
            return int(samples)

        washout_frames = duration_frames(washout_duration)
        train_frames = duration_frames(train_duration)
        test_frames = duration_frames(test_duration)
        
        train_stop = washout_frames + train_frames
        
        required_len = washout_frames + train_frames + test_frames
        if len(X_full) < required_len:
            raise ValueError(
                f"Simulation too short! Need {required_len} frames "
                f"({washout_duration + train_duration + test_duration:.2f}s), "
                f"but simulation only has {len(X_full)} frames ({len(X_full) * dt:.2f}s)."
            )

        # 3. Fit Scaling on Exactly the Rows Used by the IPC Regressions
        # Exclude initial washout and any additional unavailable input history.
        train_start = max(washout_frames, benchmark_args['tau_s'] * benchmark_args['k_delay'])
        if train_start >= train_stop or test_frames < 2:
            raise ValueError('Need nonempty training and at least two test rows after history/washout.')
        scaler_X = StandardScaler().fit(X_full[train_start:train_stop])
        # Keep original row indices for the delayed targets. Only fit(), above,
        # estimates statistics; transform() cannot learn from held-out rows.
        X_std = scaler_X.transform(X_full)

        # 4. Run memory task
        results = memory_task(
            X=X_std,
            u_input=u_input,
            washout=washout_frames,
            train_stop=train_stop,
            test_duration=test_frames,
            tau_s=benchmark_args['tau_s'],
            n_s=benchmark_args['n_s'],
            k_delay=benchmark_args['k_delay'],
            ridge=benchmark_args.get('ridge', 1e-6),
            input_bounds=benchmark_args.get('input_bounds', (-1., 1.))
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
            'sample_dt': dt,
            'washout': washout_duration,
            'train_duration': train_duration,
            'test_duration': test_duration,
            'feature_type': trainer.features.__class__.__name__,
            'input_bounds': benchmark_args.get('input_bounds', (-1., 1.)),
            'target_basis': 'orthonormal_legendre',
            'standardization': 'training_rows_only'
        }

        # 5. Train and save readouts if requested
        save_readouts_for = benchmark_args.get('save_readouts_for')
        if save_readouts_for:
            basis_names = results['basis_names']  # This is already a list of strings
            exponents = results['exponents']
            basis_to_exp = {name: exp for name, exp in zip(basis_names, exponents)}

            u = scale_iid_input(u_input, benchmark_args.get('input_bounds', (-1.,1.)))
            k_delay = benchmark_args['k_delay']
            max_lag = benchmark_args['tau_s'] * k_delay
            if washout_frames < max_lag:
                raise ValueError('Saving readouts requires washout >= maximum input history.')

            for basis_name in save_readouts_for:
                if basis_name not in basis_to_exp:
                    print(f"Warning: Basis function '{basis_name}' not found. Skipping.")
                    continue

                exp_vector = basis_to_exp[basis_name]
                y_target_full = legendre_target(u, exp_vector, k_delay)
                # Trainer removes washout; padding is never used for fitting.
                y_reshaped = np.nan_to_num(y_target_full).reshape(-1, 1)

                print(f"Training and saving readout for: {basis_name}")
                safe_name = basis_name.replace(' ', '_').replace('^', 'p').replace('(', '').replace(')', '').replace('-', 'm')
                task_name = f"memory_{safe_name}"
                
                training_result = trainer.train(y_reshaped, task_name=task_name)
                training_result.save()

        return self
