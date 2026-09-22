"""Training-only scaler checks, including saved preprocessing and IPC history."""
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from openprc.reservoir.training.trainer import Trainer
from openprc.reservoir.readout.ridge import Ridge
from openprc.analysis.benchmarks.memory_benchmark import MemoryBenchmark
from openprc.analysis.tasks.imitation import memory_task


class TrainingScalingChecks(unittest.TestCase):
    def setUp(self):
        self.states = np.random.default_rng(34).normal(size=(120, 3))
        self.states[:, 2] = 2.0  # Constant feature should remain finite.
        self.changed = self.states.copy()
        self.changed[:20] += 1000
        self.changed[70:] -= 500

    def features(self, states):
        return SimpleNamespace(transform=lambda loader: states,
                               get_feature_info=lambda loader: None)

    def test_trainer_ignores_washout_test_and_trailing_rows(self):
        targets = (self.states[:, 0] * 2 - self.states[:, 1])[:, None]
        with tempfile.TemporaryDirectory() as folder:
            loader = SimpleNamespace(dt=1., sim_path='synthetic.h5')
            results = []
            for states in (self.states, self.changed):
                trainer = Trainer(self.features(states), Ridge(), folder, loader,
                                  washout=20., train_duration=50., test_duration=30.)
                results.append(trainer.train(targets))
            a, b = results
            expected = StandardScaler().fit(self.states[20:70])
            for result in results:
                np.testing.assert_allclose(result.scaler_params['X_mean'], expected.mean_)
                np.testing.assert_allclose(result.scaler_params['X_scale'], expected.scale_)
            np.testing.assert_array_equal(a.cache['train'][0], b.cache['train'][0])
            np.testing.assert_array_equal(a.readout.weights, b.readout.weights)
            np.testing.assert_allclose(b.cache['test'][0][:, 1:], expected.transform(self.changed[70:100]))
            with h5py.File(b.save()) as f:
                np.testing.assert_array_equal(f['preprocessing/X_mean'][:], expected.mean_)
                np.testing.assert_array_equal(f['preprocessing/X_scale'][:], expected.scale_)

    def test_benchmark_scaler_excludes_additional_input_history(self):
        u = np.random.default_rng(42).uniform(-1, 1, 120)
        captures = []
        def capture(**kwargs):
            captures.append(kwargs['X'].copy())
            return memory_task(**kwargs)
        with tempfile.TemporaryDirectory() as folder:
            for states in (self.states, self.changed):
                # Nominal washout is 10, but max input history is 20.
                trainer = Trainer(self.features(states), Ridge(), folder,
                                  SimpleNamespace(dt=1.), washout=10.,
                                  train_duration=60., test_duration=30.)
                with patch('openprc.analysis.benchmarks.memory_benchmark.memory_task', side_effect=capture):
                    result = MemoryBenchmark().run(trainer, u, tau_s=20, n_s=1, k_delay=1)
                self.assertEqual(result.metadata['standardization'], 'training_rows_only')
            expected = StandardScaler().fit(self.states[20:70])
            np.testing.assert_array_equal(captures[0][20:70], captures[1][20:70])
            np.testing.assert_allclose(captures[1], expected.transform(self.changed))
            np.testing.assert_allclose(captures[0][20:70].mean(axis=0), 0, atol=1e-15)


if __name__ == '__main__':
    unittest.main()
