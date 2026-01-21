"""
demlat Base Model
=================
Abstract interface that all physics solvers must implement.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """
    The contract for physics solvers.

    Attributes:
        n_nodes (int): Number of particles.
        precision (dtype): float32 or float64.
        backend (str): 'cpu' or 'cuda'.
    """

    def __init__(self, experiment, backend: str = 'cpu', precision: str = 'float32'):
        self.exp = experiment
        self.backend = backend
        self.dtype = np.float32 if precision == 'float32' else np.float64

        # Load initial state from experiment geometry
        # (Concrete implementations will upload this to GPU/Optimize layout)
        self.n_nodes = self._load_initial_state()

    def _load_initial_state(self) -> int:
        """Helper to verify geometry exists and return node count."""
        # Using the IOManager indirectly via the experiment object would be cleaner,
        # but for now we trust the experiment path logic.
        import h5py
        with h5py.File(self.exp.paths['geometry'], 'r') as f:
            return f['nodes/positions'].shape[0]

    @abstractmethod
    def step(self, t: float, dt: float, actuation_state: dict):
        """
        Advances the simulation by one time step.

        Args:
            t: Current simulation time.
            dt: Time delta.
            actuation_state: Dict mapping node indices to actuation vectors (forces/positions).

        Returns:
            dict: The current state {'positions': (N,3), 'velocities': (N,3), ...}
        """
        pass

    @abstractmethod
    def reset(self):
        """Resets the solver to t=0 state."""
        pass
