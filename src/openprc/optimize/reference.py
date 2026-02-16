"""
Reference Data Loader
======================

Loads reference trajectories from DEMLAT simulation.h5 files
for use as calibration targets.
"""

import h5py
import numpy as np
import jax.numpy as jnp
from pathlib import Path


class ReferenceData:
    """
    Loads and holds a reference trajectory from a DEMLAT simulation.h5 file.

    The reference is used as the ground truth that the optimizer tries
    to match by tuning theta.
    """

    def __init__(self, path):
        """
        Parameters
        ----------
        path : str or Path
            Path to simulation.h5 (the output of a DEMLAT run).
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Reference file not found: {self.path}")

        self._load()

    def _load(self):
        """Read positions, time, and metadata from simulation.h5."""
        with h5py.File(self.path, 'r') as f:
            # Time array
            self.time = np.array(f['time_series/time'])
            self.n_frames = len(self.time)

            # Positions: (T, N, 3)
            self.positions = np.array(f['time_series/nodes/positions'])
            self.n_nodes = self.positions.shape[1]

            # Velocities (if available)
            if 'time_series/nodes/velocities' in f:
                self.velocities = np.array(f['time_series/nodes/velocities'])
            else:
                self.velocities = None

            # Bar strains (if available)
            if 'time_series/elements/bars/strain' in f:
                self.strains = np.array(f['time_series/elements/bars/strain'])
            else:
                self.strains = None

            # Initial state (frame 0)
            self.x0 = self.positions[0]
            self.v0 = self.velocities[0] if self.velocities is not None \
                else np.zeros_like(self.x0)

            # Metadata
            self.dt_save = float(self.time[1] - self.time[0]) if self.n_frames > 1 else 0.0

    def get_trajectory_jax(self, start_frame=0, end_frame=None, node_subset=None):
        """
        Return reference positions as a JAX array.

        Parameters
        ----------
        start_frame : int
            First frame index (inclusive).
        end_frame : int or None
            Last frame index (exclusive).  None = all.
        node_subset : array-like or None
            Indices of nodes to include.  None = all.

        Returns
        -------
        x_ref : jnp.array  (T, N, 3) or (T, N_sub, 3)
        """
        end = end_frame or self.n_frames
        pos = self.positions[start_frame:end]
        if node_subset is not None:
            pos = pos[:, node_subset, :]
        return jnp.array(pos, dtype=jnp.float32)

    def get_initial_state_jax(self):
        """Return (x0, v0) as JAX arrays."""
        return (
            jnp.array(self.x0, dtype=jnp.float32),
            jnp.array(self.v0, dtype=jnp.float32),
        )

    def get_time_jax(self):
        return jnp.array(self.time, dtype=jnp.float32)

    def __repr__(self):
        return (
            f"ReferenceData({self.path.name}: "
            f"{self.n_frames} frames, {self.n_nodes} nodes, "
            f"dt_save={self.dt_save:.4f}s)"
        )