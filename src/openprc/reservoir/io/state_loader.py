import h5py
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d

class StateLoader:
    def __init__(self, sim_path):
        self.sim_path = Path(sim_path)
        if not self.sim_path.exists():
            raise FileNotFoundError(f"Simulation not found: {self.sim_path}")

        with h5py.File(self.sim_path, 'r') as f:
            if 'time_series' not in f:
                raise ValueError("Invalid simulation.h5 structure")
            
            # 1. Load Time Array
            self.time = f['time_series/time'][:]
            self.total_frames = self.time.shape[0]
            
            # 2. Get dt (Time Step) from attributes or calculate it
            # Different schema versions might store it in different places, 
            # so we check attributes first, then fallback to calculation.
            if 'dt' in f.attrs:
                self.dt = float(f.attrs['dt'])
            elif 'save_interval' in f.attrs:
                self.dt = float(f.attrs['save_interval'])
            elif 'simulation' in f and 'dt' in f['simulation'].attrs:
                self.dt = float(f['simulation'].attrs['dt'])
            else:
                # Fallback: Calculate from the first two frames
                self.dt = float(self.time[1] - self.time[0])

            # Paths
            base_dir = self.sim_path.parent.parent 
            self.input_dir = base_dir / "input"
            self.signal_path = self.input_dir / "signals.h5"

    def get_node_positions(self, node_ids="all", dims="all", reshape_output=True):
        with h5py.File(self.sim_path, 'r') as f:
            data = f['time_series/nodes/positions']
            
            nodes_slice = slice(None) if node_ids == "all" else node_ids
            dims_slice = slice(None) if dims == "all" else dims

            positions = data[:, nodes_slice, dims_slice]
            
            if reshape_output:
                return positions.reshape(self.total_frames, -1)
            return positions

    def get_bar_lengths(self, bar_ids="all"):
        with h5py.File(self.sim_path, 'r') as f:
            data = f['time_series/elements/bars/lengths']
            if bar_ids == "all":
                return data[:].reshape(self.total_frames, -1)
            return data[:, bar_ids].reshape(self.total_frames, -1)

    def get_bar_extensions(self, bar_ids="all"):
        with h5py.File(self.sim_path, 'r') as f:
            lengths_t = f['time_series/elements/bars/lengths'][:]
            initial_lengths = lengths_t[0] # Lengths at t=0
            extensions = lengths_t - initial_lengths
            
            if bar_ids == "all":
                return extensions.reshape(self.total_frames, -1)
            return extensions[:, bar_ids].reshape(self.total_frames, -1)

    def get_actuation_signal(self, actuator_idx=0, dof=None):
        """
        Loads a specific actuation signal from simulation.h5 based on the new
        data structure where signals are stored by node ID.
        
        Args:
            actuator_idx (int): The index of the actuator signal to load, based on the
                                sorted order of actuated node IDs.
            dof (int, optional): The degree of freedom (0=x, 1=y, 2=z) to extract.
                                 If None, the full 3D signal is returned.
        """
        try:
            with h5py.File(self.sim_path, 'r') as f:
                if 'time_series/actuation_signals' in f:
                    act_group = f['time_series/actuation_signals']
                    # Get node IDs (stored as strings) and sort them numerically
                    node_ids = sorted([int(k) for k in act_group.keys()])
                    
                    if actuator_idx < 0 or actuator_idx >= len(node_ids):
                        # print(f"Actuator index {actuator_idx} out of bounds.")
                        return np.zeros(self.total_frames)

                    target_node_id = str(node_ids[actuator_idx])
                    signal_data = act_group[target_node_id][:]
                    
                    if dof is not None and dof < signal_data.shape[1]:
                        return signal_data[:, dof]
                    else:
                        return signal_data
                else:
                    # Fallback for older format or if no signals are saved
                    return np.zeros(self.total_frames)
        except Exception:
            # Fallback on any H5 read error
            return np.zeros(self.total_frames)