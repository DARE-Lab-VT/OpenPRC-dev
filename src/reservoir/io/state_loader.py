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

    def get_node_positions(self, node_ids="all"):
        with h5py.File(self.sim_path, 'r') as f:
            data = f['time_series/nodes/positions']
            if node_ids == "all":
                return data[:].reshape(self.total_frames, -1)
            return data[:, node_ids, :].reshape(self.total_frames, -1)

    def get_actuation_signal(self, actuator_idx=0):
        """
        Loads signal from signals.h5 and aligns it to simulation time using loader.dt
        """
        if not self.signal_path.exists():
            # Fallback if signal file is missing (e.g. pure gravity sim)
            return np.zeros(self.total_frames)

        with h5py.File(self.signal_path, 'r') as f:
            keys = sorted([k for k in f.keys() if k.startswith("sig_actuator")])
            if not keys: return np.zeros(self.total_frames)
            
            target_key = keys[actuator_idx] if actuator_idx < len(keys) else keys[0]
            
            # Load High-Res Data
            raw_data = f[target_key][:] 
            dt_signal = f.attrs.get('dt_base', 0.001) 
            
        # Extract X-axis (assuming 1D drive)
        u_high_res = raw_data[:, 0] if raw_data.ndim > 1 else raw_data
        t_high_res = np.arange(len(u_high_res)) * dt_signal

        # Align to Simulation Time
        interpolator = interp1d(t_high_res, u_high_res, kind='linear', fill_value="extrapolate")
        u_input = interpolator(self.time)
        
        return u_input