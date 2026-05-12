import sys
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt

# --- Path Setup ---
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

def plot_waveforms(real_path, sim_path):
    print(f"Loading Real Data: {real_path.name}")
    print(f"Loading Sim Data:  {sim_path.name}")
    
    # 1. Read Real Data
    with h5py.File(real_path, 'r') as f:
        # Extract X-axis (index 0) of Node 0 across all time steps
        real_pos = f['time_series/nodes/positions'][:, 0, 0] 
        # Zero-center the displacement
        real_disp = real_pos - real_pos[0]
        
    # 2. Read Sim Data
    with h5py.File(sim_path, 'r') as f:
        # Extract X-axis (index 0) of Node 0 across all time steps
        sim_pos = f['time_series/nodes/positions'][:, 0, 0]
        # Zero-center the displacement
        sim_disp = sim_pos - sim_pos[0]

    # 3. Create Time Arrays (Assuming 30 Hz for both)
    t_real = np.arange(len(real_disp)) / 30.0
    t_sim = np.arange(len(sim_disp)) / 30.0

    # 4. Plotting
    plt.figure(figsize=(12, 5))
    
    # Plot real data with a solid line
    plt.plot(t_real, real_disp, label='Real-life Input', color='#d62728', alpha=0.9, linewidth=2)
    
    # Plot sim data with a dashed line
    plt.plot(t_sim, sim_disp, label='Simulation Input', color='#1f77b4', alpha=0.8, linewidth=2, linestyle='--')
    
    plt.title("Input Waveform Comparison: Simulation vs Reality", fontsize=14)
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Node 0 X-Displacement (meters)", fontsize=12)
    plt.legend(loc="upper right", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # ========================================================
    # UPDATE THESE PATHS to match your specific directories!
    # ========================================================
    
    # Path to the real-life experimental data
    REAL_H5 = src_dir.parent / "data" / "experiment_data" / "topology_9_prestress" / "amp=2.5" / "sample_0" / "experiment.h5"
    
    # Path to your latest simulation output (e.g., from topo_0)
    # Adjust "sim2real_bending" / "topo_0" / "hinge_damping_0.5_amplitude_1.5" to match your trial dir
    SIM_H5 = src_dir / "experiments" / "sim2real_bending" / "topo_0" / "hinge_damping_0.5_amplitude_1.5" / "output" / "simulation.h5"
    
    if not REAL_H5.exists():
        print(f"[Error] Cannot find Real data at: {REAL_H5}")
    elif not SIM_H5.exists():
        print(f"[Error] Cannot find Sim data at: {SIM_H5}")
    else:
        plot_waveforms(REAL_H5, SIM_H5)