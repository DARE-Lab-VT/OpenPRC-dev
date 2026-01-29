"""
2D Spring-Mass System Example (ExperimentSetup API)
=====================================================
This script demonstrates how to create and simulate a 2D spring-mass system
using the high-level `ExperimentSetup` API. It generates a grid of masses
connected by springs, fixes the top corners, and runs a simulation under gravity.
"""
import sys
from pathlib import Path
import numpy as np

# Ensure we can import demlat modules
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

import demlat
from demlat.models.barhinge import BarHingeModel
from demlat.io.experiment_setup import ExperimentSetup
from demlat.utils.viz_player import visualize_experiment

def run_pipeline():
    """
    Defines, saves, and runs the entire spring-mass experiment.
    """
    # --- 1. Define Grid and Simulation Parameters ---
    ROWS, COLS = 3, 3
    SPACING = 0.053  # meters
    STIFFNESS = 100.0  # N/m
    DAMPING = 0.4
    NODE_MASS = 0.01  # kg
    OUTPUT_DIR = src_dir / "experiments" / f"spring_mass_{ROWS}x{COLS}_test"

    print(f"[Step 1] Setting up {ROWS}x{COLS} spring-mass grid in {OUTPUT_DIR}")

    # Initialize the setup helper, overwriting if it exists
    setup = ExperimentSetup(OUTPUT_DIR, overwrite=True)

    # --- 2. Configure Simulation and Physics ---
    setup.set_simulation_params(duration=30.0, dt=0.001, save_interval=0.01)
    setup.set_physics(gravity=0.0, damping=0.5)

    # --- 3. Generate Geometry and Actuation ---
    node_indices = np.zeros((ROWS, COLS), dtype=int)

    # Add nodes
    for r in range(ROWS):
        for c in range(COLS):
            pos = [c * SPACING, -r * SPACING, 0.0]
            idx = setup.add_node(pos, mass=NODE_MASS, fixed=False)
            node_indices[r, c] = idx
    
    print(f"Added {len(setup.nodes['positions'])} nodes.")

    # Add bars (springs)
    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - 1):
            idx1 = node_indices[r, c]
            idx2 = node_indices[r, c + 1]
            setup.add_bar(idx1, idx2, stiffness=STIFFNESS, damping=DAMPING)

    # Vertical
    for r in range(ROWS - 1):
        for c in range(COLS):
            idx1 = node_indices[r, c]
            idx2 = node_indices[r + 1, c]
            setup.add_bar(idx1, idx2, stiffness=STIFFNESS, damping=DAMPING)
            
    print(f"Added {len(setup.bars['indices'])} bars.")

    # --- 4. Define Fixed Nodes ---
    fixed_indices = [node_indices[ROWS - 1, 0]]
    print(f"Fixing nodes via actuation: {fixed_indices}")

    sim_params = setup.config['simulation']
    dt_sig = sim_params['dt_base']
    t = np.arange(0, sim_params['duration'], dt_sig)
    
    # Create static signals to hold corner nodes in place
    for i, idx in enumerate(fixed_indices):
        p0 = setup.nodes['positions'][idx]
        sig = np.tile(p0, (len(t), 1))
        sig_name = f"sig_fixed_corner_{i}"
        setup.add_signal(sig_name, sig, dt=dt_sig)
        setup.add_actuator(idx, sig_name, type='position')

    # --- 5. Define Actuated Nodes ---
    act_indices = [node_indices[0, COLS - 1]]
    act_indices = [idx for idx in act_indices if idx not in fixed_indices]
    print(f"Adding dynamic actuation to nodes: {act_indices}")

    for i, act_idx in enumerate(act_indices):
        # Create a sinusoidal signal in the X direction
        p0 = setup.nodes['positions'][act_idx]
        amp = 0.02  # 2 cm
        freq = 3   # 3 Hz
        omega = 2 * np.pi * freq
        
        sig = np.tile(p0, (len(t), 1))
        sig[:, 0] += amp * np.sin(omega * t) * np.sin(np.pi/4)
        sig[:, 1] += amp * np.sin(omega * t) * np.cos(np.pi/4)
        
        # Add the dynamic signal and actuator
        sig_name = f"sig_actuator_{i}"
        setup.add_signal(sig_name, sig, dt=dt_sig)
        setup.add_actuator(act_idx, sig_name, type='position')

    # --- 6. Save Experiment Files ---
    print("\n[Step 2] Saving experiment files (config.json, geometry.h5)...")
    setup.save()

    # --- 7. Run Simulation ---
    print("\n[Step 3] Running simulation...")
    exp = demlat.Experiment(OUTPUT_DIR)
    
    # Use CUDA if available, else CPU
    try:
        import pycuda.driver
        pycuda.driver.init()
        if pycuda.driver.Device.count() > 0:
            backend = 'cuda'
            print("Using CUDA backend.")
        else:
            raise ImportError("No CUDA devices found")
    except ImportError:
        backend = 'cpu'
        print("CUDA not available. Using CPU backend.")
        
    eng = demlat.Engine(BarHingeModel, backend=backend)
    eng.run(exp)
    
    print("\nDone. You can now visualize the results with:")
    print(f"python src/demlat/utils/viz_player.py {OUTPUT_DIR}")

    return OUTPUT_DIR


if __name__ == "__main__":
    output_dir = run_pipeline()
    if output_dir:
        print(f"\n[Step 4] Visualizing experiment: {output_dir}")
        visualize_experiment(output_dir)
