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


def run_pipeline(k_mat: np.ndarray = None, c_mat: np.ndarray = None, ga_generation: int = 0):
    """
    Defines, saves, and runs the entire spring-mass experiment.

    Args:
        k_mat (np.ndarray, optional): An NxN matrix of spring stiffness values,
            where N is the total number of nodes (ROWS * COLS). A non-zero
            value at (i, j) creates a spring. If None, a default grid is
            generated with default stiffness.
        c_mat (np.ndarray, optional): An NxN matrix of spring damping values.
            If None, the default damping value is used for all springs.
    """
    # --- 1. Define Grid and Simulation Parameters ---
    ROWS, COLS = 3, 3
    SPACING = 0.053  # meters
    STIFFNESS = 100.0  # N/m
    DAMPING = 0.4
    NODE_MASS = 0.01  # kg
    OUTPUT_DIR = src_dir/"experiments"/f"spring_mass_{ROWS}x{COLS}_test"/f"generation_{ga_generation}"

    print(f"[Step 1] Setting up {ROWS}x{COLS} spring-mass grid in {OUTPUT_DIR}")

    # Initialize the setup helper, overwriting if it exists
    setup = ExperimentSetup(OUTPUT_DIR, overwrite=True)

    # --- 2. Configure Simulation and Physics ---
    setup.set_simulation_params(duration=30.0, dt=0.001, save_interval=0.01)
    setup.set_physics(gravity=-9.8, damping=0.5)

    # --- 3. Generate Geometry and Actuation ---
    node_indices = np.zeros((ROWS, COLS), dtype=int)

    # Add nodes
    for r in range(ROWS):
        for c in range(COLS):
            pos = [c * SPACING, -r * SPACING, 0.0]
            idx = setup.add_node(pos, mass=NODE_MASS, fixed=False)
            node_indices[r, c] = idx
    
    print(f"Added {len(setup.nodes['positions'])} nodes.")

    # Add bars (springs) based on the provided topology
    if k_mat is not None:
        print("Using provided stiffness matrix (k_mat) to create springs.")
        num_nodes = ROWS * COLS
        if not isinstance(k_mat, np.ndarray) or k_mat.shape != (num_nodes, num_nodes):
            raise ValueError(
                f"Stiffness matrix (k_mat) must be a NumPy array with shape "
                f"({num_nodes}, {num_nodes}), but got shape {k_mat.shape if isinstance(k_mat, np.ndarray) else type(k_mat)}."
            )
        if c_mat is not None and (not isinstance(c_mat, np.ndarray) or c_mat.shape != (num_nodes, num_nodes)):
            raise ValueError(
                f"Damping matrix (c_mat) must be a NumPy array with shape "
                f"({num_nodes}, {num_nodes}), but got shape {c_mat.shape if isinstance(c_mat, np.ndarray) else type(c_mat)}."
            )

        # Iterate over the upper triangle of the matrix to add bars
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                stiffness = k_mat[i, j]
                if stiffness > 0:
                    damping = c_mat[i, j] if c_mat is not None else DAMPING
                    setup.add_bar(i, j, stiffness=stiffness, damping=damping)
    else:
        print("Generating default fully connected grid topology.")
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
    fixed_indices = []
    print(f"Fixing nodes via actuation: {fixed_indices}")

    sim_params = setup.config['simulation']
    dt_sig = sim_params['dt_base']
    t_ = np.arange(0, sim_params['duration'], dt_sig)
    
    # Create static signals to hold corner nodes in place
    if len(fixed_indices) != 0:
        for i, idx in enumerate(fixed_indices):
            p0 = setup.nodes['positions'][idx]
            sig = np.tile(p0, (len(t_), 1))
            sig_name = f"sig_fixed_corner_{i}"
            setup.add_signal(sig_name, sig, dt=dt_sig)
            setup.add_actuator(idx, sig_name, type='position')

    # --- 5. Define Actuated Nodes ---
    act_indices = [node_indices[0, 0], node_indices[0, COLS - 1], node_indices[ROWS - 1, 0], node_indices[ROWS - 1, COLS - 1]]
    act_indices = [idx for idx in act_indices if idx not in fixed_indices]
    print(f"Adding dynamic actuation to nodes: {act_indices}")

    if len(act_indices) != 0:
        for i, act_idx in enumerate(act_indices):
            # Create a sinusoidal signal in the X direction
            p0 = setup.nodes['positions'][act_idx]
            amp = 0.02  # 2 cm
            f1 = 2.11   # 2.11 Hz
            f2 = 3.73   # 3.73 Hz
            f3 = 4.33   # 4.33 Hz
            
            sig = np.tile(p0, (len(t_), 1))
            sig[:, 0] += amp * np.sin(2 * np.pi * f1 * t_) * np.sin(2 * np.pi * f2 * t_) * np.sin(2 * np.pi * f3 * t_)
            
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
