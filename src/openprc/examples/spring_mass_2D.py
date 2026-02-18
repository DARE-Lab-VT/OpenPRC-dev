"""
2D Spring-Mass System Example (ExperimentSetup API)
=====================================================
Modified for GA Integration (Strict Naming & String Path Fix)
"""
import sys
import h5py
from pathlib import Path
import numpy as np

# Ensure we can import demlat modules
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

from openprc import demlat
from openprc.demlat.models.barhinge import BarHingeModel
from openprc.demlat.io.experiment_setup import ExperimentSetup
from openprc.demlat.utils.viz_player import visualize_experiment


def run_pipeline(
    rows: int = 3, 
    cols: int = 3, 
    k_mat: np.ndarray = None, 
    c_mat: np.ndarray = None, 
    ga_generation: int = 0
):
    """
    Defines, saves, and runs the entire spring-mass experiment.
    """
    # --- 1. Define Grid and Simulation Parameters ---
    ROWS, COLS = rows, cols
    SPACING = 0.053  # meters
    STIFFNESS = 222.15  # N/m
    DAMPING = 0.8
    NODE_MASS = 0.01  # kg
    
    # STRICT Directory Naming: experiments/spring_mass_{R}x{C}_test/generation_{gen}
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
            raise ValueError("Stiffness matrix mismatch.")
        
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

    # --- 4. Add Hinges for Bending Resistance ---
    HINGE_STIFFNESS = 0.01  # N-m/rad. Kept low as a precaution.
    HINGE_DAMPING = 0.1
    # Add diagonal hinges to each quad to provide a baseline bending resistance
    print("Generating hinges for all quads in the grid.")
    for r in range(ROWS - 1):
        for c in range(COLS - 1):
            # Quad nodes
            n_tl = node_indices[r, c]
            n_tr = node_indices[r, c + 1]
            n_bl = node_indices[r + 1, c]
            n_br = node_indices[r + 1, c + 1]

            # Hinge across the diagonal n_tl -> n_br
            # Hinge 1: [j, k, i, l]
            setup.add_hinge([n_tl, n_br, n_tr, n_bl], stiffness=HINGE_STIFFNESS, damping=HINGE_DAMPING, rest_angle=np.pi)

            # Hinge across the other diagonal n_tr -> n_bl
            # The order of the last two nodes is swapped here compared to previous attempts.
            # This might be required to ensure a consistent winding order for the solver.
            # Hinge 2: [j, k, i, l]
            setup.add_hinge([n_tr, n_bl, n_br, n_tl], stiffness=HINGE_STIFFNESS, damping=HINGE_DAMPING, rest_angle=np.pi)
    
    print(f"Added {len(setup.hinges['indices'])} hinges.")


    # --- 5. Define Fixed Nodes ---
    fixed_indices = []
    print(f"Fixing nodes via actuation: {fixed_indices}")

    sim_params = setup.config['simulation']
    dt_sig = sim_params['dt_base']
    t_ = np.arange(0, sim_params['duration'], dt_sig)
    
    if len(fixed_indices) != 0:
        for i, idx in enumerate(fixed_indices):
            p0 = setup.nodes['positions'][idx]
            sig = np.tile(p0, (len(t_), 1))
            setup.add_signal(f"sig_fixed_corner_{i}", sig, dt=dt_sig)
            setup.add_actuator(idx, f"sig_fixed_corner_{i}", type='position')

    # --- 6. Define Actuated Nodes ---
    act_indices = [
        node_indices[0, 0], 
        node_indices[0, COLS - 1], 
        node_indices[ROWS - 1, 0], 
        node_indices[ROWS - 1, COLS - 1]
    ]
    act_indices = [idx for idx in act_indices if idx not in fixed_indices]
    print(f"Adding dynamic actuation to nodes: {act_indices}")

    if len(act_indices) != 0:
        for i, act_idx in enumerate(act_indices):
            p0 = setup.nodes['positions'][act_idx]
            amp = 0.02  # 2 cm
            f1 = 2.11   # 2.11 Hz
            f2 = 3.73   # 3.73 Hz
            f3 = 4.33   # 4.33 Hz
            
            sig = np.tile(p0, (len(t_), 1))
            sig[:, 0] += amp * np.sin(2 * np.pi * f1 * t_) * np.sin(2 * np.pi * f2 * t_) * np.sin(2 * np.pi * f3 * t_)
            
            setup.add_signal(f"sig_actuator_{i}", sig, dt=dt_sig)
            setup.add_actuator(act_idx, f"sig_actuator_{i}", type='position')

    # --- 6. Save Experiment Files ---
    print("\n[Step 2] Saving experiment files (config.json, geometry.h5)...")
    setup.save()

    # --- 7. Run Simulation ---
    print("\n[Step 3] Running simulation...")
    exp = demlat.Experiment(OUTPUT_DIR)
    
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
    # Return Path object for internal use, string for print/visualizer
    print(f"python src/demlat/utils/viz_player.py {OUTPUT_DIR}")

    # --- 8. Read and Return Data ---
    data = None
    res_path = OUTPUT_DIR / "output" / "simulation.h5"
    
    if res_path.exists():
        with h5py.File(res_path, 'r') as f:
            data = f['time_series/nodes/positions'][:]
    return data, OUTPUT_DIR


if __name__ == "__main__":
    # Test Run
    # High Stiffness (Stiff)
    H = 222.15
    # Low Stiffness (Soft)
    L = 65.0

    # 16x16 Weighted Adjacency Matrix
    # Rows/Cols correspond to Nodes 0-15
    K_mat = np.array([
        # 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
        [ 0,  L,  0,  0,  H,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # Node 0
        [ L,  0,  H,  0,  0,  H,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # Node 1
        [ 0,  H,  0,  H,  0,  0,  L,  0,  0,  0,  0,  0,  0,  0,  0,  0], # Node 2
        [ 0,  0,  H,  0,  0,  0,  0,  L,  0,  0,  0,  0,  0,  0,  0,  0], # Node 3
        [ H,  0,  0,  0,  0,  L,  0,  0,  L,  0,  0,  0,  0,  0,  0,  0], # Node 4
        [ 0,  H,  0,  0,  L,  0,  H,  0,  0,  L,  0,  0,  0,  0,  0,  0], # Node 5
        [ 0,  0,  L,  0,  0,  H,  0,  H,  0,  0,  H,  0,  0,  0,  0,  0], # Node 6
        [ 0,  0,  0,  L,  0,  0,  H,  0,  0,  0,  0,  H,  0,  0,  0,  0], # Node 7
        [ 0,  0,  0,  0,  L,  0,  0,  0,  0,  H,  0,  0,  L,  0,  0,  0], # Node 8
        [ 0,  0,  0,  0,  0,  L,  0,  0,  H,  0,  L,  0,  0,  L,  0,  0], # Node 9
        [ 0,  0,  0,  0,  0,  0,  H,  0,  0,  L,  0,  L,  0,  0,  H,  0], # Node 10
        [ 0,  0,  0,  0,  0,  0,  0,  H,  0,  0,  L,  0,  0,  0,  0,  H], # Node 11
        [ 0,  0,  0,  0,  0,  0,  0,  0,  L,  0,  0,  0,  0,  H,  0,  0], # Node 12
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  L,  0,  0,  H,  0,  L,  0], # Node 13
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  H,  0,  0,  L,  0,  L], # Node 14
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  H,  0,  0,  L,  0]  # Node 15
    ])

    result = run_pipeline(rows=4, cols=4, k_mat=K_mat)
    data, output_dir = result
    
    if output_dir:
        print(f"\n[Step 4] Visualizing experiment: {output_dir}")
        # FIX: Convert Path object to string for the visualizer
        visualize_experiment(str(output_dir))