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
from openprc.demlat.io.simulation_setup import SimulationSetup
from openprc.demlat.utils.animator import ShowSimulation
from scipy.interpolate import CubicSpline

def add_discretized_spring(setup, n_start, n_end, segments=3, k_total=222.15, c_total=0.8, mass_total=0.005, hinge_k=0.02, hinge_c=0.005):
    """
    Replaces a single bar with a chain of sub-bars using 4-node dihedral hinges
    to simulate in-plane bending, out-of-plane bending, and distributed mass.
    """
    pos_start = np.array(setup.nodes['positions'][n_start])
    pos_end = np.array(setup.nodes['positions'][n_end])

    # 1. Physics Scaling (Series connections)
    k_seg = k_total * segments
    c_seg = c_total * segments
    
    # We add 2 ghost nodes for every intermediate node. 
    num_intermediate = segments - 1
    total_nodes_added = num_intermediate * 3 
    node_mass = 0.0005

    chain_nodes = [n_start]

    # 2. Create intermediate nodes
    for i in range(1, segments):
        frac = i / segments
        pos_mid = pos_start * (1 - frac) + pos_end * frac
        mid_idx = setup.add_node(pos_mid.tolist(), mass=node_mass, fixed=False)
        chain_nodes.append(mid_idx)

    chain_nodes.append(n_end)

    # 3. Connect the main chain with bars
    for i in range(segments):
        setup.add_bar(chain_nodes[i], chain_nodes[i+1], stiffness=k_seg, damping=c_seg)

    # 4. Create Ghost Nodes and 4-Node Hinges
    vec = pos_end - pos_start
    length = np.linalg.norm(vec)
    if length < 1e-6:
        return chain_nodes

    u = vec / length
    n_xy = np.array([-u[1], u[0], 0.0]) # In-plane normal
    n_z = np.array([0.0, 0.0, 1.0])     # Out-of-plane normal
    ghost_offset = length / segments * 0.5 

    for i in range(1, segments):
        idx_prev = chain_nodes[i-1]
        idx_curr = chain_nodes[i]
        idx_next = chain_nodes[i+1]
        pos_curr = np.array(setup.nodes['positions'][idx_curr])

        # A. Create Ghost Z (Out of plane) & Ghost Y (In plane)
        pos_gz = pos_curr + n_z * ghost_offset
        gz_idx = setup.add_node(pos_gz.tolist(), mass=node_mass, fixed=False)

        pos_gy = pos_curr + n_xy * ghost_offset
        gy_idx = setup.add_node(pos_gy.tolist(), mass=node_mass, fixed=False)

        # B. Rigidly attach ghosts to the PREVIOUS segment (idx_prev -> idx_curr)
        # This acts like an umbrella frame that tracks the orientation of the first segment.
        # It does NOT restrict the next segment (idx_next) from moving.
        frame_k, frame_c = k_total * 0.5, c_total * 0.5
        
        setup.add_bar(gz_idx, idx_curr, stiffness=frame_k, damping=frame_c)
        setup.add_bar(gz_idx, idx_prev, stiffness=frame_k, damping=frame_c)
        setup.add_bar(gy_idx, idx_curr, stiffness=frame_k, damping=frame_c)
        setup.add_bar(gy_idx, idx_prev, stiffness=frame_k, damping=frame_c)
        setup.add_bar(gz_idx, gy_idx, stiffness=frame_k, damping=frame_c) # Lock them together

        # C. Add the 4-node Dihedral Hinges!
        # Because we leave rest_angle=None, OpenPRC will cleanly calculate the perfect rest state 
        # from the initial straight-line geometry.
        
        # In-Plane Bending (Axis is Ghost_Z -> Current Node)
        setup.add_hinge(
            [gz_idx, idx_curr, idx_prev, idx_next], 
            stiffness=hinge_k, damping=hinge_c, rest_angle=None
        )

        # Out-of-Plane Bending (Axis is Ghost_Y -> Current Node)
        setup.add_hinge(
            [gy_idx, idx_curr, idx_prev, idx_next], 
            stiffness=hinge_k, damping=hinge_c, rest_angle=None
        )

    return chain_nodes

def run_pipeline(
    rows: int = 3, 
    cols: int = 3, 
    k_mat: np.ndarray = None, 
    c_mat: np.ndarray = None, 
    ga_generation: int = 0,
    amplitude: float = 0.015,     # <--- NEW: Default to 2.5 to match baseline
    target_hz: float = 30.0     # <--- NEW: Enforce 30Hz output
):
    """
    Defines, saves, and runs the entire spring-mass experiment.
    """
    # --- 1. Define Grid and Simulation Parameters ---
    ROWS, COLS = rows, cols
    SPACING = 0.053  # meters
    STIFFNESS = 100.00  # N/m
    DAMPING = 0.8
    NODE_MASS = 0.01  # kg
    
    # STRICT Directory Naming: experiments/spring_mass_{R}x{C}_test/generation_{gen}
    OUTPUT_DIR = src_dir/"experiments"/f"spring_mass_{ROWS}x{COLS}_test"/f"generation_{ga_generation}"

    print(f"[Step 1] Setting up {ROWS}x{COLS} spring-mass grid in {OUTPUT_DIR}")

    # Initialize the setup helper, overwriting if it exists
    setup = SimulationSetup(OUTPUT_DIR, overwrite=True)

    # --- 2. Configure Simulation and Physics ---
    save_int = 1.0 / target_hz
    setup.set_simulation_params(duration=3.0, dt=0.0001, save_interval=save_int)
    setup.set_physics(gravity=-9.8, damping=0.1, enable_collision=True)

    # --- 3. Generate Geometry and Actuation ---
    node_indices = np.zeros((ROWS, COLS), dtype=int)

    # Add nodes
    for r in range(ROWS):
        for c in range(COLS):
            pos = [c * SPACING, -r * SPACING, 0.0]
            idx = setup.add_node(pos, mass=NODE_MASS, fixed=False)
            node_indices[r, c] = idx
    
    print(f"Added {len(setup.nodes['positions'])} nodes.")

    from collections import defaultdict
    node_connections = defaultdict(list)

    if k_mat is not None:
        print("Using provided stiffness matrix (k_mat) to create discretized springs.")
        num_nodes = ROWS * COLS
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                stiffness = k_mat[i, j]
                if stiffness > 0:
                    damping = c_mat[i, j] if c_mat is not None else DAMPING
                    
                    # Replace add_bar with the discretized version
                    chain = add_discretized_spring(
                        setup, i, j, 
                        segments=4,             # Start with 3 segments (2 internal nodes)
                        k_total=STIFFNESS, 
                        c_total=DAMPING,
                        hinge_k=0.005            # Tune this to match the physical flexure
                    )
                    
                    # Log the connections to build rigid joints
                    node_connections[i].append((j, chain[1]))       # i's first step towards j
                    node_connections[j].append((i, chain[-2]))      # j's first step towards i

    # --- 3.1 Add Rigid Joint Friction at Main Nodes ---
    print("Generating rigid joint constraints via structural bar bracing.")
    RIGID_BRACE_STIFFNESS = STIFFNESS * 1.5  # High stiffness to mimic rigid plastic
    RIGID_BRACE_DAMPING = 0.5

    for main_node, connections in node_connections.items():
        if len(connections) >= 2:
            for c1 in range(len(connections)):
                target_j1, step1 = connections[c1]
                for c2 in range(c1 + 1, len(connections)):
                    target_j2, step2 = connections[c2]
                    
                    # Form a rigid structural triangle! No ghost nodes or hinges needed.
                    setup.add_bar(step1, step2, stiffness=RIGID_BRACE_STIFFNESS, damping=RIGID_BRACE_DAMPING)

    # --- 4. Add Hinges for Bending Resistance ---
    HINGE_STIFFNESS = 0.02  # N-m/rad. Kept low as a precaution.
    HINGE_DAMPING = 1.0
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
    fixed_indices = [
        node_indices[0, COLS - 1], 
        node_indices[ROWS - 1, 0], 
        node_indices[ROWS - 1, COLS - 1]
    ]
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
    # Only actuate the Top-Left corner (Node 0)
    act_indices = [node_indices[0, 0]] # Assuming you actuate the top-left corner
    print(f"Adding 30Hz IID actuation to nodes: {act_indices}")

    sim_params = setup.config['simulation']
    dt_sig = sim_params['dt_base']
    t_sim = np.arange(0, sim_params['duration'], dt_sig)

    np.random.seed(42) 
    sample_hz = target_hz  # <--- Match the spline knots to your saving Hz
    sample_interval = 1.0 / sample_hz
    t_coarse = np.arange(0, sim_params['duration'] + sample_interval, sample_interval)
    
    u_coarse = np.random.uniform(low=-1.0, high=1.0, size=len(t_coarse))
    cs = CubicSpline(t_coarse, u_coarse)

    AMPLITUDE = amplitude
    u_fine = cs(t_sim) * AMPLITUDE

    for i, idx in enumerate(act_indices):
        p0 = setup.nodes['positions'][idx]
        sig = np.tile(p0, (len(t_sim), 1))
        
        # Apply the 30Hz IID spline exclusively to the X-axis displacement
        sig[:, 0] += u_fine 
        
        setup.add_signal(f"sig_iid_input_{i}", sig, dt=dt_sig)
        setup.add_actuator(idx, f"sig_iid_input_{i}", type='position')
        
    # --- 6. Save Experiment Files ---
    print("\n[Step 2] Saving experiment files (config.json, geometry.h5)...")
    setup.save()

    # --- 7. Run Simulation ---
    print("\n[Step 3] Running simulation...")
    exp = demlat.Simulation(OUTPUT_DIR)
    
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


def compute_global_effective_stiffness(adj_matrix, nx=4, ny=4):
    """
    Computes effective stiffness via a virtual uniaxial tensile test.
    """
    num_nodes = nx * ny
    dof = 2 * num_nodes  # 2 Degrees of Freedom per node (x, y)
    
    # 1. Initialize Global Stiffness Matrix K (Size: 32x32)
    K = np.zeros((dof, dof))
    
    # Generate geometric coordinates for a unit grid
    coords = []
    for r in range(ny):
        for c in range(nx):
            coords.append(np.array([c, r])) # Unit distance = 1.0
            
    # 2. Assemble K from Adjacency Matrix
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            k_val = adj_matrix[i, j]
            
            if k_val > 0: # If connected
                # Calculate vector direction (Cosine/Sine)
                vec = coords[j] - coords[i]
                length = np.linalg.norm(vec)
                c = vec[0] / length # Cosine
                s = vec[1] / length # Sine
                
                # Element Stiffness Matrix (2D Truss Element)
                block = k_val * np.array([
                    [c*c, c*s],
                    [c*s, s*s]
                ])
                
                # Indices in the large matrix
                ix, iy = 2*i, 2*i+1
                jx, jy = 2*j, 2*j+1
                
                # Diagonal (Positive)
                K[ix:ix+2, ix:ix+2] += block
                K[jx:jx+2, jx:jx+2] += block
                
                # Off-Diagonal (Negative)
                K[ix:ix+2, jx:jx+2] -= block
                K[jx:jx+2, ix:ix+2] -= block

    # --- THE FIX STARTS HERE ---
    # Add a tiny stiffness (epsilon) to the diagonal.
    # This prevents the "Singular Matrix" error by ensuring every node 
    # has a tiny resistance to motion in all directions (like air resistance or weak hinges).
    epsilon = 1e-6 * np.max(np.abs(K))  # Scale epsilon relative to your stiffness
    indices = np.arange(dof)
    K[indices, indices] += epsilon
    # --- THE FIX ENDS HERE ---

    # 3. Apply Boundary Conditions (Virtual Pull Test)
    # Left Nodes (Fixed): Indices 0, 4, 8, 12
    fixed_nodes = [0, 4, 8, 12]
    fixed_dofs = []
    for n in fixed_nodes:
        fixed_dofs.extend([2*n, 2*n+1]) # Fix X and Y
        
    # Right Nodes (Pulled): Indices 3, 7, 11, 15
    pulled_nodes = [3, 7, 11, 15]
    pulled_dofs_x = [2*n for n in pulled_nodes]
    
    # 4. Partition the Matrix
    # We want to solve F = K * u
    all_dofs = np.arange(dof)
    prescribed_dofs = fixed_dofs + pulled_dofs_x
    free_dofs = np.setdiff1d(all_dofs, prescribed_dofs)
    
    # Construct Force Vector F (initially zero)
    # Construct Displacement Vector u
    u = np.zeros(dof)
    
    # Set defined displacement (delta = 1.0)
    delta = 1.0
    u[pulled_dofs_x] = delta
    
    # 5. Solve for Free DOFs
    # K_free_free * u_free = - K_free_prescribed * u_prescribed
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    K_fp = K[np.ix_(free_dofs, prescribed_dofs)]
    u_p = u[prescribed_dofs]
    
    # Solve linear system
    u_free = np.linalg.solve(K_ff, -K_fp @ u_p)
    u[free_dofs] = u_free
    
    # 6. Calculate Reaction Force
    total_reaction_force = 0
    for dof_idx in pulled_dofs_x:
        force = K[dof_idx, :] @ u
        total_reaction_force += force
        
    K_eff = total_reaction_force / delta
    
    return K_eff, K


if __name__ == "__main__":
    # Test Run
    # High Stiffness (Stiff)
    H = 222.15
    # Low Stiffness (Soft)
    L = 65.08

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

    K_eff, K_global = compute_global_effective_stiffness(K_mat)

    print(f"Global Effective Stiffness (X-direction): {K_eff:.2f} N/m")

    result = run_pipeline(rows=4, cols=4, k_mat=K_mat)
    data, output_dir = result
    
    if output_dir:
        print(f"\n[Step 4] Visualizing experiment: {output_dir}")
        # FIX: Convert Path object to string for the visualizer
        ShowSimulation(str(output_dir))