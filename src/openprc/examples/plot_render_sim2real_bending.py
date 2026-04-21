import sys
from pathlib import Path
import numpy as np
import sys
import shutil
import os
from pathlib import Path
import numpy as np

# --- Windows File Lock Fix ---
original_rmtree = shutil.rmtree

def safe_rmtree(path, ignore_errors=False, onerror=None):
    try:
        original_rmtree(path, ignore_errors=ignore_errors, onerror=onerror)
    except PermissionError as e:
        if "WinError 32" in str(e) or "used by another process" in str(e):
            print(f"[Warning] Ignored Windows file lock on: {path}")
        else:
            raise

shutil.rmtree = safe_rmtree
# --- Path Setup ---
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

from openprc.optimization.search_spaces.fourier_series_2D import FourierSeries2D
from openprc.examples.spring_mass_2D_plate import run_pipeline
from openprc.demlat.utils.animator import ShowSimulation

def simulate_and_visualize(topo_name="topo_test", removed_springs=[]):
    """
    Sets up the topology, runs ONLY the simulation, and launches the GUI.
    """
    ROWS, COLS = 4, 4
    TARGET_STIFFNESS = 222.15  
    TARGET_DAMPING = 0.8
    
    fourier = FourierSeries2D(ROWS, COLS)
    
    # 1. Generate baseline matrices
    c_mat, k_mat = fourier.build_full_neighbor_topology(ROWS, COLS, rigid_outer_frame=False)
    k_mat = np.where(k_mat > 0, TARGET_STIFFNESS, 0.0)
    c_mat = np.where(c_mat > 0, TARGET_DAMPING, 0.0)
    
    # 2. Remove specified springs
    for (node_i, node_j) in removed_springs:
        k_mat[node_i, node_j] = 0.0
        k_mat[node_j, node_i] = 0.0
        c_mat[node_i, node_j] = 0.0
        c_mat[node_j, node_i] = 0.0

    print(f"\n--- Running Physics Simulation for: {topo_name} ---")
    
    # 3. Run the pipeline (this only takes a few seconds)
    # We pass topo_name as ga_generation so it saves in a named folder
    data, output_dir = run_pipeline(
        rows=ROWS, 
        cols=COLS, 
        k_mat=k_mat, 
        c_mat=c_mat, 
        ga_generation=topo_name,
        amplitude=0.015,
        target_hz=30.0
    )
    
    # 4. Launch the GUI Visualizer
    print(f"\n[INFO] Simulation complete. Launching GUI visualizer...")
    if output_dir:
        ShowSimulation(str(output_dir))

if __name__ == "__main__":
    # Test a specific topology directly here
    TOPOLOGY_NO = 2
    target_key = f"topo_{TOPOLOGY_NO}"
    topologies_dict = {
        "topo_0": [], # Baseline: Fully connected
        "topo_1": [(5, 6), (6, 7), (8, 9)], # Example: Removed springs between 0-1 and 2-3
        "topo_2": [(2, 6), (4, 5), (8, 9), (10, 11)], # Example: Removed three internal springs
        "topo_3": [(1, 5), (2, 6), (9, 13), (10, 14)],
        "topo_4": [(1, 5), (2, 6), (5, 9), (6, 10), (9, 13), (10, 14)],
        "topo_5": [(4, 5), (5, 6), (6, 7), (8, 9), (9, 10), (10, 11)]
    }

    simulate_and_visualize(target_key, topologies_dict[target_key])