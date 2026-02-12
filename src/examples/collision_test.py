"""
Yoshimura Equilibrium Visualizer
================================
Iterates through saved stable equilibria, injects them into the
simulation experiment, and visualizes them one by one.
"""

import h5py
import numpy as np
from pathlib import Path
import demlat
from demlat.models.barhinge import BarHingeModel
from demlat.utils.viz_player import visualize_experiment

# Import your existing setup function
# (Assuming the previous script is named 'yoshimura_test.py')
# If it's in the same file, you don't need this import.

DEMO_DIR = Path("experiments/yoshimura_equlilibira_test")
EQUILIBRIA_PATH = Path("experiments/yoshimura_equilibrium") / "output" / "equilibria.h5"


def setup(beta=35):
    from demlat.io.experiment_setup import ExperimentSetup
    from examples.Yoshimura import Yoshimura

    """Setup the Yoshimura experiment"""
    print("\n[Setup] Creating Yoshimura Experiment...")

    # Initialize Setup
    setup = ExperimentSetup(DEMO_DIR, overwrite=True)

    # Simulation parameters
    duration = 5.0
    dt = 0.0005
    save_interval = 0.005
    setup.set_simulation_params(duration=duration, dt=dt, save_interval=save_interval)
    setup.set_physics(gravity=0.0, damping=0.1)

    # Build Geometry
    beta = np.deg2rad(beta)
    n = 3
    d = np.tan(beta)
    k_axial = 1000.0
    k_fold = 0.00
    k_facet = 0.02
    mass = 0.01
    damping = 2.0

    print(f"\nYoshimura Parameters:")
    print(f"  n={n}, beta={np.rad2deg(beta):.2f}°")

    # Generate geometry
    module = Yoshimura(n, beta, randomize=False)
    nodes, bars, hinges, faces, params = module.get_geometry()

    print(f"\nGenerated Geometry:")
    print(f"  Nodes: {len(nodes)}")
    print(f"  Bars: {len(bars)}")
    print(f"  Hinges: {len(hinges)}")
    print(f"  Faces: {len(faces)}")

    # Add nodes to setup
    for i, node_pos in enumerate(nodes):
        if i == 0:
            setup.add_node(node_pos, mass=mass, fixed=False)
        else:
            setup.add_node(node_pos, mass=mass, fixed=False)

    # Add bars to setup
    for bar in bars:
        setup.add_bar(bar[0], bar[1], stiffness=k_axial * bar[2], rest_length=bar[2], damping=damping)

    # Add hinges to setup
    for hinge in hinges:
        if hinge[5] == 'fold':
            k = k_fold
        elif hinge[5] == 'facet':
            k = k_facet
        else:
            k = 0.0

        if k > 0.0:
            setup.add_hinge(nodes=[hinge[0], hinge[1], hinge[2], hinge[3]], stiffness=k, rest_angle=hinge[4])

    for face in faces:
        setup.add_face(face)
    setup.save()


def patch_geometry_with_equilibrium(exp_dir, positions):
    """
    Overwrites the node positions in the input geometry file
    with the equilibrium positions.
    """
    geo_path = exp_dir / "input" / "geometry.h5"

    with h5py.File(geo_path, 'r+') as f:
        # Overwrite positions
        if 'nodes/positions' in f:
            del f['nodes/positions']
        f.create_dataset('nodes/positions', data=positions)

        # Ensure velocities are zero (start from rest)
        if 'nodes/velocities' in f:
            del f['nodes/velocities']
        n_nodes = len(positions)
        f.create_dataset('nodes/velocities', data=np.zeros((n_nodes, 3)))

    print(f"  -> Injected equilibrium positions into {geo_path}")


def run_simulation(exp_dir):
    """Runs a short simulation to verify stability."""
    print("  -> Running verification simulation...")
    exp = demlat.Experiment(exp_dir)
    # Use 'cuda' if available, else 'cpu' or 'jax'
    eng = demlat.Engine(BarHingeModel, backend='cuda')
    eng.run(exp)


def visualize_sequence(state=0):
    if not EQUILIBRIA_PATH.exists():
        print(f"Error: {EQUILIBRIA_PATH} not found. Run the finder first.")
        return

    # 1. Load all stable equilibria
    stable_eqs = []
    print(f"\n[Loader] Reading {EQUILIBRIA_PATH}...")

    with h5py.File(EQUILIBRIA_PATH, 'r') as f:
        grp = f['equilibria']
        for key in sorted(grp.keys()):
            item = grp[key]
            if item.attrs['stability'] == 'stable':
                pos = item['positions'][:]
                # metrics for display
                z_range = pos[:, 2].max() - pos[:, 2].min()
                res = item.attrs['residual']
                source = item.attrs['source_guess']
                stable_eqs.append((key, pos, z_range, res, source))

    print(f"[Loader] Found {len(stable_eqs)} stable configurations.")

    if not stable_eqs:
        return

    # 2. Loop and Visualize
    for i, (key, pos, z_h, res, src) in enumerate(stable_eqs):
        if i == state:
            print(f"\n{'=' * 60}")
            print(f"Visualizing Stable State {i + 1}/{len(stable_eqs)} (ID: {key})")
            print(f"  Height (Z): {z_h:.4f}")
            print(f"  Residual:   {res:.2e}")
            print(f"  Source:     {src}")
            print(f"{'=' * 60}")

            # A. Reset Experiment Topology (Force=False ensures no external loads)
            # We assume beta=35 matches your finder's geometry
            setup(beta=35)

            # B. Inject Equilibrium Positions
            patch_geometry_with_equilibrium(DEMO_DIR, pos)

            # C. Run Short Simulation
            # This proves the state is actually stable in the dynamic engine
            run_simulation(DEMO_DIR)

            # D. Visualize
            print("  -> Launching Visualizer (Close window to see next)...")
            visualize_experiment(DEMO_DIR)

            # Optional: Pause
            # choice = input(f"  [Enter] for next, [q] to quit: ")
            # if choice.lower() == 'q':
            #     break

        else:
            pass


if __name__ == "__main__":
    import sys
    print(sys.argv)

    visualize_sequence(state=int(sys.argv[1]))


