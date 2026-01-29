"""
Yoshimura-Ori Geometry Test
===========================
Generates large number of experiments for analyzing beta and d relation
"""
import numpy as np
from examples.yoshimura_ori_geometry import *
from pathlib import Path
import h5py

import demlat
from demlat.io.experiment_setup import ExperimentSetup
from demlat.utils.viz_player import visualize_experiment
from examples.pipeline_yoshimura_test import setup_actuation
from examples.pipeline_yoshimura_test import show_pe
from demlat.models.barhinge import BarHingeModel

EXP_DIR_ROOT = Path("experiments/yoshimura_beta_d_sweep/")


def run_and_save_experiment(dir_name, n, beta):
    # setup
    path = EXP_DIR_ROOT / dir_name
    setup = ExperimentSetup(path, overwrite=True)

    # Simulation parameters
    duration = 20.0
    dt = 0.0005
    save_interval = 0.01

    # Configure Simulation
    setup.set_simulation_params(duration=duration, dt=dt, save_interval=save_interval)
    setup.set_physics(gravity=0.0, damping=0.2)

    d = np.tan(beta) - 0.001

    # Build Geometry

    print(f"\nYoshimura Parameters:")
    print(f"  n={n}, beta={np.rad2deg(beta):.2f}°")

    # Generate geometry
    nodes, bars, faces, params = generate_yoshimura_geometry(n, beta, d=d, gamma=0.0, psi=0.0)
    mass = 0.01,
    damping = 1.0
    k_axial = 1000.0

    print(f"\nGenerated Geometry:")
    print(f"  Nodes: {len(nodes)}")
    print(f"  Bars: {len(bars)}")
    print(f"  Faces: {len(faces)}")

    node_info = {
        'base_corners': [2 * i for i in range(n)],
        'top_corners': [4 * n + 2 * i for i in range(n)],
        'n': n,
        'total_nodes': len(nodes)
    }

    # Add nodes to setup
    for i, node_pos in enumerate(nodes):
        setup.add_node(node_pos, mass=mass, fixed=False)

    # Add bars to setup
    for i, j, length in bars:
        setup.add_bar(i, j, stiffness=k_axial, rest_length=length, damping=damping)

    k_fold = 1.0
    k_facet = 0.0

    # Add hinges to setup
    add_yoshimura_hinges(
        setup,
        nodes,
        faces,
        n,
        params,
        k_fold=k_fold,
        k_facet=k_facet
    )

    n, beta, d, gamma, psi = params
    d = np.tan(beta)

    # Setup Actuation
    setup_actuation(
        setup,
        node_info,
        min_pos=0.0,
        max_pos=d,
        frequency=0.2,
        duration=duration
    )

    # Save Everything
    setup.save()

    # Save Visualization Faces
    with h5py.File(path / "input" / "visualization.h5", 'w') as f:
        f.create_dataset("faces", data=np.array(faces, dtype=np.int32))

    print(f"\nSaved to: {path}")

    print("\n[Step 2] Running Simulation...")

    exp = demlat.Experiment(path)
    eng = demlat.Engine(BarHingeModel, backend='cuda')
    eng.run(exp)

    print("\nSimulation complete!")


if __name__ == "__main__":
    n = 3
    betas = np.linspace(30, 45, 1000)
    for beta in betas:
        dir_name = "exp_beta_" + str(beta) + "_n_" + str(n)
        try:
            run_and_save_experiment(dir_name, n, np.deg2rad(beta))
        except Exception as e:
            print(e)

    beta = betas[499]
    dir_name = "exp_beta_" + str(beta) + "_n_" + str(n)

    visualize_experiment(EXP_DIR_ROOT / dir_name)
    show_pe(EXP_DIR_ROOT / dir_name)
