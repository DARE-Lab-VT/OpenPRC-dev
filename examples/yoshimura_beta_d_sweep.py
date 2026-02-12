"""
Yoshimura-Ori beta d sweep
===========================
Generates large number of experiments for analyzing beta and d relation
"""
import numpy as np
from examples.yoshimura_ori_geometry import *
from pathlib import Path
import h5py

import openprc.demlat
from openprc.demlat.io.simulation_setup import SimulationSetup
from openprc.demlat.utils.viz_player import visualize_experiment
from pipeline_yoshimura_test import setup_actuation
from pipeline_yoshimura_test import show_pe
from openprc.demlat.models.barhinge import BarHingeModel

EXP_DIR_ROOT = Path("experiments/yoshimura_beta_d_sweep/")


def run_and_save_experiment(dir_name, n, beta):
    # setup
    path = EXP_DIR_ROOT / dir_name
    setup = SimulationSetup(path, overwrite=True)

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
    nodes, bars, hinges, faces, params = generate_yoshimura_geometry(n, beta, d=d, gamma=0.0, psi=0.0)
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

    for hinge in hinges:
        i = hinge[0]
        j = hinge[1]
        k = hinge[2]
        l = hinge[3]
        rest_angle = hinge[4]
        edge_type = hinge[5]
        if edge_type == 'fold' and k_fold >= 0.0:
            setup.add_hinge(nodes=[i, j, k, l], stiffness=k_fold, rest_angle=rest_angle)
        elif edge_type == 'facet' and k_facet >= 0.0:
            setup.add_hinge(nodes=[i, j, k, l], stiffness=k_facet, rest_angle=rest_angle)

    for face in faces:
        setup.add_face(face)

    n, beta, d, gamma, psi = params
    d = np.tan(beta)

    # Setup Actuation
    min_pos=0.0,
    max_pos=d,
    frequency=0.2,

    # Get initial positions
    positions = setup.nodes['positions']

    # Signal parameters
    dt_sig = 0.001
    t = np.arange(0, duration, dt_sig)
    omega = 2 * np.pi * frequency

    # Create signals for each top corner
    base_corners = node_info['base_corners']
    top_corners = node_info['top_corners']

    print(f"\nSetting up actuation:")
    print(f"  Fixed base corners: {base_corners}")
    print(f"  Actuated top corners: {top_corners}")

    # Create sinusoidal signals for top corners
    for i, idx in enumerate(top_corners):
        p0 = positions[idx]

        # generate a signal to go from min_pos to max_pos from p0[2] with given frequency
        sig = np.zeros((len(t), 3), dtype=np.float32)
        sig[:, 0] = p0[0]
        sig[:, 1] = p0[1]
        sig[:, 2] = (max_pos - min_pos) * (1 + np.cos(omega * t)) / 2

        sig_name = f"sig_top_corner_{i}"
        setup.add_signal(sig_name, sig, dt=dt_sig)
        setup.add_actuator(idx, sig_name, type='position')

    # add zero actuation to base corners
    for i, idx in enumerate(base_corners):
        p0 = positions[idx]

        # generate a signal to go from min_pos to max_pos from p0[2] with given frequency
        sig = np.zeros((len(t), 3), dtype=np.float32)
        sig[:, 0] = p0[0]
        sig[:, 1] = p0[1]
        sig[:, 2] = p0[2]

        sig_name = f"sig_base_corner_{i}"
        setup.add_signal(sig_name, sig, dt=dt_sig)
        setup.add_actuator(idx, sig_name, type='position')

    # Save Everything
    setup.save()

    print("\n[Step 2] Running Simulation...")

    exp = demlat.Simulation(path)
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
