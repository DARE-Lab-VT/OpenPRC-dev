"""
Yoshimura-Ori Geometry Test
===========================
Generates Yoshimura origami geometry in folded configuration.
"""
from examples.yoshimura_ori_geometry import *
from pathlib import Path
from demlat.io.experiment_setup import ExperimentSetup

import matplotlib.pyplot as plt
import numpy as np
from demlat.utils.plot_timeseries import SimulationPlotter

trapezoid = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

DEMO_DIR = Path("experiments/yoshimura_test")


def create_yoshimura_geometry(setup: ExperimentSetup,
                              n=4,
                              beta=np.pi / 6,
                              d=None,
                              gamma=0.0,
                              psi=0.0,
                              k_axial=100.0,
                              k_fold=0.0,
                              k_facet=0.0,
                              mass=0.01,
                              damping=2.0):
    print(f"\nYoshimura Parameters:")
    print(f"  n={n}, beta={np.rad2deg(beta):.2f}°")

    # Generate geometry
    nodes, bars, hinges, faces, params = generate_yoshimura_geometry(n, beta, d=d, gamma=gamma, psi=psi)

    print(f"\nGenerated Geometry:")
    print(f"  Nodes: {len(nodes)}")
    print(f"  Bars: {len(bars)}")
    print(f"  Hinges: {len(hinges)}")
    print(f"  Faces: {len(faces)}")

    node_info = {
        'base_corners': [2 * i for i in range(n)],
        'top_corners': [4 * n + 2 * i for i in range(n)],
        'n': n,
        'total_nodes': len(nodes),
        'beta': beta,
        'd': d,
    }

    # Add nodes to setup
    for i, node_pos in enumerate(nodes):
        if i in node_info['base_corners']:
            setup.add_node(node_pos, mass=mass, fixed=False)
        else:
            setup.add_node(node_pos, mass=mass, fixed=False)

    # Add bars to setup
    for bar in bars:
        i = bar[0]
        j = bar[1]
        length = bar[2]
        if len(bar) > 3:
            stiffness = bar[3]
        else:
            stiffness = 1.0
        setup.add_bar(i, j, stiffness=stiffness * k_axial, rest_length=length, damping=damping)

    # Add hinges to setup
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

    return faces, node_info, params


def setup_actuation(setup: ExperimentSetup, node_info: dict, min_pos=0.0, max_pos=1.0, frequency=0.5, duration=10.0,
                    drivers=3):
    # Get initial positions
    positions = setup.nodes['positions']

    # Signal parameters
    dt_sig = 0.001
    t = np.arange(0, duration, dt_sig)
    omega = 2 * np.pi * frequency

    # Create signals for each top corner
    base_corners = node_info['base_corners']
    top_corners = node_info['top_corners']
    beta = node_info['beta']

    print(f"\nSetting up actuation:")
    print(f"  Fixed base corners: {base_corners}")
    print(f"  Actuated top corners: {top_corners}")

    # Create sinusoidal signals for top corners
    for i, idx in enumerate(top_corners):
        if i < drivers:
            p0 = positions[idx]

            # generate a signal to go from min_pos to max_pos from p0[2] with given frequency
            sig = np.zeros((len(t), 3), dtype=np.float32)
            sig[:, 0] = p0[0]
            sig[:, 1] = p0[1]
            sig[:, 2] = np.tan(beta) - (max_pos - min_pos) * (1 - np.cos(omega * t)) / 4

            sig_name = f"sig_top_corner_{i}"
            setup.add_signal(sig_name, sig, dt=dt_sig)
            setup.add_actuator(idx, sig_name, type='position')

    # add zero actuation to base corners
    for i, idx in enumerate(base_corners):
        if i < drivers:
            p0 = positions[idx]

            # generate a signal to go from min_pos to max_pos from p0[2] with given frequency
            sig = np.zeros((len(t), 3), dtype=np.float32)
            sig[:, 0] = p0[0]
            sig[:, 1] = p0[1]
            sig[:, 2] = 0 + (max_pos - min_pos) * (1 - np.cos(omega * t)) / 4

            sig_name = f"sig_base_corner_{i}"
            setup.add_signal(sig_name, sig, dt=dt_sig)
            setup.add_actuator(idx, sig_name, type='position')


def setup_experiment(beta, drivers=3):
    """Setup the Yoshimura experiment"""
    print("\n[Setup] Creating Yoshimura Experiment...")

    # Initialize Setup
    setup = ExperimentSetup(DEMO_DIR, overwrite=True)

    # Simulation parameters
    duration = 20.0
    dt = 0.0005
    save_interval = 0.01

    # Configure Simulation
    setup.set_simulation_params(duration=duration, dt=dt, save_interval=save_interval)
    setup.set_physics(gravity=0.0, damping=0.2)

    beta = np.deg2rad(beta)
    d = np.tan(beta)

    # Build Geometry
    faces, node_info, params = create_yoshimura_geometry(
        setup,
        n=3,
        beta=beta,
        d=d,
        k_axial=1000.0,
        k_fold=5.0,
        k_facet=10.0,
        mass=0.01,
        damping=3.0
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
        duration=duration,
        drivers=drivers
    )

    # Save Everything
    setup.save()


def run_simulation():
    """Run the simulation"""
    print("\n[Step 2] Running Simulation...")
    import demlat
    from demlat.models.barhinge import BarHingeModel

    exp = demlat.Experiment(DEMO_DIR)
    eng = demlat.Engine(BarHingeModel, backend='cuda')
    eng.run(exp)

    print("\nSimulation complete!")


def show_pe(demo_dir):
    # Plot displacement vs potential energy with hysteresis

    plotter = SimulationPlotter(demo_dir / "output" / "simulation.h5")

    # Get data
    time, _ = plotter.get_dataset("time")
    positions, _ = plotter.get_dataset("nodes/positions")
    potential_energy, _ = plotter.get_dataset("system/potential_energy")
    strain_energy, _ = plotter.get_dataset("elements/bars/potential_energy")
    # potential_energy = np.sum(strain_energy, axis=1)
    print(np.array(potential_energy).shape, np.array(strain_energy).shape)

    # Flatten arrays if needed
    time = np.asarray(time).flatten()
    potential_energy = np.asarray(potential_energy).flatten()

    # Skip first 5 seconds (transient)
    t_start = 10.0
    t_end = 20.0
    mask = time >= t_start
    mask &= time <= t_end
    time = time[mask]
    positions = positions[mask]
    potential_energy = potential_energy[mask]

    # Get driven node displacement
    n = 0
    driven_node_idx = 4 * n
    z0 = positions[0, driven_node_idx, 2]
    displacement = z0 - positions[:, driven_node_idx, 2]
    displacement = np.asarray(displacement).flatten()

    plt.figure()
    plt.plot(time, displacement)
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.title('Yoshimura Origami: Displacement vs Time')

    plt.figure()
    plt.plot(displacement, potential_energy)
    plt.xlabel('displacement (m)')
    plt.ylabel('potential energy (J)')
    plt.title('Yoshimura Origami: Potential Energy vs Displacement')

    plt.figure()
    # derivative of pe vs displacement
    dPE_ddisp = np.gradient(potential_energy, displacement)
    plt.plot(displacement, dPE_ddisp)
    plt.xlabel('displacement (m)')
    plt.ylabel('dPE/ddisp (N)')
    plt.title('Yoshimura Origami: dPE/ddisp vs Displacement')
    plt.show()


if __name__ == "__main__":
    setup_experiment(beta=35.0, drivers=3)
    run_simulation()
    from demlat.utils.viz_player import visualize_experiment

    visualize_experiment(DEMO_DIR)
    # show_pe(DEMO_DIR)
