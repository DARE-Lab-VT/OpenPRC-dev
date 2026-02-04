"""
Yoshimura-Ori Geometry Test
===========================
Generates Yoshimura origami geometry in folded configuration.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from demlat.utils.plot_timeseries import SimulationPlotter

trapezoid = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

DEMO_DIR = Path("experiments/yoshimura_test")


def setup(beta, drivers):
    from demlat.io.experiment_setup import ExperimentSetup
    from examples.yoshimura_ori_geometry import generate_yoshimura_geometry

    """Setup the Yoshimura experiment"""
    print("\n[Setup] Creating Yoshimura Experiment...")

    # Initialize Setup
    setup = ExperimentSetup(DEMO_DIR, overwrite=True)

    # Simulation parameters
    duration = 5.0
    dt = 0.0005
    save_interval = 0.01
    setup.set_simulation_params(duration=duration, dt=dt, save_interval=save_interval)
    setup.set_physics(gravity=0.0, damping=0.2)

    beta = np.deg2rad(beta)

    # Build Geometry
    n = 3
    gamma = 0.0
    d = np.tan(beta)
    psi = 0.0
    k_axial = 2000.0
    k_fold = 0.01
    k_facet = 5.0
    mass = 0.01
    damping = 2.0

    print(f"\nYoshimura Parameters:")
    print(f"  n={n}, beta={np.rad2deg(beta):.2f}°")

    # Generate geometry
    nodes, bars, hinges, faces, params = generate_yoshimura_geometry(n, beta, d=None, gamma=gamma, psi=psi)

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
        setup.add_bar(bar[0], bar[1], stiffness=k_axial, rest_length=bar[2], damping=damping)

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

    # Setup Actuation
    min_pos = 0.0
    max_pos = d
    frequency = 0.2
    duration = duration
    drivers = drivers

    positions = setup.nodes['positions']

    # Signal parameters
    dt_sig = 0.001
    t = np.arange(0, duration, dt_sig)
    omega = 2 * np.pi * frequency
    half_duration = duration / 2.0

    # Create signals for each top corner
    base_corners = node_info['base_corners']
    top_corners = node_info['top_corners']
    beta = node_info['beta']

    print(f"\nSetting up actuation:")
    print(f"  Fixed base corners: {base_corners}")
    print(f"  Actuated top corners: {top_corners}")
    use_force_actuators = True

    if use_force_actuators:
        # Force actuation: constant force for first half, then zero
        force_magnitude = 300.0  # Adjust this value as needed

        # Create force signal (force on for first half, off for second half)
        force_signal = np.zeros((len(t), 3), dtype=np.float32)
        mask = t <= half_duration

        # Top corners: apply upward force (+z)
        for i, idx in enumerate(top_corners):
            if i < drivers:
                sig = force_signal.copy()
                sig[mask, 2] = force_magnitude  # Upward force

                sig_name = f"sig_force_top_{i}"
                setup.add_signal(sig_name, sig, dt=dt_sig)
                setup.add_actuator(idx, sig_name, type='force')

        # Base corners: apply downward force (-z)
        for i, idx in enumerate(base_corners):
            if i < drivers:
                sig = force_signal.copy()
                sig[mask, 2] = -force_magnitude  # Downward force

                sig_name = f"sig_force_base_{i}"
                setup.add_signal(sig_name, sig, dt=dt_sig)
                setup.add_actuator(idx, sig_name, type='force')

    else:
        # Original position actuation
        min_pos = 0.0
        max_pos = d
        omega = 2 * np.pi * frequency

        # Create sinusoidal signals for top corners
        for i, idx in enumerate(top_corners):
            if i < drivers:
                p0 = positions[idx]

                sig = np.zeros((len(t), 3), dtype=np.float32)
                sig[:, 0] = p0[0]
                sig[:, 1] = p0[1]
                sig[:, 2] = np.tan(beta) - (max_pos - min_pos) * (1 - np.cos(omega * t)) / 4

                sig_name = f"sig_top_corner_{i}"
                setup.add_signal(sig_name, sig, dt=dt_sig)
                setup.add_actuator(idx, sig_name, type='position')

        # Add zero actuation to base corners
        for i, idx in enumerate(base_corners):
            if i < drivers:
                p0 = positions[idx]

                sig = np.zeros((len(t), 3), dtype=np.float32)
                sig[:, 0] = p0[0]
                sig[:, 1] = p0[1]
                sig[:, 2] = 0 + (max_pos - min_pos) * (1 - np.cos(omega * t)) / 4

                sig_name = f"sig_base_corner_{i}"
                setup.add_signal(sig_name, sig, dt=dt_sig)
                setup.add_actuator(idx, sig_name, type='position')

    # Save Everything
    setup.save()


def run():
    """Run the simulation"""
    print("\n[Step 2] Running Simulation...")
    import demlat
    from demlat.models.barhinge import BarHingeModel

    exp = demlat.Experiment(DEMO_DIR)
    eng = demlat.Engine(BarHingeModel, backend='cpu')
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


def show(pe):
    from demlat.utils.viz_player import visualize_experiment
    # if pe:
    #     show_pe(DEMO_DIR)
    visualize_experiment(DEMO_DIR)


if __name__ == "__main__":
    setup(beta=35.0, drivers=3)
    run()
    show(1)
