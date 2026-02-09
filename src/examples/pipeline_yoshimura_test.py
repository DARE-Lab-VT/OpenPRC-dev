"""
Yoshimura-Ori Geometry Test
===========================
Generates Yoshimura origami geometry in folded configuration.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

trapezoid = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

DEMO_DIR = Path("experiments/yoshimura_test")


def setup(beta, drivers, force=False, amplitude=4.0):
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
    k_fold = 0.0
    k_facet = 0.01
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

    # Setup Actuation
    min_pos = 0.0
    max_pos = d
    frequency = 1.5
    duration = duration
    drivers = drivers

    positions = setup.nodes['positions']

    # Signal parameters
    dt_sig = 0.001
    t = np.arange(0, duration, dt_sig)
    omega = 2 * np.pi * frequency


    if force:
        # Force actuation: smooth ramp from 0 to force_magnitude
        force_magnitude = amplitude  # Adjust this value as needed
        ramp_duration = 0.5  # Time to reach full force (seconds)
        half_duration = duration / 2.0

        # Create force signal
        force_signal = np.zeros((len(t), 3), dtype=np.float32)

        # Create smooth ramp: 0 → force_magnitude over ramp_duration, then hold
        for i, time in enumerate(t):
            if time <= ramp_duration:
                # Linear ramp from 0 to force_magnitude
                force_signal[i, 2] = force_magnitude * (time / ramp_duration)
            else:
                # Zero after half_duration
                force_signal[i, 2] = 0.0

        # Top corners: apply upward force (+z)
        for i in range(drivers):
            sig = -force_signal.copy()
            sig_name = f"sig_force_top_{i}"
            setup.add_signal(sig_name, sig, dt=dt_sig)
            setup.add_actuator(module.top_idx(2 * i), sig_name, type='force')

        # Base corners: apply downward force (-z)
        for i in range(drivers):
            sig = force_signal.copy()  # Positive = downward
            sig_name = f"sig_force_base_{i}"
            setup.add_signal(sig_name, sig, dt=dt_sig)
            setup.add_actuator(module.base_idx(2 * i), sig_name, type='force')

        # # Mid centers force outward
        # for i in range(drivers):
        #     sig = np.zeros((len(t), 3), dtype=np.float32)
        #     sig[:, 0] = -force_signal.copy()[:, 2] * np.cos(2 * np.pi / n * (i - 1)) / 5.0
        #     sig[:, 1] = -force_signal.copy()[:, 2] * np.sin(2 * np.pi / n * (i - 1)) / 5.0
        #     sig_name = f"sig_force_mid_{i}"
        #     setup.add_signal(sig_name, sig, dt=dt_sig)
        #     setup.add_actuator(module.mid_idx(2 * i), sig_name, type='force')

    else:
        # Original position actuation
        min_pos = 0.0
        max_pos = d
        omega = 2 * np.pi * frequency

        # Create sinusoidal signals for top corners
        for i in range(drivers):
            p0 = positions[module.top_idx(2 * i)]

            sig = np.zeros((len(t), 3), dtype=np.float32)
            sig[:, 0] = p0[0]
            sig[:, 1] = p0[1]
            sig[:, 2] = np.tan(beta) - (max_pos - min_pos) * (1 - np.cos(omega * t)) / 4

            sig_name = f"sig_top_corner_{i}"
            setup.add_signal(sig_name, sig, dt=dt_sig)
            setup.add_actuator(module.top_idx(2 * i), sig_name, type='position')

        # Add zero actuation to base corners
        for i in range(drivers):
            p0 = positions[module.base_idx(2 * i)]

            sig = np.zeros((len(t), 3), dtype=np.float32)
            sig[:, 0] = p0[0]
            sig[:, 1] = p0[1]
            sig[:, 2] = 0 + (max_pos - min_pos) * (1 - np.cos(omega * t)) / 4

            sig_name = f"sig_base_corner_{i}"
            setup.add_signal(sig_name, sig, dt=dt_sig)
            setup.add_actuator(module.base_idx(2 * i), sig_name, type='position')

    # Save Everything
    setup.save()


def run():
    """Run the simulation"""
    print("\n[Step 2] Running Simulation...")
    import demlat
    from demlat.models.barhinge import BarHingeModel

    exp = demlat.Experiment(DEMO_DIR)
    eng = demlat.Engine(BarHingeModel, backend='cuda')
    eng.run(exp)

    print("\nSimulation complete!")


def show_pe(demo_dir):
    from demlat.utils.plot_timeseries import SimulationPlotter

    plotter = SimulationPlotter(demo_dir / "output" / "simulation.h5")
    time, _ = plotter.get_dataset("time")
    potential_energy, _ = plotter.get_dataset("system/potential_energy")
    kinetic_energy, _ = plotter.get_dataset("system/kinetic_energy")
    time = np.asarray(time).flatten()
    potential_energy = np.asarray(potential_energy).flatten()
    kinetic_energy = np.asarray(kinetic_energy).flatten()

    plt.figure()
    plt.plot(time, potential_energy, '-r', label='pe')
    plt.plot(time, kinetic_energy, '-b', label='ke')
    plt.plot(time, potential_energy + kinetic_energy, '-g', label='pe+ke')
    plt.xlabel('Time (s)')
    plt.ylabel('Potential Energy (J)')
    plt.legend()
    plt.show()


def show(pe):
    from demlat.utils.viz_player import visualize_experiment
    if pe:
        show_pe(DEMO_DIR)
    visualize_experiment(DEMO_DIR)


if __name__ == "__main__":
    setup(beta=35, drivers=1, force=True, amplitude=5.6)
    run()
    show(0)
