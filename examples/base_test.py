"""
Yoshimura-Ori Geometry Test
===========================
Generates Yoshimura origami geometry in folded configuration.
"""
from pathlib import Path
import numpy as np

DEMO_DIR = Path("experiments/base_test")


def setup():
    from openprc.demlat.io.simulation_setup import SimulationSetup
    print("\n[Setup] Creating Simulation...")

    setup = SimulationSetup(DEMO_DIR, overwrite=True)

    setup.set_simulation_params(duration=2.0, dt=0.0005, save_interval=0.01)
    setup.set_physics(gravity=0.0, damping=0.2)

    phi = 0

    # nodes
    setup.add_node([-1.0, 0.0, 0.0], mass=0.01, fixed=False)
    setup.add_node([1.0, 0.0, 0.0], mass=0.01, fixed=False)
    setup.add_node([0.0, np.cos(phi / 2), np.sin(phi / 2)], mass=0.01, fixed=False)
    setup.add_node([0.0, -np.cos(phi / 2), np.sin(phi / 2)], mass=0.01, fixed=False)

    # bars
    # setup.add_bar(0, 1, stiffness=10.0, rest_length=2.0, damping=0.1)
    setup.add_bar(0, 2, stiffness=10.0, rest_length=2 ** 0.5, damping=0.1)
    setup.add_bar(2, 1, stiffness=10.0, rest_length=2 ** 0.5, damping=0.1)
    setup.add_bar(1, 3, stiffness=10.0, rest_length=2 ** 0.5, damping=0.1)
    setup.add_bar(3, 0, stiffness=10.0, rest_length=2 ** 0.5, damping=0.1)

    # hinges
    setup.add_hinge(nodes=[0, 1, 2, 3], stiffness=1.0, rest_angle=np.pi / 2)
    setup.add_hinge(nodes=[2, 3, 0, 1], stiffness=2.0, rest_angle=np.pi / 2)

    # faces
    # setup.add_face([0, 1, 2])
    # setup.add_face([0, 1, 3])

    # actuation
    pass

    setup.save()


def run():
    """Run the simulation"""
    print("\n[Step 2] Running Simulation...")
    import openprc.demlat
    from openprc.demlat.models.barhinge import BarHingeModel

    exp = openprc.demlat.Simulation(DEMO_DIR)
    eng = openprc.demlat.Engine(BarHingeModel, backend='cuda')
    eng.run(exp)

    print("\nSimulation complete!")


def show():
    from openprc.demlat.utils.viz_player import visualize_experiment
    visualize_experiment(DEMO_DIR)


def show_stats():
    from openprc.demlat.utils.plot_timeseries import SimulationPlotter
    import matplotlib.pyplot as plt
    plotter = SimulationPlotter(DEMO_DIR / "output" / "simulation.h5")

    # Get data
    time, _ = plotter.get_dataset("time")
    positions, _ = plotter.get_dataset("nodes/positions")
    potential_energy, _ = plotter.get_dataset("system/potential_energy")
    kinetic_energy, _ = plotter.get_dataset("system/kinetic_energy")
    strain_energy_bar, _ = plotter.get_dataset("elements/bars/potential_energy")
    strain_energy_hinge, _ = plotter.get_dataset("elements/hinges/potential_energy")
    damping_loss, _ = plotter.get_dataset("system/damping_loss")
    potential_energy1 = np.sum(strain_energy_bar, axis=1)

    print(potential_energy.shape)

    plt.plot(time, potential_energy1, '-r', label='potential energy1')
    plt.plot(time, strain_energy_hinge, '-k', label='potential energy')
    plt.plot(time, kinetic_energy, '-b', label='kinetic energy')
    plt.plot(time, damping_loss, '-y', label='damping loss')
    # plt.plot(time, potential_energy.flatten() + kinetic_energy.flatten(), '-g', label='total energy')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (J)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    setup()
    run()
    show()
    # show_stats()
