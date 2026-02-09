"""
Yoshimura-Ori Geometry Test
===========================
Generates Yoshimura origami geometry in folded configuration.
"""
from pathlib import Path
import numpy as np

DEMO_DIR = Path("experiments/double_pendulum")


def setup():
    from demlat.io.experiment_setup import ExperimentSetup
    print("\n[Setup] Creating Experiment...")

    setup = ExperimentSetup(DEMO_DIR, overwrite=True)

    setup.set_simulation_params(duration=30.0, dt=0.0005, save_interval=0.01)
    setup.set_physics(gravity=-9.8, damping=0.0)

    phi1 = np.pi / 3
    phi2 = np.pi / 4

    # nodes
    setup.add_node([0.0, 0.0, 0.0], mass=0.1, fixed=True)
    setup.add_node([np.cos(phi1), 0.0, np.sin(phi1)], mass=1.0, fixed=False)
    setup.add_node([np.cos(phi1) + np.cos(phi1 + phi2), 0.0, np.sin(phi1 + phi2)], mass=1.0, fixed=False)

    # bars
    setup.add_bar(0, 1, stiffness=-1.0, rest_length=1.0, damping=0.0)
    setup.add_bar(1, 2, stiffness=-1.0, rest_length=1.0, damping=0.0)

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


def show():
    from demlat.utils.viz_player import visualize_experiment
    visualize_experiment(DEMO_DIR)


def show_stats():
    from demlat.utils.plot_timeseries import SimulationPlotter
    import matplotlib.pyplot as plt
    plotter = SimulationPlotter(DEMO_DIR / "output" / "simulation.h5")

    # Get data
    time, _ = plotter.get_dataset("time")
    positions, _ = plotter.get_dataset("nodes/positions")
    potential_energy, _ = plotter.get_dataset("system/potential_energy")
    kinetic_energy, _ = plotter.get_dataset("system/kinetic_energy")
    strain_energy_bar, _ = plotter.get_dataset("elements/bars/potential_energy")


    print(potential_energy.shape)

    plt.plot(time, potential_energy, '-r', label='potential energy')
    plt.plot(time, kinetic_energy, '-b', label='kinetic energy')
    plt.plot(time, potential_energy + kinetic_energy, '-k', label='total energy')

    plt.xlabel('Time (s)')
    plt.ylabel('Energy (J)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    setup()
    run()
    show()
    show_stats()
