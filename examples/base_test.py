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

    setup.set_simulation_params(duration=60.0, dt=0.001, save_interval=0.01)
    setup.set_physics(gravity=0.0, damping=0.1)

    setup.add_node([0, 0, 0], 5.0, True)
    setup.add_node([1, -0.5, 0], 2.5, False)
    setup.add_node([1, 0.5, 0], 2.5, False)
    setup.add_node([2, 0, 0], 5.0, False)

    setup.add_bar(0, 1, 10.0, 2.0)
    setup.add_bar(0, 2, 10.0, 2.0)
    setup.add_bar(1, 3, 10.0, 2.0)
    setup.add_bar(2, 3, 10.0, 2.0)
    setup.add_bar(1, 2, 10.0, 2.0)

    setup.add_hinge([1, 2, 0, 3], 0.5, 0.0, np.pi / 2)

    # faces
    # setup.add_face([0, 1, 2])
    # setup.add_face([0, 1, 3])

    # actuation

    setup.save()


def run():
    """Run the simulation"""
    print("\n[Step 2] Running Simulation...")
    import openprc.demlat
    from openprc.demlat.models.barhinge import BarHingeModel

    sim = openprc.demlat.Simulation(DEMO_DIR)
    eng = openprc.demlat.Engine(BarHingeModel, backend='jax')
    eng.run(sim)
    sim.show()

    print("\nSimulation complete!")


def show_stats():
    from openprc.demlat.utils.data_parser import SimulationData
    import matplotlib.pyplot as plt
    plotter = SimulationData(DEMO_DIR / "output" / "simulation.h5")

    # Get data
    time, _ = plotter.get_dataset("time")

    potential_energy, _ = plotter.get_dataset("system/potential_energy")
    kinetic_energy, _ = plotter.get_dataset("system/kinetic_energy")

    strains, _ = plotter.get_dataset("elements/bars/strain")
    kes, _ = plotter.get_dataset("nodes/kinetic_energy")
    te, _ = plotter.get_dataset("system/total_energy")

    # plt.plot(time, potential_energy, '-b', label='potential energy')
    # plt.plot(time, kinetic_energy, '-y', label='kinetic energy')
    plt.plot(time, strains, label='strains')

    plt.plot(time, te.flatten(), '--r', label='total energy')

    plt.xlabel('Time (s)')
    plt.ylabel('Energy (J)')
    plt.legend()
    plt.show()

    from openprc.analysis import correlation as corr
    red = corr.Redundancy(strains)

    print(red.correlation)  # N×N Pearson matrix (Result with p-values)
    print(red.rank)  # effective rank via Shannon entropy of eigenspectrum


if __name__ == "__main__":
    setup()
    run()
    show_stats()
