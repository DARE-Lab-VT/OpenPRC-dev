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

    setup.set_simulation_params(duration=60, dt=0.001, save_interval=0.01)
    setup.set_physics(gravity=0.0, damping=0.0)

    # Equilibrium positions: 0, 1.5, 3.0, 4.5 (rest_length = 1.5)
    # Displaced by x0 = (0.489, -0.218, -0.150, -0.268)
    setup.add_node([0.489, 0.0, 0.0], mass=1.0, fixed=False)  # node 0
    setup.add_node([1.282, 0.0, 0.0], mass=0.6, fixed=False)  # node 1
    setup.add_node([2.850, 0.0, 0.0], mass=0.6, fixed=False)  # node 2
    setup.add_node([4.232, 0.0, 0.0], mass=1.0, fixed=False)  # node 3

    # Springs (symmetric outer, different center)
    setup.add_bar(0, 1, stiffness=1.5, rest_length=1.5, damping=0.0)
    setup.add_bar(1, 2, stiffness=1.8, rest_length=1.5, damping=0.0)
    setup.add_bar(2, 3, stiffness=1.5, rest_length=1.5, damping=0.0)

    # hinges
    # setup.add_hinge(nodes=[0, 1, 2, 3], stiffness=1.0, rest_angle=np.pi / 2)
    # setup.add_hinge(nodes=[2, 3, 0, 1], stiffness=2.0, rest_angle=np.pi / 2)

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
    eng = openprc.demlat.Engine(BarHingeModel, backend='cuda')
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
    plt.plot(time, strains, '-y', label='strains')

    plt.plot(time, te.flatten(), '--r', label='total energy')

    plt.xlabel('Time (s)')
    plt.ylabel('Energy (J)')
    plt.legend()
    plt.show()

    from openprc.analysis import correlation as corr
    red = corr.Redundancy(strains)

    print(red.correlation)  # N×N Pearson matrix (Result with p-values)
    print(red.partial)  # N×N partial correlation (direct connections)
    print(red.acf)  # per-channel autocorrelation

    print(red.rank)  # effective rank via Shannon entropy of eigenspectrum
    print(red.condition)  # condition number (high = redundant)
    print(red.eigenvalues)  # raw eigenvalues, sorted descending


if __name__ == "__main__":
    setup()
    run()
    show_stats()
