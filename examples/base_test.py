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

    setup.set_simulation_params(duration=2 * np.pi, dt=0.0001, save_interval=0.01)
    setup.set_physics(gravity=0.0, damping=0.0)

    setup.add_node([0.000000, 0.0, 0.0], mass=1.0, fixed=False)
    setup.add_node([1.500000, 0.0, 0.0], mass=1.0, fixed=False)
    setup.add_node([3.000000, 0.0, 0.0], mass=1.0, fixed=False)
    setup.add_node([4.500000, 0.0, 0.0], mass=1.0, fixed=False)
    setup.add_node([6.000000, 0.0, 0.0], mass=1.0, fixed=False)
    setup.add_node([7.500000, 0.0, 0.0], mass=1.0, fixed=False)
    setup.add_node([9.000000, 0.0, 0.0], mass=1.0, fixed=False)
    setup.add_node([10.500000, 0.0, 0.0], mass=1.0, fixed=False)
    setup.add_node([12.000000, 0.0, 0.0], mass=1.0, fixed=False)
    setup.add_node([13.500000, 0.0, 0.0], mass=1.0, fixed=False)
    setup.add_node([15.000000, 0.0, 0.0], mass=1.0, fixed=False)

    # Designed springs (material properties)
    setup.add_bar(0, 1, stiffness=18.6935467411, rest_length=1.4605268928, damping=0.0)
    setup.add_bar(1, 2, stiffness=21.3768914187, rest_length=1.5073930448, damping=0.0)
    setup.add_bar(2, 3, stiffness=14.2320551208, rest_length=1.4965072664, damping=0.0)
    setup.add_bar(3, 4, stiffness=39.9769985396, rest_length=1.5102938493, damping=0.0)
    setup.add_bar(4, 5, stiffness=17.7300860720, rest_length=1.4725502149, damping=0.0)
    setup.add_bar(5, 6, stiffness=8.6141863717, rest_length=1.4483881234, damping=0.0)
    setup.add_bar(6, 7, stiffness=36.6758241498, rest_length=1.5078897916, damping=0.0)
    setup.add_bar(7, 8, stiffness=4.9410746039, rest_length=1.4747220074, damping=0.0)
    setup.add_bar(8, 9, stiffness=8.9110854393, rest_length=1.5209853006, damping=0.0)
    setup.add_bar(9, 10, stiffness=21.3482515431, rest_length=1.5303193287, damping=0.0)

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
