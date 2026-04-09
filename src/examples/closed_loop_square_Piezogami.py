"""
Piezogami Kirigami geometry test
=====================================================
This example demonstrates how to use the ClosedLoopSquare geometry
class to export nodes and bars to an external setup system.
"""
from pathlib import Path
import numpy as np

from src.examples.closed_loop_square import ClosedLoopSquare

DEMO_DIR = Path("experiments/base_test")


def setup():
    from src.demlat.io.experiment_setup import ExperimentSetup
    print("\n[Setup] Creating Experiment...")

    setup = ExperimentSetup(DEMO_DIR, overwrite=True)

    simulation_time = 5
    setup.set_simulation_params(duration=simulation_time, dt=0.0003, save_interval=0.01)
    setup.set_physics(gravity=0, damping=0.1)

    # Create the geometry
    Piezogami = ClosedLoopSquare(
        num_loops=4,  # number of concentric loops
        side=130e-2,  # sample side length, m
        width=14.5e-2,  # ligament width for within-loop bars, m
        joint_width=100e-2,  # ligament width for inter-loop joint bars, m
        cut_width=1.5e-2,  # cut width, m
        mass=5e-2,  # sample total mass, kg
        E=2e3,  # Young's modulus, Pa
        thickness=2e-3,  # thickness, m
        damping=1  # damping coefficient
    )
    # Show geometry summary
    Piezogami.summary()

    # nodes
    for node in Piezogami.nodes:
        is_fixed = (node.boundary_condition == 'fixed')
        setup.add_node(
            node.pos.tolist(),
            mass=node.mass,
            fixed=is_fixed
        )

    # bars
    for bar in Piezogami.bars:
        setup.add_bar(
            bar.node_a_idx,
            bar.node_b_idx,
            stiffness=bar.stiffness,
            rest_length=bar.rest_length *0.8,
            damping=bar.damping
        )



    # hinges
    # for hinge in Piezogami.hinges:
    #     # print(hinge.stiffness)
    #     setup.add_hinge(
    #         nodes=[hinge.node_a_idx,
    #                hinge.node_b_idx,
    #                hinge.node_c_idx,
    #                hinge.node_d_idx],
    #         stiffness=hinge.stiffness,
    #         rest_angle=hinge.rest_angle
    #     )
        # print(
        #     f"hinge index = {hinge.index}\n node a = {hinge.node_a_idx}\n node b = {hinge.node_b_idx}\n node c = {hinge.node_c_idx}\n node d = {hinge.node_d_idx}\n, stiffness = {hinge.stiffness}, rest_angle = {hinge.rest_angle}")

    # faces
    # setup.add_face([0, 1, 2])
    # setup.add_face([0, 1, 3])


    # actuation
    #
    # Signal Parameters
    dt_sig = 0.001
    frequency = 2.75
    amplitude = 0.5
    t = np.arange(0, simulation_time, dt_sig)
    omega = 2 * np.pi * frequency
    input_signal = amplitude * np.sin(omega * t)

    fixed_nodes = [Piezogami.nodes[0, 0], Piezogami.nodes[0, 1], Piezogami.nodes[0, 2], Piezogami.nodes[0, 3]]

    for node in fixed_nodes:
        sig = np.zeros((len(t), 3), dtype=np.float32)
        sig[:, 0] = node.pos[0]
        sig[:, 1] = node.pos[1]
        sig[:, 2] = node.pos[2] + input_signal

        sig_name = f"input_sinal_loop{node.loop}_node{node.angle}"
        setup.add_signal(sig_name, sig, dt=dt_sig)
        setup.add_actuator(node.index, sig_name, type='position')

    setup.save()


def run():
    """Run the simulation"""
    print("\n[Step 2] Running Simulation...")
    import src.demlat
    from src.demlat.models.barhinge import BarHingeModel

    exp = src.demlat.Experiment(DEMO_DIR)
    eng = src.demlat.Engine(BarHingeModel, backend='cpu')
    eng.run(exp)

    print("\nSimulation complete!")


def show():
    from src.demlat.utils.viz_player import visualize_experiment
    visualize_experiment(DEMO_DIR)


def show_stats():
    from src.demlat.utils.plot_timeseries import SimulationPlotter
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
    show_stats()
