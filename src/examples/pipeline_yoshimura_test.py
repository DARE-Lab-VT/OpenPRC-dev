"""
Yoshimura-Ori Dynamic Simulation
=================================
Demonstrates Yoshimura origami structure under gravity with fixed base
and optional force actuation on top nodes.
"""
import numpy as np
from pathlib import Path
import sys
import shutil

# Ensure import
sys.path.insert(0, str(Path(__file__).parent.parent))

import demlat
from demlat.models.barhinge import BarHingeModel
from demlat.io.experiment_setup import ExperimentSetup

DEMO_DIR = Path("experiments/yoshimura_dynamic_test")


def create_yoshimura_geometry(setup: ExperimentSetup, n=3, beta=np.deg2rad(30),
                              n_layers=3, k_axial=1000.0, k_fold=10.0,
                              k_facet=50.0, mass=0.01):
    """
    Generates Yoshimura-Ori geometry based on the kinematics model.
    """

    # Geometric parameters
    r = 1 / (2 * np.sin(np.pi / n))
    w = 0.5 * np.tan(beta)

    # Initial configuration parameters
    psi = 0.0
    gamma = 0.0  # Start flat
    d = 0.0

    # Transformation matrix
    def get_transform(psi, gamma, d):
        return np.array([
            [np.cos(psi) ** 2 + np.cos(gamma) * np.sin(psi) ** 2,
             (1 - np.cos(gamma)) * np.sin(psi) * np.cos(psi),
             np.sin(gamma) * np.sin(psi),
             d * np.sin(psi) * np.sin(gamma / 2)],
            [(1 - np.cos(gamma)) * np.sin(psi) * np.cos(psi),
             np.sin(psi) ** 2 + np.cos(gamma) * np.cos(psi) ** 2,
             -np.sin(gamma) * np.cos(psi),
             -d * np.cos(psi) * np.sin(gamma / 2)],
            [-np.sin(gamma) * np.sin(psi),
             np.sin(gamma) * np.cos(psi),
             np.cos(gamma),
             d * np.cos(gamma / 2)],
            [0, 0, 0, 1]
        ])

    node_indices = []
    c_axial = 5.0

    # Expected edge lengths
    polygon_edge = 2 * r * np.sin(np.pi / n)
    diagonal_edge = np.sqrt(w ** 2 + 0.25)  # Edge from corner to facet center

    T = np.eye(4)

    for layer in range(n_layers):
        layer_nodes = {'base': [], 'mid': [], 'top': []}

        # === CREATE NODES ===

        # Base nodes
        if layer == 0:
            for i in range(n):
                angle = 2 * np.pi * i / n
                pos = [r * np.sin(angle), -r * np.cos(angle), 0, 1]
                pos_transformed = T @ np.array(pos)
                node_idx = setup.add_node(
                    pos_transformed[:3],
                    mass=mass,
                    fixed=True  # Fix all base nodes
                )
                layer_nodes['base'].append(node_idx)
        else:
            layer_nodes['base'] = node_indices[layer - 1]['top']

        # Top nodes
        T_unit = get_transform(psi, gamma, d)
        for i in range(n):
            angle = 2 * np.pi * i / n
            pos = [r * np.sin(angle), -r * np.cos(angle), 0, 1]
            pos_top = T_unit @ np.array(pos)
            pos_transformed = T @ pos_top
            node_idx = setup.add_node(
                pos_transformed[:3],
                mass=mass,
                fixed=False
            )
            layer_nodes['top'].append(node_idx)

        # Mid nodes - following your kinematics structure
        # Each quadrilateral facet has: facet_center (even) and edge_mid (odd)
        for i in range(n):
            base_i = layer_nodes['base'][i]
            base_next = layer_nodes['base'][(i + 1) % n]
            top_i = layer_nodes['top'][i]
            top_next = layer_nodes['top'][(i + 1) % n]

            # Get actual positions
            pos_base_i = setup.nodes['positions'][base_i]
            pos_base_next = setup.nodes['positions'][base_next]
            pos_top_i = setup.nodes['positions'][top_i]
            pos_top_next = setup.nodes['positions'][top_next]

            # Mid node 2*i: Facet center (center of quadrilateral)
            facet_center = (pos_base_i + pos_base_next + pos_top_i + pos_top_next) / 4
            facet_idx = setup.add_node(facet_center, mass=mass, fixed=False)
            layer_nodes['mid'].append(facet_idx)

            # Mid node 2*i+1: Edge midpoint (midpoint of edge base_next--top_next)
            edge_mid = (pos_base_next + pos_top_next) / 2
            edge_idx = setup.add_node(edge_mid, mass=mass, fixed=False)
            layer_nodes['mid'].append(edge_idx)

        node_indices.append(layer_nodes)

        # === ADD BARS ===

        # 1. Top and base polygon edges
        for i in range(n):
            # Top polygon
            setup.add_bar(
                layer_nodes['top'][i],
                layer_nodes['top'][(i + 1) % n],
                stiffness=k_axial,
                rest_length=polygon_edge,
                damping=c_axial
            )

            # Base polygon (only for first layer)
            if layer == 0:
                setup.add_bar(
                    layer_nodes['base'][i],
                    layer_nodes['base'][(i + 1) % n],
                    stiffness=k_axial,
                    rest_length=polygon_edge,
                    damping=c_axial
                )

        # 2. Bars within each facet
        for i in range(n):
            base_i = layer_nodes['base'][i]
            base_next = layer_nodes['base'][(i + 1) % n]
            top_i = layer_nodes['top'][i]
            top_next = layer_nodes['top'][(i + 1) % n]
            facet_center = layer_nodes['mid'][2 * i]
            edge_mid = layer_nodes['mid'][2 * i + 1]

            # Bars from corners to facet center
            setup.add_bar(base_i, facet_center, stiffness=k_axial,
                          rest_length=diagonal_edge, damping=c_axial)
            setup.add_bar(base_next, facet_center, stiffness=k_axial,
                          rest_length=diagonal_edge, damping=c_axial)
            setup.add_bar(top_i, facet_center, stiffness=k_axial,
                          rest_length=diagonal_edge, damping=c_axial)
            setup.add_bar(top_next, facet_center, stiffness=k_axial,
                          rest_length=diagonal_edge, damping=c_axial)

            # Bars from edge_mid to adjacent corners
            setup.add_bar(base_next, edge_mid, stiffness=k_axial,
                          rest_length=w, damping=c_axial)
            setup.add_bar(top_next, edge_mid, stiffness=k_axial,
                          rest_length=w, damping=c_axial)

            # Bars connecting facet centers to edge mids
            setup.add_bar(facet_center, edge_mid, stiffness=k_axial,
                          rest_length=diagonal_edge, damping=c_axial)

            # Connect to previous facet's edge_mid (creates the boundary)
            prev_edge_mid = layer_nodes['mid'][2 * i - 1]
            setup.add_bar(facet_center, prev_edge_mid, stiffness=k_axial,
                          rest_length=diagonal_edge, damping=c_axial)

            # Bars from top/base to previous edge_mid
            setup.add_bar(top_i, prev_edge_mid, stiffness=k_axial,
                          rest_length=w, damping=c_axial)
            setup.add_bar(base_i, prev_edge_mid, stiffness=k_axial,
                          rest_length=w, damping=c_axial)

        # === ADD HINGES ===

        # Facet planarity hinges
        for i in range(n):
            base_i = layer_nodes['base'][i]
            base_next = layer_nodes['base'][(i + 1) % n]
            top_i = layer_nodes['top'][i]
            top_next = layer_nodes['top'][(i + 1) % n]
            facet_center = layer_nodes['mid'][2 * i]

            # Two triangular faces per quadrilateral, keep them coplanar
            setup.add_hinge(
                [base_i, facet_center, top_i, base_next],
                stiffness=k_facet,
                rest_angle=np.pi
            )
            setup.add_hinge(
                [top_next, facet_center, base_next, top_i],
                stiffness=k_facet,
                rest_angle=np.pi
            )

        T = T @ T_unit

    return node_indices[-1]['top'], node_indices


def step_1_setup_experiment():
    print("\n[Step 1] Setting up Yoshimura Experiment...")

    setup = ExperimentSetup(DEMO_DIR, overwrite=True)

    # 1. Configure Simulation
    setup.set_simulation_params(duration=5.0, dt=0.0005, save_interval=0.01)
    setup.set_physics(gravity=-9.8, damping=0.1)

    # 2. Build Geometry
    n = 3  # Triangle base
    beta = np.deg2rad(30)
    n_layers = 4

    top_nodes, all_nodes = create_yoshimura_geometry(
        setup,
        n=n,
        beta=beta,
        n_layers=n_layers,
        k_axial=1000.0,
        k_fold=20.0,
        k_facet=100.0,
        mass=0.01
    )

    # 3. Add actuators (optional: apply forces to top nodes)
    # Example: Apply sinusoidal vertical force
    dt_sig = 0.001
    t = np.arange(0, 5.0, dt_sig)

    # Ramp force smoothly
    force_magnitude = 2.0  # Newtons
    ramp = np.clip(t / 1.0, 0, 1.0)  # 1 second ramp
    force_signal = force_magnitude * ramp * np.sin(2 * np.pi * 0.5 * t) * 0.0

    for i, node_idx in enumerate(top_nodes):
        sig = np.zeros((len(t), 3), dtype=np.float32)
        sig[:, 2] = force_signal  # Vertical force

        sig_name = f"force_top_{i}"
        setup.add_signal(sig_name, sig, dt=dt_sig)
        setup.add_actuator(node_idx, sig_name, type='force')

    # 4. Save
    setup.save()

    print(f"Created Yoshimura structure with {n} sides, {n_layers} layers")
    print(f"Bottom nodes fixed, top nodes have applied forces")


def step_2_run_simulation():
    print("\n[Step 2] Running Simulation...")
    exp = demlat.Experiment(DEMO_DIR)
    eng = demlat.Engine(BarHingeModel, backend='cuda')
    eng.run(exp)


def check_results():
    sim_path = DEMO_DIR / "output" / "simulation.h5"
    if not sim_path.exists():
        print("Error: simulation.h5 was not created.")
        return

    import h5py
    with h5py.File(sim_path, 'r') as f:
        print("\n[Check] simulation.h5 Attributes:")
        for k, v in f.attrs.items():
            print(f"  - {k}: {v}")

        if 'time_series/nodes/positions' in f:
            frames = f['time_series/nodes/positions'].shape[0]
            nodes = f['time_series/nodes/positions'].shape[1]
            print(f"  - Frames: {frames}")
            print(f"  - Nodes: {nodes}")


if __name__ == "__main__":
    step_1_setup_experiment()
    step_2_run_simulation()
    check_results()
