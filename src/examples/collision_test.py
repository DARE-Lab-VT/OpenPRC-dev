"""
Yoshimura-Ori with Collision Detection Demo
============================================
Demonstrates collision detection in Yoshimura origami compression.

This example shows:
1. How to enable collision detection in setup
2. How to mark specific nodes as collidable
3. Collision during dynamic compression
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

trapezoid = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

DEMO_DIR = Path("experiments/yoshimura_collision_test")


def setup_with_collision(beta, drivers, force=False, enable_collision=True):
    """
    Setup Yoshimura with collision detection enabled.

    Args:
        beta: Sector angle (degrees)
        drivers: Number of actuated corners
        force: Use force actuation instead of position
        enable_collision: Enable collision detection
    """
    from demlat.io.experiment_setup import ExperimentSetup
    from examples.Yoshimura import Yoshimura

    print("\n[Setup] Creating Yoshimura Experiment with Collision Detection...")

    # Initialize Setup
    setup = ExperimentSetup(DEMO_DIR, overwrite=True)

    # Simulation parameters
    duration = 2.0
    dt = 0.0001
    save_interval = 0.005
    setup.set_simulation_params(duration=duration, dt=dt, save_interval=save_interval)

    # Physics with collision enabled
    setup.set_physics(
        gravity=0.0,
        damping=0.2,
        enable_collision=enable_collision,  # Enable collision detection
        collision_radius=0.01,  # 2cm collision radius
        collision_restitution=1.0  # 30% bounciness
    )

    # Build Geometry
    beta = np.deg2rad(beta)
    n = 3  # Number of sectors
    d = np.tan(beta)
    k_axial = 1000.0
    k_fold = 0.1
    k_facet = 10.0
    mass = 0.01
    damping = 3.0

    print(f"\nYoshimura Parameters:")
    print(f"  n={n}, beta={np.rad2deg(beta):.2f}°")
    print(f"  Collision: {'Enabled' if enable_collision else 'Disabled'}")

    # Generate geometry
    module = Yoshimura(n, beta, randomize=False)
    nodes, bars, hinges, faces, params = module.get_geometry()

    print(f"\nGenerated Geometry:")
    print(f"  Nodes: {len(nodes)}")
    print(f"  Bars: {len(bars)}")
    print(f"  Hinges: {len(hinges)}")
    print(f"  Faces: {len(faces)}")

    # ========================================================================
    # NODE SETUP WITH COLLISION ATTRIBUTES
    # ========================================================================

    # Strategy: Make mid-level and top nodes collidable
    # This prevents face-through-face penetration during compression

    n_levels = 2  # base, mid, top
    nodes_per_level = len(nodes) // n_levels

    for i, node_pos in enumerate(nodes):
        # Determine if node should be collidable
        level = i // nodes_per_level

        # Make mid and top level nodes collidable
        # Base nodes don't need collision (they're being compressed)
        is_collidable = (level >= 1) and enable_collision

        # Add node with collision attribute
        setup.add_node(
            node_pos,
            mass=mass,
            fixed=False,
            collidable=False  # NEW: collision attribute
        )

    print(f"\nCollision Configuration:")
    print(f"  Collidable nodes: {sum(1 for i in range(len(nodes)) if i >= nodes_per_level)}")
    print(f"  Non-collidable: {nodes_per_level}")

    # ========================================================================
    # BAR AND HINGE SETUP (unchanged)
    # ========================================================================

    for bar in bars:
        setup.add_bar(
            bar[0], bar[1],
            stiffness=k_axial,
            rest_length=bar[2],
            damping=damping
        )

    for hinge in hinges:
        if hinge[5] == 'fold':
            k = k_fold
        elif hinge[5] == 'facet':
            k = k_facet
        else:
            k = 0.0

        if k > 0.0:
            setup.add_hinge(
                nodes=[hinge[0], hinge[1], hinge[2], hinge[3]],
                stiffness=k,
                rest_angle=hinge[4],
                damping=-(k == k_fold)
            )

    for face in faces:
        setup.add_face(face)

    # ========================================================================
    # ACTUATION SETUP
    # ========================================================================

    positions = setup.nodes['positions']
    dt_sig = 0.001
    t = np.arange(0, duration, dt_sig)
    frequency = 0.5
    omega = 2 * np.pi * frequency

    if force:
        # Force actuation with aggressive compression
        force_magnitude = 200.0
        ramp_duration = 1.5

        force_signal = np.zeros((len(t), 3), dtype=np.float32)
        for i, time in enumerate(t):
            if time <= ramp_duration:
                force_signal[i, 2] = force_magnitude * (time / ramp_duration)
            elif time <= duration / 2.0:
                force_signal[i, 2] = force_magnitude
            else:
                force_signal[i, 2] = 0.0

        # Top corners: upward force
        for i in range(drivers):
            sig = -force_signal.copy()
            sig_name = f"sig_force_top_{i}"
            setup.add_signal(sig_name, sig, dt=dt_sig)
            setup.add_actuator(module.top_idx(2 * i), sig_name, type='force')

        # Base corners: downward force
        for i in range(drivers):
            sig = force_signal.copy()
            sig_name = f"sig_force_base_{i}"
            setup.add_signal(sig_name, sig, dt=dt_sig)
            setup.add_actuator(module.base_idx(2 * i), sig_name, type='force')

    else:
        # Position actuation
        min_pos = 0.0
        max_pos = d

        for i in range(drivers):
            p0 = positions[module.top_idx(2 * i)]
            sig = np.zeros((len(t), 3), dtype=np.float32)
            sig[:, 0] = p0[0]
            sig[:, 1] = p0[1]
            sig[:, 2] = np.tan(beta) - (max_pos - min_pos) * (1 - np.cos(omega * t)) / 4

            sig_name = f"sig_top_corner_{i}"
            setup.add_signal(sig_name, sig, dt=dt_sig)
            setup.add_actuator(module.top_idx(2 * i), sig_name, type='position')

        for i in range(drivers):
            p0 = positions[module.base_idx(2 * i)]
            sig = np.zeros((len(t), 3), dtype=np.float32)
            sig[:, 0] = p0[0]
            sig[:, 1] = p0[1]
            sig[:, 2] = 0 + (max_pos - min_pos) * (1 - np.cos(omega * t)) / 4

            sig_name = f"sig_base_corner_{i}"
            setup.add_signal(sig_name, sig, dt=dt_sig)
            setup.add_actuator(module.base_idx(2 * i), sig_name, type='position')

    # Save
    setup.save()
    print("\nSetup complete!")


def run():
    """Run the simulation with collision detection"""
    print("\n[Step 2] Running Simulation with Collision Detection...")
    import demlat
    from demlat.models.barhinge import BarHingeModel

    exp = demlat.Experiment(DEMO_DIR)
    eng = demlat.Engine(BarHingeModel, backend='cuda')
    eng.run(exp)

    print("\nSimulation complete!")


def analyze_collisions(demo_dir):
    """Analyze collision behavior during simulation"""
    from demlat.utils.plot_timeseries import SimulationPlotter

    plotter = SimulationPlotter(demo_dir / "output" / "simulation.h5")

    # Get data
    time, _ = plotter.get_dataset("time")
    positions, _ = plotter.get_dataset("nodes/positions")

    time = np.asarray(time).flatten()

    # Compute minimum inter-node distances over time
    n_nodes = positions.shape[1]
    min_distances = []

    for frame in positions:
        min_dist = float('inf')
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                dist = np.linalg.norm(frame[i] - frame[j])
                min_dist = min(min_dist, dist)
        min_distances.append(min_dist)

    min_distances = np.array(min_distances)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Minimum distance over time
    axes[0].plot(time, min_distances, label='Min node distance')
    axes[0].axhline(y=0.04, color='r', linestyle='--', label='2 × collision radius')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Distance (m)')
    axes[0].set_title('Minimum Inter-Node Distance (Collision Indicator)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Height compression
    top_node = 0  # Adjust based on your geometry
    heights = positions[:, top_node, 2]
    axes[1].plot(time, heights)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Height (m)')
    axes[1].set_title('Structure Height (Compression)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Statistics
    print("\n=== Collision Analysis ===")
    print(f"Minimum distance reached: {min_distances.min():.4f} m")
    print(f"Collision radius: 0.02 m")
    print(f"Detection threshold (2×radius): 0.04 m")
    print(f"Frames below threshold: {np.sum(min_distances < 0.04)}/{len(min_distances)}")


def compare_with_without_collision():
    """
    Compare simulations with and without collision detection.
    Useful for validating the collision system.
    """
    print("\n" + "=" * 60)
    print("COMPARISON: With vs Without Collision Detection")
    print("=" * 60)

    # Run without collision
    print("\n[1/2] Running WITHOUT collision detection...")
    setup_with_collision(beta=35, drivers=3, force=True, enable_collision=False)
    run()

    # Save results
    import shutil
    shutil.copytree(
        DEMO_DIR / "output",
        DEMO_DIR / "output_no_collision",
        dirs_exist_ok=True
    )

    # Run with collision
    print("\n[2/2] Running WITH collision detection...")
    setup_with_collision(beta=35, drivers=3, force=True, enable_collision=True)
    run()

    print("\nComparison complete! Analyze results in:")
    print(f"  - {DEMO_DIR / 'output'} (with collision)")
    print(f"  - {DEMO_DIR / 'output_no_collision'} (without collision)")


def visualize():
    """Visualize the simulation"""
    from demlat.utils.viz_player import visualize_experiment
    visualize_experiment(DEMO_DIR)


if __name__ == "__main__":
    import sys

    if 0:
        # Comparison mode
        compare_with_without_collision()
    else:
        # Normal mode with collision
        setup_with_collision(beta=35, drivers=3, force=False, enable_collision=True)
        run()

        # Optional: analyze collisions
        # analyze_collisions(DEMO_DIR)

        # Visualize
        visualize()
