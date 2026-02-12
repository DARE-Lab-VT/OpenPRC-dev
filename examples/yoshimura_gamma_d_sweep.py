"""
Yoshimura Gamma-D Sweep with Hilbert Curve
===========================================
Uses a Hilbert space-filling curve to densely sample (gamma, d) space
in a single continuous trajectory.
"""
import numpy as np
from examples.yoshimura_ori_geometry import *
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import openprc.demlat
from openprc.demlat.io.experiment_setup import ExperimentSetup
from openprc.demlat.models.barhinge import BarHingeModel

EXP_DIR_ROOT = Path("experiments/yoshimura_hilbert_sweep/")


def hilbert_curve(order):
    """
    Generate 2D Hilbert curve coordinates.

    Parameters
    ----------
    order : int
        Order of the Hilbert curve (resolution = 2^order)
        order=3 -> 8x8 grid (64 points)
        order=4 -> 16x16 grid (256 points)
        order=5 -> 32x32 grid (1024 points)

    Returns
    -------
    coords : np.ndarray
        Array of shape (N, 2) with normalized coordinates in [0, 1]
    """
    # Hilbert curve generation using bit interleaving
    n = 2 ** order

    def hilbert_d2xy(n, d):
        """Convert Hilbert distance to (x, y) coordinates."""
        x = y = 0
        s = 1
        while s < n:
            rx = 1 & (d // 2)
            ry = 1 & (d ^ rx)
            if ry == 0:
                if rx == 1:
                    x = s - 1 - x
                    y = s - 1 - y
                x, y = y, x
            x += s * rx
            y += s * ry
            d //= 4
            s *= 2
        return x, y

    # Generate all points along the curve
    num_points = n * n
    coords = np.zeros((num_points, 2))

    for i in range(num_points):
        x, y = hilbert_d2xy(n, i)
        coords[i] = [x / (n - 1), y / (n - 1)]  # Normalize to [0, 1]

    return coords


def visualize_hilbert_curve(hilbert_order, gamma_range=(0.0, np.pi / 6),
                            d_range=(0.01, 0.1), beta=None, save_path=None):
    """
    Visualize the Hilbert curve in both normalized and physical (gamma, d) space.

    Parameters
    ----------
    hilbert_order : int
        Order of Hilbert curve
    gamma_range : tuple
        (min_gamma, max_gamma) in radians
    d_range : tuple
        (min_d, max_d)
    beta : float, optional
        If provided, shows the theoretical d_max line
    save_path : Path, optional
        Where to save the figure
    """
    # Generate Hilbert curve
    hilbert_coords = hilbert_curve(hilbert_order)
    resolution = 2 ** hilbert_order

    print(f"\nHilbert Curve Visualization")
    print(f"  Order: {hilbert_order}")
    print(f"  Resolution: {resolution}x{resolution}")
    print(f"  Total points: {len(hilbert_coords)}")

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 6))

    # --- Plot 1: Normalized Hilbert curve [0,1] x [0,1] ---
    ax1 = fig.add_subplot(131)

    # Draw the curve
    ax1.plot(hilbert_coords[:, 0], hilbert_coords[:, 1],
             'b-', linewidth=1, alpha=0.6, label='Hilbert path')

    # Color points by traversal order
    scatter1 = ax1.scatter(hilbert_coords[:, 0], hilbert_coords[:, 1],
                           c=np.arange(len(hilbert_coords)),
                           s=20, cmap='viridis', alpha=0.8, edgecolors='black', linewidth=0.5)

    # Mark start and end
    ax1.scatter(hilbert_coords[0, 0], hilbert_coords[0, 1],
                c='lime', s=200, marker='o', edgecolors='black',
                linewidths=3, label='Start', zorder=10)
    ax1.scatter(hilbert_coords[-1, 0], hilbert_coords[-1, 1],
                c='red', s=200, marker='s', edgecolors='black',
                linewidths=3, label='End', zorder=10)

    ax1.set_xlabel('X (normalized)', fontsize=12)
    ax1.set_ylabel('Y (normalized)', fontsize=12)
    ax1.set_title(f'Hilbert Curve (Order {hilbert_order})\n{resolution}x{resolution} grid',
                  fontsize=14, fontweight='bold')
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    # ax1.legend(loc='upper right', fontsize=10)

    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Traversal Order', fontsize=11)

    # --- Plot 2: Mapped to (gamma, d) space ---
    ax2 = fig.add_subplot(132)

    gamma_min, gamma_max = gamma_range
    d_min, d_max = d_range

    # Map to physical space
    gamma_physical = gamma_min + hilbert_coords[:, 0] * (gamma_max - gamma_min)
    d_physical = d_min + hilbert_coords[:, 1] * (d_max - d_min)

    # Draw the path
    ax2.plot(np.rad2deg(gamma_physical), d_physical,
             'b-', linewidth=1, alpha=0.6, label='Hilbert path')

    # Color by order
    scatter2 = ax2.scatter(np.rad2deg(gamma_physical), d_physical,
                           c=np.arange(len(hilbert_coords)),
                           s=20, cmap='viridis', alpha=0.8, edgecolors='black', linewidth=0.5)

    # Mark start and end
    ax2.scatter(np.rad2deg(gamma_physical[0]), d_physical[0],
                c='lime', s=200, marker='o', edgecolors='black',
                linewidths=3, label='Start', zorder=10)
    ax2.scatter(np.rad2deg(gamma_physical[-1]), d_physical[-1],
                c='red', s=200, marker='s', edgecolors='black',
                linewidths=3, label='End', zorder=10)

    # Show theoretical limit if beta provided
    if beta is not None:
        d_max_theory = np.tan(beta)
        ax2.axhline(d_max_theory, color='red', linestyle='--',
                    linewidth=2, alpha=0.5, label=f'd_max = tan(β) = {d_max_theory:.3f}')

    ax2.set_xlabel('γ (degrees)', fontsize=12)
    ax2.set_ylabel('d', fontsize=12)
    ax2.set_title(f'Hilbert Curve in (γ, d) Space\n' +
                  f'γ ∈ [{np.rad2deg(gamma_min):.1f}°, {np.rad2deg(gamma_max):.1f}°], ' +
                  f'd ∈ [{d_min:.3f}, {d_max:.3f}]',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    # ax2.legend(loc='upper right', fontsize=10)

    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Traversal Order', fontsize=11)

    # --- Plot 3: Grid reconstruction visualization ---
    ax3 = fig.add_subplot(133)

    # Create a 2D grid showing which cells are visited
    visited_grid = np.zeros((resolution, resolution))

    for idx, coord in enumerate(hilbert_coords):
        i = int(coord[1] * (resolution - 1))
        j = int(coord[0] * (resolution - 1))
        visited_grid[i, j] = idx

    # Plot as heatmap
    im3 = ax3.imshow(visited_grid, cmap='viridis', origin='lower',
                     extent=[np.rad2deg(gamma_min), np.rad2deg(gamma_max),
                             d_min, d_max],
                     aspect='auto', interpolation='nearest')

    # Overlay grid lines
    gamma_edges = np.linspace(np.rad2deg(gamma_min), np.rad2deg(gamma_max), resolution + 1)
    d_edges = np.linspace(d_min, d_max, resolution + 1)

    for g in gamma_edges:
        ax3.axvline(g, color='white', linewidth=0.3, alpha=0.3)
    for d in d_edges:
        ax3.axhline(d, color='white', linewidth=0.3, alpha=0.3)

    ax3.set_xlabel('γ (degrees)', fontsize=12)
    ax3.set_ylabel('d', fontsize=12)
    ax3.set_title(f'Grid Cell Visitation Order\n{resolution}x{resolution} cells',
                  fontsize=14, fontweight='bold')

    cbar3 = plt.colorbar(im3, ax=ax3)
    cbar3.set_label('Visitation Order', fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")

    plt.show()

    return hilbert_coords


def plot_hilbert_coverage_animation_frames(hilbert_order, gamma_range=(0.0, np.pi / 6),
                                           d_range=(0.01, 0.1), n_frames=10, save_dir=None):
    """
    Create frames showing progressive coverage of the Hilbert curve.
    Useful for understanding how the space is traversed.

    Parameters
    ----------
    hilbert_order : int
        Order of Hilbert curve
    gamma_range : tuple
        (min_gamma, max_gamma)
    d_range : tuple
        (min_d, max_d)
    n_frames : int
        Number of frames to generate
    save_dir : Path, optional
        Directory to save frames
    """
    hilbert_coords = hilbert_curve(hilbert_order)
    resolution = 2 ** hilbert_order
    total_points = len(hilbert_coords)

    gamma_min, gamma_max = gamma_range
    d_min, d_max = d_range

    # Map to physical space
    gamma_physical = gamma_min + hilbert_coords[:, 0] * (gamma_max - gamma_min)
    d_physical = d_min + hilbert_coords[:, 1] * (d_max - d_min)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Generate frames
    frame_points = np.linspace(10, total_points, n_frames, dtype=int)

    for frame_idx, n_points in enumerate(frame_points):
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot traversed path
        ax.plot(np.rad2deg(gamma_physical[:n_points]), d_physical[:n_points],
                'b-', linewidth=1.5, alpha=0.7, label='Traversed path')

        # Plot untraversed points
        if n_points < total_points:
            ax.scatter(np.rad2deg(gamma_physical[n_points:]), d_physical[n_points:],
                       c='lightgray', s=10, alpha=0.3, label='Unvisited')

        # Plot traversed points colored by order
        scatter = ax.scatter(np.rad2deg(gamma_physical[:n_points]),
                             d_physical[:n_points],
                             c=np.arange(n_points),
                             s=30, cmap='viridis', alpha=0.8,
                             edgecolors='black', linewidth=0.5)

        # Current position
        ax.scatter(np.rad2deg(gamma_physical[n_points - 1]), d_physical[n_points - 1],
                   c='red', s=200, marker='*', edgecolors='black',
                   linewidths=2, label='Current', zorder=10)

        ax.set_xlabel('γ (degrees)', fontsize=14)
        ax.set_ylabel('d', fontsize=14)
        ax.set_title(f'Hilbert Curve Traversal (Order {hilbert_order})\n' +
                     f'Progress: {n_points}/{total_points} points ' +
                     f'({100 * n_points / total_points:.1f}%)',
                     fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        # ax.legend(loc='upper right', fontsize=12)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Traversal Order', fontsize=12)

        plt.tight_layout()

        if save_dir:
            frame_path = save_dir / f"hilbert_frame_{frame_idx:03d}.png"
            plt.savefig(frame_path, dpi=150, bbox_inches='tight')
            print(f"Saved frame {frame_idx + 1}/{n_frames}: {frame_path}")

        plt.close()

    print(f"\nGenerated {n_frames} frames showing Hilbert curve traversal")


def create_hilbert_trajectory(duration, dt_sig, n, beta,
                              gamma_range=(0.0, np.pi / 6),
                              d_range=None,
                              hilbert_order=4,
                              smooth_transitions=True,
                              transition_time=0.1):
    """
    Create a trajectory following a Hilbert curve in (gamma, d) space.

    Parameters
    ----------
    duration : float
        Total duration
    dt_sig : float
        Signal time step
    n : int
        Number of polygon sides
    beta : float
        Sector angle
    gamma_range : tuple
        (min_gamma, max_gamma) in radians
    d_range : tuple, optional
        (min_d, max_d)
    hilbert_order : int
        Hilbert curve order (resolution = 2^order)
    smooth_transitions : bool
        If True, use smooth interpolation between Hilbert points
    transition_time : float
        Time to transition between adjacent Hilbert points (if smoothing)

    Returns
    -------
    t : np.ndarray
        Time array
    gamma_t : np.ndarray
        Gamma trajectory
    d_t : np.ndarray
        D trajectory
    hilbert_coords : np.ndarray
        Original Hilbert curve coordinates for reference
    """
    # Generate Hilbert curve in [0, 1] x [0, 1]
    hilbert_coords = hilbert_curve(hilbert_order)
    num_hilbert_points = len(hilbert_coords)

    print(f"Hilbert curve order: {hilbert_order}")
    print(f"Grid resolution: {2 ** hilbert_order} x {2 ** hilbert_order}")
    print(f"Total points: {num_hilbert_points}")

    # Compute d range from beta if not provided
    if d_range is None:
        d_max = np.tan(beta) * 0.95
        d_min = 0.01
        d_range = (d_min, d_max)

    gamma_min, gamma_max = gamma_range
    d_min, d_max = d_range

    # Map Hilbert coordinates to (gamma, d) space
    hilbert_gamma = gamma_min + hilbert_coords[:, 0] * (gamma_max - gamma_min)
    hilbert_d = d_min + hilbert_coords[:, 1] * (d_max - d_min)

    t = np.arange(0, duration, dt_sig)

    if smooth_transitions:
        # Smooth interpolation between Hilbert points
        # Each point gets transition_time seconds
        time_per_point = transition_time
        total_hilbert_time = num_hilbert_points * time_per_point

        if total_hilbert_time > duration:
            print(f"Warning: Hilbert traversal ({total_hilbert_time:.1f}s) exceeds duration ({duration:.1f}s)")
            print(f"Adjusting transition time to {duration / num_hilbert_points:.4f}s per point")
            time_per_point = duration / num_hilbert_points

        # Create time array for each Hilbert point
        hilbert_times = np.arange(num_hilbert_points) * time_per_point

        # Interpolate smoothly
        gamma_t = np.interp(t, hilbert_times, hilbert_gamma,
                            left=hilbert_gamma[0], right=hilbert_gamma[-1])
        d_t = np.interp(t, hilbert_times, hilbert_d,
                        left=hilbert_d[0], right=hilbert_d[-1])

        # Apply smoothing to reduce acceleration (optional)
        from scipy.ndimage import gaussian_filter1d
        sigma = int(0.05 / dt_sig)  # 50ms smoothing window
        gamma_t = gaussian_filter1d(gamma_t, sigma, mode='nearest')
        d_t = gaussian_filter1d(d_t, sigma, mode='nearest')

    else:
        # Step function - jump between points
        hilbert_times = np.linspace(0, duration, num_hilbert_points)

        # Find which Hilbert point we're at for each time
        indices = np.searchsorted(hilbert_times, t, side='right') - 1
        indices = np.clip(indices, 0, num_hilbert_points - 1)

        gamma_t = hilbert_gamma[indices]
        d_t = hilbert_d[indices]

    # Clamp to valid ranges
    gamma_t = np.clip(gamma_t, gamma_min, gamma_max)
    d_t = np.clip(d_t, d_min, d_max)

    return t, gamma_t, d_t, hilbert_coords


def compute_top_position(base_pos, psi, gamma, d):
    """Compute transformed top position."""
    T = general_transform_matrix(psi, gamma, d)
    top_homog = T @ base_pos
    return top_homog[:3]


def run_hilbert_sweep(n, beta, hilbert_order=4, psi=0.0, gamma_range=(0.0, np.pi / 6),
                      d_range=(0.01, 0.5), visualize_curve=True):
    """
    Run gamma-d sweep using Hilbert curve trajectory.

    Parameters
    ----------
    n : int
        Number of polygon sides
    beta : float
        Sector angle (radians)
    hilbert_order : int
        Hilbert curve order (4 = 16x16 = 256 points, 5 = 32x32 = 1024 points)
    psi : float
        Fixed rotation angle (radians)
    visualize_curve : bool
        If True, generate visualization of Hilbert curve before running simulation
    """

    dir_name = f"exp_n{n}_beta{np.rad2deg(beta):.1f}deg_hilbert{hilbert_order}"
    path = EXP_DIR_ROOT / dir_name
    setup = ExperimentSetup(path, overwrite=True)

    # Simulation parameters
    # Duration should be long enough to traverse the Hilbert curve smoothly
    resolution = 2 ** hilbert_order
    num_points = resolution * resolution
    transition_time = 0.05  # 50ms per point
    duration = num_points * transition_time * 1.2  # Add 20% buffer

    dt = 0.0005
    save_interval = 0.02

    print(f"\n{'=' * 60}")
    print(f"Hilbert Curve Gamma-D Sweep")
    print(f"{'=' * 60}")
    print(f"n={n}, beta={np.rad2deg(beta):.2f}°")
    print(f"Hilbert order: {hilbert_order} ({resolution}x{resolution} = {num_points} points)")
    print(f"Duration: {duration:.1f}s")
    print(f"{'=' * 60}\n")

    # Visualize Hilbert curve BEFORE running simulation
    if visualize_curve:
        print("\n[Step 0] Visualizing Hilbert Curve Mapping...")
        viz_path = path / "hilbert_curve_visualization.png"
        path.mkdir(parents=True, exist_ok=True)
        visualize_hilbert_curve(hilbert_order, gamma_range, d_range, beta, save_path=viz_path)

        # Optionally create animation frames
        # frames_dir = path / "hilbert_frames"
        # plot_hilbert_coverage_animation_frames(hilbert_order, gamma_range, d_range,
        #                                        n_frames=10, save_dir=frames_dir)

    # Configure Simulation
    setup.set_simulation_params(duration=duration, dt=dt, save_interval=save_interval)
    setup.set_physics(gravity=0.0, damping=0.2)

    # Start at a valid configuration
    gamma_init = 0.0
    d_init = np.tan(beta) * 0.5

    print("\n[Step 1] Generating Geometry...")

    # Generate initial geometry
    nodes, bars, hinges, faces, params = generate_yoshimura_geometry(
        n, beta, d=d_init, gamma=gamma_init, psi=psi
    )

    mass = 0.01
    damping = 1.0
    k_axial = 1000.0

    print(f"Generated Geometry:")
    print(f"  Nodes: {len(nodes)}")
    print(f"  Bars: {len(bars)}")
    print(f"  Faces: {len(faces)}")

    node_info = {
        'base_corners': [2 * i for i in range(n)],
        'top_corners': [4 * n + 2 * i for i in range(n)],
        'n': n,
        'total_nodes': len(nodes)
    }

    # Add nodes
    for node_pos in nodes:
        setup.add_node(node_pos, mass=mass, fixed=False)

    # Add bars
    for i, j, length in bars:
        setup.add_bar(i, j, stiffness=k_axial, rest_length=length, damping=damping)

    # Add hinges
    k_fold = 1.0
    k_facet = 0.0

    # for hinge in hinges:
    #     i, j, k, l, rest_angle, edge_type = hinge
    #     if edge_type == 'fold' and k_fold >= 0.0:
    #         setup.add_hinge(nodes=[i, j, k, l], stiffness=k_fold, rest_angle=rest_angle)
    #     elif edge_type == 'facet' and k_facet >= 0.0:
    #         setup.add_hinge(nodes=[i, j, k, l], stiffness=k_facet, rest_angle=rest_angle)

    # Add visualization faces
    for face in faces:
        setup.add_face(face)

    # Create Hilbert curve trajectory
    dt_sig = 0.001

    t, gamma_t, d_t, hilbert_coords = create_hilbert_trajectory(
        duration, dt_sig, n, beta,
        gamma_range=gamma_range,
        d_range=d_range,
        hilbert_order=hilbert_order,
        smooth_transitions=True,
        transition_time=transition_time
    )

    # Get base polygon positions
    r = 1 / (2 * np.sin(np.pi / n))
    base_polygon = np.array([
        [r * np.sin(2 * np.pi / n * i),
         -r * np.cos(2 * np.pi / n * i),
         0, 1]
        for i in range(n)
    ]).T

    positions = setup.nodes['positions']
    base_corners = node_info['base_corners']
    top_corners = node_info['top_corners']

    print(f"\nSetting up actuation:")
    print(f"  Gamma range: {np.rad2deg(gamma_range[0]):.1f}° to {np.rad2deg(gamma_range[1]):.1f}°")
    print(f"  D range: {d_range[0]:.4f} to {d_range[1]:.4f}")

    # Create actuation signals for top corners
    for corner_idx, node_idx in enumerate(top_corners):
        base_pos = base_polygon[:, corner_idx]

        sig = np.zeros((len(t), 3), dtype=np.float32)

        for i in range(len(t)):
            top_pos = compute_top_position(base_pos, psi, gamma_t[i], d_t[i])
            sig[i] = top_pos

        sig_name = f"sig_top_corner_{corner_idx}"
        setup.add_signal(sig_name, sig, dt=dt_sig)
        setup.add_actuator(node_idx, sig_name, type='position')

    # Fix base corners
    for corner_idx, node_idx in enumerate(base_corners):
        p0 = positions[node_idx]

        sig = np.zeros((len(t), 3), dtype=np.float32)
        sig[:, 0] = p0[0]
        sig[:, 1] = p0[1]
        sig[:, 2] = p0[2]

        sig_name = f"sig_base_corner_{corner_idx}"
        setup.add_signal(sig_name, sig, dt=dt_sig)
        setup.add_actuator(node_idx, sig_name, type='position')

    # Save trajectory parameters
    traj_file = path / "input" / "trajectory_params.h5"
    with h5py.File(traj_file, 'w') as f:
        f.create_dataset('time', data=t)
        f.create_dataset('gamma', data=gamma_t)
        f.create_dataset('d', data=d_t)
        f.create_dataset('hilbert_coords', data=hilbert_coords)
        f.attrs['n'] = n
        f.attrs['beta'] = beta
        f.attrs['psi'] = psi
        f.attrs['trajectory_type'] = 'hilbert'
        f.attrs['hilbert_order'] = hilbert_order
        f.attrs['hilbert_resolution'] = resolution

    # Save
    setup.save()

    print("\n[Step 2] Running Simulation...")

    exp = openprc.demlat.Experiment(path)
    eng = openprc.demlat.Engine(BarHingeModel, backend='cuda')
    eng.run(exp)

    print("\nSimulation complete!")

    return path


if __name__ == "__main__":
    # for beta in [31.7]:
    #     # Test with different Hilbert orders
    #     n = 3
    #     beta = np.deg2rad(beta)
    #     order = 7
    #
    #     # Visualize just the curve without running simulation
    #     print("Generating Hilbert curve visualization...")
    #     gamma_range = (0.0, np.pi / 4)
    #     d_range = (0.00, np.tan(beta))
    #
    #     # visualize_hilbert_curve(hilbert_order=order, gamma_range=gamma_range,
    #     #                         d_range=d_range, beta=beta)
    #
    #     # Run the actual sweep
    #     # Start with order 4 (16x16 = 256 points, ~13 seconds)
    #     exp_path = run_hilbert_sweep(n, beta, hilbert_order=order, psi=0.0, gamma_range=gamma_range,
    #                                  d_range=d_range, visualize_curve=False)
    #
    #     print(f"\nExperiment saved to: {exp_path}")

    from openprc.demlat.utils.viz_player import visualize_experiment

    n = 3
    beta = np.deg2rad(31.7)
    order = 7

    # # Visualize just the curve without running simulation
    # print("Generating Hilbert curve visualization...")
    # gamma_range = (0.0, np.pi / 4)
    # d_range = (0.00, np.tan(beta))
    #
    # visualize_hilbert_curve(hilbert_order=order, gamma_range=gamma_range,
    #                         d_range=d_range, beta=beta)

    # Run the actual sweep
    # Start with order 4 (16x16 = 256 points, ~13 seconds)
    exp_path = EXP_DIR_ROOT / f"exp_n{n}_beta{np.rad2deg(beta):.1f}deg_hilbert{order}"
    visualize_experiment(exp_path)
