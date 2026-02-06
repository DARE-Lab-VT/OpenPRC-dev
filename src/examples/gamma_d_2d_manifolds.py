"""
Plot Hilbert Curve Sweep Results
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import griddata
from demlat.utils.plot_timeseries import SimulationPlotter
from scipy.optimize import fsolve
from scipy.ndimage import minimum_filter



def get_params(n=3, limit='vertex', beta=None):
    beta_ = beta
    gamma_ = None
    d_ = None

    if limit == 'vertex':
        # Two Pop Out
        a = 1 - np.cos(np.pi / n)
        b = 1 + np.cos(np.pi / n)

        if n % 2 == 1:
            tan_beta = np.sqrt((b - np.sqrt(b ** 2 - 4 * a ** 2)) / (2 * a))
            beta_ = np.arctan(tan_beta)
            diag_length = (1 + np.cos(np.pi / n)) / (2 * np.sin(np.pi / n))

        else:
            term1 = 2 / (np.sin(np.pi / n) ** 2)
            term2 = 1 - np.sqrt(np.cos(np.pi / n) * (2 - np.cos(np.pi / n)))
            tan_beta = np.sqrt(term1 * term2)
            beta_ = np.arctan(tan_beta)
            diag_length = 1 / (np.sin(np.pi / n))

        gamma_ = 2 * np.arcsin(np.tan(beta_) / (2 * diag_length))
        d_ = np.sin(gamma_ / 2) / np.sin(np.pi / n)
    elif limit == 'edge':
        # One Pop Out
        def equations(x, diag, sub_n):
            beta_, gamma_, theta = x

            P = np.array([1 / 2, 1 / 2 * np.tan(beta_), 0])
            Q = np.array([np.sin(theta), np.cos(theta), 0])
            R = np.array(
                [1 / 2 + np.cos(2 * np.pi / sub_n), np.sin(2 * np.pi / sub_n) * np.cos(gamma_ / 2) + np.tan(beta_) / 2,
                 np.sin(2 * np.pi / sub_n) * np.sin(gamma_ / 2)])

            eq1 = diag * np.sin(gamma_ / 2) - np.tan(beta_) / 2
            eq2 = np.sum((P - Q) * (P - Q)) - (1 + np.tan(beta_) ** 2) / 4
            eq3 = np.sum((Q - R) * (Q - R)) - (1 + np.tan(beta_) ** 2) / 4
            return [eq1, eq2, eq3]

        if n % 2 == 1:
            diag = (1 + np.cos(np.pi / n)) / (2 * np.sin(np.pi / n))
        else:
            diag = 1 / np.tan(np.pi / n)

        x0 = np.array([np.pi / (2 * n), 0.5, 1.4])
        sol = fsolve(equations, x0, args=(diag, n))

        beta_ = sol[0]
        gamma_ = sol[1]
        if n % 2 == 1:
            d_ = 2 * (diag - 1 / (2 * np.sin(np.pi / n))) * np.sin(gamma_ / 2)
        else:
            # d = np.tan(beta) / 2
            d_ = diag * np.sin(gamma_ / 2)
    elif limit == 'flat':
        beta_ = np.pi / (2 * n)
        gamma_ = 0
        d_ = 0
    elif limit == 'folded' and beta_ is not None:
        gamma_ = 0
        d_ = (np.tan(beta_) ** 2 - np.tan(np.pi / (2 * n)) ** 2) ** 0.5
    elif limit == 'deployed' and beta_ is not None:
        gamma_ = 0
        d_ = np.tan(beta_)
    else:
        raise ValueError('limit must be either vertex, edge or flat')

    return beta_, gamma_, d_


def generate_parameter_list(n_):
    # beta, psi, gamma, d
    parameter_list = np.zeros((2 * (n_ + 1), 5))
    # beta0, _, _ = get_params(n_, limit='edge')
    beta0, _, _ = get_params(n_, limit='vertex')
    _, _, d = get_params(n_, limit='folded', beta=beta0)

    parameter_list[0, 0] = beta0
    parameter_list[0, 1] = 0
    parameter_list[0, 2] = 0
    parameter_list[0, 3] = d
    parameter_list[0, 4] = 0

    _, _, d = get_params(n_, limit='deployed', beta=beta0)

    parameter_list[-1, 0] = beta0
    parameter_list[-1, 1] = 0
    parameter_list[-1, 2] = 0
    parameter_list[-1, 3] = d
    parameter_list[-1, 4] = 3

    beta, gamma, d = get_params(n_, limit='edge')
    for i in range(1, n_ + 1):
        parameter_list[i, 0] = beta
        parameter_list[i, 1] = np.pi / n_ + (i - 1) * 2 * np.pi / n_
        parameter_list[i, 2] = gamma
        parameter_list[i, 3] = d
        parameter_list[i, 4] = 1

    beta, gamma, d = get_params(n_, limit='vertex')
    for i in range(n_ + 1, 2 * n_ + 1):
        parameter_list[i, 0] = beta
        parameter_list[i, 1] = (i - n_) * 2 * np.pi / n_
        parameter_list[i, 2] = gamma
        parameter_list[i, 3] = d
        parameter_list[i, 4] = 2

    return parameter_list


def plot_hilbert_manifold(exp_path):
    """
    Plot 2D potential energy manifold from Hilbert curve sweep.
    Uses the full dense trajectory with proper interpolation to grid.
    """
    exp_path = Path(exp_path)

    params = generate_parameter_list(3)

    # Initialize plotter
    plotter = SimulationPlotter(exp_path / "output" / "simulation.h5")

    # Load trajectory parameters
    traj_file = exp_path / "input" / "trajectory_params.h5"
    with h5py.File(traj_file, 'r') as f:
        t_traj = f['time'][:]
        gamma_traj = f['gamma'][:]
        d_traj = f['d'][:]
        hilbert_coords = f['hilbert_coords'][:]
        n = f.attrs['n']
        beta = f.attrs['beta']
        hilbert_order = f.attrs['hilbert_order']
        resolution = f.attrs['hilbert_resolution']

    print(f"Hilbert Curve Sweep Analysis")
    print(f"  n={n}, beta={np.rad2deg(beta):.2f}°")
    print(f"  Hilbert order: {hilbert_order}")
    print(f"  Hilbert grid resolution: {resolution}x{resolution} = {len(hilbert_coords)} points")
    print(f"  Trajectory signal points: {len(t_traj)}")

    # Get simulation data
    time, _ = plotter.get_dataset("time")
    strain_energy, _ = plotter.get_dataset("elements/bars/potential_energy")
    potential_energy = np.sum(strain_energy, axis=1)

    # Flatten
    time = np.asarray(time).flatten()
    potential_energy = np.asarray(potential_energy).flatten()

    print(f"  Simulation output points: {len(time)}")
    print(f"  Energy range: {potential_energy.min():.6f} to {potential_energy.max():.6f}")

    # Interpolate gamma and d to simulation time points
    gamma_sim = np.interp(time, t_traj, gamma_traj)
    d_sim = np.interp(time, t_traj, d_traj)

    # Create a DENSE grid for visualization (matching or exceeding Hilbert resolution)
    # Use the actual Hilbert resolution, not a coarser one
    grid_resolution = resolution  # Use same resolution as Hilbert curve

    gamma_min, gamma_max = gamma_traj.min(), gamma_traj.max()
    d_min, d_max = d_traj.min(), d_traj.max()

    gamma_grid = np.linspace(gamma_min, gamma_max, grid_resolution)
    d_grid = np.linspace(d_min, d_max, grid_resolution)
    gamma_mesh, d_mesh = np.meshgrid(gamma_grid, d_grid)

    # Interpolate energy onto the dense grid using all simulation points
    # This uses the FULL trajectory, not just Hilbert nodes
    print(f"\nInterpolating {len(gamma_sim)} trajectory points onto {grid_resolution}x{grid_resolution} grid...")

    points = np.column_stack([gamma_sim, d_sim])
    energy_grid = griddata(points, potential_energy,
                           (gamma_mesh, d_mesh),
                           method='cubic',
                           fill_value=np.nan)
    
    # Also interpolate time onto the grid to find corresponding times for minima
    time_grid = griddata(points, time,
                         (gamma_mesh, d_mesh),
                         method='nearest')

    # Fill any NaN values with nearest neighbor interpolation
    if np.any(np.isnan(energy_grid)):
        print(f"  Filling {np.sum(np.isnan(energy_grid))} NaN values with nearest neighbor...")
        energy_grid_nearest = griddata(points, potential_energy,
                                       (gamma_mesh, d_mesh),
                                       method='nearest')
        nan_mask = np.isnan(energy_grid)
        energy_grid[nan_mask] = energy_grid_nearest[nan_mask]

    print(f"  Grid energy range: {np.nanmin(energy_grid):.6f} to {np.nanmax(energy_grid):.6f}")

    # Create figure
    fig, axes = plt.subplots(1, 1, figsize=(7, 6))

    # Plot 1: Energy manifold
    ax1 = axes
    im1 = ax1.contourf(np.rad2deg(gamma_mesh), d_mesh, energy_grid,
                       levels=1000, cmap='jet')

    # # Add contour lines for clarity
    # contours = ax1.contour(np.rad2deg(gamma_mesh), d_mesh, energy_grid,
    #                        levels=10, colors='black', linewidths=0.5, alpha=0.3)

    ax1.set_xlabel('γ (degrees)', fontsize=14)
    ax1.set_ylabel('d', fontsize=14)
    ax1.set_title(f'Potential Energy Manifold (Hilbert {resolution}x{resolution})\n' +
                  f'n={n}, β={np.rad2deg(beta):.1f}°, interpolated from {len(time)} points',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # # plot params with legend
    for param in params:
        beta, psi, gamma, d, idx = param
        ax1.plot(np.rad2deg(gamma), d, 'r*')
    
    # draw and print all the local minimas
    # Use minimum filter to find local minima
    local_min = minimum_filter(energy_grid, size=3) == energy_grid
    
    # Filter out edge cases and high energy points if needed
    # For now, just get all local minima
    min_indices = np.where(local_min)
    
    print("\nLocal Minima found:")
    print(f"{'Index':<5} {'Gamma (deg)':<15} {'d':<10} {'Energy':<15} {'Time (s)':<10}")
    print("-" * 65)
    
    for i in range(len(min_indices[0])):
        row, col = min_indices[0][i], min_indices[1][i]
        
        # Get coordinates and energy
        gamma_val = gamma_grid[col]
        d_val = d_grid[row]
        energy_val = energy_grid[row, col]
        time_val = time_grid[row, col]
        
        # Print
        print(f"{i:<5} {np.rad2deg(gamma_val):<15.4f} {d_val:<10.4f} {energy_val:<15.6f} {time_val:<10.3f}")
        
        # Plot on the manifold
        ax1.plot(np.rad2deg(gamma_val), d_val, 'w+', markersize=10, markeredgewidth=2)
        ax1.plot(np.rad2deg(gamma_val), d_val, 'k+', markersize=10, markeredgewidth=1)

    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Potential Energy', fontsize=12, fontweight='bold')

    # # Plot 2: Complete Hilbert curve trajectory in parameter space
    # ax2 = axes[1]
    #
    # # Downsample for visualization if trajectory is too dense
    # max_plot_points = 5000
    # if len(gamma_traj) > max_plot_points:
    #     step = len(gamma_traj) // max_plot_points
    #     gamma_plot = gamma_traj[::step]
    #     d_plot = d_traj[::step]
    #     colors_plot = np.arange(len(gamma_plot))
    #     print(f"  Downsampling trajectory for plot: {len(gamma_traj)} -> {len(gamma_plot)} points")
    # else:
    #     gamma_plot = gamma_traj
    #     d_plot = d_traj
    #     colors_plot = np.arange(len(gamma_plot))

    # # Plot the complete trajectory
    # ax2.plot(np.rad2deg(gamma_plot), d_plot,
    #          'k-', alpha=0.15, linewidth=0.5, label='Trajectory path', zorder=1)
    #
    # # Show trajectory with color indicating progression
    # scatter = ax2.scatter(np.rad2deg(gamma_plot), d_plot,
    #                       c=colors_plot, s=8, cmap='viridis', alpha=0.7, zorder=2)
    #
    # # # Mark start and end
    # # ax2.scatter(np.rad2deg(gamma_traj[0]), d_traj[0],
    # #             c='lime', s=200, marker='o',
    # #             edgecolors='black', linewidths=2.5, label='Start', zorder=10)
    # # ax2.scatter(np.rad2deg(gamma_traj[-1]), d_traj[-1],
    # #             c='red', s=200, marker='s',
    # #             edgecolors='black', linewidths=2.5, label='End', zorder=10)
    #
    # # Overlay the Hilbert grid points
    # gamma_hilbert_grid = gamma_min + hilbert_coords[:, 0] * (gamma_max - gamma_min)
    # d_hilbert_grid = d_min + hilbert_coords[:, 1] * (d_max - d_min)
    # ax2.scatter(np.rad2deg(gamma_hilbert_grid), d_hilbert_grid,
    #             c='white', s=15, marker='+', alpha=0.4,
    #             linewidths=0.5, label=f'Hilbert grid ({len(hilbert_coords)} pts)', zorder=3)
    #
    # ax2.set_xlabel('γ (degrees)', fontsize=14)
    # ax2.set_ylabel('d', fontsize=14)
    # ax2.set_title(f'Hilbert Curve Trajectory\n' +
    #               f'{resolution}x{resolution} grid, {len(t_traj)} trajectory samples, {len(time)} simulation points',
    #               fontsize=13, fontweight='bold')
    # # ax2.legend(fontsize=10, loc='upper right')
    # ax2.grid(True, alpha=0.3, linestyle='--')
    #
    # cbar2 = plt.colorbar(scatter, ax=ax2)
    # cbar2.set_label('Trajectory Progression', fontsize=12)
    #
    plt.tight_layout()

    fig_path = exp_path / "hilbert_manifold.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to: {fig_path}")

    plt.show()

    # Statistics
    min_idx = np.argmin(potential_energy)
    max_idx = np.argmax(potential_energy)

    print(f"\n{'=' * 60}")
    print(f"Energy Minimum: {potential_energy[min_idx]:.6f}")
    print(f"  at γ = {np.rad2deg(gamma_sim[min_idx]):.2f}°, d = {d_sim[min_idx]:.4f}")
    print(f"  at time t = {time[min_idx]:.3f}s")
    print(f"\nEnergy Maximum: {potential_energy[max_idx]:.6f}")
    print(f"  at γ = {np.rad2deg(gamma_sim[max_idx]):.2f}°, d = {d_sim[max_idx]:.4f}")
    print(f"  at time t = {time[max_idx]:.3f}s")
    print(f"\nEnergy Statistics:")
    print(f"  Mean: {potential_energy.mean():.6f}")
    print(f"  Std Dev: {potential_energy.std():.6f}")
    print(f"  Range: {potential_energy.max() - potential_energy.min():.6f}")

    # Check coverage
    unique_gamma = len(np.unique(np.round(gamma_sim, 6)))
    unique_d = len(np.unique(np.round(d_sim, 6)))
    print(f"\nTrajectory Coverage:")
    print(f"  Unique γ values: {unique_gamma}")
    print(f"  Unique d values: {unique_d}")
    print(f"  Theoretical grid: {resolution}x{resolution}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    exp_path = Path("experiments/yoshimura_hilbert_sweep/exp_n3_beta31.7deg_hilbert7")
    plot_hilbert_manifold(exp_path)
