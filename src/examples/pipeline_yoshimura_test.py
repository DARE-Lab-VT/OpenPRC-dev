"""
Yoshimura-Ori Geometry Test
===========================
Generates Yoshimura origami geometry in folded configuration.
"""
from examples.yoshimura_ori_geometry import *
from pathlib import Path
from demlat.io.experiment_setup import ExperimentSetup

import matplotlib.pyplot as plt
import numpy as np
from demlat.utils.plot_timeseries import SimulationPlotter

trapezoid = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

DEMO_DIR = Path("experiments/yoshimura_test")


def create_yoshimura_geometry(setup: ExperimentSetup,
                              n=4,
                              beta=np.pi / 6,
                              d=None,
                              gamma=0.0,
                              psi=0.0,
                              k_axial=100.0,
                              k_fold=0.00,
                              k_facet=0.0,
                              mass=0.01,
                              damping=2.0):

    print(f"\nYoshimura Parameters:")
    print(f"  n={n}, beta={np.rad2deg(beta):.2f}°")

    # Generate geometry
    nodes, bars, hinges, faces, params = generate_yoshimura_geometry(n, beta, d=d, gamma=gamma, psi=psi)

    print(f"\nGenerated Geometry:")
    print(f"  Nodes: {len(nodes)}")
    print(f"  Bars: {len(bars)}")
    print(f"  Hinges: {len(hinges)}")
    print(f"  Faces: {len(faces)}")

    node_info = {
        'base_corners': [2 * i for i in range(n)],
        'top_corners': [4 * n + 2 * i for i in range(n)],
        'n': n,
        'total_nodes': len(nodes)
    }

    # Add nodes to setup
    for i, node_pos in enumerate(nodes):
        setup.add_node(node_pos, mass=mass, fixed=False)

    # Add bars to setup
    for bar in bars:
        i = bar[0]
        j = bar[1]
        length = bar[2]
        if len(bar) > 3:
            stiffness = bar[3]
        else:
            stiffness = 1.0
        setup.add_bar(i, j, stiffness=stiffness * k_axial, rest_length=length, damping=damping)

    # Add hinges to setup
    for hinge in hinges:
        i = hinge[0]
        j = hinge[1]
        k = hinge[2]
        l = hinge[3]
        rest_angle = hinge[4]
        edge_type = hinge[5]
        if edge_type == 'fold' and k_fold >= 0.0:
            setup.add_hinge(nodes=[i, j, k, l], stiffness=k_fold, rest_angle=rest_angle)
        elif edge_type == 'facet' and k_facet >= 0.0:
            setup.add_hinge(nodes=[i, j, k, l], stiffness=k_facet, rest_angle=rest_angle)

    for face in faces:
        setup.add_face(face)

    return faces, node_info, params


def setup_actuation(setup: ExperimentSetup, node_info: dict, min_pos=0.0, max_pos=1.0, frequency=0.5, duration=10.0):
    # Get initial positions
    positions = setup.nodes['positions']

    # Signal parameters
    dt_sig = 0.001
    t = np.arange(0, duration, dt_sig)
    omega = 2 * np.pi * frequency

    # Create signals for each top corner
    base_corners = node_info['base_corners']
    top_corners = node_info['top_corners']

    print(f"\nSetting up actuation:")
    print(f"  Fixed base corners: {base_corners}")
    print(f"  Actuated top corners: {top_corners}")

    # Create sinusoidal signals for top corners
    for i, idx in enumerate(top_corners):
        p0 = positions[idx]

        # generate a signal to go from min_pos to max_pos from p0[2] with given frequency
        sig = np.zeros((len(t), 3), dtype=np.float32)
        sig[:, 0] = p0[0]
        sig[:, 1] = p0[1]
        sig[:, 2] = (max_pos - min_pos) * (1 + np.cos(omega * t)) / 2

        sig_name = f"sig_top_corner_{i}"
        setup.add_signal(sig_name, sig, dt=dt_sig)
        setup.add_actuator(idx, sig_name, type='position')

    # add zero actuation to base corners
    for i, idx in enumerate(base_corners):
        p0 = positions[idx]

        # generate a signal to go from min_pos to max_pos from p0[2] with given frequency
        sig = np.zeros((len(t), 3), dtype=np.float32)
        sig[:, 0] = p0[0]
        sig[:, 1] = p0[1]
        sig[:, 2] = p0[2]

        sig_name = f"sig_base_corner_{i}"
        setup.add_signal(sig_name, sig, dt=dt_sig)
        setup.add_actuator(idx, sig_name, type='position')


def setup_experiment(beta):
    """Setup the Yoshimura experiment"""
    print("\n[Setup] Creating Yoshimura Experiment...")

    # Initialize Setup
    setup = ExperimentSetup(DEMO_DIR, overwrite=True)

    # Simulation parameters
    duration = 20.0
    dt = 0.0005
    save_interval = 0.01

    # Configure Simulation
    setup.set_simulation_params(duration=duration, dt=dt, save_interval=save_interval)
    setup.set_physics(gravity=0.0, damping=0.2)

    beta = np.deg2rad(beta)
    d = np.tan(beta) - 0.001

    # Build Geometry
    faces, node_info, params = create_yoshimura_geometry(
        setup,
        n=3,
        beta=beta,
        d=d,
        k_axial=1000.0,
        k_fold=1.0,
        k_facet=0.0,
        mass=0.01,
        damping=3.0
    )

    n, beta, d, gamma, psi = params
    d = np.tan(beta)

    # Setup Actuation
    setup_actuation(
        setup,
        node_info,
        min_pos=0.0,
        max_pos=d,
        frequency=0.2,
        duration=duration
    )

    # Save Everything
    setup.save()


def run_simulation():
    """Run the simulation"""
    print("\n[Step 2] Running Simulation...")
    import demlat
    from demlat.models.barhinge import BarHingeModel

    exp = demlat.Experiment(DEMO_DIR)
    eng = demlat.Engine(BarHingeModel, backend='cuda')
    eng.run(exp)

    print("\nSimulation complete!")


def show_pe(demo_dir):
    # Plot displacement vs potential energy with hysteresis

    plotter = SimulationPlotter(demo_dir / "output" / "simulation.h5")

    # Get data
    time, _ = plotter.get_dataset("time")
    positions, _ = plotter.get_dataset("nodes/positions")
    # potential_energy, _ = plotter.get_dataset("system/potential_energy")
    strain_energy, _ = plotter.get_dataset("elements/bars/potential_energy")
    potential_energy = np.sum(strain_energy, axis=1)
    print(np.array(potential_energy).shape, np.array(strain_energy).shape)

    # Flatten arrays if needed
    time = np.asarray(time).flatten()
    potential_energy = np.asarray(potential_energy).flatten()

    # Skip first 5 seconds (transient)
    t_start = 0.0
    t_end = 20.0
    mask = time >= t_start
    mask &= time <= t_end
    time = time[mask]
    positions = positions[mask]
    potential_energy = potential_energy[mask]

    # Get driven node displacement
    n = 4
    driven_node_idx = 4 * n
    z0 = positions[0, driven_node_idx, 2]
    displacement = z0 - positions[:, driven_node_idx, 2]
    displacement = np.asarray(displacement).flatten()

    # # Detect loading/unloading based on displacement rate
    # d_disp = np.gradient(displacement, time)
    # loading = d_disp > 0
    # unloading = d_disp <= 0
    #
    # # Bin data for averaging
    # n_bins = 100
    # disp_min, disp_max = displacement.min(), displacement.max()
    # bin_edges = np.linspace(disp_min, disp_max, n_bins + 1)
    # bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    #
    # def compute_envelope_stats(disp, energy):
    #     """Compute mean, min, max per bin."""
    #     means = np.full(n_bins, np.nan)
    #     mins = np.full(n_bins, np.nan)
    #     maxs = np.full(n_bins, np.nan)
    #
    #     bin_idx = np.digitize(disp, bin_edges) - 1
    #     bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    #
    #     for i in range(n_bins):
    #         mask = bin_idx == i
    #         if np.sum(mask) > 0:
    #             vals = energy[mask]
    #             means[i] = np.mean(vals)
    #             mins[i] = np.min(vals)
    #             maxs[i] = np.max(vals)
    #
    #     return means, mins, maxs
    #
    # # Compute stats for loading and unloading
    # load_mean, load_min, load_max = compute_envelope_stats(
    #     displacement[loading], potential_energy[loading]
    # )
    # unload_mean, unload_min, unload_max = compute_envelope_stats(
    #     displacement[unloading], potential_energy[unloading]
    # )
    #
    # # Remove NaN for plotting
    # def clean_data(x, y_mean, y_min, y_max):
    #     valid = ~np.isnan(y_mean)
    #     return x[valid], y_mean[valid], y_min[valid], y_max[valid]
    #
    # bc_load, lm, lmin, lmax = clean_data(bin_centers, load_mean, load_min, load_max)
    # bc_unload, um, umin, umax = clean_data(bin_centers, unload_mean, unload_min, unload_max)
    #
    # # Compute derivative of PE vs displacement
    # # We'll use the mean curves for this
    # dPE_ddisp_load = np.gradient(lm, bc_load)
    # dPE_ddisp_unload = np.gradient(um, bc_unload)
    #
    # # Plot
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    #
    # # Subplot 1: PE vs Displacement
    # # Loading: line + envelope
    # ax1.fill_between(bc_load, lmin, lmax, alpha=0.25, color='royalblue', label='Loading envelope')
    # ax1.plot(bc_load, lm, 'royalblue', lw=2, label='Loading (mean)')
    #
    # # Unloading: line + envelope
    # ax1.fill_between(bc_unload, umin, umax, alpha=0.25, color='orangered', label='Unloading envelope')
    # ax1.plot(bc_unload, um, 'orangered', lw=2, label='Unloading (mean)')
    #
    # # Calculate hysteresis area (between mean curves)
    # # Interpolate to common grid for area calculation
    # common_disp = np.linspace(
    #     max(bc_load.min(), bc_unload.min()),
    #     min(bc_load.max(), bc_unload.max()),
    #     200
    # )
    # load_interp = np.interp(common_disp, bc_load, lm)
    # unload_interp = np.interp(common_disp, bc_unload, um)
    # hysteresis_area = np.abs(trapezoid(load_interp - unload_interp, common_disp))
    #
    # ax1.text(0.05, 0.95, f'Hysteresis Area ≈ {hysteresis_area:.4f} J',
    #         transform=ax1.transAxes, fontsize=11, verticalalignment='top',
    #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    #
    # ax1.set_ylabel('Total Potential Energy (J)', fontsize=12)
    # ax1.set_title('Yoshimura Origami: Displacement vs Potential Energy (t > 5s)', fontsize=14)
    # ax1.grid(True, alpha=0.3)
    # ax1.legend(loc='upper right')
    #
    # # Subplot 2: dPE/ddisp vs Displacement
    # ax2.plot(bc_load, dPE_ddisp_load, 'royalblue', lw=2, label='dPE/ddisp (Loading)')
    # ax2.plot(bc_unload, dPE_ddisp_unload, 'orangered', lw=2, label='dPE/ddisp (Unloading)')
    #
    # ax2.set_xlabel('Displacement (m)', fontsize=12)
    # ax2.set_ylabel('d(PE)/d(disp) (N)', fontsize=12)
    # ax2.set_title('Derivative of Potential Energy vs Displacement', fontsize=14)
    # ax2.grid(True, alpha=0.3)
    # ax2.legend(loc='upper right')
    #
    # fig.tight_layout()
    # plt.savefig(DEMO_DIR / "displacement_vs_energy_hysteresis.png", dpi=150)
    # plt.show()

    plt.figure()
    plt.plot(time, displacement)
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.title('Yoshimura Origami: Displacement vs Time')

    plt.figure()
    plt.plot(displacement, potential_energy)
    plt.xlabel('displacement (m)')
    plt.ylabel('potential energy (J)')
    plt.title('Yoshimura Origami: Potential Energy vs Displacement')

    plt.figure()
    # derivative of pe vs displacement
    dPE_ddisp = np.gradient(potential_energy, displacement)
    plt.plot(displacement, dPE_ddisp)
    plt.xlabel('displacement (m)')
    plt.ylabel('dPE/ddisp (N)')
    plt.title('Yoshimura Origami: dPE/ddisp vs Displacement')
    plt.show()


if __name__ == "__main__":
    setup_experiment(35)
    run_simulation()
    from demlat.utils.viz_player import visualize_experiment
    visualize_experiment(DEMO_DIR)
    show_pe(DEMO_DIR)
