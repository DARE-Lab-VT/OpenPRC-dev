"""
Yoshimura Equilibrium Analysis
===============================
Finds all equilibria of the Yoshimura origami bar-hinge model
and sweeps over facet stiffness to trace bifurcations.

Usage:
    python pipeline_yoshimura_equilibrium.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


DEMO_DIR = Path("experiments/yoshimura_equilibrium")


# =============================================================
# 1. SETUP — Build experiment geometry (no actuation needed)
# =============================================================

def setup(beta=35, k_facet=0.02):
    from demlat.io.experiment_setup import ExperimentSetup
    from examples.Yoshimura import Yoshimura

    print("\n[Setup] Creating Yoshimura Geometry for Equilibrium Analysis...")

    setup = ExperimentSetup(DEMO_DIR, overwrite=True)

    # Minimal simulation params (needed by ExperimentSetup, not used by finder)
    setup.set_simulation_params(duration=1.0, dt=0.001, save_interval=0.01)
    setup.set_physics(gravity=0.0, damping=0.0)

    # Geometry
    beta_rad = np.deg2rad(beta)
    n = 3
    k_axial = 1000.0
    k_fold = 0.0
    mass = 0.01
    damping = 0.0  # irrelevant for equilibrium

    print(f"  n={n}, beta={beta:.1f}°, k_axial={k_axial}, k_facet={k_facet}, k_fold={k_fold}")

    module = Yoshimura(n, beta_rad, randomize=False)
    nodes, bars, hinges, faces, _ = module.get_geometry()

    print(f"  Nodes: {len(nodes)}, Bars: {len(bars)}, Hinges: {len(hinges)}, Faces: {len(faces)}")

    for i, pos in enumerate(nodes):
        setup.add_node(pos, mass=mass, fixed=False)

    for bar in bars:
        setup.add_bar(bar[0], bar[1], stiffness=k_axial * bar[2], rest_length=bar[2], damping=damping)

    for hinge in hinges:
        if hinge[5] == 'fold':
            k = k_fold
        elif hinge[5] == 'facet':
            k = k_facet
        else:
            k = 0.0

        if k > 0.0:
            setup.add_hinge(nodes=[hinge[0], hinge[1], hinge[2], hinge[3]],
                            stiffness=k, rest_angle=hinge[4])

    for face in faces:
        setup.add_face(face)

    # No actuators — pure free system
    setup.save()
    print(f"  Saved to {DEMO_DIR}")


# =============================================================
# 2. FIND — Run the equilibrium finder
# =============================================================

def find(num_random=100, tol=1e-8, max_iter=300, perturbation_scale=0.5):
    from demlat.analysis.equilibria import EquilibriumFinder

    print("\n[Find] Searching for equilibria...")

    finder = EquilibriumFinder.from_experiment(DEMO_DIR)

    results = finder.find_all(
        num_random=num_random,
        perturbation_scale=perturbation_scale,
        tol=tol,
        max_iter=max_iter,
        uniqueness_tol=1e-3,
    )

    results.summary()

    # Save
    out_path = DEMO_DIR / "output" / "equilibria.h5"
    finder.save_results(results, out_path)

    return results


# =============================================================
# 3. SWEEP — Vary k_facet and track equilibria count/energy
# =============================================================

def sweep(k_values=None, num_random=80, tol=1e-7):
    from demlat.analysis.equilibria import EquilibriumFinder
    import h5py

    if k_values is None:
        k_values = np.logspace(-3, 0, 15)  # 0.001 to 1.0

    print(f"\n[Sweep] Scanning k_facet over {len(k_values)} values: "
          f"[{k_values[0]:.4f} ... {k_values[-1]:.4f}]")

    sweep_data = {
        'k_values': [],
        'n_equilibria': [],
        'n_stable': [],
        'energies': [],       # list of arrays
        'stabilities': [],    # list of arrays
    }

    geom_path = DEMO_DIR / "input" / "geometry.h5"
    config_path = DEMO_DIR / "input" / "config.json"

    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    physics = config.get('global_physics', config.get('physics', {}))

    for k_facet in k_values:
        print(f"\n{'─'*50}")
        print(f"  k_facet = {k_facet:.6f}")
        print(f"{'─'*50}")

        # Load geometry and override hinge stiffness
        geometry = {}
        with h5py.File(geom_path, 'r') as f:
            geometry['nodes'] = f['nodes/positions'][:]
            if 'nodes/masses' in f:
                geometry['masses'] = f['nodes/masses'][:]
            if 'elements/bars/indices' in f:
                geometry['bars'] = {
                    'indices': f['elements/bars/indices'][:].astype(np.int32),
                }
                for attr in ['stiffness', 'rest_length', 'prestress']:
                    key = f'elements/bars/{attr}'
                    if key in f:
                        geometry['bars'][attr] = f[key][:]
            if 'elements/hinges/indices' in f:
                geometry['hinges'] = {
                    'indices': f['elements/hinges/indices'][:].astype(np.int32),
                }
                for attr in ['stiffness', 'damping']:
                    key = f'elements/hinges/{attr}'
                    if key in f:
                        geometry['hinges'][attr] = f[key][:]

                # HDF5 stores 'angle', internal dict uses 'rest_angle'
                if 'elements/hinges/angle' in f:
                    geometry['hinges']['rest_angle'] = f['elements/hinges/angle'][:]
                elif 'elements/hinges/rest_angle' in f:
                    geometry['hinges']['rest_angle'] = f['elements/hinges/rest_angle'][:]

        # Override all hinge stiffnesses
        if 'hinges' in geometry:
            geometry['hinges']['stiffness'][:] = k_facet

        finder = EquilibriumFinder(geometry, physics)
        results = finder.find_all(
            num_random=num_random,
            tol=tol,
            max_iter=300,
            uniqueness_tol=1e-3,
        )

        sweep_data['k_values'].append(k_facet)
        sweep_data['n_equilibria'].append(len(results.equilibria))
        sweep_data['n_stable'].append(results.n_stable)
        sweep_data['energies'].append([eq.energy for eq in results.equilibria])
        sweep_data['stabilities'].append([eq.stability for eq in results.equilibria])

        print(f"  → {len(results.equilibria)} equilibria "
              f"({results.n_stable} stable, {results.n_unstable} unstable/saddle)")

    # Save sweep results
    sweep_path = DEMO_DIR / "output" / "sweep_results.npz"
    np.savez(sweep_path,
             k_values=np.array(sweep_data['k_values']),
             n_equilibria=np.array(sweep_data['n_equilibria']),
             n_stable=np.array(sweep_data['n_stable']),
             energies=np.array(sweep_data['energies'], dtype=object),
             stabilities=np.array(sweep_data['stabilities'], dtype=object))
    print(f"\n[Saved] Sweep results -> {sweep_path}")

    return sweep_data


# =============================================================
# 4. PLOT — Visualize results
# =============================================================

def plot_single(results=None):
    """Plot equilibria from a single run."""
    from demlat.analysis.equilibria import EquilibriumFinder

    if results is None:
        results = EquilibriumFinder.load_results(DEMO_DIR / "output" / "equilibria.h5")

    if len(results.equilibria) == 0:
        print("No equilibria found.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- A. Energy landscape ---
    ax = axes[0]
    energies = [eq.energy for eq in results.equilibria]
    colors = ['tab:green' if eq.stability == 'stable' else
              'tab:red' if eq.stability == 'saddle' else
              'tab:orange' for eq in results.equilibria]
    ax.barh(range(len(energies)), energies, color=colors)
    ax.set_xlabel('Potential Energy')
    ax.set_ylabel('Equilibrium Index')
    ax.set_title('Equilibria Energy Levels')
    ax.axvline(x=0, color='k', linewidth=0.5, linestyle='--')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='tab:green', label='Stable'),
        Patch(facecolor='tab:red', label='Saddle'),
        Patch(facecolor='tab:orange', label='Unstable'),
    ]
    ax.legend(handles=legend_elements, loc='best')

    # --- B. Morse index distribution ---
    ax = axes[1]
    indices = [eq.index for eq in results.equilibria]
    ax.hist(indices, bins=range(max(indices) + 2), color='steelblue',
            edgecolor='black', align='left')
    ax.set_xlabel('Morse Index (# negative eigenvalues)')
    ax.set_ylabel('Count')
    ax.set_title('Morse Index Distribution')

    # --- C. Eigenvalue spectrum of each equilibrium ---
    ax = axes[2]
    for i, eq in enumerate(results.equilibria):
        eigs = eq.eigenvalues
        # Only plot non-zero eigenvalues (skip rigid body modes)
        sig = eigs[np.abs(eigs) > 1e-4]
        ax.scatter([i] * len(sig), sig, s=8, alpha=0.6,
                   c=['tab:red' if e < 0 else 'tab:blue' for e in sig])
    ax.axhline(y=0, color='k', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Equilibrium Index')
    ax.set_ylabel('Hessian Eigenvalue')
    ax.set_title('Eigenvalue Spectrum')

    plt.tight_layout()
    plt.savefig(DEMO_DIR / "output" / "equilibria_analysis.png", dpi=150)
    plt.show()


def plot_sweep(sweep_data=None):
    """Plot bifurcation diagram from sweep."""

    if sweep_data is None:
        data = np.load(DEMO_DIR / "output" / "sweep_results.npz", allow_pickle=True)
        sweep_data = {
            'k_values': data['k_values'],
            'n_equilibria': data['n_equilibria'],
            'n_stable': data['n_stable'],
            'energies': data['energies'],
            'stabilities': data['stabilities'],
        }

    k_vals = sweep_data['k_values']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- A. Number of equilibria vs k_facet ---
    ax = axes[0]
    ax.plot(k_vals, sweep_data['n_equilibria'], 'o-', color='steelblue', label='Total')
    ax.plot(k_vals, sweep_data['n_stable'], 's-', color='tab:green', label='Stable')
    ax.set_xscale('log')
    ax.set_xlabel('k_facet')
    ax.set_ylabel('Number of Equilibria')
    ax.set_title('Bifurcation: Equilibria Count vs Stiffness')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- B. Energy bifurcation diagram ---
    ax = axes[1]
    for i, (k, energies, stabs) in enumerate(
            zip(k_vals, sweep_data['energies'], sweep_data['stabilities'])):
        for e, s in zip(energies, stabs):
            color = 'tab:green' if s == 'stable' else 'tab:red'
            marker = 'o' if s == 'stable' else 'x'
            ax.plot(k, e, marker, color=color, markersize=5, alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel('k_facet')
    ax.set_ylabel('Equilibrium Energy')
    ax.set_title('Energy Bifurcation Diagram')
    ax.grid(True, alpha=0.3)

    # Manual legend
    ax.plot([], [], 'o', color='tab:green', label='Stable')
    ax.plot([], [], 'x', color='tab:red', label='Saddle/Unstable')
    ax.legend()

    plt.tight_layout()
    plt.savefig(DEMO_DIR / "output" / "bifurcation_diagram.png", dpi=150)
    plt.show()


def plot_3d(eq_index=0):
    """Quick 3D scatter of a single equilibrium configuration."""
    from demlat.analysis.equilibria import EquilibriumFinder

    results = EquilibriumFinder.load_results(DEMO_DIR / "output" / "equilibria.h5")

    if eq_index >= len(results.equilibria):
        print(f"Only {len(results.equilibria)} equilibria found.")
        return

    eq = results.equilibria[eq_index]
    pos = eq.positions

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=40, c=pos[:, 2], cmap='coolwarm')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Equilibrium [{eq_index}] — {eq.stability} (E={eq.energy:.4f})')

    # Equal aspect ratio
    max_range = np.array([pos[:, 0].ptp(), pos[:, 1].ptp(), pos[:, 2].ptp()]).max() / 2.0
    mid = pos.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.tight_layout()
    plt.show()


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":
    # --- Single analysis at default parameters ---
    setup(beta=35, k_facet=0.02)
    results = find(num_random=100, tol=1e-8)
    plot_single(results)

    # # --- Stiffness sweep (uncomment to run) ---
    # sweep_data = sweep(k_values=np.logspace(-3, 0, 12), num_random=80)
    # plot_sweep(sweep_data)
    #
    # # --- 3D view of specific equilibrium ---
    # plot_3d(eq_index=0)