"""
Yoshimura Equilibrium Analysis
===============================
Finds all equilibria of the Yoshimura origami bar-hinge model
using dynamic relaxation from multiple initial guesses.

Usage:
    python pipeline_yoshimura_equilibrium.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

DEMO_DIR = Path("experiments/yoshimura_equilibrium")


# =============================================================
# 1. SETUP
# =============================================================

def setup(beta=35, k_facet=0.02):
    from openprc.demlat.io.experiment_setup import ExperimentSetup
    from Yoshimura import Yoshimura

    print("\n[Setup] Creating Yoshimura Geometry...")

    setup = ExperimentSetup(DEMO_DIR, overwrite=True)
    setup.set_simulation_params(duration=5.0, dt=0.0001, save_interval=0.01)
    setup.set_physics(gravity=0.0, damping=0.1)

    beta_rad = np.deg2rad(beta)
    n = 3
    k_axial = 1000.0
    k_fold = 0.0
    mass = 0.01
    damping = 2.0

    print(f"  n={n}, beta={beta:.1f}°, k_axial={k_axial}, k_facet={k_facet}, k_fold={k_fold}")

    module = Yoshimura(n, beta_rad, randomize=False)
    nodes, bars, hinges, faces, _ = module.get_geometry()

    print(f"  Nodes: {len(nodes)}, Bars: {len(bars)}, Hinges: {len(hinges)}, Faces: {len(faces)}")

    for pos in nodes:
        setup.add_node(pos, mass=mass, fixed=False)

    for bar in bars:
        setup.add_bar(bar[0], bar[1], stiffness=k_axial * bar[2],
                      rest_length=bar[2], damping=damping)

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

    setup.save()
    print(f"  Saved to {DEMO_DIR}")


# =============================================================
# 2. FIND EQUILIBRIA
# =============================================================

def find(backend='jax', num_random=30, max_steps=10000, tol=5.0e-5,
         dt=0.0005, damping=0.90, verbose=True):
    from openprc.demlat.analysis.equilibria import EquilibriumFinder

    print("\n[Find] Searching for equilibria...")

    finder = EquilibriumFinder.from_experiment(DEMO_DIR, backend=backend)

    results = finder.find_all(
        num_random=num_random,
        dt=dt,
        damping=damping,
        max_steps=max_steps,
        tol=tol,
        uniqueness_tol=1e-3,
        classify=True,
        verbose=verbose
    )

    results.summary()

    out_path = DEMO_DIR / "output" / "equilibria.h5"
    finder.save_results(results, out_path)

    return results


# =============================================================
# 3. PLOT
# =============================================================

def plot_results(results=None):
    """Plot found equilibria."""
    from openprc.demlat.analysis.equilibria import EquilibriumFinder

    if results is None:
        results = EquilibriumFinder.load_results(DEMO_DIR / "output" / "equilibria.h5")

    if not results.equilibria:
        print("No equilibria found.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # A. Z-range comparison
    ax = axes[0]
    z_ranges = [eq.positions[:, 2].max() - eq.positions[:, 2].min()
                for eq in results.equilibria]
    colors = ['tab:green' if eq.stability == 'stable' else 'tab:red'
              for eq in results.equilibria]
    ax.barh(range(len(z_ranges)), z_ranges, color=colors)
    ax.set_xlabel('Z Range (height)')
    ax.set_ylabel('Equilibrium Index')
    ax.set_title('Equilibria: Height Comparison')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor='tab:green', label='Stable'),
                       Patch(facecolor='tab:red', label='Unstable/Saddle')])

    # B. Eigenvalue spectrum
    ax = axes[1]
    for i, eq in enumerate(results.equilibria):
        if eq.eigenvalues is not None:
            eigs = eq.eigenvalues
            ax.scatter([i] * len(eigs), eigs, s=8, alpha=0.6,
                       c=['tab:red' if e > 0 else 'tab:blue' for e in eigs])
    ax.axhline(y=0, color='k', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Equilibrium Index')
    ax.set_ylabel('Linearized Eigenvalue')
    ax.set_title('Stability Spectrum')

    # C. Residuals
    ax = axes[2]
    residuals = [eq.residual for eq in results.equilibria]
    ax.bar(range(len(residuals)), residuals, color='steelblue')
    ax.set_yscale('log')
    ax.set_xlabel('Equilibrium Index')
    ax.set_ylabel('Max Residual Acceleration')
    ax.set_title('Convergence Quality')

    plt.tight_layout()
    plt.savefig(DEMO_DIR / "output" / "equilibria_analysis.png", dpi=150)
    plt.show()


def plot_3d(eq_index=0):
    """3D scatter of equilibrium vs reference."""
    from openprc.demlat.analysis.equilibria import EquilibriumFinder
    import h5py

    results = EquilibriumFinder.load_results(DEMO_DIR / "output" / "equilibria.h5")
    with h5py.File(DEMO_DIR / "output" / "equilibria.h5", 'r') as f:
        ref = f['reference/positions'][:]

    if eq_index >= len(results.equilibria):
        print(f"Only {len(results.equilibria)} equilibria found.")
        return

    eq = results.equilibria[eq_index]
    pos = eq.positions

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(ref[:, 0], ref[:, 1], ref[:, 2], s=40, c=ref[:, 2], cmap='coolwarm')
    ax1.set_title('Reference Config')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=40, c=pos[:, 2], cmap='coolwarm')
    ax2.set_title(f'Equilibrium [{eq_index}] — {eq.stability}')

    for ax in [ax1, ax2]:
        all_pts = np.vstack([ref, pos])
        rng = np.ptp(all_pts, axis=0).max() / 2.0
        mid = all_pts.mean(axis=0)
        ax.set_xlim(mid[0] - rng, mid[0] + rng)
        ax.set_ylim(mid[1] - rng, mid[1] + rng)
        ax.set_zlim(mid[2] - rng, mid[2] + rng)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    plt.tight_layout()
    plt.show()


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":
    setup(beta=35, k_facet=0.02)
    results = find(backend='jax', num_random=500, max_steps=50000)
    plot_results(results)
    # plot_3d(0)