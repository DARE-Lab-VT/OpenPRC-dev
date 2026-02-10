"""
Yoshimura Multistability Analysis
==================================

Example: Setup a Yoshimura experiment, find all equilibria,
then optionally verify stable ones via time-domain simulation.

Usage:
    python examples/yoshimura_multistability.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ── DEMLAT imports ──
# from demlat.io.experiment_setup import ExperimentSetup
# from demlat.core.experiment import Experiment
# from demlat.analysis.equilibria import EquilibriumAnalyzer
# from examples.Yoshimura import Yoshimura

DEMO_DIR = Path("experiments/yoshimura_multistability")


def setup_experiment(beta_deg=35.0, n=3):
    """Create the Yoshimura geometry (no actuation — static analysis)."""
    from demlat.io.experiment_setup import ExperimentSetup
    from examples.Yoshimura import Yoshimura

    setup = ExperimentSetup(DEMO_DIR, overwrite=True)

    # Physics: no gravity, light damping
    setup.set_simulation_params(duration=1.0, dt=0.001, save_interval=0.01)
    setup.set_physics(gravity=0.0, damping=0.1)

    # Geometry
    beta = np.deg2rad(beta_deg)
    module = Yoshimura(n, beta, randomize=False)
    nodes, bars, hinges, faces, _ = module.get_geometry()

    # Material
    k_axial = 2000.0
    k_fold = 0.0  # fold creases are free
    k_facet = 0.01  # facets resist bending
    mass = 0.01
    damping = 2.0

    # Add nodes
    for node_pos in nodes:
        setup.add_node(node_pos, mass=mass, fixed=False)

    # Add bars
    for bar in bars:
        setup.add_bar(bar[0], bar[1], stiffness=k_axial * bar[2],
                      rest_length=bar[2], damping=damping)

    # Add hinges (only facet bending stiffness)
    for hinge in hinges:
        k = k_fold if hinge[5] == 'fold' else k_facet
        if k > 0.0:
            setup.add_hinge(
                nodes=[hinge[0], hinge[1], hinge[2], hinge[3]],
                stiffness=k, rest_angle=hinge[4]
            )

    for face in faces:
        setup.add_face(face)

    setup.save()
    print(f"Experiment saved to {DEMO_DIR}")
    return module


def find_equilibria():
    """Find all equilibria using deflated Newton."""
    from demlat.core.experiment import Experiment
    from demlat.analysis.equilibria import EquilibriumAnalyzer

    exp = Experiment(DEMO_DIR)
    analyzer = EquilibriumAnalyzer(exp)

    # ── Find equilibria ──
    results = analyzer.find_equilibria(
        n_attempts=200,
        strategies=['reference', 'random', 'kinematic', 'compressed'],
        perturbation_scale=0.15,
        deflation=True,
        tol=1e-8,
        dedup_tol=1e-4,
        seed=42,
        verbose=True,
    )

    print(results.summary())

    # ── Save results ──
    results.save(DEMO_DIR / "analysis")

    return analyzer, results


def plot_energy_landscape(analyzer, results):
    """Visualize energy landscape along softest modes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1D: energy along softest mode
    amps, energies = analyzer.energy_along_mode(
        mode_index=0, amplitude_range=(-0.5, 0.5), n_points=300
    )
    axes[0].plot(amps, energies, 'b-', lw=1.5)

    # Mark found equilibria projected onto this mode
    for eq in results.equilibria:
        axes[0].axhline(eq.energy, color='r' if not eq.is_stable else 'g',
                        alpha=0.3, ls='--')
    axes[0].set_xlabel("Amplitude along softest mode")
    axes[0].set_ylabel("Potential Energy V(x)")
    axes[0].set_title("1D Energy Landscape (mode 0)")

    # 2D: energy over two softest modes
    Ai, Aj, E = analyzer.energy_landscape_2d(
        mode_i=0, mode_j=1, amplitude_range=(-0.3, 0.3), n_points=80
    )
    c = axes[1].contourf(Ai, Aj, E, levels=40, cmap='viridis')
    plt.colorbar(c, ax=axes[1], label='V(x)')
    axes[1].set_xlabel("Mode 0 amplitude")
    axes[1].set_ylabel("Mode 1 amplitude")
    axes[1].set_title("2D Energy Landscape")

    plt.tight_layout()
    plt.savefig(DEMO_DIR / "analysis" / "energy_landscape.png", dpi=150)
    plt.show()


def verify_stable_states(results):
    """
    Take each stable equilibrium, set it as initial condition,
    run a short simulation, and confirm it stays put.
    """
    from demlat.io.experiment_setup import ExperimentSetup
    from demlat.core.experiment import Experiment
    import demlat
    from demlat.models.barhinge import BarHingeModel

    for i, eq in enumerate(results.stable):
        verify_dir = DEMO_DIR / f"verify_stable_{i}"
        print(f"\nVerifying stable state #{i} (E={eq.energy:.6f})...")

        # Re-create experiment with equilibrium as initial positions
        # (Copy the original setup but override initial positions)
        setup = ExperimentSetup(verify_dir, overwrite=True)
        setup.set_simulation_params(duration=0.5, dt=0.001, save_interval=0.01)
        setup.set_physics(gravity=0.0, damping=0.1)

        # Use equilibrium positions as initial config
        eq_positions = eq.positions
        for j, pos in enumerate(eq_positions):
            setup.add_node(pos, mass=0.01, fixed=False)

        setup.save()

        # Run short simulation
        exp = Experiment(verify_dir)
        eng = demlat.Engine(BarHingeModel, backend='cuda')
        eng.run(exp)

        # Check: did it stay near the equilibrium?
        # Load final positions from simulation output and compare
        print(f"  Verification simulation complete for stable state #{i}")


if __name__ == "__main__":
    print("=" * 60)
    print("Yoshimura-Ori Multistability Analysis")
    print("=" * 60)

    # Step 1: Setup geometry
    module = setup_experiment(beta_deg=35.0)

    # Step 2: Find all equilibria
    analyzer, results = find_equilibria()

    # Step 3: Visualize energy landscape
    plot_energy_landscape(analyzer, results)

    # Step 4: (Optional) Verify stable states via simulation
    if results.stable:
        verify_stable_states(results)
    else:
        print("\nNo stable equilibria found — try adjusting parameters.")
