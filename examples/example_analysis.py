"""
Analysis Module — Comprehensive Example
=========================================
Covers both analysis sub-modules:

  Part A — Correlation diagnostics
    Demonstrates corr.Linear and corr.Nonparametric on the output of a
    driven spring-chain simulation (or synthetic data if no simulation
    is available).  Shows every diagnostic: Pearson, CCF, partial
    correlation, Spearman, distance correlation.

  Part B — Multistability / Equilibrium finding
    Builds a simple bistable bar-hinge structure (two-panel "butterfly")
    and uses EquilibriumFinder to discover both stable configurations
    via multi-start dynamic relaxation.

Run either part independently:
    python example_analysis.py --part A
    python example_analysis.py --part B
    python example_analysis.py           (runs both)
"""

import argparse
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# ═══════════════════════════════════════════════════════════════════════════
# Part A — Correlation Diagnostics
# ═══════════════════════════════════════════════════════════════════════════

def run_part_A():
    """Correlation diagnostics between a scalar input and a multi-channel state."""
    print("\n" + "=" * 60)
    print("Part A — Correlation Diagnostics")
    print("=" * 60)

    from openprc.analysis import correlation as corr

    # ── Option 1: load real simulation if it exists ───────────────────────────
    sim_path = Path("experiments/example_reservoir/output/simulation.h5")
    if sim_path.exists():
        print(f"  Using simulation: {sim_path}")
        from openprc.reservoir import StateLoader, features as feat
        loader = StateLoader(sim_path)

        # x = actuation signal (first actuator, x-dof)
        x = loader.get_actuation_signal(actuator_idx=0, dof=0)

        # y = bar extensions (multi-channel reservoir state)
        y = feat.BarStrains().transform(loader)

        # Trim to equal length
        T = min(len(x), y.shape[0])
        x, y = x[:T].astype(np.float64), y[:T].astype(np.float64)

    else:
        # ── Option 2: synthetic data — sine drive + damped resonators ─────────
        print("  No simulation found — generating synthetic data.")
        rng = np.random.default_rng(0)
        T   = 2000
        t   = np.linspace(0, 20.0, T)
        x   = np.sin(2 * np.pi * 0.5 * t) + 0.1 * rng.standard_normal(T)
        # 8 resonators: each responds at a different lag + nonlinear coupling
        phases = rng.uniform(0, 2 * np.pi, 8)
        lags   = np.arange(1, 9)
        y = np.column_stack([
            0.7 * np.roll(x, lag) + 0.3 * np.roll(x, lag) ** 2
            + 0.05 * rng.standard_normal(T)
            for lag in lags
        ])

    print(f"  Input x : shape {x.shape}")
    print(f"  State y : shape {y.shape}")

    # ── Linear diagnostics ────────────────────────────────────────────────────
    print("\n  [A.1] Linear correlation (Pearson + CCF)...")
    lin = corr.Linear(x, y, lag_sweep=True, max_lag=50,
                      channel_names=[f"ch{i}" for i in range(y.shape[1])])

    pearson_r = lin.pearson.values
    print(f"  Zero-lag Pearson r  : min={pearson_r.min():.3f}  "
          f"max={pearson_r.max():.3f}  mean={pearson_r.mean():.3f}")

    peak_r = lin.peak.values
    print(f"  Best-lag Pearson r  : min={peak_r.min():.3f}  "
          f"max={peak_r.max():.3f}  mean={peak_r.mean():.3f}")

    # Significant channels at p < 0.05
    sig = lin.pearson.significant(alpha=0.05)
    print(f"  Significant channels (α=0.05): {sig.sum()} / {y.shape[1]}")

    # ACF of first channel
    acf_vals = lin.acf.values         # (max_lag, N)
    print(f"  ACF shape (lags × channels): {acf_vals.shape}")

    # ── Nonparametric diagnostics ─────────────────────────────────────────────
    print("\n  [A.2] Nonparametric correlation (Spearman + distance corr)...")
    # Use a subset of y to keep dcor affordable (O(T²) per channel)
    T_sub  = min(500, len(x))
    x_sub  = x[:T_sub]
    y_sub  = y[:T_sub, :4]   # first 4 channels only for dcor

    nr = corr.Nonparametric(x_sub, y_sub,
                            channel_names=[f"ch{i}" for i in range(4)])

    spearman_rho = nr.spearman.values
    print(f"  Spearman ρ  : {spearman_rho.round(3)}")

    dcor_vals = nr.dcor.values
    print(f"  Distance correlation: {dcor_vals.round(3)}")

    # ── Redundancy (y-internal structure) ────────────────────────────────────
    print("\n  [A.3] Redundancy (internal y structure)...")
    red = corr.Redundancy(y[:, :6])
    df  = red.correlation.to_dataframe()
    print(f"  Pairwise correlation matrix ({df.shape[0]}×{df.shape[1]}):")
    print(f"  Mean |r| off-diagonal: "
          f"{np.abs(df.values - np.eye(len(df))).mean():.3f}")
    print(f"  Effective rank : {red.rank:.2f}")
    print(f"  Condition number: {red.condition:.2f}")
    print(f"  Channel groups (r>0.9): {red.groups(threshold=0.9)}")

    print("\n  Part A complete.")


# ═══════════════════════════════════════════════════════════════════════════
# Part B — Equilibrium Finding
# ═══════════════════════════════════════════════════════════════════════════

def run_part_B():
    """Build a bistable two-panel structure and find all equilibria."""
    print("\n" + "=" * 60)
    print("Part B — Multistability / Equilibrium Finding")
    print("=" * 60)

    try:
        import jax  # EquilibriumFinder requires JAX
    except ImportError:
        print("  JAX not installed — skipping Part B.")
        print("  Install with:  pip install openprc[jax]")
        return

    from openprc.demlat import SimulationSetup, Simulation, Engine
    from openprc.analysis import EquilibriumFinder

    EXP = Path("experiments/example_bistable")

    # ── Build a bistable "butterfly" bar-hinge structure ─────────────────────
    # Four nodes forming two triangular panels sharing a central fold:
    #
    #       n2        n3
    #        \       /
    #         n0---n1        (central fold bar)
    #
    # The dihedral angle between the two panels has two symmetric rest states.
    # We give it a slightly pre-stressed fold stiffness that allows snap-through.

    print("\n  Building bistable butterfly structure...")
    setup = SimulationSetup(EXP, overwrite=True)
    setup.set_simulation_params(duration=2.0, dt=0.0005, save_interval=0.01)
    setup.set_physics(gravity=0.0, damping=0.1)

    # Central fold: n0–n1 along x-axis
    n0 = setup.add_node([0.0,  0.0, 0.0], mass=1.0, fixed=True)
    n1 = setup.add_node([1.0,  0.0, 0.0], mass=1.0, fixed=True)

    # Left wing tip
    n2 = setup.add_node([0.5, -0.8, 0.3], mass=0.5)
    # Right wing tip
    n3 = setup.add_node([0.5,  0.8, 0.3], mass=0.5)

    # Bars: two triangular panels
    K_PANEL = 2000.0
    K_FOLD  = 500.0
    C = 3.0

    setup.add_bar(n0, n1, stiffness=K_PANEL, damping=C)   # fold axis
    setup.add_bar(n0, n2, stiffness=K_PANEL, damping=C)   # left panel
    setup.add_bar(n1, n2, stiffness=K_PANEL, damping=C)
    setup.add_bar(n0, n3, stiffness=K_PANEL, damping=C)   # right panel
    setup.add_bar(n1, n3, stiffness=K_PANEL, damping=C)

    # Bistable hinge: rest angle slightly off flat → two symmetric snap positions
    # axis j–k = n0–n1; wing_i = n2; wing_l = n3
    TARGET_ANGLE = np.deg2rad(30.0)    # off-flat rest angle (symmetric bistability)
    setup.add_hinge([n0, n1, n2, n3], stiffness=K_FOLD, rest_angle=TARGET_ANGLE)

    setup.save()

    # ── Run a short baseline simulation to produce simulation.h5 ─────────────
    print("  Running baseline simulation (JAX backend)...")
    Engine(backend="jax").run(Simulation(EXP))

    # ── Find all equilibria ───────────────────────────────────────────────────
    print("\n  Searching for equilibria (multi-start dynamic relaxation)...")
    finder  = EquilibriumFinder.from_experiment(EXP, backend='jax')
    results = finder.find_all(
        num_random=20,
        noise_scale=0.15,     # larger perturbations to escape the basin
        dt=0.0005,
        damping=0.95,
        max_steps=30000,
        tol=1e-6,
        accept_tol=1e-3,
        classify=True,
        verbose=False,
    )

    results.summary()

    # ── Save and inspect ──────────────────────────────────────────────────────
    out_path = EXP / "equilibria.h5"
    finder.save_results(results, str(out_path))
    print(f"\n  Saved equilibria to {out_path}")

    loaded = finder.load_results(str(out_path))
    print(f"  Re-loaded {len(loaded.equilibria)} equilibria "
          f"({loaded.n_stable} stable, {loaded.n_unstable} unstable)")

    print("\n  Part B complete.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["A", "B"], default=None,
                        help="Run only Part A or Part B (default: both)")
    args = parser.parse_args()

    if args.part in (None, "A"):
        run_part_A()
    if args.part in (None, "B"):
        run_part_B()
