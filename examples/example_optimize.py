"""
Optimize Module — Comprehensive Example
=========================================
Demonstrates JAX-based gradient parameter calibration (openprc.optimize).

Workflow
--------
  Step 1 — Create a "ground truth" experiment with known stiffness values.
  Step 2 — Create a perturbed copy with wrong stiffness (the starting guess).
  Step 3 — Run Calibration to recover bar stiffness & hinge stiffness via
            gradient descent against the reference trajectory.
  Step 4 — Compare initial vs. calibrated loss and inspect recovered params.

Requires:  pip install openprc[jax]
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import jax
except ImportError:
    print("JAX is not installed.  Run:  pip install openprc[jax]")
    sys.exit(0)

from openprc.demlat import SimulationSetup, Simulation, Engine
from openprc.optimize import Calibration

# ── Paths ─────────────────────────────────────────────────────────────────────
REF_EXP   = Path("experiments/example_calibration_ref")    # ground truth
CAL_EXP   = Path("experiments/example_calibration_guess")  # perturbed guess

# ── Ground-truth parameters ───────────────────────────────────────────────────
K_BAR_TRUE    = 1200.0   # [N/m]
K_HINGE_TRUE  = 80.0     # [N·m/rad]
C_BAR         = 3.0

# ── Starting guess (perturbed) ────────────────────────────────────────────────
K_BAR_GUESS   = 600.0    # 50 % off
K_HINGE_GUESS = 160.0    # 2× off


# ─────────────────────────────────────────────────────────────────────────────
def _build_diamond(exp_dir, k_bar, k_hinge, overwrite=True):
    """
    Four-node diamond structure (nodes at compass points).
    One sinusoidal force on the top node; bottom node fixed.
    """
    setup = SimulationSetup(exp_dir, overwrite=overwrite)
    setup.set_simulation_params(duration=3.0, dt=0.001, save_interval=0.01)
    setup.set_physics(gravity=-9.81, damping=0.05)

    # Nodes: bottom fixed, left/right free, top free
    n_bot = setup.add_node([ 0.0,  0.0, 0.0], fixed=True)
    n_l   = setup.add_node([-0.5,  0.0, 0.5], mass=0.4)
    n_r   = setup.add_node([ 0.5,  0.0, 0.5], mass=0.4)
    n_top = setup.add_node([ 0.0,  0.0, 1.0], mass=0.6)

    for a, b in [(n_bot, n_l), (n_bot, n_r), (n_l, n_top), (n_r, n_top)]:
        setup.add_bar(a, b, stiffness=k_bar, damping=C_BAR)

    # One hinge across the top of the diamond
    setup.add_hinge([n_l, n_r, n_top, n_bot], stiffness=k_hinge)

    # Sinusoidal z-force on top node
    dt_sig = 0.001
    t      = np.arange(0, 3.0, dt_sig)
    sig    = np.zeros((len(t), 3), dtype=np.float32)
    sig[:, 2] = 4.0 * np.sin(2 * np.pi * 1.5 * t)
    setup.add_signal("z_drive", sig, dt=dt_sig)
    setup.add_actuator(n_top, "z_drive", type="force", dof=[0, 0, 1])

    setup.save()


# ─────────────────────────────────────────────────────────────────────────────
def step_1_ground_truth():
    print("\n[Step 1] Building and running ground-truth experiment...")
    _build_diamond(REF_EXP, K_BAR_TRUE, K_HINGE_TRUE)
    Engine(backend="jax").run(Simulation(REF_EXP))
    ref_sim = REF_EXP / "output" / "simulation.h5"
    print(f"  Reference trajectory: {ref_sim}")
    return str(ref_sim)


# ─────────────────────────────────────────────────────────────────────────────
def step_2_initial_guess():
    print("\n[Step 2] Building perturbed starting-guess experiment...")
    _build_diamond(CAL_EXP, K_BAR_GUESS, K_HINGE_GUESS)
    print(f"  bar_stiffness  : true={K_BAR_TRUE}   guess={K_BAR_GUESS}")
    print(f"  hinge_stiffness: true={K_HINGE_TRUE}  guess={K_HINGE_GUESS}")


# ─────────────────────────────────────────────────────────────────────────────
def step_3_calibrate(ref_sim_path):
    print("\n[Step 3] Running calibration (JAX gradient descent)...")

    cal = Calibration(backend="jax")   # BarHingeModel is implicit

    # Load the perturbed geometry to calibrate
    cal.load_geometry(str(CAL_EXP))

    # Load the reference trajectory to match
    cal.load_reference(ref_sim_path)

    # Specify which parameters to optimize
    cal.optimize_params(
        bar_stiffness=True,
        hinge_stiffness=True,
        bar_damping=False,     # keep damping fixed
    )

    # Loose bounds to prevent unphysical values
    cal.set_bounds(
        bar_stiffness=(50.0, 5000.0),
        hinge_stiffness=(1.0, 1000.0),
    )

    # Run optimisation
    result = cal.run(
        max_iterations=200,
        lr=5e-2,
        cost="mse",
        optimizer="adam",
        patience=40,
        save_interval=20,
        verbose=True,
    )

    return cal, result


# ─────────────────────────────────────────────────────────────────────────────
def step_4_evaluate(cal, result):
    print("\n[Step 4] Results:")
    result.summary()

    # Save calibrated geometry
    out_path = str(CAL_EXP / "calibrated_geometry.h5")
    cal.save(out_path)
    print(f"\n  Calibrated geometry saved → {out_path}")

    if len(result.loss_history) > 1:
        reduction = (1.0 - result.final_loss / result.loss_history[0]) * 100
        print(f"  Loss reduction: {reduction:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ref_path    = step_1_ground_truth()
    step_2_initial_guess()
    cal, result = step_3_calibrate(ref_path)
    step_4_evaluate(cal, result)
