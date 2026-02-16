"""
Example: Calibration Workflow
==============================

Demonstrates the full calibration pipeline:
  1. Set up a "ground truth" simulation with known parameters
  2. Run it to generate a reference trajectory
  3. Start with wrong parameters and calibrate to recover the originals
"""

from pathlib import Path
import numpy as np

DEMO_DIR = Path("experiments/calibration_test")
TRUTH_DIR = DEMO_DIR / "ground_truth"
CALIBRATE_DIR = DEMO_DIR / "initial_guess"


# ============================================================
# Step 1: Create ground truth simulation (known parameters)
# ============================================================

def setup_ground_truth():
    from openprc.demlat.io.simulation_setup import SimulationSetup

    print("\n[1] Setting up ground truth simulation...")
    setup = SimulationSetup(TRUTH_DIR, overwrite=True)
    setup.set_simulation_params(duration=5.0, dt=0.001, save_interval=0.01)
    setup.set_physics(gravity=-9.81, damping=0.1)

    # 4-node truss
    setup.add_node([0, 0, 0], 5.0, True)  # fixed
    setup.add_node([1, -0.5, 0], 2.5, False)
    setup.add_node([1, 0.5, 0], 2.5, False)
    setup.add_node([2, 0, 0], 5.0, False)

    # Ground truth stiffnesses: [10, 10, 10, 10, 10]
    setup.add_bar(0, 1, stiffness=10.0, damping=2.0)
    setup.add_bar(0, 2, stiffness=10.0, damping=2.0)
    setup.add_bar(1, 3, stiffness=10.0, damping=2.0)
    setup.add_bar(2, 3, stiffness=10.0, damping=2.0)
    setup.add_bar(1, 2, stiffness=10.0, damping=2.0)

    # Ground truth hinge stiffness: 0.5
    setup.add_hinge([1, 2, 0, 3], stiffness=0.5, damping=0.0, rest_angle=np.pi / 2)

    setup.save()


def run_ground_truth():
    import openprc.demlat
    from openprc.demlat.models.barhinge import BarHingeModel

    print("\n[2] Running ground truth simulation...")
    sim = openprc.demlat.Simulation(TRUTH_DIR)
    eng = openprc.demlat.Engine(BarHingeModel, backend='jax')
    eng.run(sim)
    print(f"    Reference saved to: {TRUTH_DIR / 'output' / 'simulation.h5'}")


# ============================================================
# Step 2: Create initial guess (wrong parameters)
# ============================================================

def setup_initial_guess():
    from openprc.demlat.io.simulation_setup import SimulationSetup
    import shutil

    print("\n[3] Setting up initial guess (wrong stiffnesses)...")
    setup = SimulationSetup(CALIBRATE_DIR, overwrite=True)
    setup.set_simulation_params(duration=5.0, dt=0.001, save_interval=0.01)
    setup.set_physics(gravity=-9.81, damping=0.1)

    # Same topology
    setup.add_node([0, 0, 0], 5.0, True)
    setup.add_node([1, -0.5, 0], 2.5, False)
    setup.add_node([1, 0.5, 0], 2.5, False)
    setup.add_node([2, 0, 0], 5.0, False)

    # WRONG stiffnesses: [50, 50, 50, 50, 50] (5x too high)
    setup.add_bar(0, 1, stiffness=50.0, damping=2.0)
    setup.add_bar(0, 2, stiffness=50.0, damping=2.0)
    setup.add_bar(1, 3, stiffness=50.0, damping=2.0)
    setup.add_bar(2, 3, stiffness=50.0, damping=2.0)
    setup.add_bar(1, 2, stiffness=50.0, damping=2.0)

    # WRONG hinge stiffness: 5.0 (10x too high)
    setup.add_hinge([1, 2, 0, 3], stiffness=5.0, damping=0.0, rest_angle=np.pi / 2)

    setup.save()


# ============================================================
# Step 3: Calibrate
# ============================================================

def calibrate():
    from openprc.optimize import Calibration
    from openprc.demlat.models.barhinge import BarHingeModel

    print("\n[4] Running calibration...")

    cal = Calibration(BarHingeModel, backend='jax')

    # Load wrong-parameter geometry
    cal.load_geometry(CALIBRATE_DIR)

    # Load ground truth trajectory as reference
    cal.load_reference(TRUTH_DIR / "output" / "simulation.h5")

    # Tell it what to optimize
    cal.optimize_params(
        bar_stiffness=True,
        hinge_stiffness=True,
        bar_damping=False,  # keep damping fixed
        hinge_rest_angle=False,  # keep rest angle fixed
    )

    # Constrain search space
    cal.set_bounds(
        bar_stiffness=(0.1, 500.0),
        hinge_stiffness=(0.01, 50.0),
    )

    # Run
    result = cal.run(
        max_iterations=300,
        lr=1.0,
        cost='mse',
        optimizer='adam',
        patience=50,
        verbose=True,
    )

    # Save optimized geometry
    output_path = DEMO_DIR / "optimized_geometry.h5"
    cal.save(output_path)
    print(f"\n    Optimized geometry saved to: {output_path}")

    # Compare
    print("\n[5] Parameter comparison:")
    print(f"    Ground truth bar_k : [10, 10, 10, 10, 10]")
    print(f"    Initial guess bar_k: [50, 50, 50, 50, 50]")
    print(f"    Calibrated bar_k   : {np.array(result.theta_optimized['bar_stiffness'])}")
    print(f"    Ground truth hinge_k: 0.5")
    print(f"    Initial guess hinge_k: 5.0")
    print(f"    Calibrated hinge_k  : {float(result.theta_optimized['hinge_stiffness'])}")

    return result


# ============================================================
# Step 4: Verify by re-running with calibrated params
# ============================================================

def verify(result):
    import openprc.demlat
    from openprc.demlat.models.barhinge import BarHingeModel
    from openprc.demlat.utils.data_parser import SimulationData

    print("\n[6] Verification: re-running with calibrated parameters...")

    # Run with optimized geometry
    verify_dir = DEMO_DIR / "verification"
    verify_dir.mkdir(parents=True, exist_ok=True)
    (verify_dir / "input").mkdir(parents=True, exist_ok=True)

    # Copy optimized geometry as the input
    import shutil
    shutil.copy2(DEMO_DIR / "optimized_geometry.h5", verify_dir / "input" / "geometry.h5")
    shutil.copy2(CALIBRATE_DIR / "input" / "config.json", verify_dir / "input" / "config.json")

    # Copy signals file (required by Simulation)
    signals_src = CALIBRATE_DIR / "input" / "signals.h5"
    if signals_src.exists():
        shutil.copy2(signals_src, verify_dir / "input" / "signals.h5")

    sim = openprc.demlat.Simulation(verify_dir)
    eng = openprc.demlat.Engine(BarHingeModel, backend='jax')
    eng.run(sim)

    # Compare trajectories
    truth = SimulationData(TRUTH_DIR / "output" / "simulation.h5")
    verify_data = SimulationData(verify_dir / "output" / "simulation.h5")

    pos_truth, _ = truth.get_dataset("time_series/nodes/positions")
    pos_verify, _ = verify_data.get_dataset("time_series/nodes/positions")

    rmse = np.sqrt(np.mean((pos_truth - pos_verify) ** 2))
    print(f"    Trajectory RMSE (truth vs calibrated): {rmse:.6e}")
    print(f"    {'PASS' if rmse < 0.01 else 'NEEDS MORE ITERATIONS'}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    setup_ground_truth()
    run_ground_truth()
    setup_initial_guess()
    result = calibrate()
    verify(result)

# ============================================================
# BONUS: Actuated Example (Position-Driven Node)
# ============================================================
# Run with:  python example_calibration.py --actuated

ACTUATED_DIR = Path("experiments/calibration_actuated")
ACTUATED_TRUTH = ACTUATED_DIR / "ground_truth"
ACTUATED_GUESS = ACTUATED_DIR / "initial_guess"


def setup_actuated_truth():
    """Ground truth with a position-driven node (sinusoidal input)."""
    from openprc.demlat.io.simulation_setup import SimulationSetup

    print("\n[A1] Setting up actuated ground truth...")
    setup = SimulationSetup(ACTUATED_TRUTH, overwrite=True)
    setup.set_simulation_params(duration=2.0, dt=0.001, save_interval=0.01)
    setup.set_physics(gravity=0.0, damping=0.1)

    # 3-node chain: [driven] ---bar--- [free] ---bar--- [free]
    setup.add_node([0, 0, 0], 1.0, False)  # node 0: will be position-driven
    setup.add_node([1, 0, 0], 1.0, False)  # node 1: free
    setup.add_node([2, 0, 0], 1.0, False)  # node 2: free

    # Ground truth stiffnesses
    setup.add_bar(0, 1, stiffness=20.0, damping=1.0)
    setup.add_bar(1, 2, stiffness=20.0, damping=1.0)

    # Sinusoidal position signal for node 0
    dt_sig = 0.001
    t = np.arange(0, 2.0, dt_sig)
    # Node 0 oscillates in x: x(t) = 0.5 * sin(2*pi*t)
    signal_x = 0.5 * np.sin(2 * np.pi * t)
    signal_3d = np.stack([signal_x, np.zeros_like(t), np.zeros_like(t)], axis=1)

    setup.add_signal("drive_signal", signal_3d, dt=dt_sig)
    setup.add_actuator(node_idx=0, signal_name="drive_signal", type='position', dof=[1, 1, 1])

    setup.save()


def setup_actuated_guess():
    """Same geometry but wrong stiffnesses."""
    from openprc.demlat.io.simulation_setup import SimulationSetup

    print("\n[A2] Setting up actuated initial guess (wrong stiffnesses)...")
    setup = SimulationSetup(ACTUATED_GUESS, overwrite=True)
    setup.set_simulation_params(duration=2.0, dt=0.001, save_interval=0.01)
    setup.set_physics(gravity=0.0, damping=0.1)

    setup.add_node([0, 0, 0], 1.0, False)
    setup.add_node([1, 0, 0], 1.0, False)
    setup.add_node([2, 0, 0], 1.0, False)

    # WRONG stiffnesses: 100.0 (5x too high)
    setup.add_bar(0, 1, stiffness=100.0, damping=1.0)
    setup.add_bar(1, 2, stiffness=100.0, damping=1.0)

    # Same actuation signal
    dt_sig = 0.001
    t = np.arange(0, 2.0, dt_sig)
    signal_x = 0.5 * np.sin(2 * np.pi * t)
    signal_3d = np.stack([signal_x, np.zeros_like(t), np.zeros_like(t)], axis=1)

    setup.add_signal("drive_signal", signal_3d, dt=dt_sig)
    setup.add_actuator(node_idx=0, signal_name="drive_signal", type='position', dof=[1, 1, 1])

    setup.save()


def calibrate_actuated():
    from openprc.optimize import Calibration
    from openprc.demlat.models.barhinge import BarHingeModel

    # First run ground truth
    import openprc.demlat
    print("\n[A3] Running actuated ground truth...")
    sim = openprc.demlat.Simulation(ACTUATED_TRUTH)
    eng = openprc.demlat.Engine(BarHingeModel, backend='jax')
    eng.run(sim)

    # Now calibrate
    print("\n[A4] Running actuated calibration...")
    cal = Calibration(BarHingeModel, backend='jax')
    cal.load_geometry(ACTUATED_GUESS)
    cal.load_reference(ACTUATED_TRUTH / "output" / "simulation.h5")

    cal.optimize_params(bar_stiffness=True)
    cal.set_bounds(bar_stiffness=(1.0, 500.0))

    result = cal.run(
        max_iterations=300,
        lr=1.0,
        cost='mse',
        optimizer='adam',
    )

    print(f"\n    Ground truth bar_k : [20, 20]")
    print(f"    Initial guess bar_k: [100, 100]")
    print(f"    Calibrated hinge_k  : {np.array(result.theta_optimized['hinge_stiffness'])}")

    return result
