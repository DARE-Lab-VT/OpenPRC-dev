"""
Simple User API Example
=======================
Demonstrates the complete DEMLat workflow:
1. Setup (Create Experiment)
2. Run (Execute Simulation)
3. Inspect (Validate Results)
"""

import sys
from pathlib import Path

# Ensure import
sys.path.insert(0, str(Path(__file__).parent.parent))

import demlat
from demlat.models.barhinge import BarHingeModel
from demlat.io.experiment_setup import ExperimentSetup

# Define Experiment Path
EXP_DIR = Path("experiments/simple_api_test")


def main():
    print("🚀 Starting DEMLat Simple API Test...\n")

    # --- 1. Setup Experiment ---
    # Use the fluent ExperimentSetup API to create the environment
    print("[1] Setting up Experiment...")
    setup = ExperimentSetup(EXP_DIR, overwrite=True)

    # Configure Simulation
    setup.set_simulation_params(duration=2.0, dt=0.001, save_interval=0.01)
    setup.set_physics(gravity=-9.81, damping=0.1)

    # Build Geometry (A simple hanging triangle)
    # Nodes: [x, y, z]
    n0 = setup.add_node([0.0, 0.0, 0.0], fixed=True)  # Anchor
    n1 = setup.add_node([1.0, 0.0, 0.0], fixed=True)  # Anchor
    n2 = setup.add_node([0.5, 0.0, -1.0], mass=1.0)   # Hanging Mass

    # Bars: Connect anchors to hanging mass
    setup.add_bar(n0, n2, stiffness=1000.0, damping=10.0)
    setup.add_bar(n1, n2, stiffness=1000.0, damping=10.0)
    # Horizontal bar for stability
    setup.add_bar(n0, n1, stiffness=5000.0, damping=10.0)

    # Save to disk (generates input/config.json, input/geometry.h5, etc.)
    setup.save()
    print(f"    Experiment created at: {EXP_DIR}")


    # --- 2. Initialize Engine ---
    # Load the experiment context
    print("\n[2] Initializing Engine...")
    exp = demlat.Experiment(EXP_DIR)

    # Create the engine with the desired physics model and backend
    engine = demlat.Engine(
        model_class=BarHingeModel,
        backend="auto",  # Will prefer CUDA, fallback to CPU
        buffer_size=50
    )


    # --- 3. Run Simulation ---
    # This handles validation, physics loop, and post-processing automatically
    print("\n[3] Running Simulation...")
    result = engine.run(exp, auto_process=True)

    print(f"\n✅ Simulation Complete!")
    print(f"    Status: {result['status']}")
    print(f"    Frames: {result['frames']}")
    print(f"    Output: {result['path']}")


    # --- 4. Inspect Results (Optional) ---
    # Use the validator to peek at the generated file
    print("\n[4] Inspecting Output...")
    from demlat.io.validator import ExperimentValidator
    
    validator = ExperimentValidator(EXP_DIR)
    # Inspect the simulation file to show it contains valid data
    validator.inspect_file('simulation', show_data=False)


if __name__ == "__main__":
    main()
