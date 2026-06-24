import sys
import numpy as np
import h5py
from pathlib import Path

# --- Path Setup ---
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

# --- Core Library Imports ---
from openprc.reservoir.io.state_loader import StateLoader
from openprc.reservoir.features.node_features import NodePositions, NodeDisplacements
from openprc.reservoir.readout.ridge import Ridge
from openprc.reservoir.training.trainer import Trainer
from openprc.analysis.visualization.time_series import TimeSeriesComparison

# --- Import Standard Benchmark ---
from openprc.analysis.benchmarks.narma_benchmark import NARMABenchmark

def main():
    # ==========================================
    # 1. Configuration
    # ==========================================
    experiment_subpath = "spring_mass_4x4_test/generation_0"
    experiments_dir = src_dir / "experiments"
    experiment_dir = experiments_dir / experiment_subpath
    sim_path = experiment_dir / "output" / "simulation.h5"

    if not sim_path.exists():
        print(f"Simulation missing at {sim_path}!")
        print("Please run a simulation for the experiment first.")
        print("For example, you can run: python src/examples/simple_user_api.py")
        return
    
    # ==========================================
    # 2. Load & Scale Actuation
    # ==========================================
    loader = StateLoader(sim_path)
    u_raw = loader.get_actuation_signal(actuator_idx=0, dof=0)
    
    if u_raw is None: return
    if u_raw.ndim > 1: u_raw = u_raw.flatten()
    u_scaled = (u_raw - np.nanmin(u_raw)) / (np.nanmax(u_raw) - np.nanmin(u_raw)) * 0.5
    
    print(f"   Actuation Loaded (Shape: {u_scaled.shape})")

    # ==========================================
    # 3. Setup Trainer
    # ==========================================
    features = NodeDisplacements(reference_node=0, dims=[0])
    
    # The Trainer automatically standardizes (Z-score) the features (X) internally.
    trainer = Trainer(
        loader=loader,
        features=features,
        readout=Ridge(1e-5),
        experiment_dir=experiment_dir,
        washout=5.0,
        train_duration=10.0,
        test_duration=10.0
    )

    # ==========================================
    # 4. Run Standard Benchmark
    # ==========================================
    print("\n[Workflow: Standard NARMA Benchmark]")

    # Instantiate the standard class
    benchmark = NARMABenchmark(group_name="narma_benchmark")
    
    # Run it with our pre-scaled input
    # The benchmark handles generation, training, scoring, and internal saving.
    score = benchmark.run(trainer, u_scaled, order=2)
    
    # Save the metrics.h5 file
    score.save()
    
    print(f"--- Benchmark complete for: {experiment_dir.name} ---")

    # ==========================================
    # 5. Visualize Results
    # ==========================================
    visualizer = TimeSeriesComparison()
    
    # The benchmark score object contains the path to the trained readout
    if score.readout_path and visualizer:
        print("\n[Processing] Visualizing results")
        plot_path = visualizer.plot(score.readout_path, start_frame=0, end_frame=500).save()
        print(f" >> Plot saved to: {plot_path}")

    # Print Metrics
    if score.metrics:
        print("\n[Processing] Benchmark Results:")
        for key, value in score.metrics.items():
            print(f"  >> {key}: {value:.5f}")

if __name__ == "__main__":
    main()