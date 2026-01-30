import sys
import numpy as np
from pathlib import Path

# --- Path Setup ---
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

from reservoir.io.state_loader import StateLoader
from reservoir.features.node_features import NodePositions
from reservoir.readout.ridge import Ridge
from reservoir.tasks.memory.narma import NARMA
from reservoir.utils.trainer import Trainer
from analysis.benchmarks.memory_benchmark import NARMABenchmark
from analysis.visualization.time_series import TimeSeriesComparison

def run_pipeline(experiment_subpath="spring_mass_3x3_test/generation_0"):
    experiments_dir = src_dir / "experiments"
    experiment_dir = experiments_dir / experiment_subpath
    sim_path = experiment_dir / "output" / "simulation.h5"

    if not sim_path.exists():
        print(f"Simulation missing at {sim_path}!"); return

    print(f"--- Pipeline for: {experiment_dir.name} ---")
    
    # 1. Load Data
    loader = StateLoader(sim_path)
    print(f"Loaded {loader.total_frames} frames.")

    # 2. Get Input Signal (Directly from saved Actuator Position)
    print("[Input] Retrieving actuation signal from saved node trajectories...")
    u_input = loader.get_actuation_signal(actuator_idx=0)

    # 3. Initialize Trainer
    trainer = Trainer(
        features=NodePositions(),
        readout=Ridge(1e-5),
        experiment_dir=experiment_dir,
        washout=500,
        train_len=2000,
        test_len=500
    )
    
    # 4. Train (NARMA-2)
    print("\n[Processing] NARMA-2")
    
    # Pass the perfectly aligned input signal
    result = trainer.fit(loader, NARMA(2), u_input=u_input) 
    readout_path = result.save()
    
    # 5. Analyze
    benchmark = NARMABenchmark()
    score = benchmark.run(readout_path, order=2)
    score.save()
    print(f" >> Score (NRMSE): {score.nrmse:.5f}")
    
    # 6. Visualize
    visualizer = TimeSeriesComparison()
    plot_path = visualizer.plot(readout_path, start_frame=0, end_frame=500).save()
    print(f" >> Plot saved to: {plot_path}")

if __name__ == "__main__":
    run_pipeline()