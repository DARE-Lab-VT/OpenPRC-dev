"""
Reservoir Module — Comprehensive Example
==========================================
Demonstrates the full readout-training workflow on a driven spring-chain:

  Step 1 — Build & run a 1-D mass-spring chain driven by a broadband signal.
  Step 2 — Load the simulation with StateLoader.
  Step 3 — Extract state features (NodePositions, BarExtensions).
  Step 4 — Train a Ridge readout to imitate a lagged version of the input
            (standard short-term memory benchmark).
  Step 5 — Evaluate and print train/test NRMSE.
  Step 6 — Run MemoryBenchmark to compute capacity vs. lag curve.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from openprc.demlat import SimulationSetup, Simulation, Engine
from openprc.reservoir import StateLoader, Ridge, Trainer, features
from openprc.analysis.benchmarks.memory_benchmark import MemoryBenchmark

EXP       = Path("experiments/example_reservoir")
DURATION  = 40.0      # seconds — needs washout + train + test
DT        = 0.002     # integration step
SAVE_DT   = 0.01      # output frame interval (100 Hz)
BACKEND   = "cuda"    # "cpu" or "cuda"

N_MASSES  = 8         # number of free masses in the chain
K         = 500.0     # spring stiffness  [N/m]
C         = 1.5       # damping           [N·s/m]
MASS      = 0.3       # kg per node


# ─────────────────────────────────────────────────────────────────────────────
def step_1_build_and_run():
    """Build a 1-D spring chain and drive the left anchor with broadband noise."""
    print("\n[Step 1] Building spring chain...")

    setup = SimulationSetup(EXP, overwrite=True)
    setup.set_simulation_params(duration=DURATION, dt=DT, save_interval=SAVE_DT)
    setup.set_physics(gravity=0.0, damping=0.02)   # no gravity — horizontal chain

    # ── Geometry: anchor → N_MASSES free nodes → fixed wall ──────────────────
    anchor = setup.add_node([0.0, 0.0, 0.0], fixed=True)
    free_nodes = []
    for i in range(1, N_MASSES + 1):
        free_nodes.append(setup.add_node([i * 1.0, 0.0, 0.0], mass=MASS))
    wall = setup.add_node([(N_MASSES + 1) * 1.0, 0.0, 0.0], fixed=True)

    # ── Springs ───────────────────────────────────────────────────────────────
    all_nodes = [anchor] + free_nodes + [wall]
    for a, b in zip(all_nodes, all_nodes[1:]):
        setup.add_bar(a, b, stiffness=K, damping=C)

    print(f"  Nodes: {len(all_nodes)}  ({N_MASSES} free)")
    print(f"  Bars : {len(all_nodes) - 1}")

    # ── Input signal: bandlimited white noise ─────────────────────────────────
    rng    = np.random.default_rng(42)
    dt_sig = SAVE_DT
    t      = np.arange(0, DURATION, dt_sig)
    u_raw  = rng.standard_normal(len(t))

    # Low-pass filter to limit bandwidth to ~5 Hz (avoids aliasing issues)
    from scipy.signal import butter, filtfilt
    b_filt, a_filt = butter(4, 5.0 / (0.5 / dt_sig), btype='low')
    u = filtfilt(b_filt, a_filt, u_raw).astype(np.float32)
    u *= 0.4 / u.std()      # normalise amplitude to ±0.4 N

    force_sig = np.zeros((len(t), 3), dtype=np.float32)
    force_sig[:, 0] = u     # x-direction force on first free mass

    setup.add_signal("noise_drive", force_sig, dt=dt_sig)
    # Apply to the first free mass (not the fixed anchor) so the chain is driven
    setup.add_actuator(free_nodes[0], "noise_drive", type="force")

    setup.save()

    print(f"\n[Step 1] Running simulation (backend={BACKEND})...")
    Engine(backend=BACKEND).run(Simulation(EXP))
    print("  Done.")

    # Pad u by one sample to match the simulation's frame count (includes t=0)
    u_padded = np.concatenate([[u[0]], u])
    return u_padded


# ─────────────────────────────────────────────────────────────────────────────
def step_2_extract_features():
    """Load the simulation and inspect available observables."""
    print("\n[Step 2] Loading state...")

    sim_file = EXP / "output" / "simulation.h5"
    loader = StateLoader(sim_file)

    print(f"  Frames : {loader.total_frames}")
    print(f"  dt     : {loader.dt:.4f} s")

    X_pos = features.NodePositions(node_ids="all", dims="all").transform(loader)
    X_str = features.BarStrains().transform(loader)

    print(f"  NodePositions feature shape : {X_pos.shape}  (T × n_nodes×3)")
    print(f"  BarStrains    feature shape : {X_str.shape}  (T × n_bars)")
    return loader


# ─────────────────────────────────────────────────────────────────────────────
def step_3_train_readout(loader, u_input):
    """Train a Ridge readout to predict a τ-lag delayed version of the input."""
    print("\n[Step 3] Training ridge readout...")

    # Target: input delayed by τ steps (short-term memory task)
    TAU        = 5      # lag in frames at SAVE_DT = 0.01 s → 50 ms
    y_full     = np.roll(u_input, TAU)
    y_full[:TAU] = 0.0  # mask wrap-around

    feat   = features.BarStrains()               # use bar strains as state
    readout = Ridge(regularization=1e-3)

    trainer = Trainer(
        features=feat,
        readout=readout,
        experiment_dir=str(EXP),
        loader=loader,
        washout=3.0,        # discard first 3 s (transient)
        train_duration=25.0,
        test_duration=8.0,
    )

    result = trainer.train(y_full, task_name=f"Lag{TAU}Memory")
    result.save()

    # cache layout: {'train': (X, y_true, y_pred), 'test': (X, y_true, y_pred)}
    _, y_train_true, y_train_pred = result.cache['train']
    _, y_test_true,  y_test_pred  = result.cache['test']

    def nrmse(pred, true):
        return np.sqrt(np.mean((pred - true) ** 2)) / (true.std() + 1e-8)

    print(f"  τ = {TAU} frames ({TAU * loader.dt * 1000:.0f} ms)")
    print(f"  train NRMSE = {nrmse(y_train_pred, y_train_true):.4f}")
    print(f"  test  NRMSE = {nrmse(y_test_pred,  y_test_true):.4f}")
    return trainer


# ─────────────────────────────────────────────────────────────────────────────
def step_4_memory_benchmark(trainer, u_input):
    """Run the MemoryBenchmark to compute memory capacity vs. lag."""
    print("\n[Step 4] Running MemoryBenchmark...")

    bench = MemoryBenchmark()
    bench.run(
        trainer,
        u_input,
        tau_s=20,       # max lag to test (frames)
        n_s=1,          # linear-only (degree-1 polynomial)
        k_delay=1,      # step size for lag sweep
        ridge=1e-3,
    )

    if hasattr(bench, 'metrics') and bench.metrics:
        m = bench.metrics
        print(f"  Linear MC     : {float(m['linear_memory_capacity']):.3f}")
        print(f"  Nonlinear MC  : {float(m['nonlinear_memory_capacity']):.3f}")
        print(f"  Total capacity: {float(m['total_capacity']):.3f}")
        caps = m['capacities']   # per-basis-function capacity array
        print(f"  Capacity array: {np.array(caps).round(3)}")
    else:
        print("  (metrics unavailable — check MemoryBenchmark.run output)")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    u_input = step_1_build_and_run()
    loader  = step_2_extract_features()
    trainer = step_3_train_readout(loader, u_input)
    step_4_memory_benchmark(trainer, u_input)
