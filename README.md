# OpenPRC: Physical Reservoir Computing Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache_2.0-green.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-accelerated-76B900.svg)](https://developer.nvidia.com/cuda-zone)
[![arXiv](https://img.shields.io/badge/arXiv-2604.07423-b31b1b.svg)](https://arxiv.org/abs/2604.07423)

**OpenPRC** is a modular, GPU-accelerated Python framework for simulating and evaluating physical reservoir computers — mechanical systems that process information through their intrinsic dynamics.

If you use OpenPRC in your research, please cite:

```bibtex
@article{phalak2026openprc,
  title={OpenPRC: A Unified Open-Source Framework for Physics-to-Task Evaluation in Physical Reservoir Computing},
  author={Phalak, Yogesh and Lor, Wen Sin and Khairnar, Apoorva and Jantzen, Benjamin and Naughton, Noel and Li, Suyi},
  journal={arXiv preprint arXiv:2604.07423},
  year={2026}
}
```

---

## Simulation Capabilities

OpenPRC supports diverse mechanical substrates ranging from compliant mass-spring networks to rigid-foldable origami. Below are examples of validated simulation outputs:

<table>
  <tr>
    <td align="center">
      <img src="media/soft_reservoir_network.gif" width="350px" alt="Soft Reservoir Network"/><br/>
      <b>Soft Reservoir Network</b><br/>
      Mass-spring lattice under dynamic actuation
    </td>
    <td align="center">
      <img src="media/miura_ori_tessallation.gif" width="350px" alt="Miura-Ori Tessellation"/><br/>
      <b>Miura-Ori Tessellation</b><br/>
      Rigid-foldable origami pattern
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="media/kirigami_cut.gif" width="350px" alt="Kirigami Structure"/><br/>
      <b>Kirigami Structure</b><br/>
      Compliant network with geometric cuts
    </td>
    <td align="center">
      <img src="media/k_cone.gif" width="350px" alt="K-Cone Origami"/><br/>
      <b>K-Cone Origami</b><br/>
      Non-periodic origami configuration
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="media/bistable_slab.gif" width="350px" alt="Bistable Slab"/><br/>
      <b>Bistable Slab</b><br/>
      Multistable mechanical metamaterial
    </td>
    <td align="center">
      <img src="media/tepered_spring.gif" width="350px" alt="Tapered Spring"/><br/>
      <b>Tapered Spring</b><br/>
      Nonlinear elastic element dynamics
    </td>
  </tr>
</table>

---

## Installation

```bash
pip install openprc

# With GPU support
pip install openprc[cuda]

# With all optional dependencies
pip install openprc[full]
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations |
| `h5py` | HDF5 I/O |
| `scipy` | Numerical methods |
| `pycuda` *(optional)* | GPU acceleration |
| `scikit-learn` *(optional)* | ML utilities |
| `matplotlib` *(optional)* | Visualization |

---

## Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   demlat    │────▶│   reservoir │────▶│   analysis  │────▶│  optimize   │
│  (Physics)  │     │  (Learning) │     │  (Metrics)  │     │  (Search)   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

---

## Quick Start

```python
import openprc
from openprc import demlat, reservoir, analysis

# 1. Create a lattice geometry
geometry = demlat.meshing.rectangular_lattice(
    nx=10, ny=10,
    spacing=0.01,
    stiffness=1e4,
    damping=0.1,
)
geometry.save("./my_experiment/input/geometry.h5")

# 2. Define input signal and run simulation
experiment = (
    demlat.SimulationSetup("./my_experiment")
    .add_signal("input", demlat.signals.white_noise(duration=10.0, amplitude=0.1))
    .wire_actuator(node_idx=0, signal="input", type="force", direction=[1, 0, 0])
    .set_duration(10.0)
    .build()
)

engine = demlat.Engine(backend="cuda")
engine.run(experiment)

# 3. Train reservoir readout and evaluate on NARMA-10
task = reservoir.tasks.NARMA(order=10)
trainer = reservoir.Trainer(
    features=reservoir.features.AllNodePositions(),
    readout=reservoir.readout.Ridge(regularization=1e-6),
)
result = trainer.fit("./my_experiment/output/simulation.h5", task)
print(f"NARMA-10 NRMSE: {result.metrics.nrmse:.4f}")

# 4. Compute memory capacity
mc = analysis.correlation.MemoryCapacity(max_delay=50)
mc_result = mc.compute("./my_experiment/output/simulation.h5")
print(f"Total Memory Capacity: {mc_result.total_linear:.2f}")

# 5. Visualize memory capacity decay and 3D dynamics
analysis.visualization.memory_profiles.plot(
    mc_result,
    title="Memory Profile",
    save_path="./my_experiment/output/mc_profile.png"
)
analysis.visualization.trajectories.plot_3d(
    "./my_experiment/output/simulation.h5",
    node_ids=[45, 50, 55],
    color_by="energy"
)
```

---

## API Reference

### `demlat` — Physics Simulation

```python
from openprc import demlat

# Build and run an experiment
experiment = (
    demlat.SimulationSetup("./experiments/lattice_01")
    .load_geometry("lattice_10x10.h5")
    .add_signal("chirp", demlat.signals.chirp(f0=0.1, f1=10, duration=5.0))
    .wire_actuator(node_idx=0, signal="chirp", type="force", direction=[0, 0, 1])
    .set_duration(10.0)
    .set_integrator("rk4_hybrid", dt=1e-4)
    .build()
)

engine = demlat.Engine(backend="cuda")  # or backend="cpu"
result = engine.run(experiment)         # writes simulation.h5
```

### `reservoir` — Readout Training

```python
from openprc import reservoir

states = reservoir.StateLoader("./experiments/lattice_01/output/simulation.h5")
features = reservoir.features.Composite([
    reservoir.features.NodePositions(node_ids=[10, 15, 20]),
    reservoir.features.BarStrains(bar_ids="all"),
    reservoir.features.PolynomialExpansion(degree=2),
])

task = reservoir.tasks.NARMA(order=10, length=5000)
readout = reservoir.readout.Ridge(regularization=1e-6)

trainer = reservoir.Trainer(
    features=features,
    readout=readout,
    washout=500,
    train_split=0.8,
)
result = trainer.fit(states, task)
result.save("./experiments/lattice_01/output/readout.h5")
print(f"Test NRMSE: {result.metrics.nrmse:.4f}")
```

### `analysis` — Metrics & Benchmarking

```python
from openprc import analysis

# Memory capacity
mc = analysis.correlation.MemoryCapacity(max_delay=100)
mc_result = mc.compute("./experiments/lattice_01/output/simulation.h5")
print(f"Total Linear MC:    {mc_result.total_linear:.2f}")
print(f"Total Nonlinear MC: {mc_result.total_nonlinear:.2f}")

# Full benchmark suite
suite = analysis.benchmarks.StandardSuite()
report = suite.run("./experiments/lattice_01/")
report.save("./experiments/lattice_01/output/metrics.h5")
report.export_pdf("./experiments/lattice_01/output/benchmark_report.pdf")
```

### `optimize` — Topology & Parameter Search

```python
from openprc import optimize

space = optimize.SearchSpace(
    topology=optimize.spaces.GridLattice(
        nx=(5, 20),
        ny=(5, 20),
        connectivity=["4-neighbor", "8-neighbor", "hexagonal"],
    ),
    physics=optimize.spaces.Physics(
        stiffness=(1e2, 1e6, "log"),
        damping=(0.01, 0.5),
    ),
    actuation=optimize.spaces.Actuation(
        num_actuators=(1, 10),
        placement="boundary",
    ),
)

objective = optimize.objectives.Composite([
    (0.7, optimize.objectives.MemoryCapacity(max_delay=50)),
    (0.3, optimize.objectives.EnergyEfficiency()),
])

campaign = optimize.Campaign(
    search_space=space,
    objective=objective,
    optimizer=optimize.algorithms.CMAES(population_size=20, max_evaluations=500),
    output_dir="./optimization_runs/mc_search_01/",
    n_parallel=4,
)

result = campaign.run()
print(f"Best MC: {result.best.fitness:.2f}")
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
