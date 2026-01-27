# OpenPRC: Physical Reservoir Computing Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Development Status](https://img.shields.io/badge/status-active_development-orange.svg)](https://github.com/yourusername/OpenPRC-dev)
[![License](https://img.shields.io/badge/license-Apache_2.0-green.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-accelerated-76B900.svg)](https://developer.nvidia.com/cuda-zone)
[![HDF5](https://img.shields.io/badge/HDF5-data_schema-blue.svg)](https://www.hdfgroup.org/)

**OpenPRC** is a modular, GPU-accelerated Python framework for simulating and optimizing physical reservoir computers—mechanical systems that process information through their intrinsic dynamics. This repository contains the active development branch implementing the complete architecture specification.

**Backend:** PyCUDA • NumPy • HDF5 Schema • Hybrid RK4-Constraint Relaxation

---

## Development Status

> **Note:** This is the development repository (`OpenPRC-dev`) where individual modules are being implemented and tested. The `demlat` subpackage (Discrete Element Modeling) is feature-complete and production-ready.

**Module Implementation Progress:**

| Module | Status | Description |
|--------|--------|-------------|
| **demlat** |  Complete | GPU-accelerated physics simulation engine |
| **reservoir** | In Progress | Reservoir computing interfaces and readout layers |
| **analysis** | In Progress | Performance metrics and dimensionality analysis |
| **optimize** | Planned | Evolutionary topology and parameter search |

---

## Simulation Capabilities

The `demlat` module supports diverse mechanical systems ranging from compliant networks to rigid-foldable origami. Below are examples of validated simulation outputs:

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

All simulations shown were executed using the `demlat.models.barhinge` (truss networks) and `demlat.models.origami` (rigid-foldable) physics models with CUDA acceleration.

---

## Architecture Overview

**Open Physical Reservoir Computing Framework**

Version 0.1.0 | Architecture Specification

---

## Summary

OpenPRC is a high-performance, open-source Python framework designed to serve as the foundational academic tool for the simulation and optimization of Physical Reservoir Computers (PRC). It features a modular architecture that strictly separates geometry definition from physical execution, utilizing a unified HDF5 data schema to model diverse mechanical substrates—from compliant mass-spring webs to multistable origami structures—within a single environment.

Powered by GPU-accelerated hybrid solvers that combine explicit Runge-Kutta integration with iterative constraint relaxation, OpenPRC enables researchers to treat complex mechanical dynamics as a black-box function, facilitating the evolutionary optimization of topology and physical parameters to maximize computational memory and information processing capacity.

---

## 1. Design Philosophy

### 1.1 Core Principles

| Principle | Implementation |
| --- | --- |
| **Separation of Concerns** | Geometry definition, physics execution, analysis, and learning are isolated into independent modules |
| **Reproducibility First** | Self-contained experiment directories with versioned schemas ensure exact replication |
| **Backend Agnosticism** | Physics models expose a unified interface; CPU/GPU implementations are interchangeable |
| **Schema-Driven Data** | HDF5 with strict schemas serves as the universal interchange format |
| **Composable Pipelines** | Each module consumes and produces well-defined artifacts, enabling flexible workflows |

### 1.2 The OpenPRC Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   demlat    │────▶│   reservoir │────▶│   analysis  │────▶│  optimize   │
│  (Physics)  │     │  (Learning) │     │  (Metrics)  │     │  (Search)   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
  simulation.h5      readout.h5          metrics.h5         candidates.h5

```

---

## 2. Package Structure

```bash
openprc/
│
├── __init__.py                    # Public API surface
├── _version.py                    # Semantic versioning
│
├── demlat/                        # Module 1: Discrete Element Modeling
│   ├── __init__.py
│   ├── io/
│   ├── core/
│   ├── models/
│   └── utils/
│
├── reservoir/                     # Module 2: Reservoir Computing
│   ├── __init__.py
│   ├── readout/
│   ├── tasks/
│   └── io/
│
├── analysis/                      # Module 3: Data Analysis & Benchmarking
│   ├── __init__.py
│   ├── correlation/
│   ├── dimensionality/
│   ├── benchmarks/
│   └── visualization/
│
├── optimize/                      # Module 4: Optimization & Search
│   ├── __init__.py
│   ├── objectives/
│   ├── algorithms/
│   └── constraints/
│
├── schemas/                       # Shared HDF5 schema definitions
│   ├── geometry.py
│   ├── simulation.py
│   ├── readout.py
│   └── metrics.py
│
└── common/                        # Cross-cutting utilities
    ├── logging.py
    ├── config.py
    ├── exceptions.py
    └── validators.py

```

---

## 3. Module Specifications

---

### 3.1 `demlat` — Discrete Element Modeling

**Purpose:** High-fidelity simulation of nonlinear mechanical networks as computational substrates.

### 3.1.1 Scope

The `demlat` module treats mechanical structures as graphs of discrete elements (nodes, bars, hinges, faces) and propagates their dynamics through time. It is deliberately agnostic to reservoir computing concepts—it simply produces trajectories.

### 3.1.2 Submodule Architecture

```bash
demlat/
│
├── io/
│   ├── experiment_setup.py       # Fluent builder API for experiment creation
│   ├── validator.py              # Schema validation and inspection
│   └── exporters.py              # Format converters (VTK, PLY, etc.)
│
├── core/
│   ├── experiment.py             # Filesystem interface (paths, config loading)
│   ├── engine.py                 # Main simulation loop orchestrator
│   ├── base_model.py             # Abstract base class for physics models
│   ├── state_computer.py         # Post-processing (energy, strain, stress)
│   ├── scaler.py                 # Unit conversion (physical ↔ dimensionless)
│   └── exceptions.py             # Domain-specific exceptions
│
├── models/
│   ├── __init__.py               # Model registry and factory
│   │
│   ├── barhinge/                 # Truss/lattice networks
│   │   ├── model.py              # Python wrapper (BarHingeModel)
│   │   ├── solver_cpu.py         # NumPy reference implementation
│   │   ├── solver_cuda.py        # PyCUDA high-performance backend
│   │   └── kernels/
│   │       └── barhinge.cu
│   │
│   ├── origami/                  # Rigid-foldable origami structures
│   │   ├── model.py              # OrigamiModel wrapper
│   │   ├── solver_cpu.py
│   │   └── solver_cuda.py
│   │
│   └── continuum/                # Future: FEM-based substrates
│       └── ...
│
└── utils/
    ├── meshing.py                # Procedural geometry generators
    ├── topology.py               # Graph analysis utilities
    └── viz_player.py             # Quick visualization playback

```

### 3.1.3 Data Schema — Experiment Directory

```bash
Experiment_Name/
│
├── input/
│   ├── config.json               # Simulation parameters + actuator wiring
│   ├── geometry.h5               # Structural definition (nodes, elements)
│   ├── signals.h5                # Input time-series library
│   └── visualization.h5          # Rendering metadata (faces, colors)
│
├── output/
│   ├── simulation.h5             # Trajectories(positions, velocities, energies)
│   └── logs/                     # Performance metrics, solver diagnostics
│
└── README.md                     # Auto-generated experiment summary

```

### 3.1.4 Data Schema —  `input/geometry.h5` (The Hardware)

Defines the physical structure. It knows **nothing** about the input signals, only which nodes *can* be actuated.

```
/ (Root)
│
├── nodes/
│   ├── positions     : [N, 3]   (float32)
│   ├── masses        : [N]      (float32)
│   ├── attributes    : [N]      (uint8)   # Bitmask
│   │                                   # 0 = Free Floating
│   │                                   # 1 = Fixed (Anchor)
│   │                                   # 2 = Position Actuator (Driver)
│   │                                   # 4 = Force Actuator (Thruster)
│   └── radius        : [N]      (float32)  (Visualization)
│
└── elements/
    │
    ├── bars/
    │   ├── indices      : [M, 2] (int32)
    │   ├── stiffness    : [M]    (float32)
    │   ├── rest_length  : [M]    (float32)
    │   ├── damping      : [M]    (float32)
    │   └── prestress    : [M]    (float64)
    │
    └── hinges/
        ├── indices      : [K, 4] (int32)
        └── stiffness    : [K]    (float32)

```

### 3.1.5 Data Schema —`input/signals.h5` (The Input Library)

A flat library of named time-series arrays. These are generic and reusable.

```python
/ (Root)
│
├── [Attributes]
│   └── dt_base : 0.001  (float32)  # Default signal timestep
│
├── sine_sweep_slow/
│   └── values  : [T]     (float32)
│
├── step_response_A/
│   └── values  : [T]     (float32)
│
└── multi_axis_force_01/
    └── values  : [T, 3]  (float32)

```

### 3.1.5 Data Schema — `input/config.json` (The Wiring)

This file maps the generic signals to the specific physical nodes.

JSON

```jsx
{
  "meta": {
    "experiment_id": "Lattice_Test_04",
    "description": "Force actuation test on floating node 5"
  },
  "simulation": {
    "duration": 10.0,
    "dt_policy": "adaptive",  // or "fixed"
    "dt_base": 0.001,
    "dt_save": 0.01,         // Stride for saving to simulation.h5
    "integrator": "rk4_hybrid"
  },
  "global_physics": {
    "gravity": [0, 0, -9.81],
    "global_damping": 0.1
  },
  "actuators": [
    {
      "node_idx": 10,
      "type": "force",           // "force" or "position"
      "dof": [0, 0, 1],          // Direction Vector (Z-axis)
      "signal_source": "signals.h5",
      "signal_name": "sine_sweep_01",
      "interpolation": "cubic", // Interpolation style for the solver
      "scale": 5.0               // Multiplier (e.g., 5 Newtons max)
    },
    {
      "node_idx": 0,
      "type": "position",
      "dof": [1, 0, 0],
      "signal_source": "signals.h5",
      "signal_name": "step_input_A"
    }
  ],
  "export_options": {
    "save_velocities": true,
    "save_strains": true,
    "save_energy_breakdown": false
  }
}
```

---

### 3.1.6 Data Schema —`simulation.h5` (The Result)

The heavy-lifter output file. To save space, it does not duplicate static data from `geometry.h5`.

```python
/ (Root)
│
├── [Attributes] (Metadata for Player/Loader)
│   ├── source_geometry       : "../input/geometry.h5"
│   ├── source_visualization  : "../input/visualization.h5"
│   ├── total_frames          : 1000  (int)
│   └── frame_rate            : 60.0  (float)
│
└── time_series/
    │
    ├── time                  : [T]       (float32)
    │
    ├── nodes/
    │   ├── positions         : [T, N, 3] (float32)
    │   ├── velocities        : [T, N, 3] (float32)
    │   ├── kinetic_energy    : [T, N]    (float32)
    │   └── potential_energy  : [T, N]    (float32) (Gravity)
    │
    ├── elements/
    │   ├── bars/
    │   │   ├── strain           : [T, M] (float32)
    │   │   ├── stress           : [T, M] (float32)
    │   │   └── potential_energy : [T, M] (float32) (Elastic)
    │   │
    │   └── hinges/
    │       ├── angle            : [T, K] (float32)
    │       ├── torsional_strain : [T, K] (float32)
    │       └── potential_energy : [T, K] (float32) (Elastic)
    │
    └── system/
        ├── kinetic_energy    : [T]       (float32) (Total)
        ├── potential_energy  : [T]       (float32) (Total Gravity + Elastic)
        └── damping_loss      : [T]       (float32) (Cumulative)
```

### 3.1.6 Data Schema — `input/visualization.h5` (Rendering Metadata)

```bash
/ (Root)
│
├── faces               : [F, 3]  (int32)   
│                                           # Indices into geometry.h5/nodes/positions
│                                           # Defines the mesh topology (Triangles)
│
├── [Optional / Generalized Extensions]
│   │
│   ├── face_colors     : [F, 4]  (uint8)   # RGBA colors for each face
│   │                                       # If missing, use default material
│   │
│   ├── face_groups     : [F]     (int32)   # Group ID per face (e.g., "Top Surface", "Hinge")
│   │                                       # Useful for toggling visibility in UI
│   │
│   └── group_names     : [G]     (string)  # Mapping of Group IDs to names
│
└── [Attributes]
    ├── topology_type   : "triangle"        # Hint for the renderer (vs quad/poly)
    └── shading_mode    : "flat"            # "flat" vs "smooth" (vertex normals)
```

### 3.1.8 Key Interfaces

```python
from openprc import demlat

# Fluent experiment creation
experiment = (
    demlat.ExperimentSetup("./experiments/lattice_01")
    .load_geometry("lattice_10x10.h5")
    .add_signal("chirp", demlat.signals.chirp(f0=0.1, f1=10, duration=5.0))
    .wire_actuator(node_idx=0, signal="chirp", type="force", direction=[0, 0, 1])
    .set_duration(10.0)
    .set_integrator("rk4_hybrid", dt=1e-4)
    .build()
)

# Execute simulation
engine = demlat.Engine(backend="cuda")
result = engine.run(experiment)  # Returns path to simulation.h5

```

---

### 3.2 `reservoir` — Reservoir Computing

**Purpose:** Transform mechanical trajectories into computational readouts through linear learning.

### 3.2.1 Scope

The `reservoir` module implements the machine learning layer of physical reservoir computing. It consumes simulation outputs, applies feature extraction, trains readout layers, and evaluates task performance. It also supports task-driven geometry synthesis.

### 3.2.2 Submodule Architecture

```bash
reservoir/
│
├── io/
│   ├── state_loader.py           # Efficient chunked loading from simulation.h5
│   ├── readout_io.py             # Save/load trained readout models
│   └── task_io.py                # Standard task dataset management
│
├── features/
│   ├── base.py                   # Abstract feature extractor
│   ├── node_features.py          # Position, velocity, acceleration
│   ├── element_features.py       # Strain, stress, curvature
│   ├── nonlinear.py              # Polynomial, time-delayed embeddings
│   └── spatial.py                # Spatial averaging, mode decomposition
│
├── readout/
│   ├── base.py                   # Abstract readout layer
│   ├── ridge.py                  # Analytical Ridge regression (default)
│   ├── elastic_net.py            # L1/L2 regularized regression
│   ├── online.py                 # Recursive least squares (streaming)
│   └── ensemble.py               # Multi-readout aggregation
│
├── tasks/
│   ├── base.py                   # Abstract task definition
│   ├── memory/
│   │   ├── narma.py              # NARMA-n benchmarks
│   │   ├── delay_line.py         # Short-term memory capacity
│   │   └── parity.py             # Nonlinear parity tasks
│   ├── classification/
│   │   ├── spoken_digits.py      # Audio classification
│   │   └── time_series.py        # Generic sequence classification
│   ├── prediction/
│   │   ├── mackey_glass.py       # Chaotic time-series prediction
│   │   └── lorenz.py             # Lorenz attractor forecasting
│   └── custom.py                 # User-defined task template
│
├── design/
│   ├── input_placement.py        # Optimal actuator node selection
│   ├── output_placement.py       # Optimal readout node selection
│   └── topology_synthesis.py     # Task-driven structure generation
│
└── utils/
    ├── washout.py                # Transient removal utilities
    ├── normalization.py          # Feature scaling strategies
    └── validation.py             # Cross-validation schemes

```

### 3.2.3 Data Schema — Readout Artifact

```bash
readout.h5
│
├── [Attributes]
│   ├── source_simulation     : "../demlat/output/simulation.h5"
│   ├── task_name             : "narma10"
│   ├── feature_config        : {...}  (JSON string)
│   └── training_timestamp    : "2026-01-20T14:32:00"
│
├── model/
│   ├── weights               : [D, O]     (float64)  # Readout weights
│   ├── bias                  : [O]        (float64)
│   ├── regularization        : scalar     (float64)
│   └── feature_indices       : [D]        (int32)    # Selected node/element IDs
│
├── training/
│   ├── input                 : [T_train]  (float32)
│   ├── target                : [T_train, O] (float32)
│   ├── prediction            : [T_train, O] (float32)
│   └── loss_curve            : [epochs]   (float32)  # If iterative
│
└── validation/
    ├── input                 : [T_val]    (float32)
    ├── target                : [T_val, O] (float32)
    ├── prediction            : [T_val, O] (float32)
    └── metrics/
        ├── nmse              : scalar     (float64)
        ├── nrmse             : scalar     (float64)
        └── correlation       : scalar     (float64)

```

### 3.2.4 Key Interfaces

```python
from openprc import reservoir

# Load simulation and define features
states = reservoir.StateLoader("./experiments/lattice_01/output/simulation.h5")
features = reservoir.features.Composite([
    reservoir.features.NodePositions(node_ids=[10, 15, 20]),
    reservoir.features.BarStrains(bar_ids="all"),
    reservoir.features.PolynomialExpansion(degree=2),
])

# Train readout on NARMA-10 task
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

---

### 3.3 `analysis` — Data Analysis & Benchmarking

**Purpose:** Comprehensive characterization of reservoir dynamics and computational properties.

### 3.3.1 Scope

The `analysis` module provides tools to understand *why* a physical reservoir performs well (or poorly). It computes information-theoretic metrics, visualizes high-dimensional dynamics, and benchmarks against standardized criteria.

### 3.3.2 Submodule Architecture

```bash
analysis/
│
├── correlation/
│   ├── memory_capacity.py        # Linear/nonlinear memory capacity profiles
│   ├── cross_correlation.py      # Input-state and state-state correlations
│   ├── mutual_information.py     # Nonlinear dependency estimation
│   └── kernel_quality.py         # Kernel rank and separation metrics
│
├── dimensionality/
│   ├── pca.py                    # Principal component analysis
│   ├── lle.py                    # Locally linear embedding
│   ├── umap.py                   # UMAP projections
│   ├── effective_dim.py          # Participation ratio, intrinsic dimensionality
│   └── attractor.py              # Lyapunov exponents, fractal dimension
│
├── benchmarks/
│   ├── standard_suite.py         # Consolidated benchmark runner
│   ├── memory_benchmark.py       # MC, NARMA suite
│   ├── separation_benchmark.py   # Kernel quality, generalization
│   ├── nonlinearity_benchmark.py # Parity, XOR, polynomial tasks
│   └── comparison.py             # Multi-reservoir comparison tools
│
├── visualization/
│   ├── trajectories.py           # 2D/3D state-space plots
│   ├── heatmaps.py               # Correlation matrices, weight maps
│   ├── memory_profiles.py        # MC decay curves
│   ├── energy_landscape.py       # Potential energy surfaces
│   └── interactive.py            # Plotly/Bokeh dashboards
│
└── reports/
    ├── generator.py              # Automated report generation
    └── templates/                # LaTeX/HTML report templates

```

### 3.3.3 Data Schema — Metrics Artifact

```bash
metrics.h5
│
├── [Attributes]
│   ├── source_simulation     : "..."
│   ├── source_readout        : "..."
│   └── analysis_timestamp    : "..."
│
├── memory_capacity/
│   ├── delays                : [D]        (int32)
│   ├── linear_mc             : [D]        (float64)
│   ├── quadratic_mc          : [D]        (float64)
│   ├── total_linear          : scalar     (float64)
│   └── total_nonlinear       : scalar     (float64)
│
├── dimensionality/
│   ├── pca_variance          : [K]        (float64)
│   ├── effective_dimension   : scalar     (float64)
│   ├── participation_ratio   : scalar     (float64)
│   └── embedding_coords      : [T, 2]     (float32)  # UMAP/t-SNE
│
├── kernel_quality/
│   ├── separation            : scalar     (float64)
│   ├── generalization        : scalar     (float64)
│   └── kernel_rank           : scalar     (int32)
│
└── benchmark_suite/
    ├── narma10_nrmse         : scalar     (float64)
    ├── narma20_nrmse         : scalar     (float64)
    ├── mackey_glass_nrmse    : scalar     (float64)
    └── parity_accuracy       : scalar     (float64)

```

### 3.3.4 Key Interfaces

```python
from openprc import analysis

# Compute memory capacity profile
mc = analysis.correlation.MemoryCapacity(max_delay=100)
mc_result = mc.compute("./experiments/lattice_01/output/simulation.h5")

print(f"Total Linear MC: {mc_result.total_linear:.2f}")
print(f"Total Nonlinear MC: {mc_result.total_nonlinear:.2f}")

# Run full benchmark suite
suite = analysis.benchmarks.StandardSuite()
report = suite.run("./experiments/lattice_01/")
report.save("./experiments/lattice_01/output/metrics.h5")
report.export_pdf("./experiments/lattice_01/output/benchmark_report.pdf")

# Interactive visualization
analysis.visualization.trajectories.plot_3d(
    "./experiments/lattice_01/output/simulation.h5",
    node_ids=[10, 15, 20],
    color_by="time"
)

```

---

### 3.4 `optimize` — Optimization & Search

**Purpose:** Automated discovery of optimal reservoir configurations for target tasks.

### 3.4.1 Scope

The `optimize` module wraps the entire OpenPRC pipeline into an objective function that optimization algorithms can query. It supports topology optimization, parameter tuning, input signal design, and multi-objective Pareto searches.

### 3.4.2 Submodule Architecture

```bash
optimize/
│
├── objectives/
│   ├── base.py                   # Abstract objective function
│   ├── memory_capacity.py        # Maximize total MC
│   ├── task_performance.py       # Minimize task-specific error
│   ├── energy_efficiency.py      # Minimize actuation energy
│   ├── robustness.py             # Maximize noise tolerance
│   └── composite.py              # Weighted multi-objective
│
├── search_spaces/
│   ├── base.py                   # Abstract search space definition
│   ├── topology.py               # Graph structure parameters
│   │   ├── node_count
│   │   ├── connectivity_pattern
│   │   └── boundary_conditions
│   ├── physics.py                # Material/element parameters
│   │   ├── stiffness_distribution
│   │   ├── damping_coefficients
│   │   └── prestress_pattern
│   ├── actuation.py              # Input configuration
│   │   ├── actuator_placement
│   │   ├── signal_parameters
│   │   └── frequency_content
│   └── readout.py                # Output configuration
│       ├── sensor_placement
│       ├── feature_selection
│       └── regularization
│
├── algorithms/
│   ├── base.py                   # Abstract optimizer
│   ├── evolutionary/
│   │   ├── cma_es.py             # Covariance Matrix Adaptation
│   │   ├── nsga2.py              # Multi-objective genetic algorithm
│   │   └── differential.py       # Differential evolution
│   ├── bayesian/
│   │   ├── gaussian_process.py   # GP-based Bayesian optimization
│   │   └── tpe.py                # Tree-structured Parzen Estimator
│   ├── gradient_free/
│   │   ├── nelder_mead.py        # Simplex method
│   │   └── pattern_search.py     # Direct search
│   └── hybrid/
│       └── surrogate_assisted.py # ML-accelerated optimization
│
├── constraints/
│   ├── base.py                   # Abstract constraint
│   ├── physical.py               # Stability, manufacturability
│   ├── budget.py                 # Computational cost limits
│   └── geometric.py              # Spatial bounds, symmetry
│
└── utils/
    ├── checkpointing.py          # Optimization state persistence
    ├── parallelization.py        # Distributed evaluation
    └── visualization.py          # Convergence plots, Pareto fronts

```

### 3.4.3 Data Schema — Optimization Artifact

```bash
optimization.h5
│
├── [Attributes]
│   ├── objective_name        : "memory_capacity"
│   ├── algorithm             : "cma_es"
│   ├── search_space_config   : {...}
│   └── start_timestamp       : "..."
│
├── history/
│   ├── generations           : [G]           (int32)
│   ├── evaluations           : [G]           (int32)
│   ├── best_fitness          : [G]           (float64)
│   ├── mean_fitness          : [G]           (float64)
│   └── population_diversity  : [G]           (float64)
│
├── candidates/
│   ├── parameters            : [N, P]        (float64)
│   ├── fitness               : [N]           (float64)
│   ├── feasible              : [N]           (bool)
│   └── experiment_paths      : [N]           (string)  # Links to experiments
│
├── best/
│   ├── parameters            : [P]           (float64)
│   ├── fitness               : scalar        (float64)
│   ├── experiment_path       : string
│   └── geometry_snapshot     : reference     # Copy of optimal geometry.h5
│
└── pareto_front/             # For multi-objective
    ├── parameters            : [M, P]        (float64)
    ├── objectives            : [M, O]        (float64)
    └── ranks                 : [M]           (int32)

```

### 3.4.4 Key Interfaces

```python
from openprc import optimize

# Define search space
space = optimize.SearchSpace(
    topology=optimize.spaces.GridLattice(
        nx=(5, 20),      # Range for grid dimensions
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

# Define objective
objective = optimize.objectives.Composite([
    (0.7, optimize.objectives.MemoryCapacity(max_delay=50)),
    (0.3, optimize.objectives.EnergyEfficiency()),
])

# Run optimization
optimizer = optimize.algorithms.CMAES(
    population_size=20,
    sigma_init=0.5,
    max_evaluations=500,
)

campaign = optimize.Campaign(
    search_space=space,
    objective=objective,
    optimizer=optimizer,
    output_dir="./optimization_runs/mc_search_01/",
    n_parallel=4,  # GPU workers
)

result = campaign.run()
print(f"Best MC: {result.best.fitness:.2f}")
print(f"Optimal config: {result.best.experiment_path}")

```

---

## 4. Cross-Module Data Flow

The diagram below illustrates how data flows through a complete OpenPRC workflow:

```bash
                                    ┌─────────────────┐
                                    │   User Input    │
                                    │  (Geometry,     │
                                    │   Signals,      │
                                    │   Config)       │
                                    └────────┬────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              demlat Module                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                      │
│  │ geometry.h5 │───▶│   Engine    │───▶│simulation.h5│                      │
│  │ signals.h5  │    │  (CUDA/CPU) │    │             │                      │
│  │ config.json │    └─────────────┘    └──────┬──────┘                      │
│  └─────────────┘                              │                             │
└───────────────────────────────────────────────┼─────────────────────────────┘
                                                │
                        ┌───────────────────────┼───────────────────────┐
                        │                       │                       │
                        ▼                       ▼                       ▼
┌─────────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────┐
│      reservoir Module       │  │     analysis Module     │  │     optimize Module     │
│                             │  │                         │  │                         │
│  ┌───────────┐              │  │  ┌───────────────┐      │  │  ┌───────────────┐      │
│  │ Features  │              │  │  │ Memory Cap.   │      │  │  │ Search Space  │      │
│  └─────┬─────┘              │  │  │ Dimensionality│      │  │  │ Objectives    │      │
│        │                    │  │  │ Benchmarks    │      │  │  │ Constraints   │      │
│        ▼                    │  │  └───────┬───────┘      │  │  └───────┬───────┘      │
│  ┌───────────┐              │  │          │              │  │          │              │
│  │  Readout  │              │  │          ▼              │  │          ▼              │
│  │  (Ridge)  │              │  │  ┌───────────────┐      │  │  ┌───────────────┐      │
│  └─────┬─────┘              │  │  │  metrics.h5   │      │  │  │ optimization  │      │
│        │                    │  │  └───────────────┘      │  │  │     .h5       │      │
│        ▼                    │  │                         │  │  └───────────────┘      │
│  ┌───────────┐              │  │                         │  │          │              │
│  │ readout.h5│              │  │                         │  │          │              │
│  └───────────┘              │  │                         │  │          ▼              │
│                             │  │                         │  │  ┌───────────────┐      │
└─────────────────────────────┘  └─────────────────────────┘  │  │ New Geometry  │──┐   │
                                                              │  └───────────────┘  │   │
                                                              └─────────────────────┼───┘
                                                                                    │
                                                              ┌─────────────────────┘
                                                              │ (Feedback Loop)
                                                              ▼
                                                        Back to demlat

```

---

## 5. Unified HDF5 Schema Summary

| Artifact | Producer | Consumers | Primary Content |
| --- | --- | --- | --- |
| `geometry.h5` | User / `optimize` | `demlat` | Nodes, elements, topology |
| `signals.h5` | User / `optimize` | `demlat` | Input time-series library |
| `config.json` | User / `optimize` | `demlat` | Simulation parameters, wiring |
| `simulation.h5` | `demlat` | `reservoir`, `analysis` | Trajectories, energies |
| `readout.h5` | `reservoir` | `analysis`, `optimize` | Trained weights, predictions |
| `metrics.h5` | `analysis` | `optimize`, User | MC profiles, benchmarks |
| `optimization.h5` | `optimize` | User | Search history, Pareto front |

---

## 6. Extension Points

OpenPRC is designed for extensibility at multiple levels:

| Extension Type | Mechanism | Example |
| --- | --- | --- |
| **New Physics Model** | Subclass `demlat.core.BaseModel` | Continuum FEM, granular media |
| **New Backend** | Implement solver interface | JAX, Taichi, custom CUDA |
| **New Task** | Subclass `reservoir.tasks.BaseTask` | Domain-specific prediction tasks |
| **New Readout** | Subclass `reservoir.readout.BaseReadout` | Neural readout, SVM |
| **New Optimizer** | Subclass `optimize.algorithms.BaseOptimizer` | Custom evolutionary strategy |
| **New Metric** | Subclass `analysis.BaseMetric` | Novel information measures |

---

## 7. Installation & Dependencies

```bash
pip install openprc

# With GPU support
pip install openprc[cuda]

# With all optional dependencies
pip install openprc[full]

```

### Core Dependencies

| Package | Purpose |
| --- | --- |
| `numpy` | Array operations |
| `h5py` | HDF5 I/O |
| `scipy` | Numerical methods |
| `numba` | JIT compilation |

### Optional Dependencies

| Package | Module | Purpose |
| --- | --- | --- |
| `pycuda` | `demlat` | GPU acceleration |
| `cupy` | `demlat` | GPU arrays |
| `scikit-learn` | `reservoir`, `analysis` | ML utilities |
| `optuna` | `optimize` | Bayesian optimization |
| `matplotlib` | `analysis` | Static visualization |
| `plotly` | `analysis` | Interactive visualization |

---

## 8. Quick Start Example

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
    demlat.ExperimentSetup("./my_experiment")
    .add_signal("input", demlat.signals.white_noise(duration=10.0, amplitude=0.1))
    .wire_actuator(node_idx=0, signal="input", type="force", direction=[1, 0, 0])
    .set_duration(10.0)
    .build()
)

engine = demlat.Engine(backend="cuda")
engine.run(experiment)

# 3. Train reservoir and evaluate
task = reservoir.tasks.NARMA(order=10)
trainer = reservoir.Trainer(
    features=reservoir.features.AllNodePositions(),
    readout=reservoir.readout.Ridge(regularization=1e-6),
)
result = trainer.fit("./my_experiment/output/simulation.h5", task)
print(f"NARMA-10 NRMSE: {result.metrics.nrmse:.4f}")

# 4. Analyze computational properties
mc = analysis.correlation.MemoryCapacity(max_delay=50)
mc_result = mc.compute("./my_experiment/output/simulation.h5")
print(f"Total Memory Capacity: {mc_result.total_linear:.2f}")

# 5. Visualize
# Plot the memory capacity decay curve (completing your cut-off line)
analysis.visualization.memory_profiles.plot(
    mc_result, 
    title="NARMA-10 Memory Profile",
    save_path="./my_experiment/output/mc_profile.png"
)

# Visualize 3D dynamics of the lattice
analysis.visualization.trajectories.plot_3d(
    "./my_experiment/output/simulation.h5",
    node_ids=[45, 50, 55],  # Visualize specific nodes
    color_by="energy"
)

```