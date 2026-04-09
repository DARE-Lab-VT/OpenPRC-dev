# DEMLAT Examples

Comprehensive test suite for the DEMLAT physics simulation package.

## Directory Structure

```
examples/
├── utils/                      # Shared utilities
│   ├── __init__.py
│   ├── geometry.py            # GeometryBuilder API
│   ├── experiment.py          # ExperimentBuilder API
│   └── validation.py          # Result validation tools
│
├── 01_bars/                   # Bar element tests
│   ├── test_soft_springs.py   # Soft spring dynamics
│   └── test_rigid_bars.py     # PBD rigid bar constraints
│
├── 02_hinges/                 # Hinge element tests
│   ├── test_soft_hinges.py    # Soft dihedral springs
│   └── test_rigid_hinges.py   # PBD rigid angle constraints
│
├── 03_combined/               # Combined element tests
│   └── test_combined_elements.py  # Origami, mechanisms, tensegrity
│
├── 04_actuation/              # Actuation system tests
│   └── test_actuation.py      # Position-driven actuation
│
├── 05_stress_tests/           # Stability & performance tests
│   └── test_stability.py      # Edge cases and scaling
│
└── run_all_examples.py        # Master runner script
```

## Quick Start

### 1. Generate All Experiments

```bash
cd examples
python run_all_examples.py
```

This creates experiment directories in `output/` within each test folder.

### 2. Run a Specific Test Category

```bash
python run_all_examples.py 01      # Bar tests only
python run_all_examples.py 03 04   # Combined and actuation tests
```

### 3. Run the Simulation

```python
from demlat import Engine
from demlat.models.barhinge import BarHingeModel
from demlat.core.experiment import Experiment

exp = Experiment("01_bars/output/test_01_undamped_spring")
Engine(BarHingeModel).run(exp)
```

### 4. Visualize Results

```python
from demlat.utils.viz_player import launch_player
launch_player("01_bars/output/test_01_undamped_spring")
```

### 5. Validate Results

```python
from utils.validation import validate_experiment
validate_experiment("01_bars/output/test_01_undamped_spring")
```

---

## Utility APIs

### GeometryBuilder

Intuitive API for constructing geometries:

```python
from utils import GeometryBuilder

geo = GeometryBuilder()

# Add nodes
geo.add_node([0, 0, 0], mass=1.0, fixed=True)
geo.add_node([1, 0, 0], mass=1.0)

# Add bars
geo.add_bar(0, 1, stiffness=1000, damping=5.0)
geo.add_bar(0, 1, rigid=True)  # PBD constraint

# Add hinges (dihedral springs)
geo.add_hinge(j, k, i, l, stiffness=100, rest_angle=np.pi)

# Prefab geometries
from utils import create_pendulum, create_chain, create_tetrahedron
geo = create_pendulum(length=1.0, n_segments=3)
```

### ExperimentBuilder

Fluent API for creating complete experiments:

```python
from utils import ExperimentBuilder

exp = ExperimentBuilder("my_experiment")
exp.set_simulation(duration=5.0, dt=0.0001, dt_save=0.01)
exp.set_physics(gravity=-9.81, damping=0.5)
exp.set_geometry(geo)

# Add actuation
exp.add_sine_signal("motor", amplitude=0.1, frequency=2.0, axis='z')
exp.add_actuator(node_idx=0, signal_name="motor")

exp.save()  # Creates experiment directory
```

---

## Test Categories

### 01_bars: Bar Elements

| Test | Description | Key Check |
|------|-------------|-----------|
| `test_01_undamped_spring` | Simple harmonic oscillator | Frequency matches √(k/m) |
| `test_02_damped_spring` | Exponential decay | Amplitude decreases |
| `test_03_chain_catenary` | Chain under gravity | Forms catenary shape |
| `test_04_prestressed` | Prestress in spring | Equilibrium shift |
| `test_05_stiffness_comparison` | Multiple stiffnesses | Different frequencies |
| `test_01_rigid_pendulum` | Rigid bar pendulum | Length exactly constant |
| `test_02_double_pendulum` | Chaotic dynamics | Both lengths constant |
| `test_03_rigid_triangle` | Rigid body | Shape preserved |
| `test_04_mixed_soft_rigid` | Hybrid system | Rigid parts stay rigid |
| `test_05_rigid_chain` | Linkage mechanism | All lengths preserved |

### 02_hinges: Hinge Elements

| Test | Description | Key Check |
|------|-------------|-----------|
| `test_01_flat_hinge_oscillation` | Hinge at π | Returns to flat |
| `test_02_prefolded_hinge` | Rest angle ≠ π | Folds to target |
| `test_03_hinge_stiffness` | Different k values | Frequency varies |
| `test_04_rigid_bars_soft_hinge` | Hybrid | Only angle changes |
| `test_05_accordion` | Multiple hinges | Wave-like motion |
| `test_01_rigid_flat_hinge` | Locked at 180° | Angle exactly π |
| `test_02_rigid_90_hinge` | L-bracket | Angle exactly π/2 |
| `test_03_mixed_hinges` | Rigid + soft | Correct behavior each |
| `test_04_rigid_pyramid` | All rigid | True rigid body |
| `test_05_rigid_hinge_load` | Under stress | Holds angle |

### 03_combined: Complex Systems

| Test | Description | Key Check |
|------|-------------|-----------|
| `test_01_miura_ori` | Origami unit cell | Symmetric folding |
| `test_02_four_bar` | Compliant mechanism | Mechanism motion |
| `test_03_tensegrity` | Struts + cables | Shape maintained |
| `test_04_hinged_beam` | Discrete bending | Beam-like sag |
| `test_05_sheet_mesh` | Cloth-like | Draping behavior |

### 04_actuation: Driven Systems

| Test | Description | Key Check |
|------|-------------|-----------|
| `test_01_sine_actuation` | Sine wave input | Tracks signal |
| `test_02_circular_actuation` | 2-DOF motion | Circular path |
| `test_03_step_response` | Step input | Transient oscillation |
| `test_04_ramp_response` | Slow ramp | Quasi-static |
| `test_05_synchronized` | Multiple actuators | Phase-locked |
| `test_06_mechanism` | Crank-slider | Motion conversion |

### 05_stress_tests: Edge Cases

| Test | Description | Key Check |
|------|-------------|-----------|
| `test_01_high_stiffness` | k = 100,000 | No explosion |
| `test_02_large_deformation` | Wild swinging | Stable |
| `test_03_fast_actuation` | 50 Hz input | No aliasing |
| `test_04_dense_constraints` | Many rigid bars | All satisfied |
| `test_05_energy_conservation` | 60s simulation | Energy stable |
| `test_06_scaling_*` | 50/500/2000 nodes | Performance |
| `test_07_mixed_stiffness` | 1000x ratio | Both correct |

---

## Validation Checks

After running simulations, validate with:

```python
from utils.validation import validate_experiment

results = validate_experiment("path/to/experiment")
# Checks: explosion, rigid_bars, rigid_hinges
```

Individual checks available:
- `check_for_explosion()` - NaN/infinity detection
- `check_rigid_bar_constraints()` - Length preservation
- `check_rigid_hinge_constraints()` - Angle preservation
- `check_energy_conservation()` - For undamped systems

---

## Expected Behaviors

### Soft Elements
- **Springs**: Oscillate at natural frequency ω = √(k/m)
- **Damped springs**: Exponential amplitude decay
- **Hinges**: Angular oscillation about rest angle

### Rigid Elements (PBD)
- **Rigid bars**: Length = rest_length ± 1e-6
- **Rigid hinges**: Angle = rest_angle ± 0.01 rad
- Multiple iterations ensure convergence

### Actuation
- Position-driven nodes exactly follow their signal
- Connected structures respond through elastic coupling

---

## Troubleshooting

**Simulation explodes (NaN)**:
- Reduce timestep `dt`
- Increase damping
- Check for degenerate geometry (zero-length bars)

**Rigid constraints drift**:
- Increase PBD iterations in solver
- Check that constraints aren't conflicting

**Slow performance**:
- Ensure CUDA backend is active
- Reduce save frequency `dt_save`
- Use rigid bars instead of very stiff springs
