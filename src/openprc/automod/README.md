# Robot Trajectory Bundle

A unified dataset of robot meshes, kinematics, and real-hardware trajectories — normalized across five robots for training physics-from-data models (specifically Physical Reservoir Computing) and visualizing results in PiViz.

## Purpose

This bundle exists because public robot datasets are scattered across repositories with incompatible formats: ROS bags, RLDS shards, LeRobot parquet files, paper-specific CSV dumps. Each uses different joint-name conventions, coordinate frames, and signal subsets. Before any auto-modeling method can be fairly evaluated across embodiments, the data has to live in one schema.

The bundle provides:

- **URDFs** for every robot (converted from MuJoCo Menagerie MJCF, or taken from upstream URDF releases where available)
- **Mesh files** (`.obj` / `.stl`) in their original format, referenced by the URDF
- **Trajectories** in a single HDF5 schema, with every signal explicitly declared in metadata so downstream code never has to guess what's present
- **A manifest** at the root indexing every robot and every trajectory with source, license, and signal information

## Robots

| Robot | Type | Source dataset | Data type | Target trajectory count |
|---|---|---|---|---|
| Unitree Go1 | Quadruped (12 DoF) | [legkilo-dataset](https://github.com/ouguangjun/legkilo-dataset) | Real hardware | 50–100 |
| Franka Emika Panda | 7-DoF arm (9 with gripper) | [DROID](https://droid-dataset.github.io/) (curated subset) | Teleop real | ~100 |
| ANYbotics ANYmal C | Quadruped (12 DoF) | *TBD — see anymal_c/*  | pending | pending |
| Unitree G1 (whole body) | Humanoid (~27 DoF) | [UnifoLM-WBT](https://huggingface.co/collections/unitreerobotics/unifolm-wbt-dataset) | Teleop real | 50–150 |
| Unitree G1 (Dex3) | Dexterous hands (~21 DoF) | [unitreerobotics HF org](https://huggingface.co/unitreerobotics) | Teleop real | 50–100 |

All trajectories are from real hardware (either autonomous execution or teleoperation). No simulation data. ANYmal C has no single canonical public dataset, so its slot is treated as "pending" — see `anymal_c/NOTES.md` (once written) or `_tools/fetch/fetch_anymal.py` for candidate sources. If ANYmal data isn't secured in time for submission, the bundle degrades to four robots cleanly.

## Directory layout

```
robot_bundle/
├── README.md                          # this file
├── SCHEMA.md                          # full schema specification
├── LICENSE.md                         # per-dataset license audit
├── manifest.json                      # index of all robots and trajectories
│
├── _schema/                           # reference JSON schemas
│   ├── manifest.schema.json
│   ├── trajectory.schema.json
│   └── robot_metadata.schema.json
│
├── _tools/                            # scripts (drafted separately)
│   ├── validate_bundle.py
│   ├── inspect_trajectory.py
│   └── fetch/
│       ├── fetch_go1.py
│       ├── fetch_panda.py
│       ├── fetch_anymal.py
│       ├── fetch_shadow.py
│       └── fetch_g1.py
│
├── go1/
│   ├── metadata.json
│   ├── urdf/
│   │   ├── go1.urdf
│   │   └── meshes/
│   │       ├── trunk.stl
│   │       └── ...
│   └── trajectories/
│       ├── trot_flat_001.h5
│       ├── trot_flat_002.h5
│       └── ...
│
├── panda/
│   ├── metadata.json
│   ├── urdf/
│   └── trajectories/
│
├── anymal_c/
├── shadow_hand/
└── g1/
```

Each robot directory is self-contained: the URDF references meshes in its own `meshes/` folder only, so a single robot can be copied out and used independently.

## Coordinate conventions

All trajectories use a consistent convention to eliminate a whole class of silent bugs:

- **Frame**: Z-up, X-forward (right-handed). The root `coordinate_frame` attribute on every HDF5 file records this explicitly. Any source dataset using a different convention is transformed at ingestion time.
- **Units**: radians for angles, meters for lengths, seconds for time, Newtons for forces, Newton-meters for torques. Recorded in the `units` attribute.
- **Quaternions**: stored as `[w, x, y, z]` (scalar-first). The `quaternion_order` attribute is always present and always `wxyz`.
- **Joint order**: defined by the `joint_names` attribute on each trajectory and expected to match the URDF joint order. Validation enforces this.
- **Floating-base split**: for floating-base robots (quadrupeds, humanoid), the 6-DoF (or 7-DoF with quaternion) base pose lives in `/base_pose`, separate from `/qpos`. `/qpos` contains only the actuated joints. This matches how real hardware logs the data (state estimator for base, encoders for joints).

## Signals

Not every trajectory has every signal. This is honest: real hardware logs vary. Every trajectory's HDF5 root attribute `signals_present` is an explicit list of which datasets exist in that file, so consumer code reads the manifest first and never has to probe-and-handle-KeyError.

Required on every trajectory: `time`, `qpos`.
Commonly present: `qvel`, `tau`, `base_pose`, `base_vel`.
Sometimes present: `contact/foot_force`, `contact/contact_flags`, `action/target_qpos`, `action/target_tau`.

Full signal definitions, shapes, and dtypes are in `SCHEMA.md`.

## Versioning

The bundle follows semantic versioning. Schema-breaking changes bump the major version (`manifest.version` in `manifest.json`). Adding trajectories or robots bumps the minor version. Any consumer code should check `manifest.version` and refuse to run against an incompatible major version.

Current version: `0.1.0` (pre-release; schema may still change before first stable release).

## Citing the source datasets

This bundle is a re-packaging; the underlying data belongs to the original releases. Any paper using this bundle must cite every source dataset whose trajectories are used. The per-robot `metadata.json` files contain `source_citation` entries for each dataset, and `LICENSE.md` reproduces the full license text or reference for each.

## Status

- [x] Schema files drafted and reviewed (Stage 1)
- [x] Writer, validator, self-tests (Stage 1 — 6/6 passing)
- [x] Go1 parser + MJCF-to-URDF converter (Stage 2 — 14/14 transform tests passing)
- [x] Go1 end-to-end run against real legkilo bags (ingested and validated)
- [x] Manifest builder (Stage 2)
- [x] FK + PiViz adapter + trajectory animator (Stage 3 — 13/13 FK math tests passing)
- [x] G1 + Panda fetch scripts + transforms (Stage 4 — 17/17 transform tests passing)
- [ ] G1 end-to-end ingest + animate (WBT mode)
- [ ] G1 end-to-end ingest + animate (Dex3 mode)
- [ ] Panda end-to-end ingest + animate (requires ~15 GB DROID download)
- [ ] ANYmal C data source decision and ingest
- [ ] Auto-generated data table for paper
- [ ] License audit complete

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install rosbags mjcf_urdf_simple_converter   # extras for Stage 2

# 2. Verify the schema and transforms work on your machine
python _tools/selftest.py                # expects 5/5 passing
python _tools/test_go1_transforms.py     # expects 14/14 passing

# 3. Convert Menagerie's Go1 MJCF to URDF (one-time)
git clone https://github.com/google-deepmind/mujoco_menagerie.git /tmp/menagerie
python _tools/fetch/fetch_go1.py \
    --bundle-dir . \
    convert-urdf \
    --menagerie-dir /tmp/menagerie

# 4. Download one legkilo bag (manual step; see go1/NOTES.md for the GDrive link)
#    ...download corridor.bag, slope.bag, etc...

# 5. Ingest each bag
python _tools/fetch/fetch_go1.py \
    --bundle-dir . \
    ingest-bag ~/Downloads/corridor.bag

# 6. Validate
python _tools/validate_bundle.py . --robot go1 --verbose

# 7. Animate one trajectory in PiViz
python _tools/kinematics/test_fk_math.py    # expect 13/13 passing
python _tools/viz/animate_trajectory.py \
    --bundle-dir . \
    --robot go1 \
    --trajectory corridor_000
```

Per-robot data-quirk notes live in `<robot>/NOTES.md` (e.g., [go1/NOTES.md](./go1/NOTES.md)).
