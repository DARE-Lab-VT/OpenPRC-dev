# Schema Specification

This document defines the exact structure of every file in the bundle. It is the authoritative reference; if the code and this document disagree, the document is wrong and should be updated.

## Overview

Three file types define the bundle:

1. **`manifest.json`** at the root — index of all robots and all trajectories.
2. **`<robot>/metadata.json`** per robot — URDF path, joint order, DoF counts, floating-base flag.
3. **`<robot>/trajectories/*.h5`** — the actual trajectory data, one file per trajectory, self-describing via root attributes.

A validator (`_tools/validate_bundle.py`, drafted separately) checks every file against the schemas defined here.

---

## 1. Trajectory HDF5 files

Each HDF5 file contains one trajectory. The file is self-describing: root attributes declare what signals are present, so consumer code reads attributes first and only accesses datasets that exist.

### Datasets

All time-series arrays have the first dimension `T` (number of timesteps). All are `float32` unless noted.

| Path | Shape | Required | Description |
|---|---|---|---|
| `/time` | `(T,)` float64 | yes | Seconds from trajectory start. Monotonically increasing. |
| `/qpos` | `(T, n_joints)` | yes | Joint positions in radians. Column order matches `joint_names`. |
| `/qvel` | `(T, n_joints)` | no | Joint velocities in rad/s. |
| `/tau` | `(T, n_joints)` | no | Joint torques in Nm (measured or estimated). |
| `/base_pose` | `(T, 7)` | floating-base only | `[x, y, z, qw, qx, qy, qz]`, meters and wxyz quaternion. |
| `/base_vel` | `(T, 6)` | no | `[vx, vy, vz, ωx, ωy, ωz]`, linear in m/s, angular in rad/s (body frame). |
| `/contact/foot_force` | `(T, n_contacts)` | no | Normal force at each contact point (N). Typically quadrupeds. |
| `/contact/foot_force_est` | `(T, n_contacts)` | no | Estimated version, when direct measurement unavailable. |
| `/contact/contact_flags` | `(T, n_contacts)` bool | no | Binary contact indicator. |
| `/action/target_qpos` | `(T, n_joints)` | no | Commanded joint positions. |
| `/action/target_tau` | `(T, n_joints)` | no | Commanded joint torques. |
| `/action/target_qvel` | `(T, n_joints)` | no | Commanded joint velocities. |

### Root attributes

Every trajectory file has these attributes attached to the root group:

| Attribute | Type | Required | Description |
|---|---|---|---|
| `schema_version` | string | yes | e.g. `"0.1.0"`; matches `manifest.version`. |
| `robot_name` | string | yes | e.g. `"go1"`. Must match a key in `manifest.json`. |
| `trajectory_id` | string | yes | Unique within the robot. Also encoded in the filename. |
| `joint_names` | string list | yes | Length `n_joints`, matches `/qpos` column order and URDF joint names. |
| `contact_names` | string list | if contact data present | e.g. `["FR_foot", "FL_foot", "RR_foot", "RL_foot"]`. |
| `frequency_hz` | float | yes | Nominal sampling rate. |
| `duration_s` | float | yes | Total duration = `/time[-1] - /time[0]`. Redundant with `/time`; included for manifest queries. |
| `n_timesteps` | int | yes | `T`. Redundant but cheap. |
| `source` | string | yes | Short source identifier, e.g. `"legkilo"`, `"droid"`, `"robopianist"`, `"unitree_hf_g1_pouring"`. |
| `source_type` | string | yes | One of: `"real_hardware"`, `"teleop_real"`, `"sim_rollout"`. |
| `source_url` | string | yes | URL to the original dataset release. |
| `source_citation` | string | yes | BibTeX key or full citation string of the source paper/release. |
| `source_license` | string | yes | SPDX identifier or short name (e.g. `"MIT"`, `"Apache-2.0"`, `"CC-BY-4.0"`). |
| `signals_present` | string list | yes | Explicit list of dataset paths present, e.g. `["qpos", "qvel", "tau", "base_pose", "contact/foot_force"]`. |
| `quaternion_order` | string | if any quaternion present | Always `"wxyz"` in this bundle. |
| `coordinate_frame` | string | yes | Always `"z_up_x_forward"` in this bundle. |
| `units` | string | yes | Always `"rad, m, s, N, Nm"` in this bundle. |
| `notes` | string | no | Free text. Task description, anomalies, etc. |

### Validation rules

1. `len(joint_names) == /qpos.shape[1]` and equals `n_dof_actuated` in `metadata.json`.
2. If `contact_names` present, `len(contact_names) == /contact/foot_force.shape[1]` (or `contact_flags`, whichever is present).
3. Every path listed in `signals_present` exists as a dataset; every dataset in the file (other than `time` and `qpos`) appears in `signals_present`.
4. `/time` is monotonically non-decreasing. No NaNs anywhere.
5. If `floating_base: true` in `metadata.json`, `base_pose` must be present.
6. Quaternions in `base_pose` are unit-norm (tolerance 1e-3).
7. `schema_version` must be compatible with `manifest.version` (same major version).

### Example (pseudo-code inspection)

```python
import h5py

with h5py.File("go1/trajectories/trot_flat_001.h5", "r") as f:
    attrs = dict(f.attrs)
    print(attrs["robot_name"])           # "go1"
    print(attrs["source_type"])          # "real_hardware"
    print(attrs["signals_present"])      # ["qpos", "qvel", "tau", "base_pose", "contact/foot_force"]
    print(attrs["joint_names"])          # ["FR_hip", "FR_thigh", "FR_calf", ...]

    qpos = f["/qpos"][:]                 # (T, 12)
    foot_force = f["/contact/foot_force"][:]   # (T, 4)
```

---

## 2. Per-robot `metadata.json`

Lives at `<robot>/metadata.json`. Describes the robot once, so trajectories don't repeat invariant information.

```json
{
  "schema_version": "0.1.0",
  "robot_name": "go1",
  "display_name": "Unitree Go1",
  "urdf_path": "urdf/go1.urdf",
  "mesh_dir": "urdf/meshes",
  "floating_base": true,
  "n_dof_actuated": 12,
  "n_dof_total": 18,
  "joint_names": [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
  ],
  "contact_names": ["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
  "urdf_source": {
    "origin": "mujoco_menagerie",
    "origin_url": "https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go1",
    "conversion_tool": "mjcf_urdf_simple_converter",
    "conversion_notes": "Tendons and sites discarded; kinematics preserved."
  },
  "mass_kg": 12.0,
  "nominal_height_m": 0.30,
  "notes": "Floating base with 6-DoF free joint at root; qpos in trajectories covers the 12 leg joints only."
}
```

### Notes on `n_dof_total` vs `n_dof_actuated`

For a floating-base robot, `n_dof_total = 6 + n_dof_actuated` (counting the floating base). `/qpos` stores only the `n_dof_actuated` entries; the base goes in `/base_pose`. This split is consistent whether the source data stored them combined (MuJoCo-style `qpos[7:]`) or separately (typical real-robot logs).

---

## 3. Top-level `manifest.json`

The single entry point. Consumers should read this first.

```json
{
  "schema_version": "0.1.0",
  "version": "0.1.0",
  "created_utc": "2026-04-22T00:00:00Z",
  "robots": {
    "go1": {
      "metadata_path": "go1/metadata.json",
      "urdf_path": "go1/urdf/go1.urdf",
      "n_dof_actuated": 12,
      "floating_base": true,
      "trajectory_count": 57,
      "trajectories": [
        {
          "id": "trot_flat_001",
          "path": "go1/trajectories/trot_flat_001.h5",
          "duration_s": 8.24,
          "frequency_hz": 500.0,
          "n_timesteps": 4120,
          "source": "legkilo",
          "source_type": "real_hardware",
          "source_license": "MIT",
          "signals_present": ["qpos", "qvel", "tau", "base_pose", "contact/foot_force"],
          "task": "trotting on flat ground",
          "split": "train"
        }
      ]
    },
    "panda": { "...": "..." },
    "anymal_c": { "...": "..." },
    "shadow_hand": { "...": "..." },
    "g1": { "...": "..." }
  },
  "sources": {
    "legkilo": {
      "citation": "Ou et al., LegKilo: Legged Robot Kinematics-Inertial-LiDAR Odometry Dataset, 2023",
      "url": "https://github.com/ouguangjun/legkilo-dataset",
      "license": "MIT"
    },
    "droid": {
      "citation": "Khazatsky et al., DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset, RSS 2024",
      "url": "https://droid-dataset.github.io/",
      "license": "MIT"
    },
    "robopianist": {
      "citation": "Zakka et al., RoboPianist: Dexterous Piano Playing with Deep Reinforcement Learning, CoRL 2023",
      "url": "https://kzakka.com/robopianist/",
      "license": "Apache-2.0"
    },
    "rp1m": {
      "citation": "Zhao et al., RP1M: A Large-Scale Motion Dataset for Piano Playing with Bi-Manual Dexterous Robot Hands, 2024",
      "url": "https://rp1m.github.io/",
      "license": "Apache-2.0"
    },
    "unitree_g1_hf": {
      "citation": "Unitree Robotics, G1 Task Datasets on Hugging Face",
      "url": "https://huggingface.co/unitreerobotics",
      "license": "Apache-2.0"
    },
    "eth_anymal": {
      "citation": "varies per paper; see per-trajectory source_citation",
      "url": "varies",
      "license": "varies"
    }
  }
}
```

### Design notes

- **`split`** per trajectory lets us define train/test partitions at bundle-creation time rather than having consumers reinvent splits. Values: `"train"`, `"test"`, `"val"`. Default `"train"`.
- **`signals_present`** is duplicated in the manifest (also in the HDF5 attrs) so the manifest alone is enough for querying. Validator checks consistency.
- **Why both `schema_version` and `version`**: `schema_version` is the schema format (file structure); `version` is the bundle content version (which trajectories are included). They evolve independently.

---

## 4. JSON Schema files

For machine validation, the `_schema/` directory contains strict JSON Schema definitions:

- `_schema/manifest.schema.json`
- `_schema/robot_metadata.schema.json`
- `_schema/trajectory.schema.json` (describes the HDF5 root-attribute shape, since HDF5 itself has no schema language)

These are drafted in Stage 1 alongside the Python validator.

---

## 5. Versioning policy

- **Major version bump**: any change that breaks existing consumer code (renaming a field, changing a unit convention, removing a required attribute, changing the wxyz/xyzw ordering).
- **Minor version bump**: additive changes (new optional fields, new robots, new trajectories).
- **Patch version bump**: fixes to existing data (correcting a mislabeled trajectory, fixing a joint-order bug in one file).

Consumer code must check `manifest.version`'s major field and refuse to proceed against incompatible bundles.
