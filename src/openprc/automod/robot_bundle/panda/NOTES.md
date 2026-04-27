# Panda — Data Notes

Quirks and decisions for the Franka Emika Panda slot of the bundle.

## Data source

Trajectories come from [DROID](https://droid-dataset.github.io/) (Khazatsky et al., RSS 2024), a 76k-episode teleop dataset collected across 13 institutions using Panda arms. Distributed via TensorFlow Datasets (TFDS) as RLDS-formatted shards.

Size: full dataset is roughly 1.7 TB. The `droid_100` variant (if enabled in TFDS) is a ~15 GB subset of 100 representative episodes. For our bundle we curate ~100 episodes via stride-based sampling over the full dataset index space.

## URDF source

Converted from MuJoCo Menagerie's `franka_emika_panda/panda.xml`. Kinematics and meshes preserved; MuJoCo-specific dynamics parameters dropped.

## Joint-name convention

Matches Menagerie and the upstream `franka_description` URDF exactly:

```
panda_joint1 ... panda_joint7, panda_finger_joint1, panda_finger_joint2
```

No remapping needed — DROID's `joint_position` field stores these in the same order.

## Gripper handling

DROID logs `gripper_position` as a single scalar in `[0, 1]` (0 = closed, 1 = fully open), not per-finger joint values. The Panda URDF has two prismatic finger joints that mirror each other.

Our expansion: `per_finger_m = gripper * 0.04` (max 4cm opening per finger). Both finger joints receive the same value. Stored in `/qpos` columns 7 and 8.

If you want arm-only (7-DoF) data without the gripper, run with `--no-gripper` — then `/qpos` is (T, 7), matching just the arm joints.

## What's missing

- **No `/tau`.** DROID does not log joint torques, even though Panda's hardware provides them. The decision was made upstream; we can't recover what wasn't recorded.
- **No `/base_pose`.** The Panda is fixed-base; `floating_base: false` in metadata.
- **No contact forces.** Not measured in DROID.
- **Action field varies.** DROID has multiple action spaces across its subsets (joint position, joint velocity, cartesian, etc). Our script writes `action/target_qpos` only when the action vector matches the qpos dimensionality; otherwise skips it. This is the safe choice but means some episodes come out without action data.

## Sampling rate

DROID logs at 15 Hz. The `time` array is synthesized from frame index × 1/15 since DROID doesn't carry per-step timestamps.

## License

DROID is released under MIT per the project page and dataset card. Recorded in each trajectory's `source_license` and in `LICENSE.md`.

## How to ingest

```bash
# Prerequisites (note: TensorFlow is heavy — 2+ GB install)
pip install 'tensorflow>=2.13' tensorflow-datasets tqdm

# One-time URDF conversion
python _tools/fetch/fetch_panda.py \
    --bundle-dir . \
    convert-urdf \
    --menagerie-dir /path/to/mujoco_menagerie

# Curated ingestion of ~100 episodes
python _tools/fetch/fetch_panda.py \
    --bundle-dir . \
    ingest \
    --target-episodes 100

# Or arm-only (no gripper)
python _tools/fetch/fetch_panda.py \
    --bundle-dir . \
    ingest \
    --target-episodes 100 \
    --no-gripper

python _tools/build_manifest.py .
python _tools/validate_bundle.py . --robot panda --verbose
```

First TFDS call downloads DROID; expect a long wait on first run.
