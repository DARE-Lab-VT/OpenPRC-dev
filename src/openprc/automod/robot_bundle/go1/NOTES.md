# Go1 — Data Notes

Robot-specific facts and conversion decisions for the Unitree Go1 slot of the bundle. Written during ingestion so the next person (including future you) doesn't have to reverse-engineer it.

## Data source

Trajectories come from the [legkilo-dataset](https://github.com/ouguangjun/legkilo-dataset) by Ou et al., RA-L 2024. Seven recorded sequences (`corridor`, `park`, `indoor`, `running`, `slope`, `rotation`, `grass`), each a ROS1 `.bag` file published on [Google Drive](https://drive.google.com/drive/folders/1Egpj7FngTTPCeQDEzlbiK3iesPPZtqiM). Download each bag manually before ingestion.

## URDF source

Converted from MuJoCo Menagerie's `unitree_go1/go1.xml` using `mjcf_urdf_simple_converter`. Kinematics and meshes are preserved; MuJoCo-specific dynamics parameters (joint damping, armature, solver settings, composite tendons) are not representable in URDF and are dropped. For visualization in PiViz and forward kinematics, this is fine.

Meshes are copied into `go1/urdf/meshes/` and the URDF's `<mesh filename=...>` entries are rewritten to relative paths so the robot directory is self-contained.

## Joint-order remap — the important one

This is the quirk that would silently ruin every downstream result if gotten wrong. The Unitree SDK publishes `motorState[20]` in a non-sequential leg order:

| SDK slot | Leg | SDK slot | Leg |
|---|---|---|---|
| 0, 1, 2 | FR hip/thigh/calf | 6, 7, 8 | **RL** hip/thigh/calf |
| 3, 4, 5 | FL hip/thigh/calf | 9, 10, 11 | **RR** hip/thigh/calf |

The Menagerie MJCF (and the upstream `go1_description.urdf`) lists joints in this order:

| URDF slot | Leg | URDF slot | Leg |
|---|---|---|---|
| 0, 1, 2 | FR | 6, 7, 8 | **RR** |
| 3, 4, 5 | FL | 9, 10, 11 | **RL** |

RR and RL are swapped between the two conventions. The remap index array in `go1_transforms.py` is:

```python
LEGKILO_TO_URDF_IDX = [0,1,2, 3,4,5, 9,10,11, 6,7,8]
```

This is a permutation of `range(12)` — enforced by a test. Joint names in `metadata.json` follow URDF order, so any consumer loading `/qpos` and reading `joint_names` gets consistent columns.

Foot forces (`footForce[4]` in the message) are indexed in SDK leg order `FR, FL, RL, RR`, so a separate remap `[0, 1, 3, 2]` reorders to our canonical `FR, FL, RR, RL`.

## Quaternion convention

Unitree publishes quaternions as `[x, y, z, w]` (scalar-last) in both the IMU and `/state_SDK` messages. Our schema stores `[w, x, y, z]` (scalar-first). Conversion happens once at ingestion; everything downstream can assume wxyz.

## Base pose source

Legkilo provides two candidates:
- `/high_state.imu.quaternion`: the built-in IMU orientation estimate (no position).
- `/state_SDK`: Unitree's built-in Kalman-filter leg+IMU odometry (`nav_msgs/Odometry`) — position, orientation, linear/angular velocity.

We use `/state_SDK` for `/base_pose` because it's a full 6-DoF pose. It is **not** MoCap ground truth — most legkilo sequences have no motion-capture setup. Expect drift on the order of a few percent over a multi-minute sequence, particularly in yaw. The `corridor` and `slope` sequences start and end at the same physical point, so end-to-end error can be computed for sanity checking, but no per-timestep ground truth exists.

`/state_SDK` publishes at 50 Hz, same as `/high_state`. In practice timestamps rarely align exactly, so we interpolate: linear for position, nearest-neighbor for quaternions (avoiding a SLERP dependency). At 50 Hz the nearest-neighbor error is at most one 20ms step, which is well below the drift floor.

## Contact forces

The `footForce[4]` and `footForceEst[4]` fields are `int16` ADC values from the foot load cells, not Newtons. Unitree does not publish a calibration curve.

Per-sequence normalization to `[0, 1]`:
1. Clip negative raw values to 0 (occasional lift-off sensor artifacts).
2. Divide by the per-sequence max.

Stored in `/contact/foot_force` and `/contact/foot_force_est` as `float32`. The HDF5 root attribute `contact_force_units` is set to `"normalized_per_sequence"` and `contact_force_scale` records the raw max for reproducibility. This means foot-force values are directly comparable within a sequence, but not across sequences — which is fine for PRC readout training on individual trajectories.

Binary `/contact/contact_flags` is computed from normalized foot force with a threshold of `0.1`. Threshold is hardcoded for now; revisit if we see false positives on soft terrain (grass sequence especially).

## Clip slicing

Each `.bag` is a continuous ~7-minute recording at 50 Hz. We slice into non-overlapping 30-second clips. Tail shorter than 30s is discarded. This yields ~14 clips per full sequence.

Train/test split: the last 2 clips of each sequence go to `test`, the rest to `train`. This keeps test clips temporally held-out from training data, which is the right thing to do for sequence-based training where leakage via autocorrelation is a concern. Values are recorded in each trajectory's manifest entry `split` field and also in the HDF5 `notes` attribute.

## Missing signals and why

| Signal | Status | Why |
|---|---|---|
| `qpos`, `qvel`, `tau` | ✓ present | From `motorState.q`, `.dq`, `.tauEst` |
| `base_pose` | ✓ present | From `/state_SDK` (not MoCap) |
| `base_vel` | ✗ skipped | Available in `/state_SDK.twist`, but not needed for initial PRC experiments; easy to add later |
| `contact/foot_force` | ✓ present | Normalized per-sequence |
| `contact/foot_force_est` | ✓ present | Unitree's built-in estimate, also normalized |
| `contact/contact_flags` | ✓ present | Thresholded from foot force |
| `action/*` | ✗ skipped | Legkilo logs *state*, not the policy's commanded targets. The robot was driven by Unitree's high-level controller; commands aren't exposed in `/high_state`. |

If the action streams matter for a future PRC experiment, they'd have to be inferred (e.g., backing out target_qpos from a low-level controller model) or collected from a fresh run with the SDK. Flagging this as an open item, not a blocker.

## License check

The legkilo repository does not carry an explicit top-level LICENSE file as of this writing. The paper is published in RA-L (IEEE), which doesn't govern the dataset separately. **Action before publication**: contact the authors (Ou et al.) to confirm redistribution terms, or switch to a source with an explicit dataset license. For now `metadata.json` records `"MIT"` as the most likely intent based on the repo's code license style, but this is marked as `TODO verify` in `fetch_go1.py`.

## How to ingest

```bash
# One-time URDF conversion (needs a clone of mujoco_menagerie)
python _tools/fetch/fetch_go1.py \
    --bundle-dir /path/to/robot_bundle \
    convert-urdf \
    --menagerie-dir /path/to/mujoco_menagerie

# Per-bag ingestion (run for each downloaded bag)
python _tools/fetch/fetch_go1.py \
    --bundle-dir /path/to/robot_bundle \
    ingest-bag /path/to/corridor.bag

# Validate after each ingest
python _tools/validate_bundle.py /path/to/robot_bundle --robot go1
```

Expected trajectory count across all 7 sequences: ~90–100 clips, depending on exact bag durations.
