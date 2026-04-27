# G1 — Data Notes

Quirks and decisions for the Unitree G1 humanoid slot of the bundle.

## Two robots, one platform

The G1 appears twice in the lineup with different joint subsets:

| Subdir | Mode flag | Joints | Data source |
|---|---|---|---|
| `g1/` | `--mode wbt` | Full body (~27 DoF) | [UnifoLM-WBT](https://huggingface.co/collections/unitreerobotics/unifolm-wbt-dataset) whole-body teleop |
| `g1_dex3/` | `--mode dex3` | Arms + hands (~21 DoF) | [unitreerobotics](https://huggingface.co/unitreerobotics) G1_Dex3_* datasets |

Both load via the same `fetch_g1.py` script and differ in which joint subset is extracted from the LeRobot observation.state columns.

## Data source

Released by Unitree on Hugging Face using the LeRobot v2.0/v2.1 format. Each dataset is a collection of parquet files with `observation.state`, `action`, and synchronized camera videos. Per-dataset sizes range from ~90k to ~500k frames. UnifoLM-WBT (March 2026 release) is the whole-body teleop flagship; G1_Dex3 datasets cover dexterous manipulation tasks (toasted bread, block stacking, camera packaging, etc).

Episodes are typically ~30 seconds at 30 fps. Files are downloaded automatically on first use by the LeRobot library (to `~/.cache/huggingface/lerobot/unitreerobotics` by default).

## URDF source

Converted from MuJoCo Menagerie's `unitree_g1/g1.xml`. Menagerie ships the 29-DoF variant. Kinematics and meshes preserved; MuJoCo-specific dynamics dropped.

## Joint-name translation

Unitree's native joint names (seen in LeRobot `observation.state.names`) look like `kLeftShoulderPitch`, `kLeftHip`, etc. These are translated to URDF names (`left_shoulder_pitch_joint`, `left_hip_pitch_joint`) at ingest time. The translation table lives in `_tools/fetch/fetch_g1.py` (`UNITREE_TO_URDF_NAME_MAP`) and covers the kG1_29 enumeration.

For hand joints (Dex3 mode), many finger joints aren't in the hardcoded map — a heuristic converts `kLeftHandThumb0` → `left_hand_thumb_0_joint` by CamelCase-to-snake_case. **Verify this against the actual Dex3 URDF** before publication — if the URDF uses different names, you'll need to extend the map.

## What's missing

- **No `/base_pose`.** LeRobot's G1 datasets don't include base-frame pose — the observation.state is joint-only. This is a real limitation for WBT (you can animate the joints but the robot stays at the origin). For paper figures showing the whole-body motion in space, a workaround is to infer approximate base pose from foot contacts and a kinematic walker model, but that's a future project.
- **No `/tau` or contact forces.** Not recorded.
- **Actions are commanded target positions.** Stored in `/action/target_qpos` when available.
- **No camera/vision data preserved.** LeRobot datasets include video streams but we don't pull them — they aren't useful for PRC training from trajectories.

## Sampling rate

30 Hz across all G1 datasets. Times are synthesized from frame_index × 1/30 since LeRobot stores frame indices, not absolute timestamps.

## Curation strategy

For each task dataset, all episodes are ingested (capped by `--max-episodes` if given). Long episodes are sliced into 30s clips matching the Go1 convention. Short episodes (<60s) become single-clip trajectories.

Train/test split: 85/15 episode-level split with contiguous tail for test, same as other robots.

## License

Per the `unitreerobotics` HF org, datasets are released under Apache-2.0. Each trajectory records this in its `source_license` attribute. Verify the specific dataset card before publication.

## How to ingest

```bash
pip install lerobot tqdm

# One-time URDF conversion (do once per mode)
python _tools/fetch/fetch_g1.py --bundle-dir . --mode wbt convert-urdf \
    --menagerie-dir /path/to/mujoco_menagerie

# WBT ingestion — pick one or more task datasets from the collection:
python _tools/fetch/fetch_g1.py --bundle-dir . --mode wbt ingest \
    --hf-repo unitreerobotics/UnifoLM_WBT_FoldClothes_Dataset \
    --max-episodes 30

python _tools/fetch/fetch_g1.py --bundle-dir . --mode wbt ingest \
    --hf-repo unitreerobotics/UnifoLM_WBT_LoadWashing_Dataset \
    --max-episodes 30

# Dex3 ingestion
python _tools/fetch/fetch_g1.py --bundle-dir . --mode dex3 convert-urdf \
    --menagerie-dir /path/to/mujoco_menagerie

python _tools/fetch/fetch_g1.py --bundle-dir . --mode dex3 ingest \
    --hf-repo unitreerobotics/G1_Dex3_ToastedBread_Dataset \
    --max-episodes 50

# After ingest
python _tools/build_manifest.py .
python _tools/validate_bundle.py . --verbose
```

First run of each dataset triggers download from HF; expect a few minutes.

## Things likely to need adjustment on first run

- **Joint name map incomplete.** If the LeRobot dataset card names joints differently from our hardcoded map, ingest will fail with a clear `Target joint names not found in source data` error listing the missing ones. Fix: add entries to `UNITREE_TO_URDF_NAME_MAP`.
- **WBT datasets may have 23-DoF vs 29-DoF variants.** Unitree supports both G1_23 and G1_29 configurations. Our URDF is 27-joint (29 minus arm wrist_yaw on some variants). The ingest script will tell you what's present.
- **Dex3 URDF:** Menagerie's G1 MJCF may or may not include the Dex3 three-fingered hand — if it doesn't, the Dex3 URDF needs a separate hand URDF appended. Check by running `convert-urdf --mode dex3` and inspecting whether hand joints appear.
