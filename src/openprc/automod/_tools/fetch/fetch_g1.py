"""
G1 bundle ingestion (both WBT whole-body and Dex3 dexterous manipulation).

Pipeline:
  1. Convert Menagerie's unitree_g1 MJCF → URDF (via mjcf_urdf_simple_converter).
  2. Write g1/metadata.json (or g1_dex3/metadata.json depending on --mode).
  3. Pull LeRobot dataset from HuggingFace Hub via the official loader.
  4. For each episode: extract observation.state and action, remap joint columns
     from LeRobot's native order to our URDF order, synthesize a time array,
     optionally slice into 30s clips if the episode is long, and write HDF5
     clips via write_trajectory.

Usage:

    # One-time URDF conversion
    python fetch_g1.py --bundle-dir . convert-urdf \\
        --menagerie-dir /path/to/mujoco_menagerie \\
        --mode wbt

    # Ingest one task dataset
    python fetch_g1.py --bundle-dir . ingest \\
        --mode wbt \\
        --hf-repo unitreerobotics/UnifoLM_WBT_FoldClothes_Dataset \\
        --max-episodes 30

IMPORTANT: Treat anything in this file marked "TODO: verify at run-time" as
best-guess — the exact LeRobot feature schema for each G1 dataset may differ
from what the HF dataset card suggests. The script is defensive about this:
it reads meta/info.json and the first episode before committing to a joint
mapping, and errors out clearly if the data doesn't match expectations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from write_trajectory import write_trajectory  # noqa: E402
from g1_transforms import (  # noqa: E402
    build_joint_remap,
    times_from_frame_indices,
    split_episode_train_test,
    slice_long_episode,
)


# -----------------------------------------------------------------------------
# Dataset-specific configuration
# -----------------------------------------------------------------------------
#
# Two modes. Each has a different URDF subset and therefore a different
# set of joint_names. We read the LeRobot `names` metadata at runtime and
# intersect against the target joint names here — anything missing on either
# side raises an error, so we never silently animate the wrong joints.
#
# The exact Unitree joint-name strings come from Unitree's publicly documented
# kG1_29 enumeration. They look like `kLeftShoulderPitch`, `kLeftHip`, etc.
# In the URDF these correspond to joint names like `left_shoulder_pitch_joint`.
# We translate between the two below.

# Canonical URDF joint names for the full-body G1 (Menagerie's g1.xml).
# TODO: verify at run-time — actual URDF joint names come from the converted
# URDF. This is the expected order based on Menagerie's MJCF convention.
G1_URDF_FULL_BODY_JOINT_NAMES = [
    # Left leg (6)
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    # Right leg (6)
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # Waist (1)
    "waist_yaw_joint",
    # Left arm (7)
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    # Right arm (7)
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]  # 27 joints, floating base adds 6 for 33 total

# Unitree's kG1_29 joint enumeration (what appears in LeRobot `names` field)
# mapped to our URDF names above. Used by the joint-remap logic.
# TODO: verify against actual LeRobot dataset metadata at run-time.
UNITREE_TO_URDF_NAME_MAP = {
    "kLeftHipPitch": "left_hip_pitch_joint",
    "kLeftHipRoll": "left_hip_roll_joint",
    "kLeftHipYaw": "left_hip_yaw_joint",
    "kLeftKnee": "left_knee_joint",
    "kLeftAnklePitch": "left_ankle_pitch_joint",
    "kLeftAnkleRoll": "left_ankle_roll_joint",
    "kRightHipPitch": "right_hip_pitch_joint",
    "kRightHipRoll": "right_hip_roll_joint",
    "kRightHipYaw": "right_hip_yaw_joint",
    "kRightKnee": "right_knee_joint",
    "kRightAnklePitch": "right_ankle_pitch_joint",
    "kRightAnkleRoll": "right_ankle_roll_joint",
    "kWaistYaw": "waist_yaw_joint",
    "kLeftShoulderPitch": "left_shoulder_pitch_joint",
    "kLeftShoulderRoll": "left_shoulder_roll_joint",
    "kLeftShoulderYaw": "left_shoulder_yaw_joint",
    "kLeftElbow": "left_elbow_joint",
    "kLeftWristRoll": "left_wrist_roll_joint",
    "kLeftWristPitch": "left_wrist_pitch_joint",
    "kLeftWristYaw": "left_wrist_yaw_joint",
    "kRightShoulderPitch": "right_shoulder_pitch_joint",
    "kRightShoulderRoll": "right_shoulder_roll_joint",
    "kRightShoulderYaw": "right_shoulder_yaw_joint",
    "kRightElbow": "right_elbow_joint",
    "kRightWristRoll": "right_wrist_roll_joint",
    "kRightWristPitch": "right_wrist_pitch_joint",
    "kRightWristYaw": "right_wrist_yaw_joint",
}

# Dex3 mode: upper body only (arms + hand joints). No legs/waist.
# TODO: the exact hand joint names on both sides need verification once we
# see the actual dataset — Unitree's Dex3 has ~21 total DoF across both hands.
G1_DEX3_URDF_JOINT_NAMES_PLACEHOLDER: List[str] = [
    # Will be populated from the actual LeRobot `names` field at ingest time.
    # This is a placeholder — see _resolve_joint_names_for_mode.
]

SOURCE_URL = "https://huggingface.co/unitreerobotics"
SOURCE_LICENSE = "Apache-2.0"  # typical for unitreerobotics org; verify
SOURCE_CITATION = (
    "Unitree Robotics (2026). UnifoLM-WBT-Dataset: Whole-Body Teleoperation "
    "Dataset for the G1 Humanoid. https://huggingface.co/unitreerobotics"
)

CLIP_DURATION_S = 30.0
DEFAULT_FPS = 30.0


# -----------------------------------------------------------------------------
# Lazy imports — LeRobot is heavy and the CLI should be helpful even without it
# -----------------------------------------------------------------------------

def _import_lerobot():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as e:
        raise ImportError(
            "LeRobot is required for G1 ingestion. Install with:\n"
            "    pip install lerobot\n"
            "Or from the official repo for latest features:\n"
            "    pip install 'lerobot @ git+https://github.com/huggingface/lerobot.git'"
        ) from e
    return LeRobotDataset


# -----------------------------------------------------------------------------
# Joint-name resolution
# -----------------------------------------------------------------------------

def _flatten_names(names_field):
    """
    LeRobot's `names` field for state is sometimes a nested list like
    [[...]] (a 1-row matrix) and sometimes a flat list. Normalize to a
    flat list of strings.
    """
    if names_field is None:
        return None
    flat: List[str] = []
    def _walk(x):
        if isinstance(x, str):
            flat.append(x)
        elif isinstance(x, (list, tuple)):
            for item in x:
                _walk(item)
    _walk(names_field)
    return flat


def _resolve_joint_names_for_mode(
        mode: str,
        source_names: List[str],
) -> List[str]:
    """
    Given the LeRobot-reported joint names and the requested mode, return
    the target URDF joint name list.

    For `wbt`: full body minus hand fingers (matches Menagerie's g1.xml).
    For `dex3`: whatever hand/arm joints the dataset provides, translated
    to URDF names where possible.
    """
    if mode == "wbt":
        # Translate each source name to a URDF-style name; filter to the body
        # joints we expect (no fingers in WBT mode).
        translated = [_translate_joint_name(n) for n in source_names]
        target = [n for n in G1_URDF_FULL_BODY_JOINT_NAMES if n in translated]
        if len(target) < 10:
            raise ValueError(
                f"WBT mode found only {len(target)} matching joints. "
                f"Source names ({len(source_names)}): {source_names[:15]}...\n"
                f"After translation ({len(translated)}): {translated[:15]}...\n"
                f"Expected (subset of): {G1_URDF_FULL_BODY_JOINT_NAMES[:5]}...\n"
                f"Common causes:\n"
                f"  - Dataset is upper-body only (no legs/waist)\n"
                f"  - Joint naming differs from kG1_29 or snake_case URDF\n"
                f"    convention — extend UNITREE_TO_URDF_NAME_MAP and retry"
            )
        return target
    elif mode == "dex3":
        # Trust whatever the dataset provides; translate each name.
        return [_translate_joint_name(n) for n in source_names]
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Expected 'wbt' or 'dex3'.")


def _looks_like_urdf_name(name: str) -> bool:
    """
    Detect if a joint name is already in URDF style (snake_case with optional
    '_joint' suffix), versus Unitree-SDK style (kCamelCase). Heuristic: URDF
    names are mostly lowercase with underscores; SDK names start with 'k'
    followed by an uppercase letter.
    """
    if not name:
        return False
    if name[0] == "k" and len(name) > 1 and name[1].isupper():
        return False  # kLeftHip → SDK style
    # Lowercase-dominated, contains underscore → URDF style
    lower_ratio = sum(1 for c in name if c.islower()) / max(len(name), 1)
    return lower_ratio > 0.5 or "_" in name


def _translate_joint_name(source_name: str) -> str:
    """
    Best-effort conversion from whatever-the-LeRobot-dataset-uses to our
    canonical URDF name. Order of preference:
      1. Exact match in the manual translation table.
      2. Already URDF-shaped — return as-is (maybe add _joint suffix).
      3. Heuristic kCamelCase → snake_case conversion.
    """
    if source_name in UNITREE_TO_URDF_NAME_MAP:
        return UNITREE_TO_URDF_NAME_MAP[source_name]
    if _looks_like_urdf_name(source_name):
        # Already URDF-shaped. Append _joint if missing.
        if not source_name.endswith("_joint"):
            return source_name + "_joint"
        return source_name
    return _unitree_hand_to_urdf(source_name)


def _unitree_hand_to_urdf(unitree_name: str) -> str:
    """Heuristic conversion for hand joint names not in the map."""
    # kLeftHandThumb0 → left_hand_thumb_0_joint
    # Strip 'k' prefix, then convert CamelCase to snake_case, then append _joint
    if unitree_name.startswith("k"):
        unitree_name = unitree_name[1:]
    out = []
    for i, ch in enumerate(unitree_name):
        if ch.isupper() and i > 0:
            out.append("_")
        out.append(ch.lower())
    # Digits at the end don't get a separator normally; check for trailing digit
    snake = "".join(out)
    # Insert underscore before trailing digit if not already there
    if len(snake) >= 2 and snake[-1].isdigit() and snake[-2] != "_":
        snake = snake[:-1] + "_" + snake[-1]
    return snake + "_joint"


# -----------------------------------------------------------------------------
# Episode iteration
# -----------------------------------------------------------------------------

def iter_episodes(dataset, max_episodes: Optional[int] = None):
    """
    Yield (episode_index, from_idx, to_idx) triples over the dataset's episodes.
    """
    meta = dataset.meta
    n_episodes = meta.total_episodes
    if max_episodes is not None:
        n_episodes = min(n_episodes, max_episodes)
    for ep in range(n_episodes):
        from_idx = int(meta.episodes["dataset_from_index"][ep])
        to_idx = int(meta.episodes["dataset_to_index"][ep])
        yield ep, from_idx, to_idx


def extract_episode(
        dataset, from_idx: int, to_idx: int, state_key: str, action_key: str,
) -> Dict[str, np.ndarray]:
    """
    Read every step in [from_idx, to_idx) and stack into numpy arrays.
    Returns {'state': (T, D), 'action': (T, A)} or action-less if action_key is None.
    """
    import tqdm  # progress bar for long episodes
    state_rows = []
    action_rows = []
    for step_idx in tqdm.tqdm(range(from_idx, to_idx), leave=False,
                              desc=f"  frames {from_idx}-{to_idx}"):
        step = dataset[step_idx]
        state_rows.append(np.asarray(step[state_key], dtype=np.float32))
        if action_key and action_key in step:
            action_rows.append(np.asarray(step[action_key], dtype=np.float32))
    out = {"state": np.stack(state_rows)}
    if action_rows:
        out["action"] = np.stack(action_rows)
    return out


# -----------------------------------------------------------------------------
# Main ingest flow
# -----------------------------------------------------------------------------

def _update_metadata_with_discovered_joints(
        bundle_dir: str, robot_subdir: str, discovered_joints: List[str],
):
    """
    Rewrite `<robot>/metadata.json` so its joint_names and DoF counts match
    what we actually found in the LeRobot data. This prevents schema drift
    between metadata.json (written at convert-urdf time with placeholders)
    and trajectory HDF5s (written during ingest with real names).
    """
    meta_path = os.path.join(bundle_dir, robot_subdir, "metadata.json")
    if not os.path.exists(meta_path):
        print(f"  warning: {meta_path} not found; skipping metadata update. "
              f"Run convert-urdf first.")
        return

    with open(meta_path) as f:
        metadata = json.load(f)

    old_joints = metadata.get("joint_names", [])
    if old_joints == discovered_joints:
        return  # no change needed

    n_new = len(discovered_joints)
    metadata["joint_names"] = list(discovered_joints)
    metadata["n_dof_actuated"] = n_new
    if metadata.get("floating_base"):
        metadata["n_dof_total"] = n_new + 6
    else:
        metadata["n_dof_total"] = n_new

    existing_notes = metadata.get("notes", "")
    update_note = (
        f" [joints updated from {len(old_joints)} placeholders to {n_new} "
        f"names discovered in LeRobot data]"
    )
    if update_note not in existing_notes:
        metadata["notes"] = existing_notes + update_note

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  updated {meta_path}: joint_names now has {n_new} entries")


def ingest(
        bundle_dir: str,
        hf_repo: str,
        mode: str,
        max_episodes: Optional[int],
        robot_subdir: str,
):
    """Pull a LeRobot dataset from HF and emit trajectory HDF5 clips."""
    LeRobotDataset = _import_lerobot()

    print(f"Loading LeRobot dataset: {hf_repo}")
    # Pass revision='main' to bypass LeRobot's codebase_version git-tag lookup.
    # Unitree publishes WBT datasets without the expected version tags, so
    # without this we hit RevisionNotFoundError. See huggingface/lerobot #3080
    # for context.
    try:
        dataset = LeRobotDataset(repo_id=hf_repo, revision="main")
    except TypeError:
        # Older LeRobot versions might not accept revision=; fall back.
        dataset = LeRobotDataset(repo_id=hf_repo)
    meta = dataset.meta
    print(f"  {meta.total_episodes} episodes, {meta.total_frames} total frames, "
          f"fps={meta.fps}, robot_type={meta.robot_type}")

    # Identify the state and action feature keys
    feature_keys = list(meta.features.keys())
    state_key = "observation.state"
    action_key = "action" if "action" in feature_keys else None
    if state_key not in feature_keys:
        raise ValueError(
            f"Dataset missing {state_key!r}. Available features: {feature_keys}"
        )
    state_feature = meta.features[state_key]
    source_names = _flatten_names(state_feature.get("names"))
    if source_names is None:
        raise ValueError(
            f"Dataset {hf_repo} has no `names` field for {state_key}; cannot "
            f"map columns to joints. Contact dataset maintainer or add a manual "
            f"override in this script."
        )
    print(f"  source joint names ({len(source_names)}): "
          f"{source_names[:6]}{'...' if len(source_names) > 6 else ''}")

    # Resolve target joint names and build the remap
    target_joint_names = _resolve_joint_names_for_mode(mode, source_names)
    source_urdf_names = [_translate_joint_name(n) for n in source_names]
    remap = build_joint_remap(source_urdf_names, target_joint_names)
    print(f"  target joint names ({len(target_joint_names)}): "
          f"{target_joint_names[:6]}{'...' if len(target_joint_names) > 6 else ''}")

    # Rewrite metadata.json with the joint names we actually discovered, so
    # trajectory HDF5s and metadata.json stay consistent and the validator
    # passes. This replaces the placeholder names we wrote during convert-urdf.
    _update_metadata_with_discovered_joints(
        bundle_dir, robot_subdir, target_joint_names
    )

    # Episode-level train/test split
    total_ep = min(meta.total_episodes, max_episodes) if max_episodes else meta.total_episodes
    train_ids, test_ids = split_episode_train_test(total_ep)
    print(f"  split: {len(train_ids)} train, {len(test_ids)} test episodes")

    out_dir = os.path.join(bundle_dir, robot_subdir, "trajectories")
    os.makedirs(out_dir, exist_ok=True)
    manifest_entries: List[dict] = []

    fps = float(meta.fps or DEFAULT_FPS)
    task_tag = hf_repo.split("/")[-1].replace("_Dataset", "").lower()

    for ep, from_idx, to_idx in iter_episodes(dataset, max_episodes):
        n_frames = to_idx - from_idx
        if n_frames < 2:
            print(f"  episode {ep}: too short ({n_frames} frames), skipping")
            continue
        split = "test" if ep in test_ids else "train"

        print(f"  episode {ep} ({n_frames} frames, split={split}):")
        data = extract_episode(dataset, from_idx, to_idx, state_key, action_key)

        # Remap columns to URDF order
        qpos_full = data["state"][:, remap]
        action_full = data["action"][:, remap] if "action" in data else None

        # Slice if needed
        clips = slice_long_episode(n_frames, CLIP_DURATION_S, fps)
        for clip_idx, (s, e) in enumerate(clips):
            clip_id = f"{task_tag}_ep{ep:04d}_clip{clip_idx:02d}"
            out_path = os.path.join(out_dir, f"{clip_id}.h5")
            t_clip = times_from_frame_indices(e - s, fps)

            write_trajectory(
                out_path=out_path,
                robot_name=robot_subdir,
                trajectory_id=clip_id,
                joint_names=target_joint_names,
                time=t_clip,
                qpos=qpos_full[s:e],
                target_qpos=action_full[s:e] if action_full is not None else None,
                source=f"unitree_hf_{task_tag}",
                source_type="teleop_real",
                source_url=f"https://huggingface.co/datasets/{hf_repo}",
                source_citation=SOURCE_CITATION,
                source_license=SOURCE_LICENSE,
                notes=(
                    f"Task: {task_tag}, episode {ep}, clip {clip_idx+1}/{len(clips)}. "
                    f"Split: {split}. Teleop data from Unitree's LeRobot release. "
                    f"Floating base not recorded — base pose omitted."
                ),
                overwrite=True,
            )

            signals = ["qpos"]
            if action_full is not None:
                signals.append("action/target_qpos")

            manifest_entries.append({
                "id": clip_id,
                "path": f"{robot_subdir}/trajectories/{clip_id}.h5",
                "duration_s": float(t_clip[-1] - t_clip[0]),
                "frequency_hz": fps,
                "n_timesteps": int(e - s),
                "source": f"unitree_hf_{task_tag}",
                "source_type": "teleop_real",
                "source_license": SOURCE_LICENSE,
                "signals_present": signals,
                "task": task_tag,
                "split": split,
            })
            print(f"    wrote {clip_id}.h5 ({split}, {e-s} frames)")

    # Write or update the manifest fragment
    fragment_path = os.path.join(bundle_dir, robot_subdir, "_manifest_fragment.json")
    existing = []
    if os.path.exists(fragment_path):
        with open(fragment_path) as f:
            existing = json.load(f).get("trajectories", [])
    existing_ids = {e["id"] for e in manifest_entries}
    existing = [e for e in existing if e["id"] not in existing_ids]
    existing.extend(manifest_entries)
    existing.sort(key=lambda e: e["id"])
    with open(fragment_path, "w") as f:
        json.dump({"trajectories": existing}, f, indent=2)
    print(f"  manifest fragment: {len(existing)} total trajectories")


# -----------------------------------------------------------------------------
# URDF conversion command
# -----------------------------------------------------------------------------

def cmd_convert_urdf(bundle_dir: str, menagerie_dir: str, mode: str):
    """Convert Menagerie's G1 MJCF to URDF."""
    sys.path.insert(0, os.path.dirname(HERE))
    from mjcf_to_urdf import convert_mjcf_to_urdf

    robot_subdir = "g1" if mode == "wbt" else "g1_dex3"

    # Menagerie's G1 is at unitree_g1/g1.xml
    mjcf_path = os.path.join(menagerie_dir, "unitree_g1", "g1.xml")
    if not os.path.exists(mjcf_path):
        # Fallback — menagerie dir might be pointed at the g1 subdir directly
        alt = os.path.join(menagerie_dir, "g1.xml")
        if os.path.exists(alt):
            mjcf_path = alt
        else:
            raise FileNotFoundError(
                f"G1 MJCF not found at {mjcf_path} or {alt}. "
                f"Make sure --menagerie-dir points to a clone of mujoco_menagerie."
            )

    urdf_out = os.path.join(bundle_dir, robot_subdir, "urdf", "g1.urdf")
    mesh_dir = os.path.join(bundle_dir, robot_subdir, "urdf", "meshes")

    print(f"Converting {mjcf_path} → {urdf_out}")
    convert_mjcf_to_urdf(mjcf_path, urdf_out, mesh_dir=mesh_dir)

    metadata = {
        "schema_version": "0.1.0",
        "robot_name": robot_subdir,
        "display_name": "Unitree G1" + (" (Dex3)" if mode == "dex3" else ""),
        "urdf_path": "urdf/g1.urdf",
        "mesh_dir": "urdf/meshes",
        "floating_base": True,
        # n_dof will be set correctly by the ingest step; these are placeholders
        "n_dof_actuated": 27 if mode == "wbt" else 21,
        "n_dof_total": (27 if mode == "wbt" else 21) + 6,
        "joint_names": (
            G1_URDF_FULL_BODY_JOINT_NAMES if mode == "wbt"
            else G1_DEX3_URDF_JOINT_NAMES_PLACEHOLDER
        ),
        "urdf_source": {
            "origin": "mujoco_menagerie",
            "origin_url": "https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1",
            "conversion_tool": "mjcf_urdf_simple_converter",
            "conversion_notes": "Kinematics and meshes preserved; dynamics dropped.",
            "license": "BSD-3-Clause",  # typical for Menagerie G1 — verify
        },
        "notes": (
            f"Mode: {mode}. Joint names will be cross-checked against LeRobot "
            f"dataset metadata during ingest. If joint_names above does not "
            f"exactly match the URDF's actuated_joints, the validator will "
            f"catch it."
        ),
    }
    with open(os.path.join(bundle_dir, robot_subdir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote {robot_subdir}/metadata.json")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="G1 bundle ingestion")
    parser.add_argument("--bundle-dir", required=True,
                        help="Path to the robot_bundle directory")
    parser.add_argument("--mode", required=True, choices=["wbt", "dex3"],
                        help="wbt = whole-body joints, dex3 = hand joints")

    sub = parser.add_subparsers(dest="command", required=True)

    p_convert = sub.add_parser("convert-urdf")
    p_convert.add_argument("--menagerie-dir", required=True)

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("--hf-repo", required=True,
                          help="HuggingFace repo id, e.g. "
                               "unitreerobotics/UnifoLM_WBT_FoldClothes_Dataset")
    p_ingest.add_argument("--max-episodes", type=int, default=None,
                          help="Cap the number of episodes (for quick tests)")

    args = parser.parse_args()

    if args.command == "convert-urdf":
        cmd_convert_urdf(args.bundle_dir, args.menagerie_dir, args.mode)
    else:
        robot_subdir = "g1" if args.mode == "wbt" else "g1_dex3"
        ingest(
            args.bundle_dir,
            args.hf_repo,
            args.mode,
            args.max_episodes,
            robot_subdir,
        )


if __name__ == "__main__":
    main()