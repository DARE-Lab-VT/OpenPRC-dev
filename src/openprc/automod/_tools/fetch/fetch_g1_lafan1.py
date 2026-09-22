"""
LAFAN1 retargeted motion data → G1 bundle ingest.

Unitree published LAFAN1 (Ubisoft La Forge) motion-capture sequences retargeted
onto the G1 humanoid via IK + interaction-mesh optimization. The data lives at
huggingface.co/datasets/unitreerobotics/LAFAN1_Retargeting_Dataset as CSV
files, one per motion sequence.

Column layout (G1 variant, 30 FPS, 36 columns total):
    [0:3]   root xyz position (meters)
    [3:7]   root quaternion (x, y, z, w) — note: xyzw, not wxyz
    [7:36]  29 actuated joint positions in URDF-declared order:
        left hip pitch/roll/yaw, left knee, left ankle pitch/roll,   (6)
        right hip pitch/roll/yaw, right knee, right ankle pitch/roll, (6)
        waist yaw/roll/pitch,                                         (3)
        left shoulder pitch/roll/yaw, left elbow,                     (4)
        left wrist roll/pitch/yaw,                                    (3)
        right shoulder pitch/roll/yaw, right elbow,                   (4)
        right wrist roll/pitch/yaw                                    (3)

This is *retargeted simulation* data, not real hardware. We tag
`source_type: sim_rollout` accordingly, and note this in every trajectory's
HDF5 attributes and NOTES.md so downstream readers always know.

Motions cover walking, running, dance, falls/get-ups, jumps, and transitions —
exactly the "dynamic humanoid motion" the paper needs to show off.

Usage:
    python fetch_g1_lafan1.py --bundle-dir . ingest \\
        --motions walk1_subject1 run1_subject2 jumps1_subject1 \\
        --max-clips-per-motion 4

    # Or list all available motions
    python fetch_g1_lafan1.py --bundle-dir . list
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from write_trajectory import write_trajectory  # noqa: E402
from g1_transforms import (  # noqa: E402
    times_from_frame_indices,
    split_episode_train_test,
    slice_long_episode,
)

# -----------------------------------------------------------------------------
# Constants from the LAFAN1 dataset card (G1 config)
# -----------------------------------------------------------------------------

LAFAN1_G1_JOINT_NAMES: List[str] = [
    # Left leg (6)
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    # Right leg (6)
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # Waist (3)
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # Left arm (7)
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    # Right arm (7)
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

LAFAN1_FPS = 30.0
LAFAN1_HF_REPO = "lvhaidong/LAFAN1_Retargeting_Dataset"
LAFAN1_G1_SUBDIR = "g1"

SOURCE_NAME = "lafan1_retargeted_g1"
SOURCE_URL = f"https://huggingface.co/datasets/{LAFAN1_HF_REPO}"
SOURCE_CITATION = (
    "Harvey, F. G., Yurick, M., Nowrouzezahrai, D., & Pal, C. (2020). "
    "Robust Motion In-betweening (LAFAN1). ACM Trans. Graph. 39(4). "
    "Retargeted to Unitree G1 via IK + interaction-mesh optimization "
    "(Unitree Robotics, 2024). Distributed via community mirror at "
    f"https://huggingface.co/datasets/{LAFAN1_HF_REPO}"
)
# LAFAN1 mocap is CC-BY-NC-ND 4.0; retargeting code is MIT.
# The retargeted CSVs inherit the more-restrictive CC-BY-NC-ND license.
SOURCE_LICENSE = "CC-BY-NC-ND-4.0"

CLIP_DURATION_S = 30.0


# -----------------------------------------------------------------------------
# HF helpers
# -----------------------------------------------------------------------------

def _hf_api():
    try:
        from huggingface_hub import HfApi
        return HfApi()
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from e


def _hf_download(repo: str, path: str, cache_dir: Optional[str] = None) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id=repo, filename=path, repo_type="dataset",
        cache_dir=cache_dir,
    )


def list_motions() -> List[str]:
    """Enumerate all available G1 motion sequences in the LAFAN1 repo."""
    api = _hf_api()
    files = api.list_repo_files(LAFAN1_HF_REPO, repo_type="dataset")
    g1_csvs = sorted(
        os.path.splitext(os.path.basename(f))[0]
        for f in files
        if f.startswith(f"{LAFAN1_G1_SUBDIR}/") and f.endswith(".csv")
    )
    return g1_csvs


# -----------------------------------------------------------------------------
# Parsing and conversion
# -----------------------------------------------------------------------------

def parse_lafan1_csv(
    csv_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a LAFAN1-retargeted G1 CSV.

    Returns:
        time:      (T,) seconds at 30 Hz
        base_pose: (T, 7) xyz + wxyz quaternion (converted from xyzw)
        qpos:     (T, 29) joint positions in URDF order

    The CSV has 36 columns. The first row is often a header-like row of
    numbers rather than a string header, so we parse assuming no header.
    """
    import pandas as pd

    # Try with no header first; if parsing fails fall back to header=0.
    try:
        df = pd.read_csv(csv_path, header=None, dtype=np.float64)
    except ValueError:
        df = pd.read_csv(csv_path, header=0, dtype=np.float64)

    arr = df.to_numpy()
    if arr.shape[1] != 36:
        raise ValueError(
            f"expected 36 columns (7 root + 29 joints), got {arr.shape[1]} "
            f"in {csv_path}"
        )

    T = arr.shape[0]
    time = times_from_frame_indices(T, LAFAN1_FPS)

    # Root: columns 0..6 are x, y, z, qx, qy, qz, qw
    root_xyz = arr[:, 0:3].astype(np.float32)
    root_quat_xyzw = arr[:, 3:7].astype(np.float32)
    # Convert xyzw → wxyz to match our schema
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]
    base_pose = np.concatenate([root_xyz, root_quat_wxyz], axis=1)

    # Re-normalize quaternions (mocap retargeting may have small numerical drift)
    quat_norms = np.linalg.norm(base_pose[:, 3:7], axis=1, keepdims=True)
    base_pose[:, 3:7] = base_pose[:, 3:7] / np.maximum(quat_norms, 1e-8)

    # Joints: columns 7..35 are the 29 actuated joints in URDF order
    qpos = arr[:, 7:36].astype(np.float32)

    return time, base_pose, qpos


# -----------------------------------------------------------------------------
# Main ingest
# -----------------------------------------------------------------------------

def ingest(
    bundle_dir: str,
    motion_names: List[str],
    max_clips_per_motion: Optional[int],
    cache_dir: Optional[str],
):
    if not motion_names:
        raise ValueError(
            "No motions specified. Run `fetch_g1_lafan1.py list` to see available."
        )

    out_dir = os.path.join(bundle_dir, "g1", "trajectories")
    os.makedirs(out_dir, exist_ok=True)

    # Update metadata.json with the canonical 29-joint list
    _update_metadata(bundle_dir, "g1", LAFAN1_G1_JOINT_NAMES)

    manifest_entries: List[dict] = []

    # Process each motion, slice into clips, route train vs test at the clip level
    all_clips: List[Tuple[str, int, np.ndarray, np.ndarray, np.ndarray]] = []

    for motion in motion_names:
        print(f"\nFetching {motion}.csv ...")
        try:
            csv_path = _hf_download(
                LAFAN1_HF_REPO,
                f"{LAFAN1_G1_SUBDIR}/{motion}.csv",
                cache_dir,
            )
        except Exception as e:
            print(f"  failed to download: {e}")
            continue

        try:
            time, base_pose, qpos = parse_lafan1_csv(csv_path)
        except Exception as e:
            print(f"  failed to parse: {e}")
            continue

        print(f"  {len(time)} frames, {len(time)/LAFAN1_FPS:.1f}s")
        clips = slice_long_episode(len(time), CLIP_DURATION_S, LAFAN1_FPS)

        if max_clips_per_motion is not None:
            clips = clips[:max_clips_per_motion]

        for clip_idx, (s, e) in enumerate(clips):
            all_clips.append((motion, clip_idx, time[s:e], base_pose[s:e], qpos[s:e]))

    if not all_clips:
        print("\nNo clips produced. Nothing to write.")
        return

    # Train/test split over all clips, deterministic tail for test
    n_clips = len(all_clips)
    train_ids, test_ids = split_episode_train_test(n_clips)
    print(f"\n{n_clips} clips total: {len(train_ids)} train, {len(test_ids)} test")

    for clip_i, (motion, clip_idx, t_src, bp_src, q_src) in enumerate(all_clips):
        split = "test" if clip_i in test_ids else "train"
        clip_id = f"lafan1_{motion}_clip{clip_idx:02d}"
        out_path = os.path.join(out_dir, f"{clip_id}.h5")

        # Rebase time so clips start at zero
        t_clip = t_src - t_src[0]

        write_trajectory(
            out_path=out_path,
            robot_name="g1",
            trajectory_id=clip_id,
            joint_names=LAFAN1_G1_JOINT_NAMES,
            time=t_clip,
            qpos=q_src,
            base_pose=bp_src,
            source=SOURCE_NAME,
            source_type="sim_rollout",
            source_url=SOURCE_URL,
            source_citation=SOURCE_CITATION,
            source_license=SOURCE_LICENSE,
            notes=(
                f"LAFAN1 motion '{motion}' retargeted to Unitree G1 via "
                f"IK+interaction-mesh optimization (not real hardware). "
                f"Clip {clip_idx+1}. Split: {split}. "
                f"Base pose is included — the robot translates through world "
                f"frame as it walks/moves."
            ),
            overwrite=True,
        )

        manifest_entries.append({
            "id": clip_id,
            "path": f"g1/trajectories/{clip_id}.h5",
            "duration_s": float(t_clip[-1] - t_clip[0]),
            "frequency_hz": LAFAN1_FPS,
            "n_timesteps": int(len(t_clip)),
            "source": SOURCE_NAME,
            "source_type": "sim_rollout",
            "source_license": SOURCE_LICENSE,
            "signals_present": ["qpos", "base_pose"],
            "task": motion,
            "split": split,
        })
        print(f"  wrote {clip_id}.h5 ({split})")

    # Merge into manifest fragment
    fragment_path = os.path.join(bundle_dir, "g1", "_manifest_fragment.json")
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
    print(f"\nTotal G1 trajectories in fragment: {len(existing)}")


def _update_metadata(bundle_dir: str, robot_subdir: str, joint_names: List[str]):
    """Update metadata.json with the LAFAN1 canonical joint list."""
    meta_path = os.path.join(bundle_dir, robot_subdir, "metadata.json")
    if not os.path.exists(meta_path):
        print(f"  warning: {meta_path} not found; skipping metadata update. "
              f"Run convert-urdf first.")
        return

    with open(meta_path) as f:
        metadata = json.load(f)

    old_joints = metadata.get("joint_names", [])
    if old_joints == joint_names:
        return

    n = len(joint_names)
    metadata["joint_names"] = list(joint_names)
    metadata["n_dof_actuated"] = n
    metadata["n_dof_total"] = n + 6 if metadata.get("floating_base") else n

    note = f" [joints updated to LAFAN1 canonical 29-joint G1 config]"
    if note not in metadata.get("notes", ""):
        metadata["notes"] = metadata.get("notes", "") + note

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  updated {meta_path}: {n} joints")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LAFAN1 retargeted G1 motion ingestion"
    )
    parser.add_argument("--bundle-dir", required=True)

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List available motion names in the HF repo")

    p_ingest = sub.add_parser("ingest", help="Ingest selected motions")
    p_ingest.add_argument(
        "--motions", nargs="+", default=None,
        help="Motion names (without .csv), e.g. walk1_subject1 run1_subject2. "
             "If omitted, all motions are ingested.",
    )
    p_ingest.add_argument(
        "--max-clips-per-motion", type=int, default=None,
        help="Cap the number of clips per motion (for quick tests)",
    )
    p_ingest.add_argument("--cache-dir", default=None)

    args = parser.parse_args()

    if args.command == "list":
        names = list_motions()
        print(f"{len(names)} G1 motions available in {LAFAN1_HF_REPO}:\n")
        for n in names:
            print(f"  {n}")
    elif args.command == "ingest":
        motions = args.motions
        if motions is None:
            print("No --motions given, discovering all available...")
            motions = list_motions()
            print(f"  found {len(motions)}")
        ingest(
            args.bundle_dir,
            motions,
            args.max_clips_per_motion,
            args.cache_dir,
        )


if __name__ == "__main__":
    main()
