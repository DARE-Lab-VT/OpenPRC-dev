"""
Panda (Franka) ingestion via DROID in LeRobot format.

Source: IPEC-COMMUNITY/droid_lerobot — the full DROID teleop dataset
re-published as LeRobot v2 parquet (92k episodes, 27M frames, 15 Hz, real
Franka Panda teleop across 13 institutions).

Why this source instead of raw DROID:
  - Parquet-per-episode means we download exactly what we need (2 MB per
    episode, not TB-scale).
  - Same parse pattern as our LAFAN1/WBT scripts — no TensorFlow dep.
  - 15 Hz is real Franka control-rate data (teleop), not the downsampled
    3-5 Hz of many LeRobot arm datasets.

Schema (from meta/info.json at IPEC-COMMUNITY/droid_lerobot):
  observation.state: (8,) float32 — 7 arm joint positions + 1 gripper
  action:            (7,) float32 — 6 EEF delta + 1 gripper (NOT joint-space!)
  fps: 15
  episodes: 92233 total across 31k tasks

IMPORTANT: The action is end-effector delta, not joint target. We DO NOT
write action/target_qpos for this robot — doing so would be mislabeling. If
you need commanded joints, they'd have to be derived from the EEF action
and the gripper state, which requires an IK pass at ingest time (not worth
doing for a first pass).

Usage:
    python fetch_panda.py --bundle-dir . convert-urdf \\
        --menagerie-dir /path/to/mujoco_menagerie

    # Ingest ~100 curated episodes (about 2 MB of downloads)
    python fetch_panda.py --bundle-dir . ingest \\
        --target-episodes 100

The curation is stride-based over all 92k episodes for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from write_trajectory import write_trajectory  # noqa: E402
from panda_transforms import (  # noqa: E402
    PANDA_URDF_JOINT_NAMES,
    PANDA_GRIPPER_JOINT_NAMES,
    expand_gripper_position,
    select_subset_by_stride,
)
from g1_transforms import times_from_frame_indices  # noqa: E402


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

DROID_HF_REPO = "IPEC-COMMUNITY/droid_lerobot"
DROID_FPS = 15.0
DROID_TOTAL_EPISODES = 92233  # from info.json as of April 2026
DROID_EPISODES_PER_CHUNK = 1000

SOURCE_NAME = "droid"
SOURCE_URL = "https://huggingface.co/datasets/IPEC-COMMUNITY/droid_lerobot"
SOURCE_CITATION = (
    "Khazatsky, A., et al. (2024). DROID: A Large-Scale In-the-Wild Robot "
    "Manipulation Dataset. RSS 2024. Distributed in LeRobot format via "
    "https://huggingface.co/datasets/IPEC-COMMUNITY/droid_lerobot"
)
SOURCE_LICENSE = "MIT"  # DROID is MIT; verify community re-host terms before publication


# -----------------------------------------------------------------------------
# HF helpers
# -----------------------------------------------------------------------------

def _hf_download(path: str, cache_dir: Optional[str] = None) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id=DROID_HF_REPO, filename=path, repo_type="dataset",
        cache_dir=cache_dir,
    )


def _episode_parquet_path(episode_idx: int) -> str:
    """
    DROID_lerobot uses one parquet per episode, under chunked directories.
    Path template (from info.json): data/chunk-{chunk:03d}/episode_{ep:06d}.parquet
    """
    chunk = episode_idx // DROID_EPISODES_PER_CHUNK
    return f"data/chunk-{chunk:03d}/episode_{episode_idx:06d}.parquet"


def load_episode(episode_idx: int, cache_dir: Optional[str]):
    """Load one episode parquet and return a pandas DataFrame."""
    import pyarrow.parquet as pq
    local = _hf_download(_episode_parquet_path(episode_idx), cache_dir)
    return pq.read_table(local).to_pandas()


# -----------------------------------------------------------------------------
# Main ingest
# -----------------------------------------------------------------------------

def ingest(
    bundle_dir: str,
    target_episodes: int,
    cache_dir: Optional[str],
):
    """Pull curated subset of DROID episodes and write trajectory HDF5s."""
    print(f"DROID direct parquet ingest from {DROID_HF_REPO}")
    print(f"  target: {target_episodes} episodes sampled across the full "
          f"{DROID_TOTAL_EPISODES}-episode dataset via stride")

    selected = select_subset_by_stride(DROID_TOTAL_EPISODES, target_episodes)
    print(f"  selected episode indices (first 5): "
          f"{list(selected[:5])}...{list(selected[-3:])}")

    out_dir = os.path.join(bundle_dir, "panda", "trajectories")
    os.makedirs(out_dir, exist_ok=True)

    # Joint name list: arm only (Menagerie's Panda URDF doesn't expose fingers)
    joint_names = list(PANDA_URDF_JOINT_NAMES)
    n_joints = len(joint_names)  # 7

    # Deterministic train/test split: last 15% of selected episodes → test
    from g1_transforms import split_episode_train_test
    train_ranks, test_ranks = split_episode_train_test(len(selected))

    manifest_entries: List[dict] = []
    n_written = 0
    n_failed = 0

    for rank, ep_idx in enumerate(selected):
        try:
            df = load_episode(int(ep_idx), cache_dir)
        except Exception as e:
            print(f"  episode {ep_idx}: failed to download/read — {e}")
            n_failed += 1
            continue

        if "observation.state" not in df.columns:
            print(f"  episode {ep_idx}: unexpected columns {list(df.columns)[:5]}...")
            n_failed += 1
            continue

        # observation.state is (T, 8): cols 0..6 = arm joint positions,
        # col 7 = gripper position (scalar in [0, 1]). We keep only the arm.
        state = np.array(df["observation.state"].tolist(), dtype=np.float32)
        T = state.shape[0]
        if T < 2:
            n_failed += 1
            continue

        qpos = state[:, :7]  # arm joints only, matches URDF
        assert qpos.shape == (T, n_joints), f"qpos shape {qpos.shape} != ({T}, {n_joints})"

        t_clip = times_from_frame_indices(T, DROID_FPS)
        duration_s = float(t_clip[-1] - t_clip[0])

        split = "test" if rank in test_ranks else "train"
        clip_id = f"droid_ep{int(ep_idx):06d}"
        out_path = os.path.join(out_dir, f"{clip_id}.h5")

        write_trajectory(
            out_path=out_path,
            robot_name="panda",
            trajectory_id=clip_id,
            joint_names=joint_names,
            time=t_clip,
            qpos=qpos,
            # Action is EEF delta, not joint target — skip to avoid mislabeling.
            source=SOURCE_NAME,
            source_type="teleop_real",
            source_url=SOURCE_URL,
            source_citation=SOURCE_CITATION,
            source_license=SOURCE_LICENSE,
            notes=(
                f"DROID episode {int(ep_idx)} of {DROID_TOTAL_EPISODES}. "
                f"Split: {split}. 7 arm joint positions + gripper. "
                f"Action (EEF delta) not stored — not joint-space."
            ),
            overwrite=True,
        )

        manifest_entries.append({
            "id": clip_id,
            "path": f"panda/trajectories/{clip_id}.h5",
            "duration_s": duration_s,
            "frequency_hz": DROID_FPS,
            "n_timesteps": int(T),
            "source": SOURCE_NAME,
            "source_type": "teleop_real",
            "source_license": SOURCE_LICENSE,
            "signals_present": ["qpos"],
            "task": "droid_manipulation",
            "split": split,
        })
        n_written += 1
        if n_written % 10 == 0:
            print(f"  wrote {n_written} / {len(selected)} episodes "
                  f"({n_failed} failed)")

    print(f"\nDone: {n_written} episodes written, {n_failed} failed")

    # Merge into manifest fragment
    fragment_path = os.path.join(bundle_dir, "panda", "_manifest_fragment.json")
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
    print(f"Total Panda trajectories: {len(existing)}")


# -----------------------------------------------------------------------------
# URDF conversion (identical to original fetch_panda.py)
# -----------------------------------------------------------------------------

def cmd_convert_urdf(bundle_dir: str, menagerie_dir: str):
    sys.path.insert(0, os.path.dirname(HERE))
    from mjcf_to_urdf import convert_mjcf_to_urdf

    mjcf_path = os.path.join(menagerie_dir, "franka_emika_panda", "panda.xml")
    if not os.path.exists(mjcf_path):
        raise FileNotFoundError(f"MJCF not found: {mjcf_path}")

    urdf_out = os.path.join(bundle_dir, "panda", "urdf", "panda.urdf")
    mesh_dir = os.path.join(bundle_dir, "panda", "urdf", "meshes")

    print(f"Converting {mjcf_path} → {urdf_out}")
    convert_mjcf_to_urdf(mjcf_path, urdf_out, mesh_dir=mesh_dir)

    joint_names = list(PANDA_URDF_JOINT_NAMES)  # arm only; no gripper in Menagerie URDF
    metadata = {
        "schema_version": "0.1.0",
        "robot_name": "panda",
        "display_name": "Franka Emika Panda",
        "urdf_path": "urdf/panda.urdf",
        "mesh_dir": "urdf/meshes",
        "floating_base": False,
        "n_dof_actuated": len(joint_names),
        "n_dof_total": len(joint_names),
        "joint_names": joint_names,
        "urdf_source": {
            "origin": "mujoco_menagerie",
            "origin_url": "https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda",
            "conversion_tool": "mjcf_urdf_simple_converter",
            "license": "Apache-2.0",
        },
        "notes": (
            "Fixed-base 7-DoF arm. No gripper in URDF (Menagerie MJCF models "
            "the Panda hand with tendons which don't round-trip to URDF). "
            "DROID state column 7 (gripper scalar) is discarded at ingest."
        ),
    }
    with open(os.path.join(bundle_dir, "panda", "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print("Wrote panda/metadata.json")


def main():
    parser = argparse.ArgumentParser(description="Panda (DROID LeRobot) ingestion")
    parser.add_argument("--bundle-dir", required=True)

    sub = parser.add_subparsers(dest="command", required=True)

    p_convert = sub.add_parser("convert-urdf")
    p_convert.add_argument("--menagerie-dir", required=True)

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("--target-episodes", type=int, default=100,
                           help="How many episodes to sample from DROID (default: 100)")
    p_ingest.add_argument("--cache-dir", default=None)

    args = parser.parse_args()

    if args.command == "convert-urdf":
        cmd_convert_urdf(args.bundle_dir, args.menagerie_dir)
    else:
        ingest(args.bundle_dir, args.target_episodes, args.cache_dir)


if __name__ == "__main__":
    main()
