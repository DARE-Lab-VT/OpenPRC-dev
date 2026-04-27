"""
G1 Dex3 ingestion (upper-body + dexterous hands).

Source: Unitree's G1_Dex3_* task datasets on HuggingFace (13 available, e.g.
ToastedBread, BlockStacking, PickApple). Each is LeRobot v2 parquet format with:
  - observation.state: (T, 28) — 14 arm joints + 14 hand finger joints
  - action:            (T, 28) — same layout, commanded positions
  - 30 fps, ~300-500 episodes per dataset

Schema pattern:
  - Joint names are semantic (kLeftShoulderPitch, kLeftHandThumb0, etc.)
  - We translate them to URDF names using the same map as fetch_g1.py
  - Finger names follow kCamelCase → snake_case heuristic

URDF source:
  - Menagerie's g1_with_hands.xml (full-body + three-finger hands, ~43 joints)
  - Dataset only covers 28 upper-body+hand joints; legs/waist stay at URDF
    neutral during animation. This looks like the G1 standing still at the
    hips while doing a tabletop manipulation task — visually correct for the
    recorded tasks which are all stationary tabletop work.

Separate bundle subdir: `g1_dex3/` — keeps it distinct from `g1/` (which is
already populated with LAFAN1 whole-body mocap). Two different URDFs, two
different joint sets, so they deserve to be separate robots in the bundle.

Usage:
    # One-time URDF conversion
    python fetch_g1_dex3.py --bundle-dir . convert-urdf \\
        --menagerie-dir /tmp/menagerie

    # Ingest a task dataset (default: ToastedBread)
    python fetch_g1_dex3.py --bundle-dir . ingest \\
        --hf-repo unitreerobotics/G1_Dex3_ToastedBread_Dataset \\
        --max-episodes 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from typing import List, Optional

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
# Reuse the proven translation and metadata helpers from fetch_g1.py
from openprc.automod._tools.fetch.fetch_g1 import (  # noqa: E402
    _translate_joint_name,
    _flatten_names,
    _update_metadata_with_discovered_joints,
)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

ROBOT_SUBDIR = "g1_dex3"

SOURCE_URL_BASE = "https://huggingface.co/datasets"
SOURCE_LICENSE = "Apache-2.0"  # typical for unitreerobotics org; verify
CLIP_DURATION_S = 30.0

AVAILABLE_DEX3_DATASETS = [
    "unitreerobotics/G1_Dex3_BlockStacking_Dataset",
    "unitreerobotics/G1_Dex3_CameraPackaging_Dataset",
    "unitreerobotics/G1_Dex3_GraspSquare_Dataset",
    "unitreerobotics/G1_Dex3_ObjectPlacement_Dataset",
    "unitreerobotics/G1_Dex3_PickApple_Dataset",
    "unitreerobotics/G1_Dex3_PickBottle_Dataset",
    "unitreerobotics/G1_Dex3_PickCharger_Dataset",
    "unitreerobotics/G1_Dex3_PickDoll_Dataset",
    "unitreerobotics/G1_Dex3_PickGum_Dataset",
    "unitreerobotics/G1_Dex3_PickSnack_Dataset",
    "unitreerobotics/G1_Dex3_PickTissue_Dataset",
    "unitreerobotics/G1_Dex3_Pouring_Dataset",
    "unitreerobotics/G1_Dex3_ToastedBread_Dataset",
]


# -----------------------------------------------------------------------------
# HF helpers
# -----------------------------------------------------------------------------

def _hf_download(repo: str, path: str, cache_dir: Optional[str] = None) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id=repo, filename=path, repo_type="dataset", cache_dir=cache_dir,
    )


def _list_repo_files(hf_repo: str) -> List[str]:
    """List all files in an HF dataset repo. Cached by the caller."""
    from huggingface_hub import HfApi
    return HfApi().list_repo_files(hf_repo, repo_type="dataset")


def load_episode_table(hf_repo: str, cache_dir: Optional[str]):
    """
    Load meta/episodes/ parquet (v3 format) which provides per-episode
    (dataset_from_index, dataset_to_index) boundaries into the consolidated
    data shards.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    files = _list_repo_files(hf_repo)
    ep_files = sorted(f for f in files
                      if f.startswith("meta/episodes") and f.endswith(".parquet"))
    if not ep_files:
        raise RuntimeError(
            f"No meta/episodes/*.parquet in {hf_repo}. v2 layout? "
            f"Repo has meta files: {[f for f in files if f.startswith('meta/')]}"
        )
    tables = []
    for f in ep_files:
        local = _hf_download(hf_repo, f, cache_dir)
        tables.append(pq.read_table(local))
    combined = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    return combined.to_pandas()


def load_data_shards(hf_repo: str, cache_dir: Optional[str]):
    """
    Download and concatenate all data/*.parquet shards into one DataFrame.
    v3 format packs many episodes per shard; episode boundaries come from
    the episode table.
    """
    import pyarrow.parquet as pq
    import pandas as pd
    files = _list_repo_files(hf_repo)
    shards = sorted(f for f in files
                    if f.startswith("data/") and f.endswith(".parquet"))
    if not shards:
        raise RuntimeError(
            f"No data/*.parquet shards in {hf_repo}. "
            f"Repo file roots: {sorted({f.split('/')[0] for f in files})}"
        )
    print(f"  {len(shards)} data shard(s) to fetch")
    dfs = []
    for i, f in enumerate(shards):
        print(f"    shard {i + 1}/{len(shards)}: {f}")
        local = _hf_download(hf_repo, f, cache_dir)
        dfs.append(pq.read_table(local).to_pandas())
    return pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]


def load_info(hf_repo: str, cache_dir: Optional[str]) -> dict:
    local = _hf_download(hf_repo, "meta/info.json", cache_dir)
    with open(local) as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# URDF actuated-joint extraction (for name alignment)
# -----------------------------------------------------------------------------

def read_urdf_actuated_joints(urdf_path: str) -> List[str]:
    """Return actuated joint names in URDF declaration order."""
    tree = ET.parse(urdf_path)
    return [
        j.get("name") for j in tree.getroot().findall("joint")
        if j.get("type", "fixed") != "fixed" and j.get("name")
    ]


# -----------------------------------------------------------------------------
# Main ingest
# -----------------------------------------------------------------------------

def ingest(
    bundle_dir: str,
    hf_repo: str,
    max_episodes: Optional[int],
    cache_dir: Optional[str],
):
    print(f"Dex3 direct parquet ingest from {hf_repo}")

    # Load dataset metadata
    info = load_info(hf_repo, cache_dir)
    fps = float(info.get("fps", 30.0))
    total_ep = int(info.get("total_episodes", 0))
    total_frames = int(info.get("total_frames", 0))
    chunks_size = int(info.get("chunks_size", 1000))
    print(f"  fps={fps}, total_episodes={total_ep}, total_frames={total_frames}")

    # Feature schema
    features = info.get("features", {})
    if "observation.state" not in features:
        raise RuntimeError(
            f"Dataset missing observation.state. Features: {list(features.keys())}"
        )
    state_feat = features["observation.state"]
    source_names = _flatten_names(state_feat.get("names"))
    if source_names is None:
        raise RuntimeError(
            f"observation.state has no names. state_feat: {state_feat}"
        )
    print(f"  state shape: {state_feat.get('shape')}")
    print(f"  source joint names ({len(source_names)}): {source_names[:6]}...")

    has_action = "action" in features

    # Translate Unitree SDK names → URDF names
    source_urdf_names = [_translate_joint_name(n) for n in source_names]

    # Load URDF and find which of our translated names exist in it.
    # This is the intersection — the 28 state joints must all appear in the
    # URDF's actuated joints. If any don't, we fail early with a clear message.
    meta_path = os.path.join(bundle_dir, ROBOT_SUBDIR, "metadata.json")
    if not os.path.exists(meta_path):
        raise RuntimeError(
            f"{meta_path} not found. Run convert-urdf first."
        )
    with open(meta_path) as f:
        robot_meta = json.load(f)
    urdf_path = os.path.join(bundle_dir, ROBOT_SUBDIR, robot_meta["urdf_path"])
    urdf_joints = read_urdf_actuated_joints(urdf_path)

    missing_from_urdf = [n for n in source_urdf_names if n not in urdf_joints]
    if missing_from_urdf:
        raise RuntimeError(
            f"{len(missing_from_urdf)} translated joint names are not in the URDF "
            f"({len(urdf_joints)} actuated joints there):\n"
            f"  missing: {missing_from_urdf[:10]}{'...' if len(missing_from_urdf) > 10 else ''}\n"
            f"  URDF has (first 15): {urdf_joints[:15]}\n"
            f"Fix: extend UNITREE_TO_URDF_NAME_MAP in fetch_g1.py with the "
            f"missing mappings, or verify the URDF was built from the right MJCF."
        )

    # Target joint list for trajectories: the 28 Dex3 joints in source order,
    # translated to URDF names. These are a subset of urdf_joints.
    target_joint_names = source_urdf_names
    remap = build_joint_remap(source_urdf_names, target_joint_names)
    # remap is identity here (target == source-translated), but we keep it for
    # uniformity and in case future datasets have different source orderings.
    print(f"  target joint names ({len(target_joint_names)}): "
          f"{target_joint_names[:5]}... (all present in URDF)")

    # Update metadata.json with the 28-joint subset — this is what the
    # trajectory HDF5s will use, and the validator checks consistency.
    _update_metadata_with_discovered_joints(
        bundle_dir, ROBOT_SUBDIR, target_joint_names
    )

    # Select episodes
    n_ep = total_ep if max_episodes is None else min(total_ep, max_episodes)
    train_ranks, test_ranks = split_episode_train_test(n_ep)

    task_tag = hf_repo.split("/")[-1].replace("_Dataset", "").lower()
    out_dir = os.path.join(bundle_dir, ROBOT_SUBDIR, "trajectories")
    os.makedirs(out_dir, exist_ok=True)

    # Load episode boundaries and all data shards (v3 consolidated format)
    print("  loading episode boundary table...")
    episodes_df = load_episode_table(hf_repo, cache_dir)
    print(f"  {len(episodes_df)} episode entries, "
          f"columns: {list(episodes_df.columns)[:8]}")
    from_col = next((c for c in ("dataset_from_index", "from_index")
                     if c in episodes_df.columns), None)
    to_col = next((c for c in ("dataset_to_index", "to_index")
                   if c in episodes_df.columns), None)
    if from_col is None or to_col is None:
        raise RuntimeError(
            f"Episode table missing expected boundary columns. "
            f"Got: {list(episodes_df.columns)}"
        )

    print("  downloading data shards (this is most of the download time)...")
    data = load_data_shards(hf_repo, cache_dir)
    print(f"  combined shards: {len(data)} rows, "
          f"columns: {list(data.columns)[:8]}")

    if "observation.state" not in data.columns:
        raise RuntimeError(
            f"data shards missing observation.state column. "
            f"Got: {list(data.columns)}"
        )

    manifest_entries: List[dict] = []
    n_written = 0
    n_failed = 0

    for rank in range(n_ep):
        row = episodes_df.iloc[rank]
        from_idx = int(row[from_col])
        to_idx = int(row[to_col])
        T = to_idx - from_idx
        if T < 2:
            n_failed += 1
            continue

        ep_data = data.iloc[from_idx:to_idx]
        state = np.array(ep_data["observation.state"].tolist(), dtype=np.float32)
        if state.shape[0] != T:
            print(f"  ep {rank}: shape mismatch "
                  f"(slice {T} frames but state has {state.shape[0]}), skipping")
            n_failed += 1
            continue

        # Apply remap (identity here, but symbolic for future)
        qpos = state[:, remap]

        target_qpos = None
        if has_action and "action" in ep_data.columns:
            act = np.array(ep_data["action"].tolist(), dtype=np.float32)
            if act.shape == state.shape:
                target_qpos = act[:, remap]

        # Slice if long
        clips = slice_long_episode(T, CLIP_DURATION_S, fps)
        split = "test" if rank in test_ranks else "train"

        for clip_idx, (s, e) in enumerate(clips):
            clip_id = f"dex3_{task_tag}_ep{rank:04d}_clip{clip_idx:02d}"
            out_path = os.path.join(out_dir, f"{clip_id}.h5")
            t_clip = times_from_frame_indices(e - s, fps)

            write_trajectory(
                out_path=out_path,
                robot_name=ROBOT_SUBDIR,
                trajectory_id=clip_id,
                joint_names=target_joint_names,
                time=t_clip,
                qpos=qpos[s:e],
                target_qpos=target_qpos[s:e] if target_qpos is not None else None,
                source=f"unitree_hf_{task_tag}",
                source_type="teleop_real",
                source_url=f"{SOURCE_URL_BASE}/{hf_repo}",
                source_citation=(
                    f"Unitree Robotics (2025). G1 Dex3 {task_tag} Dataset. "
                    f"Hugging Face: {hf_repo}"
                ),
                source_license=SOURCE_LICENSE,
                notes=(
                    f"Task: {task_tag}, episode {rank}, clip {clip_idx+1}/{len(clips)}. "
                    f"Split: {split}. Upper-body + Dex3 hands (28 DoF). "
                    f"Legs and waist stay at URDF neutral — not recorded "
                    f"because Dex3 tasks are stationary tabletop work."
                ),
                overwrite=True,
            )

            signals = ["qpos"]
            if target_qpos is not None:
                signals.append("action/target_qpos")

            manifest_entries.append({
                "id": clip_id,
                "path": f"{ROBOT_SUBDIR}/trajectories/{clip_id}.h5",
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
            n_written += 1

        if (rank + 1) % 10 == 0:
            print(f"  processed {rank + 1}/{n_ep} episodes ({n_written} clips, "
                  f"{n_failed} failed)")

    print(f"\nDone: {n_written} clips, {n_failed} episodes failed")

    # Merge manifest fragment
    fragment_path = os.path.join(bundle_dir, ROBOT_SUBDIR, "_manifest_fragment.json")
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
    print(f"Total {ROBOT_SUBDIR} trajectories: {len(existing)}")


# -----------------------------------------------------------------------------
# URDF conversion
# -----------------------------------------------------------------------------

def cmd_convert_urdf(bundle_dir: str, menagerie_dir: str):
    sys.path.insert(0, os.path.dirname(HERE))
    from mjcf_to_urdf import convert_mjcf_to_urdf

    # Use the with-hands variant — it has the three-finger Dex3-compatible
    # geometry attached to the G1's wrist ends.
    mjcf_path = os.path.join(menagerie_dir, "unitree_g1", "g1_with_hands.xml")
    if not os.path.exists(mjcf_path):
        raise FileNotFoundError(
            f"MJCF not found: {mjcf_path}. Expected g1_with_hands.xml in "
            f"Menagerie's unitree_g1 directory."
        )

    urdf_out = os.path.join(bundle_dir, ROBOT_SUBDIR, "urdf", "g1_with_hands.urdf")
    mesh_dir = os.path.join(bundle_dir, ROBOT_SUBDIR, "urdf", "meshes")

    print(f"Converting {mjcf_path} → {urdf_out}")
    convert_mjcf_to_urdf(mjcf_path, urdf_out, mesh_dir=mesh_dir)

    urdf_joints = read_urdf_actuated_joints(urdf_out)
    print(f"URDF has {len(urdf_joints)} actuated joints")

    # Placeholder metadata — will be updated at ingest time with the actual
    # 28-joint subset the dataset uses.
    metadata = {
        "schema_version": "0.1.0",
        "robot_name": ROBOT_SUBDIR,
        "display_name": "Unitree G1 + Dex3 hands",
        "urdf_path": "urdf/g1_with_hands.urdf",
        "mesh_dir": "urdf/meshes",
        "floating_base": True,
        "n_dof_actuated": len(urdf_joints),  # full URDF count; will be
        "n_dof_total": len(urdf_joints) + 6,  # narrowed at ingest time
        "joint_names": urdf_joints,           # placeholder
        "urdf_source": {
            "origin": "mujoco_menagerie",
            "origin_url": "https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1",
            "mjcf_file": "g1_with_hands.xml",
            "conversion_tool": "mjcf_urdf_simple_converter",
            "license": "BSD-3-Clause",
        },
        "notes": (
            "Full G1 with three-finger Dex3 hands attached. Trajectory data "
            "covers only the 28 upper-body+hand joints (see joint_names after "
            "first ingest). Legs and waist remain at URDF default pose in "
            "animations since Dex3 datasets record stationary tabletop tasks."
        ),
    }
    with open(os.path.join(bundle_dir, ROBOT_SUBDIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote {ROBOT_SUBDIR}/metadata.json (will be updated by ingest)")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="G1 Dex3 ingestion")
    parser.add_argument("--bundle-dir", required=True)

    sub = parser.add_subparsers(dest="command", required=True)

    p_conv = sub.add_parser("convert-urdf")
    p_conv.add_argument("--menagerie-dir", required=True)

    p_ing = sub.add_parser("ingest")
    p_ing.add_argument(
        "--hf-repo",
        default="unitreerobotics/G1_Dex3_ToastedBread_Dataset",
        help=f"HF dataset repo id. Available datasets:\n" +
             "\n".join(f"  {r}" for r in AVAILABLE_DEX3_DATASETS),
    )
    p_ing.add_argument("--max-episodes", type=int, default=None)
    p_ing.add_argument("--cache-dir", default=None)

    sub.add_parser("list", help="List available Dex3 datasets on HF")

    args = parser.parse_args()

    if args.command == "convert-urdf":
        cmd_convert_urdf(args.bundle_dir, args.menagerie_dir)
    elif args.command == "ingest":
        ingest(args.bundle_dir, args.hf_repo, args.max_episodes, args.cache_dir)
    elif args.command == "list":
        print(f"{len(AVAILABLE_DEX3_DATASETS)} Dex3 datasets available:")
        for r in AVAILABLE_DEX3_DATASETS:
            print(f"  {r}")


if __name__ == "__main__":
    main()
