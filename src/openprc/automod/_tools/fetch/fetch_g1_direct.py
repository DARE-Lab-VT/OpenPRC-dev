"""
Direct-parquet G1 ingestion.

Bypasses LeRobot's loader entirely — talks straight to the parquet files on
HuggingFace Hub. This works for both flat-layout and nested-subdirectory
LeRobot datasets (the UnifoLM-WBT collection puts everything under a named
subdirectory, which LeRobot's loader doesn't expect).

What we download:
  <prefix>/meta/info.json          — feature schema, joint names, fps
  <prefix>/meta/episodes/…         — episode boundary parquet
  <prefix>/data/chunk-000/*.parquet — actual trajectory data

We skip:
  videos/*.mp4 — we don't use the camera streams

Usage:
    python fetch_g1_direct.py --bundle-dir . --mode wbt \\
        --hf-repo unitreerobotics/G1_WBT_Inspire_Pickup_Pillow_MainCamOnly \\
        --max-episodes 3

    # If the repo has a nested prefix (WBT datasets do), specify it:
    python fetch_g1_direct.py --bundle-dir . --mode wbt \\
        --hf-repo unitreerobotics/G1_WBT_Inspire_Pickup_Pillow_MainCamOnly \\
        --dataset-prefix G1_WB_Dex5_Pickup_Pillow \\
        --max-episodes 3

    # If --dataset-prefix is not given, the script auto-discovers it by listing
    # the repo and finding where meta/info.json lives.
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
from g1_transforms import (  # noqa: E402
    build_joint_remap,
    times_from_frame_indices,
    split_episode_train_test,
    slice_long_episode,
)

# Reuse the translation logic from fetch_g1.py so we don't duplicate.
from openprc.automod._tools.fetch.fetch_g1 import (  # noqa: E402
    _translate_joint_name,
    _resolve_joint_names_for_mode,
    _update_metadata_with_discovered_joints,
    _flatten_names,
    SOURCE_LICENSE,
    SOURCE_CITATION,
    CLIP_DURATION_S,
)


# -----------------------------------------------------------------------------
# HF Hub helpers
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
    """Download one file from a HF dataset repo; return the local path."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id=repo, filename=path, repo_type="dataset",
        cache_dir=cache_dir,
    )


def _discover_prefix(repo: str) -> str:
    """
    Find the subdirectory inside a HF dataset repo that contains meta/info.json.
    Returns '' (empty string) for flat-layout repos; returns e.g. 'SubDir' for
    nested ones.
    """
    api = _hf_api()
    files = api.list_repo_files(repo, repo_type="dataset")
    info_paths = [f for f in files if f.endswith("meta/info.json")]
    if not info_paths:
        raise RuntimeError(
            f"No meta/info.json found in {repo}. This doesn't look like a "
            f"LeRobot-format dataset. Files at root: "
            f"{sorted({f.split('/')[0] for f in files})[:10]}"
        )
    if len(info_paths) > 1:
        print(f"  warning: multiple meta/info.json found: {info_paths}; using first")
    # Strip the 'meta/info.json' suffix
    prefix = info_paths[0][: -len("meta/info.json")].rstrip("/")
    if prefix:
        print(f"  discovered nested dataset prefix: {prefix!r}")
    return prefix


def _paths(prefix: str, *parts: str) -> str:
    """Join HF-repo-relative path parts respecting an optional prefix."""
    p = "/".join(parts)
    if prefix:
        p = f"{prefix}/{p}"
    return p


def _read_urdf_actuated_joints(bundle_dir: str, robot_subdir: str) -> List[str]:
    """
    Parse the robot's URDF and return the list of actuated (non-fixed) joint
    names in declaration order. Used when the LeRobot dataset doesn't name
    its joints and we need to align state columns to URDF joints by position.
    """
    import xml.etree.ElementTree as ET
    # Read metadata to find the URDF path
    meta_path = os.path.join(bundle_dir, robot_subdir, "metadata.json")
    if not os.path.exists(meta_path):
        raise RuntimeError(
            f"Expected {meta_path} but it doesn't exist. Run convert-urdf first."
        )
    with open(meta_path) as f:
        meta = json.load(f)
    urdf_path = os.path.join(bundle_dir, robot_subdir, meta["urdf_path"])
    if not os.path.exists(urdf_path):
        raise RuntimeError(f"URDF not found at {urdf_path}")

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    actuated = []
    for joint in root.findall("joint"):
        jtype = joint.get("type", "fixed")
        if jtype == "fixed":
            continue
        name = joint.get("name")
        if name:
            actuated.append(name)
    return actuated


# -----------------------------------------------------------------------------
# Parquet loading
# -----------------------------------------------------------------------------

def _load_pyarrow():
    try:
        import pyarrow.parquet as pq
        return pq
    except ImportError as e:
        raise ImportError(
            "pyarrow is required for direct parquet loading. Install with: "
            "pip install pyarrow"
        ) from e


def load_info(repo: str, prefix: str, cache_dir: Optional[str]) -> dict:
    """Download and load meta/info.json."""
    info_path = _hf_download(repo, _paths(prefix, "meta", "info.json"), cache_dir)
    with open(info_path) as f:
        return json.load(f)


def load_episode_table(repo: str, prefix: str, cache_dir: Optional[str]):
    """
    Load meta/episodes/... which gives per-episode (from_idx, to_idx) boundaries.

    LeRobot v3 stores this as one-or-more parquet files under meta/episodes/.
    Each row has episode_index, dataset_from_index, dataset_to_index,
    and other episode-level metadata.
    """
    pq = _load_pyarrow()
    api = _hf_api()
    files = api.list_repo_files(repo, repo_type="dataset")
    ep_prefix = _paths(prefix, "meta", "episodes")
    ep_files = sorted(f for f in files if f.startswith(ep_prefix) and f.endswith(".parquet"))
    if not ep_files:
        raise RuntimeError(f"No episode parquet files found under {ep_prefix}")

    tables = []
    for f in ep_files:
        local = _hf_download(repo, f, cache_dir)
        tables.append(pq.read_table(local))
    import pyarrow as pa
    combined = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    return combined.to_pandas()


def list_data_shards(repo: str, prefix: str) -> List[str]:
    """Return the list of data/ parquet shard paths in the repo."""
    api = _hf_api()
    files = api.list_repo_files(repo, repo_type="dataset")
    data_prefix = _paths(prefix, "data")
    shards = sorted(f for f in files if f.startswith(data_prefix) and f.endswith(".parquet"))
    if not shards:
        raise RuntimeError(f"No data parquet files found under {data_prefix}")
    return shards


def load_data_shard(repo: str, shard_path: str, cache_dir: Optional[str]):
    """Load one data shard as pandas DataFrame."""
    pq = _load_pyarrow()
    local = _hf_download(repo, shard_path, cache_dir)
    return pq.read_table(local).to_pandas()


# -----------------------------------------------------------------------------
# Main ingest
# -----------------------------------------------------------------------------

def ingest(
    bundle_dir: str,
    hf_repo: str,
    mode: str,
    max_episodes: Optional[int],
    robot_subdir: str,
    dataset_prefix: Optional[str],
    cache_dir: Optional[str],
):
    print(f"Direct parquet ingest from {hf_repo}")

    if dataset_prefix is None:
        prefix = _discover_prefix(hf_repo)
    else:
        prefix = dataset_prefix.rstrip("/")
        print(f"  using explicit prefix: {prefix!r}")

    print("Downloading info.json...")
    info = load_info(hf_repo, prefix, cache_dir)
    fps = float(info.get("fps", 30.0))
    total_episodes = int(info.get("total_episodes", 0))
    total_frames = int(info.get("total_frames", 0))
    print(f"  fps={fps}, total_episodes={total_episodes}, total_frames={total_frames}")

    # Extract feature schema. Unitree's G1 WBT uses a split schema:
    #   observation.state.robot_q_current — body joint positions (what we want for WBT)
    #   observation.state.hand_state       — dexterous hand joint positions (Dex mode)
    #   observation.state.ee_state         — EE pose (not a joint signal)
    #   action.robot_q_desired             — commanded body joint positions
    #   action.hand_cmd                    — commanded hand joint positions
    # Older LeRobot uploads use a single 'observation.state' / 'action' blob.
    # Handle both.
    features = info.get("features", {})

    if mode == "wbt":
        state_candidates = ["observation.state.robot_q_current", "observation.state"]
        action_candidates = ["action.robot_q_desired", "action"]
    elif mode == "dex3":
        state_candidates = ["observation.state.hand_state", "observation.state"]
        action_candidates = ["action.hand_cmd", "action"]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    state_key = next((k for k in state_candidates if k in features), None)
    if state_key is None:
        raise RuntimeError(
            f"info.json has no state feature matching mode={mode!r}. "
            f"Tried: {state_candidates}. "
            f"Available features: {list(features.keys())}"
        )
    print(f"  using state feature: {state_key!r}")

    action_key = next((k for k in action_candidates if k in features), None)
    print(f"  using action feature: {action_key!r}" if action_key else "  no action feature")

    state_feat = features[state_key]
    source_names = _flatten_names(state_feat.get("names"))
    if source_names is None:
        raise RuntimeError(
            f"{state_key} has no 'names' field. Available: {state_feat}"
        )
    print(f"  observation.state shape: {state_feat.get('shape')}")
    print(f"  source joint names ({len(source_names)}): "
          f"{source_names[:8]}{'...' if len(source_names) > 8 else ''}")

    # Detect placeholder names like 'robot_q_current_0', 'robot_q_current_1', ...
    # which contain no joint identity (just positional indices). When that
    # happens we fall back to reading joint names from the URDF we just
    # generated, and trust that columns 0..N-1 of the state vector correspond
    # to the URDF's actuated_joints in URDF order (Unitree's documented
    # kG1_29 ordering). Anything beyond column N is dropped.
    placeholder_prefixes = ("robot_q_current_", "robot_q_desired_", "q_", "joint_")
    looks_like_placeholders = all(
        any(n.startswith(p) for p in placeholder_prefixes) for n in source_names[:3]
    )
    if looks_like_placeholders:
        print(f"  source names look like positional placeholders; "
              f"falling back to URDF joint order")
        urdf_joints = _read_urdf_actuated_joints(bundle_dir, robot_subdir)
        print(f"  URDF has {len(urdf_joints)} actuated joints: "
              f"{urdf_joints[:5]}...")
        if len(urdf_joints) > len(source_names):
            raise RuntimeError(
                f"URDF has {len(urdf_joints)} actuated joints but source data "
                f"only has {len(source_names)} columns. Cannot align."
            )
        # Override source_names: first len(urdf_joints) columns correspond to
        # URDF joints, remaining columns are discarded.
        source_names = list(urdf_joints) + [
            f"_unused_{i}" for i in range(len(urdf_joints), len(source_names))
        ]
        print(f"  using URDF joint names for first {len(urdf_joints)} columns, "
              f"discarding {len(source_names) - len(urdf_joints)} trailing columns")

    has_action = action_key is not None

    # Resolve joint name translation
    if looks_like_placeholders:
        # URDF joints are our ground truth. All actuated joints are included.
        target_joint_names = [n for n in source_names if not n.startswith("_unused_")]
        source_urdf_names = list(source_names)  # already in URDF form for real cols
    else:
        target_joint_names = _resolve_joint_names_for_mode(mode, source_names)
        source_urdf_names = [_translate_joint_name(n) for n in source_names]
    remap = build_joint_remap(source_urdf_names, target_joint_names)
    print(f"  target joint names ({len(target_joint_names)}): "
          f"{target_joint_names[:6]}...")

    # Rewrite metadata.json with actual joint list
    _update_metadata_with_discovered_joints(bundle_dir, robot_subdir, target_joint_names)

    # Episode table
    print("Downloading episode table...")
    episodes_df = load_episode_table(hf_repo, prefix, cache_dir)
    print(f"  loaded {len(episodes_df)} episode entries. "
          f"columns: {list(episodes_df.columns)[:8]}")

    n_ep = len(episodes_df)
    if max_episodes is not None:
        n_ep = min(n_ep, max_episodes)

    # Data shards
    shards = list_data_shards(hf_repo, prefix)
    print(f"  {len(shards)} data shard(s)")

    # Load all shards into a combined DataFrame. For small --max-episodes this
    # may download more than strictly needed (shards contain multiple episodes),
    # but it's simpler than tracking which episodes live in which shard.
    print("Downloading data shards...")
    dfs = []
    for i, shard in enumerate(shards):
        print(f"  shard {i+1}/{len(shards)}: {shard}")
        dfs.append(load_data_shard(hf_repo, shard, cache_dir))
    import pandas as pd
    data = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    print(f"  combined: {len(data)} rows, columns: {list(data.columns)[:8]}")

    # Episode → row mapping. Column name might be episode_index or dataset index.
    ep_col = None
    for c in ("episode_index", "dataset_from_index"):
        if c in episodes_df.columns:
            ep_col = c
            break

    # Get from_idx / to_idx per episode
    from_col = "dataset_from_index" if "dataset_from_index" in episodes_df.columns else "from_index"
    to_col = "dataset_to_index" if "dataset_to_index" in episodes_df.columns else "to_index"
    if from_col not in episodes_df.columns or to_col not in episodes_df.columns:
        raise RuntimeError(
            f"Episode table missing expected columns. Got: {list(episodes_df.columns)}. "
            f"Need from_index/to_index or dataset_from_index/dataset_to_index."
        )

    # Split train/test
    train_ids, test_ids = split_episode_train_test(n_ep)
    out_dir = os.path.join(bundle_dir, robot_subdir, "trajectories")
    os.makedirs(out_dir, exist_ok=True)

    task_tag = hf_repo.split("/")[-1].replace("_Dataset", "").replace("_MainCamOnly", "").lower()
    manifest_entries = []

    for ep_i in range(n_ep):
        row = episodes_df.iloc[ep_i]
        from_idx = int(row[from_col])
        to_idx = int(row[to_col])
        n_frames = to_idx - from_idx
        if n_frames < 2:
            continue

        split = "test" if ep_i in test_ids else "train"
        print(f"  episode {ep_i} ({n_frames} frames, split={split})")

        ep_data = data.iloc[from_idx:to_idx]
        # Column name in parquet matches the feature key (e.g.
        # observation.state.robot_q_current for WBT).
        if state_key not in ep_data.columns:
            raise RuntimeError(
                f"state_key {state_key!r} not found in parquet columns: "
                f"{list(ep_data.columns)[:12]}..."
            )
        state_raw = np.array(ep_data[state_key].tolist(), dtype=np.float32)
        qpos_full = state_raw[:, remap]

        target_qpos = None
        if has_action and action_key in ep_data.columns:
            act_raw = np.array(ep_data[action_key].tolist(), dtype=np.float32)
            if act_raw.shape[1] >= len(remap):
                target_qpos = act_raw[:, remap]

        # Slice long episodes
        clips = slice_long_episode(n_frames, CLIP_DURATION_S, fps)
        for clip_idx, (s, e) in enumerate(clips):
            clip_id = f"{task_tag}_ep{ep_i:04d}_clip{clip_idx:02d}"
            out_path = os.path.join(out_dir, f"{clip_id}.h5")
            t_clip = times_from_frame_indices(e - s, fps)

            write_trajectory(
                out_path=out_path,
                robot_name=robot_subdir,
                trajectory_id=clip_id,
                joint_names=target_joint_names,
                time=t_clip,
                qpos=qpos_full[s:e].astype(np.float32),
                target_qpos=target_qpos[s:e] if target_qpos is not None else None,
                source=f"unitree_hf_{task_tag}",
                source_type="teleop_real",
                source_url=f"https://huggingface.co/datasets/{hf_repo}",
                source_citation=SOURCE_CITATION,
                source_license=SOURCE_LICENSE,
                notes=(
                    f"Task: {task_tag}, episode {ep_i}, clip {clip_idx+1}/{len(clips)}. "
                    f"Split: {split}. Direct parquet load (bypassed LeRobot loader). "
                    f"Floating base not recorded."
                ),
                overwrite=True,
            )

            signals = ["qpos"]
            if target_qpos is not None:
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

    # Write manifest fragment
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
    print(f"\nTotal {robot_subdir} trajectories: {len(existing)}")


def main():
    parser = argparse.ArgumentParser(
        description="Direct-parquet G1 ingest (bypasses LeRobot loader)"
    )
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--mode", required=True, choices=["wbt", "dex3"])
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--dataset-prefix", default=None,
        help="Subdirectory within the HF repo containing meta/ data/ (auto-detected if omitted)",
    )
    parser.add_argument("--cache-dir", default=None,
                        help="HF cache dir (default: ~/.cache/huggingface/hub)")
    args = parser.parse_args()

    robot_subdir = "g1" if args.mode == "wbt" else "g1_dex3"
    ingest(
        args.bundle_dir,
        args.hf_repo,
        args.mode,
        args.max_episodes,
        robot_subdir,
        args.dataset_prefix,
        args.cache_dir,
    )


if __name__ == "__main__":
    main()
