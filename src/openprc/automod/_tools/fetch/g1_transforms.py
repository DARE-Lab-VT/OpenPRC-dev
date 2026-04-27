"""
Pure-function transforms for G1 data ingestion.

Unlike Go1 (where legkilo and URDF had different joint orderings), for G1
the LeRobot dataset preserves Unitree's native joint names via the `names`
field in observation.state metadata. So the main transform is:
  - Reorder the LeRobot column order → URDF joint order (mapping by name)
  - Construct a time array from frame_index + fps (LeRobot doesn't store
    absolute timestamps, just relative frame indices)

Shared by fetch_g1.py for both --mode wbt (whole-body) and --mode dex3
(dexterous manipulation) since both use the same LeRobot v2.0+ schema.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def build_joint_remap(
    source_joint_names: Sequence[str],
    target_joint_names: Sequence[str],
) -> np.ndarray:
    """
    Build an index array `remap` such that
        target_qpos[:, i] = source_qpos[:, remap[i]]

    Missing joints (target names not in source) raise ValueError — we don't
    silently zero-pad, because that would produce physically wrong animations.

    Source/target need not have the same length: if target is a subset of
    source, the remap picks out just those columns.
    """
    source_idx = {name: i for i, name in enumerate(source_joint_names)}
    remap = np.empty(len(target_joint_names), dtype=np.int64)
    missing = []
    for i, name in enumerate(target_joint_names):
        if name not in source_idx:
            missing.append(name)
        else:
            remap[i] = source_idx[name]
    if missing:
        raise ValueError(
            f"Target joint names not found in source data: {missing}. "
            f"Source has {len(source_joint_names)} joints: "
            f"{list(source_joint_names)[:20]}"
            f"{'...' if len(source_joint_names) > 20 else ''}"
        )
    return remap


def times_from_frame_indices(
    n_frames: int, fps: float, t0: float = 0.0
) -> np.ndarray:
    """
    LeRobot episodes have integer frame_index columns and a per-dataset fps
    value, but no absolute timestamps. Synthesize a time array at uniform
    intervals. This is an approximation — real capture jitter is folded into
    the fps-assumed spacing — but for 30 fps teleop it's well within the
    precision we care about for PRC training.
    """
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    dt = 1.0 / fps
    return t0 + np.arange(n_frames, dtype=np.float64) * dt


def split_episode_train_test(
    n_episodes: int, test_fraction: float = 0.15
) -> Tuple[List[int], List[int]]:
    """
    Deterministic train/test split over episode indices: the last
    ⌈test_fraction * n_episodes⌉ episodes go to test, the rest to train.

    Using a contiguous tail (rather than random) means consecutive in-context
    episodes don't leak between train and test, which matters if the collector
    recorded similar sessions back-to-back.
    """
    if n_episodes < 2:
        raise ValueError(f"need at least 2 episodes to split, got {n_episodes}")
    n_test = max(1, int(np.ceil(test_fraction * n_episodes)))
    n_test = min(n_test, n_episodes - 1)  # keep at least 1 train
    train = list(range(n_episodes - n_test))
    test = list(range(n_episodes - n_test, n_episodes))
    return train, test


def slice_long_episode(
    n_frames: int, clip_duration_s: float, fps: float
) -> List[Tuple[int, int]]:
    """
    If an episode is much longer than clip_duration_s, slice into non-overlapping
    clips of that duration. Short tail is discarded. Used when WBT episodes run
    several minutes but we want 30s clips to match Go1.

    Returns list of (start_idx, end_idx_exclusive). Returns a single full-range
    clip if the episode is shorter than 2 * clip_duration (no point slicing).
    """
    if n_frames < 2:
        return []
    frames_per_clip = int(round(clip_duration_s * fps))
    if n_frames < 2 * frames_per_clip:
        return [(0, n_frames)]
    n_clips = n_frames // frames_per_clip
    return [
        (k * frames_per_clip, (k + 1) * frames_per_clip)
        for k in range(n_clips)
    ]
