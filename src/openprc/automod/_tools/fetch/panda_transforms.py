"""
Pure-function transforms for Franka Panda / DROID data ingestion.

DROID logs joint_position at 15 Hz across 76k teleop episodes from 13 institutions.
The Panda URDF has 7 actuated arm joints + 2 finger joints (gripper). DROID
logs both.

Since the Panda is a fixed-base arm, there is no floating-base handling — no
/base_pose is written. `joint_names` matches the Franka standard:
  panda_joint1 ... panda_joint7, panda_finger_joint1, panda_finger_joint2
"""

from __future__ import annotations

from typing import List

import numpy as np


# Canonical URDF joint names for the Franka Emika Panda.
# Matches Menagerie's franka_emika_panda/panda.xml, which uses unprefixed
# `joint1..joint7` (not the `panda_joint1..joint7` style from the upstream
# Franka ROS package). Menagerie also does not expose finger joints as
# URDF-actuated — the gripper is modeled with tendons in MJCF and those
# don't round-trip to URDF, so we drop the gripper from /qpos.
PANDA_URDF_JOINT_NAMES: List[str] = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]

# Retained for scripts that want to reconstruct gripper state separately,
# though the current ingest path just skips the gripper.
PANDA_GRIPPER_JOINT_NAMES: List[str] = []


def expand_gripper_position(
    gripper: np.ndarray, max_opening_m: float = 0.04,
) -> np.ndarray:
    """
    DROID's `gripper_position` is a scalar in [0, 1] (0 = closed, 1 = fully open).
    The Panda URDF has two prismatic finger joints, each mirroring the other.
    Convert scalar gripper to (T, 2) per-finger positions in meters.

    `max_opening_m` is the per-finger travel; 4 cm is the standard Panda hand.
    """
    if gripper.ndim == 2 and gripper.shape[1] == 1:
        gripper = gripper[:, 0]
    g = np.clip(gripper.astype(np.float32), 0.0, 1.0)
    per_finger = g * max_opening_m  # each finger opens by this amount
    return np.stack([per_finger, per_finger], axis=1)


def select_subset_by_stride(
    n_episodes: int,
    target_count: int,
    seed: int = 0,
) -> np.ndarray:
    """
    For a dataset with 76k episodes, we want to curate ~100. A simple and
    reproducible scheme: sort episodes by index and take every Nth one.
    """
    if target_count >= n_episodes:
        return np.arange(n_episodes)
    stride = n_episodes / target_count
    idx = (np.arange(target_count) * stride).astype(np.int64)
    return idx
