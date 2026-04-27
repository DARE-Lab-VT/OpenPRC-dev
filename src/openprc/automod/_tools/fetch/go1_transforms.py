"""
Pure-function transforms for Go1 data ingestion.

No I/O, no h5py, no rosbags — just numpy math that turns
raw legkilo-shaped arrays into bundle-canonical shapes.

Separated from fetch_go1.py so the transforms can be unit-tested
in environments that don't have h5py or rosbags installed.
"""

from __future__ import annotations

import numpy as np


# URDF joint order (matches Menagerie / upstream unitree_ros go1_description).
# This is the canonical order; /qpos columns in our HDF5 follow this order.
GO1_URDF_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]

# Legkilo motorState[] uses Unitree-SDK order:
#   [0..2]  FR   [3..5]  FL   [6..8]  RL   [9..11] RR
# Our URDF order is:
#   [0..2]  FR   [3..5]  FL   [6..8]  RR   [9..11] RL
# Remap: URDF index i ← SDK index LEGKILO_TO_URDF_IDX[i]
LEGKILO_TO_URDF_IDX = np.array([
    0, 1, 2,      # FR
    3, 4, 5,      # FL
    9, 10, 11,    # RR (from SDK slot 9..11)
    6, 7, 8,      # RL (from SDK slot 6..8)
], dtype=np.int64)

GO1_CONTACT_NAMES = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]

# Foot-force channel order in legkilo matches the SDK leg order FR,FL,RL,RR.
# We need to reorder to FR,FL,RR,RL to match our canonical contact_names.
FOOT_FORCE_LEGKILO_TO_URDF_IDX = np.array([0, 1, 3, 2], dtype=np.int64)


def remap_legkilo_joints_to_urdf(sdk_motor_array: np.ndarray) -> np.ndarray:
    """Take (T, >=12) in legkilo SDK order, return (T, 12) in URDF order."""
    if sdk_motor_array.ndim != 2:
        raise ValueError(f"expected 2D array, got {sdk_motor_array.ndim}D")
    if sdk_motor_array.shape[1] < 12:
        raise ValueError(
            f"expected at least 12 columns, got {sdk_motor_array.shape[1]}"
        )
    return sdk_motor_array[:, LEGKILO_TO_URDF_IDX]


def reorder_quaternion_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    """Unitree publishes [x, y, z, w]; our schema stores [w, x, y, z]."""
    if quat_xyzw.shape[-1] != 4:
        raise ValueError(f"expected last dim 4, got shape {quat_xyzw.shape}")
    return quat_xyzw[..., [3, 0, 1, 2]]


def remap_foot_forces_to_urdf(ff_legkilo: np.ndarray) -> np.ndarray:
    """Legkilo FR,FL,RL,RR → URDF FR,FL,RR,RL."""
    if ff_legkilo.shape[-1] != 4:
        raise ValueError(f"expected 4 feet, got shape {ff_legkilo.shape}")
    return ff_legkilo[:, FOOT_FORCE_LEGKILO_TO_URDF_IDX]


def normalize_foot_force_per_sequence(
    ff_raw: np.ndarray, epsilon: float = 1e-6
) -> tuple[np.ndarray, float]:
    """
    Normalize raw int16 ADC foot-force readings to [0, 1] per-sequence.
    Returns (normalized, scale_factor) where normalized = ff_raw / scale_factor.
    Negative raw values (lift-off artifact) are clipped to 0 before scaling.
    """
    ff_clipped = np.clip(ff_raw.astype(np.float32), 0.0, None)
    scale = float(ff_clipped.max())
    if scale < epsilon:
        return np.zeros_like(ff_clipped), 1.0
    return ff_clipped / scale, scale


def compute_contact_flags(
    ff_normalized: np.ndarray, threshold: float = 0.1
) -> np.ndarray:
    """Binary contact from normalized foot force. Threshold is in normalized units."""
    return ff_normalized > threshold


def slice_into_clips(
    time: np.ndarray, clip_duration_s: float
) -> list[tuple[int, int]]:
    """
    Return a list of (start_idx, end_idx_exclusive) covering the sequence with
    consecutive non-overlapping clips of the given duration. Tail shorter than
    clip_duration_s is discarded.
    """
    clips = []
    if len(time) < 2:
        return clips
    t0 = float(time[0])
    t_end = float(time[-1])
    total = t_end - t0
    if total < clip_duration_s:
        return clips

    # Number of whole clips that fit. Use a tiny tolerance so sequences that
    # are almost exactly N*clip_duration (off by floating point) still yield N.
    tol = 1e-9
    n_clips = int(np.floor((total + tol) / clip_duration_s))

    for k in range(n_clips):
        cs = t0 + k * clip_duration_s
        ce = cs + clip_duration_s
        start_idx = int(np.searchsorted(time, cs, side="left"))
        end_idx = int(np.searchsorted(time, ce, side="left"))
        # For the last clip, end may land past the last sample — clamp it
        if end_idx > len(time):
            end_idx = len(time)
        if end_idx - start_idx >= 2:
            clips.append((start_idx, end_idx))
    return clips


def interpolate_base_pose(
    src_t: np.ndarray,
    src_pos: np.ndarray,
    src_quat_xyzw: np.ndarray,
    tgt_t: np.ndarray,
) -> np.ndarray:
    """
    Interpolate /state_SDK onto target timestamps.
    Position: per-axis linear. Quaternion: nearest-neighbor (at 50 Hz sources,
    this is within one timestep). Returns (T, 7) wxyz.
    """
    T = len(tgt_t)
    out = np.zeros((T, 7), dtype=np.float32)
    for axis in range(3):
        out[:, axis] = np.interp(tgt_t, src_t, src_pos[:, axis])
    idx = np.searchsorted(src_t, tgt_t)
    idx = np.clip(idx, 0, len(src_t) - 1)
    left_idx = np.maximum(idx - 1, 0)
    d_right = np.abs(src_t[idx] - tgt_t)
    d_left = np.abs(src_t[left_idx] - tgt_t)
    nearest_idx = np.where(d_left < d_right, left_idx, idx)
    nearest_quat_xyzw = src_quat_xyzw[nearest_idx]
    out[:, 3:7] = reorder_quaternion_xyzw_to_wxyz(nearest_quat_xyzw)
    norms = np.linalg.norm(out[:, 3:7], axis=1, keepdims=True)
    out[:, 3:7] = out[:, 3:7] / np.maximum(norms, 1e-8)
    return out
