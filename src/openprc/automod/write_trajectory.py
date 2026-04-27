"""
Trajectory HDF5 writer.

All fetch scripts route through `write_trajectory` to produce schema-compliant
trajectory files. This is the only place in the codebase that writes trajectory
HDF5 files, so if the schema changes, only this file changes.

Usage:
    from write_trajectory import write_trajectory

    write_trajectory(
        out_path="go1/trajectories/trot_flat_001.h5",
        robot_name="go1",
        trajectory_id="trot_flat_001",
        joint_names=["FR_hip_joint", "FR_thigh_joint", ...],
        time=t,                      # (T,) float64
        qpos=q,                      # (T, 12) float32
        qvel=qdot,                   # optional
        tau=tau,                     # optional
        base_pose=base,              # (T, 7) wxyz; floating-base only
        contact_foot_force=ff,       # optional
        contact_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
        source="legkilo",
        source_type="real_hardware",
        source_url="https://github.com/ouguangjun/legkilo-dataset",
        source_citation="Ou et al., LegKilo 2023",
        source_license="MIT",
        notes="Trotting on flat ground, indoor lab",
    )
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import h5py
import numpy as np

SCHEMA_VERSION = "0.1.0"
VALID_SOURCE_TYPES = {"real_hardware", "teleop_real", "sim_rollout"}
COORDINATE_FRAME = "z_up_x_forward"
UNITS = "rad, m, s, N, Nm"
QUATERNION_ORDER = "wxyz"


def _validate_shape(name: str, arr: np.ndarray, expected_shape: tuple):
    """Check array shape, allowing -1 as wildcard."""
    if arr.ndim != len(expected_shape):
        raise ValueError(
            f"{name}: expected {len(expected_shape)} dims, got {arr.ndim} "
            f"(shape {arr.shape})"
        )
    for i, (actual, expected) in enumerate(zip(arr.shape, expected_shape)):
        if expected != -1 and actual != expected:
            raise ValueError(
                f"{name}: dim {i} expected {expected}, got {actual} "
                f"(full shape {arr.shape}, expected {expected_shape})"
            )


def _as_f32(arr) -> np.ndarray:
    return np.asarray(arr, dtype=np.float32)


def _as_f64(arr) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64)


def write_trajectory(
    out_path: str,
    *,
    # Identity
    robot_name: str,
    trajectory_id: str,
    joint_names: Sequence[str],
    # Required signals
    time: np.ndarray,
    qpos: np.ndarray,
    # Optional joint-space signals
    qvel: Optional[np.ndarray] = None,
    tau: Optional[np.ndarray] = None,
    # Optional floating-base signals
    base_pose: Optional[np.ndarray] = None,   # (T, 7) xyz + wxyz
    base_vel: Optional[np.ndarray] = None,    # (T, 6) linear + angular
    # Optional contact signals
    contact_foot_force: Optional[np.ndarray] = None,
    contact_foot_force_est: Optional[np.ndarray] = None,
    contact_flags: Optional[np.ndarray] = None,
    contact_names: Optional[Sequence[str]] = None,
    contact_force_units: Optional[str] = None,
    contact_force_scale: Optional[str] = None,
    # Optional action signals
    target_qpos: Optional[np.ndarray] = None,
    target_qvel: Optional[np.ndarray] = None,
    target_tau: Optional[np.ndarray] = None,
    # Provenance (all required)
    source: str,
    source_type: str,
    source_url: str,
    source_citation: str,
    source_license: str,
    # Optional
    notes: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    """
    Write a schema-compliant trajectory HDF5 file.

    Returns the output path on success. Raises ValueError on schema violations
    before writing anything.
    """
    # ------------------------------------------------------------
    # Pre-flight validation (no file I/O until everything is valid)
    # ------------------------------------------------------------
    if source_type not in VALID_SOURCE_TYPES:
        raise ValueError(
            f"source_type must be one of {VALID_SOURCE_TYPES}, got {source_type!r}"
        )

    if os.path.exists(out_path) and not overwrite:
        raise FileExistsError(
            f"{out_path} exists; pass overwrite=True to replace it"
        )

    # Time
    time = _as_f64(time)
    _validate_shape("time", time, (-1,))
    T = time.shape[0]
    if T < 2:
        raise ValueError(f"Need at least 2 timesteps, got {T}")
    dt = np.diff(time)
    if np.any(dt < 0):
        raise ValueError("time must be monotonically non-decreasing")
    if np.any(~np.isfinite(time)):
        raise ValueError("time contains NaN or inf")

    # qpos
    qpos = _as_f32(qpos)
    _validate_shape("qpos", qpos, (T, -1))
    n_joints = qpos.shape[1]
    if n_joints != len(joint_names):
        raise ValueError(
            f"joint_names has {len(joint_names)} entries but qpos has "
            f"{n_joints} columns"
        )
    if np.any(~np.isfinite(qpos)):
        raise ValueError("qpos contains NaN or inf")

    # Collect optional joint-space signals
    signals_present = ["qpos"]

    def _check_joint_signal(name: str, arr: Optional[np.ndarray]):
        if arr is None:
            return None
        arr = _as_f32(arr)
        _validate_shape(name, arr, (T, n_joints))
        if np.any(~np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN or inf")
        signals_present.append(name)
        return arr

    qvel = _check_joint_signal("qvel", qvel)
    tau = _check_joint_signal("tau", tau)
    target_qpos = _check_joint_signal("action/target_qpos", target_qpos)
    target_qvel = _check_joint_signal("action/target_qvel", target_qvel)
    target_tau = _check_joint_signal("action/target_tau", target_tau)

    # Floating-base signals
    if base_pose is not None:
        base_pose = _as_f32(base_pose)
        _validate_shape("base_pose", base_pose, (T, 7))
        if np.any(~np.isfinite(base_pose)):
            raise ValueError("base_pose contains NaN or inf")
        # Check quaternion unit norm (wxyz in last 4)
        quat_norms = np.linalg.norm(base_pose[:, 3:7], axis=1)
        if np.any(np.abs(quat_norms - 1.0) > 1e-3):
            raise ValueError(
                f"base_pose quaternions are not unit norm "
                f"(max deviation: {np.max(np.abs(quat_norms - 1.0)):.4g})"
            )
        signals_present.append("base_pose")

    if base_vel is not None:
        base_vel = _as_f32(base_vel)
        _validate_shape("base_vel", base_vel, (T, 6))
        if np.any(~np.isfinite(base_vel)):
            raise ValueError("base_vel contains NaN or inf")
        signals_present.append("base_vel")

    # Contact signals — all must agree on n_contacts if present
    n_contacts = None
    contact_arrays = {
        "contact/foot_force": contact_foot_force,
        "contact/foot_force_est": contact_foot_force_est,
    }
    for name, arr in list(contact_arrays.items()):
        if arr is None:
            continue
        arr = _as_f32(arr)
        _validate_shape(name, arr, (T, -1))
        if n_contacts is None:
            n_contacts = arr.shape[1]
        elif arr.shape[1] != n_contacts:
            raise ValueError(
                f"{name} has {arr.shape[1]} contacts but prior contact signal "
                f"had {n_contacts}"
            )
        if np.any(~np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN or inf")
        contact_arrays[name] = arr
        signals_present.append(name)

    if contact_flags is not None:
        contact_flags = np.asarray(contact_flags, dtype=bool)
        _validate_shape("contact/contact_flags", contact_flags, (T, -1))
        if n_contacts is None:
            n_contacts = contact_flags.shape[1]
        elif contact_flags.shape[1] != n_contacts:
            raise ValueError(
                f"contact_flags has {contact_flags.shape[1]} contacts but "
                f"prior contact signal had {n_contacts}"
            )
        signals_present.append("contact/contact_flags")

    if n_contacts is not None:
        if contact_names is None:
            raise ValueError(
                "contact_names must be provided when any contact signal is present"
            )
        if len(contact_names) != n_contacts:
            raise ValueError(
                f"contact_names has {len(contact_names)} entries but contact "
                f"signals have {n_contacts} columns"
            )
        # Require contact_force_units when any force signal (not just flags) is present
        has_force_signal = (
            contact_arrays["contact/foot_force"] is not None
            or contact_arrays["contact/foot_force_est"] is not None
        )
        if has_force_signal and contact_force_units is None:
            raise ValueError(
                "contact_force_units must be provided when foot_force or "
                "foot_force_est is present (one of 'N', 'adc_raw', "
                "'normalized_per_sequence')"
            )
        if contact_force_units is not None and contact_force_units not in (
            "N", "adc_raw", "normalized_per_sequence"
        ):
            raise ValueError(
                f"contact_force_units must be one of 'N', 'adc_raw', "
                f"'normalized_per_sequence'; got {contact_force_units!r}"
            )

    # ------------------------------------------------------------
    # All validation passed — write the file
    # ------------------------------------------------------------
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with h5py.File(out_path, "w") as f:
        # Datasets
        f.create_dataset("time", data=time, compression="gzip")
        f.create_dataset("qpos", data=qpos, compression="gzip")
        if qvel is not None:
            f.create_dataset("qvel", data=qvel, compression="gzip")
        if tau is not None:
            f.create_dataset("tau", data=tau, compression="gzip")
        if base_pose is not None:
            f.create_dataset("base_pose", data=base_pose, compression="gzip")
        if base_vel is not None:
            f.create_dataset("base_vel", data=base_vel, compression="gzip")
        for name, arr in contact_arrays.items():
            if arr is not None:
                f.create_dataset(name, data=arr, compression="gzip")
        if contact_flags is not None:
            f.create_dataset("contact/contact_flags", data=contact_flags,
                             compression="gzip")
        if target_qpos is not None:
            f.create_dataset("action/target_qpos", data=target_qpos,
                             compression="gzip")
        if target_qvel is not None:
            f.create_dataset("action/target_qvel", data=target_qvel,
                             compression="gzip")
        if target_tau is not None:
            f.create_dataset("action/target_tau", data=target_tau,
                             compression="gzip")

        # Root attributes
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["robot_name"] = robot_name
        f.attrs["trajectory_id"] = trajectory_id
        f.attrs["joint_names"] = list(joint_names)
        if contact_names is not None:
            f.attrs["contact_names"] = list(contact_names)

        # Frequency: use total samples / duration (robust to clustered or
        # duplicate timestamps). The earlier "1 / median(dt)" approach gave
        # garbage when ROS bags had tiny dt outliers from packet bursts.
        duration_s = float(time[-1] - time[0])
        frequency_hz = (len(time) - 1) / duration_s if duration_s > 0 else float("nan")

        f.attrs["frequency_hz"] = frequency_hz
        f.attrs["duration_s"] = duration_s
        f.attrs["n_timesteps"] = int(T)

        f.attrs["source"] = source
        f.attrs["source_type"] = source_type
        f.attrs["source_url"] = source_url
        f.attrs["source_citation"] = source_citation
        f.attrs["source_license"] = source_license

        f.attrs["signals_present"] = signals_present

        if base_pose is not None:
            f.attrs["quaternion_order"] = QUATERNION_ORDER
        f.attrs["coordinate_frame"] = COORDINATE_FRAME
        f.attrs["units"] = UNITS

        if contact_force_units is not None:
            f.attrs["contact_force_units"] = contact_force_units
        if contact_force_scale is not None:
            f.attrs["contact_force_scale"] = contact_force_scale

        if notes is not None:
            f.attrs["notes"] = notes

    return out_path
