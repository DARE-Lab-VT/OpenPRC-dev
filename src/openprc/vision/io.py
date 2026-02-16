"""
openprc.vision.io
=================

HDF5 serialization for the full vision pipeline.

Reads and writes the complete vision data tree — source metadata,
feature maps, tracking results, and trajectory signals — using the
openprc HDF5 schema.

Schema::

    <root>.h5
    └── /vision/
        ├── /source/              # VideoSource metadata (attrs)
        ├── /feature_map/         # FeatureMap arrays + config
        │   ├── keypoints         (N, 2)
        │   ├── descriptors       (N, D)
        │   ├── sizes             (N,)
        │   ├── angles            (N,)
        │   ├── responses         (N,)
        │   └── octaves           (N,)
        ├── /tracking/            # Raw tracking output
        │   ├── positions         (T, N, 2)  float32
        │   ├── status            (T, N)     uint8
        │   └── timestamps        (T,)       float64
        ├── /trajectories/        # Calibrated trajectories
        │   ├── positions         (T, N, 2)  float64
        │   ├── positions_px      (T, N, 2)  float32
        │   ├── status            (T, N)     uint8
        │   └── timestamps        (T,)       float64
        └── /signals/             # Exported signals
            ├── data              (T, N, d)  d=1,2,3
            └── attrs: dim, units, axis, calibration_kind

Dependencies:
    - h5py
    - numpy
    - json (stdlib)

Location: openprc/vision/io.py

Example
-------
>>> from openprc.vision import io
>>>
>>> # Save full pipeline
>>> io.save("experiment.h5", source=src, feature_map=fmap,
...         tracking_result=result, trajectory_set=tset)
>>>
>>> # Load back
>>> data = io.load("experiment.h5")
>>> tset = data["trajectory_set"]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import h5py
import numpy as np

logger = logging.getLogger(__name__)

# Root group name — consistent with the openprc HDF5 schema
VISION_GROUP = "vision"


# ======================================================================
# Write
# ======================================================================


def save(
        path: Union[str, Path],
        source=None,
        feature_map=None,
        tracking_result=None,
        trajectory_set=None,
        signals: Optional[np.ndarray] = None,
        signals_meta: Optional[dict] = None,
        mode: str = "a",
) -> Path:
    """Save vision pipeline data to an HDF5 file.

    Any combination of components can be provided. Existing data
    in the vision group is overwritten per-subgroup.

    Parameters
    ----------
    path : str or Path
        Output HDF5 file path.
    source : VideoSource or None
        Video source metadata.
    feature_map : FeatureMap or None
        Detected features.
    tracking_result : TrackingResult or None
        Raw tracking output.
    trajectory_set : TrajectorySet or None
        Calibrated trajectories.
    signals : np.ndarray or None
        Pre-extracted signal array (T, N, d).
    signals_meta : dict or None
        Signal metadata (dim, units, axis, etc.).
    mode : str
        HDF5 file mode. Default 'a' (append/create).

    Returns
    -------
    Path
        The output file path.
    """
    path = Path(path)

    with h5py.File(path, mode) as f:
        # Ensure root group
        vis = f.require_group(VISION_GROUP)

        # -- Source metadata --
        if source is not None:
            _write_source(vis, source)

        # -- Feature map --
        if feature_map is not None:
            _write_feature_map(vis, feature_map)

        # -- Tracking result --
        if tracking_result is not None:
            _write_tracking(vis, tracking_result)

        # -- Trajectory set --
        if trajectory_set is not None:
            _write_trajectories(vis, trajectory_set)

        # -- Signals --
        if signals is not None:
            _write_signals(vis, signals, signals_meta or {})

    logger.info("Saved vision data → %s", path)
    return path


def _write_source(parent: h5py.Group, source) -> None:
    """Write VideoSource metadata as attributes."""
    grp = _replace_group(parent, "source")
    meta = source.metadata()
    for k, v in meta.items():
        grp.attrs[k] = v


def _write_feature_map(parent: h5py.Group, fmap) -> None:
    """Write FeatureMap arrays and config."""
    grp = _replace_group(parent, "feature_map")
    data = fmap.to_dict()

    for key in ("keypoints", "descriptors", "sizes", "angles", "responses", "octaves"):
        grp.create_dataset(key, data=data[key], compression="gzip", compression_opts=4)

    grp.attrs["detector_name"] = data["detector_name"]
    grp.attrs["detector_config"] = data["detector_config"]  # already JSON string
    grp.attrs["ref_frame_index"] = data["ref_frame_index"]
    grp.attrs["frame_shape"] = data["frame_shape"]


def _write_tracking(parent: h5py.Group, result) -> None:
    """Write raw TrackingResult."""
    grp = _replace_group(parent, "tracking")
    grp.create_dataset("positions", data=result.positions, compression="gzip")
    grp.create_dataset("status", data=result.status, compression="gzip")
    grp.create_dataset("timestamps", data=result.timestamps)

    grp.attrs["fps"] = result.fps
    grp.attrs["frame_range"] = list(result.frame_range)
    grp.attrs["tracker_config"] = json.dumps(result.tracker_config)


def _write_trajectories(parent: h5py.Group, tset) -> None:
    """Write calibrated TrajectorySet."""
    grp = _replace_group(parent, "trajectories")
    grp.create_dataset("positions", data=tset.positions, compression="gzip")
    grp.create_dataset("positions_px", data=tset.positions_px, compression="gzip")
    grp.create_dataset("status", data=tset.status, compression="gzip")
    grp.create_dataset("timestamps", data=tset.timestamps)
    grp.create_dataset("feature_ids", data=tset.feature_ids)

    grp.attrs["fps"] = tset.fps
    grp.attrs["calibration"] = json.dumps(tset.calibration.to_dict())
    grp.attrs["metadata"] = json.dumps(
        tset.metadata, default=_json_fallback
    )


def _write_signals(parent: h5py.Group, signals: np.ndarray, meta: dict) -> None:
    """Write exported signal array."""
    grp = _replace_group(parent, "signals")
    grp.create_dataset("data", data=signals, compression="gzip")
    for k, v in meta.items():
        grp.attrs[k] = v


# ======================================================================
# Read
# ======================================================================


def load(path: Union[str, Path]) -> Dict[str, Any]:
    """Load vision pipeline data from an HDF5 file.

    Returns whatever components are present in the file.

    Parameters
    ----------
    path : str or Path
        HDF5 file path.

    Returns
    -------
    dict
        Keys may include: 'source_meta', 'feature_map',
        'tracking_positions', 'tracking_status', 'tracking_timestamps',
        'trajectory_set', 'signals', 'signals_meta'.
    """
    path = Path(path)
    result = {}

    with h5py.File(path, "r") as f:
        if VISION_GROUP not in f:
            logger.warning("No '%s' group found in %s", VISION_GROUP, path)
            return result

        vis = f[VISION_GROUP]

        # -- Source --
        if "source" in vis:
            result["source_meta"] = dict(vis["source"].attrs)

        # -- Feature map --
        if "feature_map" in vis:
            result["feature_map"] = _read_feature_map(vis["feature_map"])

        # -- Tracking --
        if "tracking" in vis:
            trk = vis["tracking"]
            result["tracking_positions"] = trk["positions"][:]
            result["tracking_status"] = trk["status"][:]
            result["tracking_timestamps"] = trk["timestamps"][:]
            result["tracking_fps"] = float(trk.attrs.get("fps", 0))
            result["tracking_config"] = json.loads(
                trk.attrs.get("tracker_config", "{}")
            )

        # -- Trajectories --
        if "trajectories" in vis:
            result["trajectory_set"] = _read_trajectories(vis["trajectories"])

        # -- Signals --
        if "signals" in vis:
            sig = vis["signals"]
            result["signals"] = sig["data"][:]
            result["signals_meta"] = dict(sig.attrs)

    logger.info("Loaded vision data from %s", path)
    return result


def _read_feature_map(grp: h5py.Group):
    """Reconstruct a FeatureMap from an HDF5 group."""
    from .detectors import FeatureMap

    config_str = grp.attrs.get("detector_config", "{}")
    if isinstance(config_str, bytes):
        config_str = config_str.decode()

    frame_shape = grp.attrs.get("frame_shape", [0, 0])
    if isinstance(frame_shape, np.ndarray):
        frame_shape = tuple(frame_shape.tolist())
    else:
        frame_shape = tuple(frame_shape)

    return FeatureMap(
        keypoints=grp["keypoints"][:],
        descriptors=grp["descriptors"][:],
        sizes=grp["sizes"][:],
        angles=grp["angles"][:],
        responses=grp["responses"][:],
        octaves=grp["octaves"][:],
        detector_name=str(grp.attrs.get("detector_name", "")),
        detector_config=json.loads(config_str),
        ref_frame_index=int(grp.attrs.get("ref_frame_index", 0)),
        frame_shape=frame_shape,
    )


def _read_trajectories(grp: h5py.Group):
    """Reconstruct a TrajectorySet from an HDF5 group."""
    from .calibration import load_calibration
    from .trajectories import TrajectorySet

    cal_str = grp.attrs.get("calibration", "{}")
    if isinstance(cal_str, bytes):
        cal_str = cal_str.decode()
    calibration = load_calibration(json.loads(cal_str))

    meta_str = grp.attrs.get("metadata", "{}")
    if isinstance(meta_str, bytes):
        meta_str = meta_str.decode()
    metadata = json.loads(meta_str)

    return TrajectorySet(
        positions=grp["positions"][:],
        positions_px=grp["positions_px"][:],
        status=grp["status"][:],
        timestamps=grp["timestamps"][:],
        fps=float(grp.attrs.get("fps", 0)),
        calibration=calibration,
        feature_ids=grp["feature_ids"][:] if "feature_ids" in grp else np.array([]),
        metadata=metadata,
    )


# ======================================================================
# Utilities
# ======================================================================


def _replace_group(parent: h5py.Group, name: str) -> h5py.Group:
    """Delete and recreate a group (ensures clean overwrite)."""
    if name in parent:
        del parent[name]
    return parent.create_group(name)


def _json_fallback(obj):
    """JSON serializer fallback for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def info(path: Union[str, Path]) -> dict:
    """Quick summary of what's stored in a vision HDF5 file.

    Parameters
    ----------
    path : str or Path
        HDF5 file path.

    Returns
    -------
    dict
        Summary with keys like 'has_feature_map', 'n_features',
        'n_frames', 'has_signals', etc.
    """
    path = Path(path)
    summary = {"path": str(path), "has_vision_data": False}

    with h5py.File(path, "r") as f:
        if VISION_GROUP not in f:
            return summary

        summary["has_vision_data"] = True
        vis = f[VISION_GROUP]
        summary["subgroups"] = list(vis.keys())

        if "feature_map" in vis:
            summary["has_feature_map"] = True
            summary["n_features"] = vis["feature_map"]["keypoints"].shape[0]
            summary["detector"] = str(
                vis["feature_map"].attrs.get("detector_name", "")
            )

        if "tracking" in vis:
            summary["has_tracking"] = True
            shape = vis["tracking"]["positions"].shape
            summary["tracking_frames"] = shape[0]
            summary["tracking_features"] = shape[1]

        if "trajectories" in vis:
            summary["has_trajectories"] = True
            shape = vis["trajectories"]["positions"].shape
            summary["traj_frames"] = shape[0]
            summary["traj_features"] = shape[1]

        if "signals" in vis:
            summary["has_signals"] = True
            summary["signal_shape"] = list(vis["signals"]["data"].shape)

    return summary