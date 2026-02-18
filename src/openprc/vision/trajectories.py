"""
openprc.vision.trajectories
============================

Signal extraction from tracked feature trajectories.

Converts raw pixel-space TrackingResult data into calibrated,
research-ready timeseries signals in 1D, 2D, or 3D, with
gap handling, smoothing, and HDF5-compatible output.

Dependencies:
    - numpy
    - scipy (for interpolation and smoothing)

Location: openprc/vision/trajectories.py

Example
-------
>>> from openprc.vision.trajectories import TrajectorySet
>>>
>>> tset = TrajectorySet.from_tracking_result(result)
>>> signals_2d = tset.to_signals(dim=2)           # (T, N, 2) normalized
>>> signals_1d = tset.to_signals(dim=1, axis="x")  # (T, N) x-component
>>>
>>> # With camera calibration
>>> from openprc.vision.calibration import CameraCalibration
>>> cal = CameraCalibration.from_matrix(K)
>>> tset_cal = TrajectorySet.from_tracking_result(result, calibration=cal)
>>> physical = tset_cal.to_signals(dim=2)  # (T, N, 2) in physical units
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from scipy.interpolate import interp1d
    from scipy.ndimage import uniform_filter1d

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from .calibration import BaseCalibration, NormalizedCalibration, auto_calibration


# ======================================================================
# Enums
# ======================================================================


class GapStrategy(Enum):
    """How to handle NaN gaps in trajectories."""

    NAN = "nan"              # Leave NaN as-is
    LINEAR = "linear"        # Linear interpolation across gaps
    ZERO = "zero"            # Fill gaps with zero
    HOLD = "hold"            # Forward-fill last known value


class Axis1D(Enum):
    """Projection axis for 1D signal extraction."""

    X = "x"                  # Horizontal component
    Y = "y"                  # Vertical component
    MAGNITUDE = "magnitude"  # Displacement magnitude from initial position
    DISTANCE = "distance"    # Cumulative distance traveled


# ======================================================================
# TrajectorySet
# ======================================================================


@dataclass
class TrajectorySet:
    """Calibrated feature trajectories with signal export.

    This is the bridge between raw tracking output and the timeseries
    format consumed by openprc.analysis and openprc.reservoir.

    Attributes
    ----------
    positions : np.ndarray
        (T, N, 2) calibrated coordinates.
    positions_px : np.ndarray
        (T, N, 2) original pixel coordinates (preserved for reference).
    status : np.ndarray
        (T, N) TrackStatus codes from the tracker.
    timestamps : np.ndarray
        (T,) time in seconds.
    fps : float
        Source framerate.
    calibration : BaseCalibration
        Calibration used for coordinate transform.
    feature_ids : np.ndarray
        (N,) integer feature identifiers.
    metadata : dict
        Additional metadata for HDF5 serialization.
    """

    positions: np.ndarray
    positions_px: np.ndarray
    status: np.ndarray
    timestamps: np.ndarray
    fps: float
    calibration: BaseCalibration
    feature_ids: np.ndarray = field(default_factory=lambda: np.array([]))
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_tracking_result(
            cls,
            result,  # TrackingResult — avoid circular import
            calibration: Optional[BaseCalibration] = None,
    ) -> TrajectorySet:
        """Create from a TrackingResult with optional calibration.

        Parameters
        ----------
        result : TrackingResult
            Output from the Tracker.
        calibration : BaseCalibration or None
            If None, uses NormalizedCalibration based on the feature
            map's frame shape.

        Returns
        -------
        TrajectorySet
        """
        fmap = result.feature_map

        if calibration is None:
            calibration = NormalizedCalibration(frame_shape=fmap.frame_shape)

        # Transform all positions (handles NaN gracefully)
        pos_px = result.positions.copy()

        # Build a mask for valid points to avoid transforming NaN
        valid = ~np.isnan(pos_px[..., 0])
        pos_cal = np.full_like(pos_px, np.nan, dtype=np.float64)
        if np.any(valid):
            pos_cal[valid] = calibration.transform(pos_px[valid])

        feature_ids = np.arange(result.n_features, dtype=np.int32)

        return cls(
            positions=pos_cal,
            positions_px=pos_px,
            status=result.status.copy(),
            timestamps=result.timestamps.copy(),
            fps=result.fps,
            calibration=calibration,
            feature_ids=feature_ids,
            metadata={
                "detector": getattr(fmap, "detector_name", ""),
                "tracker_config": result.tracker_config,
                "frame_range": result.frame_range,
                "calibration_kind": calibration.kind,
                "calibration_units": calibration.units,
            },
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_frames(self) -> int:
        return self.positions.shape[0]

    @property
    def n_features(self) -> int:
        return self.positions.shape[1]

    @property
    def duration(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def units(self) -> str:
        return self.calibration.units

    # ------------------------------------------------------------------
    # Gap handling
    # ------------------------------------------------------------------

    def fill_gaps(
            self,
            strategy: Union[str, GapStrategy] = GapStrategy.LINEAR,
            max_gap: Optional[int] = None,
    ) -> TrajectorySet:
        """Fill NaN gaps in trajectories.

        Parameters
        ----------
        strategy : str or GapStrategy
            Gap filling method. Default 'linear'.
        max_gap : int or None
            Maximum gap length (in frames) to fill. Gaps longer than
            this remain NaN. None fills all gaps.

        Returns
        -------
        TrajectorySet
            New instance with gaps filled.
        """
        if isinstance(strategy, str):
            strategy = GapStrategy(strategy)

        filled = self.positions.copy()
        new_status = self.status.copy()

        for j in range(self.n_features):
            traj = filled[:, j, :]  # (T, 2) view
            valid = ~np.isnan(traj[:, 0])

            if np.all(valid) or np.sum(valid) < 2:
                continue

            gap_mask = ~valid

            # Enforce max_gap
            if max_gap is not None:
                gap_mask = self._limit_gaps(gap_mask, max_gap)

            if strategy == GapStrategy.NAN:
                continue

            elif strategy == GapStrategy.ZERO:
                traj[gap_mask] = 0.0

            elif strategy == GapStrategy.HOLD:
                # Forward fill
                for i in range(1, self.n_frames):
                    if gap_mask[i] and valid[i - 1]:
                        traj[i] = traj[i - 1]
                        valid[i] = True

            elif strategy == GapStrategy.LINEAR:
                if not _HAS_SCIPY:
                    # Fallback to numpy interp per axis
                    t = np.arange(self.n_frames)
                    idx_valid = np.where(~np.isnan(self.positions[:, j, 0]))[0]
                    for ax in range(2):
                        vals = self.positions[idx_valid, j, ax]
                        interped = np.interp(t, idx_valid, vals)
                        traj[gap_mask, ax] = interped[gap_mask]
                else:
                    t = np.arange(self.n_frames)
                    idx_valid = np.where(~np.isnan(self.positions[:, j, 0]))[0]
                    for ax in range(2):
                        vals = self.positions[idx_valid, j, ax]
                        f = interp1d(
                            idx_valid, vals,
                            kind="linear",
                            bounds_error=False,
                            fill_value="extrapolate",
                        )
                        traj[gap_mask, ax] = f(t[gap_mask])

            # Mark filled frames
            from .tracker import TrackStatus

            originally_nan = np.isnan(self.positions[:, j, 0])
            now_filled = originally_nan & ~np.isnan(traj[:, 0])
            new_status[now_filled, j] = TrackStatus.INTERPOLATED

        return TrajectorySet(
            positions=filled,
            positions_px=self.positions_px.copy(),
            status=new_status,
            timestamps=self.timestamps.copy(),
            fps=self.fps,
            calibration=self.calibration,
            feature_ids=self.feature_ids.copy(),
            metadata=self.metadata.copy(),
        )

    @staticmethod
    def _limit_gaps(gap_mask: np.ndarray, max_gap: int) -> np.ndarray:
        """Zero out gap runs longer than max_gap."""
        limited = gap_mask.copy()
        in_gap = False
        gap_start = 0

        for i in range(len(limited)):
            if limited[i]:
                if not in_gap:
                    gap_start = i
                    in_gap = True
            else:
                if in_gap:
                    if (i - gap_start) > max_gap:
                        limited[gap_start:i] = False
                    in_gap = False

        # Handle gap at end
        if in_gap and (len(limited) - gap_start) > max_gap:
            limited[gap_start:] = False

        return limited

    # ------------------------------------------------------------------
    # Smoothing
    # ------------------------------------------------------------------

    def smooth(self, window: int = 5) -> TrajectorySet:
        """Apply moving average smoothing to trajectories.

        Parameters
        ----------
        window : int
            Smoothing window size in frames. Default 5.

        Returns
        -------
        TrajectorySet
            Smoothed copy. NaN regions are preserved.
        """
        smoothed = self.positions.copy()

        for j in range(self.n_features):
            for ax in range(2):
                col = smoothed[:, j, ax]
                valid = ~np.isnan(col)
                if np.sum(valid) < window:
                    continue

                if _HAS_SCIPY:
                    col[valid] = uniform_filter1d(col[valid], size=window)
                else:
                    # Simple numpy convolution fallback
                    kernel = np.ones(window) / window
                    valid_data = col[valid]
                    if len(valid_data) >= window:
                        padded = np.pad(valid_data, window // 2, mode="edge")
                        col[valid] = np.convolve(padded, kernel, mode="valid")[
                            : len(valid_data)
                        ]

        return TrajectorySet(
            positions=smoothed,
            positions_px=self.positions_px.copy(),
            status=self.status.copy(),
            timestamps=self.timestamps.copy(),
            fps=self.fps,
            calibration=self.calibration,
            feature_ids=self.feature_ids.copy(),
            metadata=self.metadata.copy(),
        )

    # ------------------------------------------------------------------
    # Signal extraction
    # ------------------------------------------------------------------

    def to_signals(
            self,
            dim: int = 2,
            axis: Union[str, Axis1D] = Axis1D.X,
    ) -> np.ndarray:
        """Extract timeseries signals from trajectories.

        Parameters
        ----------
        dim : int
            Output dimensionality per feature:
            - 1: Scalar signal per feature (see ``axis``).
            - 2: (x, y) coordinates per feature.
            - 3: (x, y, displacement_magnitude) per feature.

        axis : str or Axis1D
            For dim=1, which projection to use:
            - 'x': Horizontal component.
            - 'y': Vertical component.
            - 'magnitude': Euclidean distance from initial position.
            - 'distance': Cumulative distance traveled.

        Returns
        -------
        np.ndarray
            Shape depends on dim:
            - dim=1: (T, N)
            - dim=2: (T, N, 2)
            - dim=3: (T, N, 3)
        """
        pos = self.positions  # (T, N, 2)

        if dim == 2:
            return pos.copy()

        elif dim == 1:
            if isinstance(axis, str):
                axis = Axis1D(axis)

            if axis == Axis1D.X:
                return pos[..., 0].copy()

            elif axis == Axis1D.Y:
                return pos[..., 1].copy()

            elif axis == Axis1D.MAGNITUDE:
                # Displacement from initial position
                ref = pos[0:1, :, :]  # (1, N, 2)
                disp = pos - ref
                return np.sqrt(np.nansum(disp ** 2, axis=-1))  # (T, N)

            elif axis == Axis1D.DISTANCE:
                # Cumulative path distance
                diff = np.diff(pos, axis=0, prepend=pos[0:1])
                step_dist = np.sqrt(np.nansum(diff ** 2, axis=-1))  # (T, N)
                return np.nancumsum(step_dist, axis=0)

        elif dim == 3:
            # (x, y, displacement_magnitude)
            ref = pos[0:1, :, :]
            disp = pos - ref
            mag = np.sqrt(np.nansum(disp ** 2, axis=-1, keepdims=True))  # (T, N, 1)
            return np.concatenate([pos, mag], axis=-1)  # (T, N, 3)

        else:
            raise ValueError(f"dim must be 1, 2, or 3, got {dim}")

    # ------------------------------------------------------------------
    # Velocity / derived signals
    # ------------------------------------------------------------------

    def velocities(self) -> np.ndarray:
        """Compute per-frame velocity vectors via finite differences.

        Returns
        -------
        np.ndarray
            (T, N, 2) velocity in calibrated units per second.
            First frame is zero.
        """
        dt = 1.0 / self.fps if self.fps > 0 else 1.0
        vel = np.gradient(self.positions, dt, axis=0)
        # NaN propagates naturally through gradient
        return vel

    def speeds(self) -> np.ndarray:
        """Scalar speed per frame (magnitude of velocity).

        Returns
        -------
        np.ndarray
            (T, N) speed in calibrated units per second.
        """
        vel = self.velocities()
        return np.sqrt(np.nansum(vel ** 2, axis=-1))

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def select_features(self, indices: np.ndarray) -> TrajectorySet:
        """Return a subset of features by index.

        Parameters
        ----------
        indices : array-like
            Feature indices to keep.

        Returns
        -------
        TrajectorySet
        """
        idx = np.asarray(indices)
        return TrajectorySet(
            positions=self.positions[:, idx, :],
            positions_px=self.positions_px[:, idx, :],
            status=self.status[:, idx],
            timestamps=self.timestamps.copy(),
            fps=self.fps,
            calibration=self.calibration,
            feature_ids=self.feature_ids[idx],
            metadata=self.metadata.copy(),
        )

    def trim(self, start_time: float = 0.0, end_time: Optional[float] = None) -> TrajectorySet:
        """Trim trajectories to a time window.

        Parameters
        ----------
        start_time : float
            Start time in seconds.
        end_time : float or None
            End time in seconds. None keeps to the end.

        Returns
        -------
        TrajectorySet
        """
        t = self.timestamps
        mask = t >= start_time
        if end_time is not None:
            mask &= t <= end_time
        idx = np.where(mask)[0]

        return TrajectorySet(
            positions=self.positions[idx],
            positions_px=self.positions_px[idx],
            status=self.status[idx],
            timestamps=self.timestamps[idx],
            fps=self.fps,
            calibration=self.calibration,
            feature_ids=self.feature_ids.copy(),
            metadata=self.metadata.copy(),
        )

    # ------------------------------------------------------------------
    # Serialization helpers (consumed by io.py)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a dict of arrays + metadata for HDF5."""
        return {
            "positions": self.positions,
            "positions_px": self.positions_px,
            "status": self.status,
            "timestamps": self.timestamps,
            "fps": self.fps,
            "feature_ids": self.feature_ids,
            "calibration": self.calibration.to_dict(),
            "metadata": self.metadata,
        }

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TrajectorySet(frames={self.n_frames}, "
            f"features={self.n_features}, "
            f"duration={self.duration:.2f}s, "
            f"units={self.units!r})"
        )