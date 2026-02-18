"""
openprc.vision.tracker
======================

Sparse optical flow tracking across video frames.

Uses OpenCV's Lucas-Kanade (KLT) pyramid tracker to follow features
detected by the detectors module through a video sequence. Produces
raw pixel-space trajectories with per-feature, per-frame status flags.

Dependencies:
    - opencv-python (cv2)
    - numpy

Location: openprc/vision/tracker.py

Example
-------
>>> from openprc.vision.source import VideoSource
>>> from openprc.vision.detectors import detect
>>> from openprc.vision.tracker import Tracker
>>>
>>> src = VideoSource("experiment.mp4")
>>> ref = src.read_frame(0)
>>> fmap = detect(ref, method="sift")
>>>
>>> tracker = Tracker(src, fmap)
>>> result = tracker.run()
>>>
>>> print(result.positions.shape)   # (T, N, 2)
>>> print(result.status.shape)      # (T, N)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import numpy as np


# ======================================================================
# Tracking status codes
# ======================================================================

class TrackStatus:
    """Integer codes for per-feature, per-frame track state."""
    TRACKED = 0       # Successfully tracked this frame
    LOST = 1          # KLT lost the feature (flow failed)
    OOB = 2           # Feature moved outside frame bounds
    DRIFT = 3         # Failed forward-backward consistency check
    INTERPOLATED = 4  # Gap-filled by interpolation (post-processing)
    REFERENCE = 5     # Reference frame (initial position)


# ======================================================================
# KLT configuration
# ======================================================================

@dataclass
class KLTConfig:
    """Configuration for the Lucas-Kanade pyramid tracker.

    Parameters
    ----------
    win_size : tuple of (int, int)
        Search window size at each pyramid level. Default (21, 21).
    max_level : int
        Maximum pyramid levels (0 = no pyramid). Default 3.
    max_iter : int
        Maximum iterations per level. Default 30.
    epsilon : float
        Convergence threshold. Default 0.01.
    fb_threshold : float or None
        Forward-backward error threshold in pixels. If set, features
        whose round-trip displacement exceeds this are marked DRIFT.
        None disables the check. Default 1.0.
    min_eigen_threshold : float
        Minimum eigenvalue threshold for cv2.calcOpticalFlowPyrLK.
        Default 1e-4.
    """

    win_size: Tuple[int, int] = (21, 21)
    max_level: int = 3
    max_iter: int = 30
    epsilon: float = 0.01
    fb_threshold: Optional[float] = 1.0
    min_eigen_threshold: float = 1e-4

    @property
    def lk_params(self) -> dict:
        """OpenCV-compatible parameter dict for calcOpticalFlowPyrLK."""
        return {
            "winSize": self.win_size,
            "maxLevel": self.max_level,
            "criteria": (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                self.max_iter,
                self.epsilon,
            ),
            "minEigThreshold": self.min_eigen_threshold,
        }

    def to_dict(self) -> dict:
        """Serialize for HDF5 metadata."""
        return {
            "win_size": list(self.win_size),
            "max_level": self.max_level,
            "max_iter": self.max_iter,
            "epsilon": self.epsilon,
            "fb_threshold": self.fb_threshold if self.fb_threshold is not None else -1,
            "min_eigen_threshold": self.min_eigen_threshold,
        }


# ======================================================================
# Tracking result
# ======================================================================

@dataclass
class TrackingResult:
    """Raw output of the tracker.

    Attributes
    ----------
    positions : np.ndarray
        (T, N, 2) float32 — pixel coordinates (x, y) per frame per feature.
        NaN where status != TRACKED and != REFERENCE.
    status : np.ndarray
        (T, N) uint8 — TrackStatus code per frame per feature.
    timestamps : np.ndarray
        (T,) float64 — time in seconds for each frame (from fps).
    fps : float
        Source video framerate.
    feature_map : object
        The FeatureMap used as the tracking seed.
    tracker_config : dict
        KLT parameters used.
    frame_range : tuple of (int, int)
        (start_frame, end_frame) indices that were tracked.
    """

    positions: np.ndarray
    status: np.ndarray
    timestamps: np.ndarray
    fps: float
    feature_map: Any  # FeatureMap — avoid circular import at runtime
    tracker_config: dict = field(default_factory=dict)
    frame_range: Tuple[int, int] = (0, 0)

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
        """Total tracked duration in seconds."""
        return self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 0.0

    # ------------------------------------------------------------------
    # Per-feature quality metrics
    # ------------------------------------------------------------------

    def track_lengths(self) -> np.ndarray:
        """Number of frames each feature was successfully tracked.

        Returns
        -------
        np.ndarray
            (N,) int array.
        """
        return np.sum(
            (self.status == TrackStatus.TRACKED)
            | (self.status == TrackStatus.REFERENCE),
            axis=0,
            )

    def track_ratios(self) -> np.ndarray:
        """Fraction of total frames each feature was tracked.

        Returns
        -------
        np.ndarray
            (N,) float array in [0, 1].
        """
        return self.track_lengths() / self.n_frames

    def filter(self, min_track_ratio: float = 0.5) -> TrackingResult:
        """Return a new result keeping only features above a quality threshold.

        Parameters
        ----------
        min_track_ratio : float
            Minimum fraction of frames a feature must be tracked. Default 0.5.

        Returns
        -------
        TrackingResult
            Filtered copy.
        """
        mask = self.track_ratios() >= min_track_ratio
        idx = np.where(mask)[0]
        return TrackingResult(
            positions=self.positions[:, idx, :],
            status=self.status[:, idx],
            timestamps=self.timestamps.copy(),
            fps=self.fps,
            feature_map=self.feature_map,
            tracker_config=self.tracker_config.copy(),
            frame_range=self.frame_range,
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        good = int(np.sum(self.track_ratios() >= 0.5))
        return (
            f"TrackingResult(frames={self.n_frames}, "
            f"features={self.n_features}, "
            f"good_tracks(>50%)={good}, "
            f"duration={self.duration:.2f}s)"
        )


# ======================================================================
# Tracker
# ======================================================================


class Tracker:
    """KLT sparse optical flow tracker.

    Tracks features from a FeatureMap through a VideoSource using
    OpenCV's pyramidal Lucas-Kanade implementation.

    Parameters
    ----------
    source : VideoSource
        Video to track through.
    feature_map : FeatureMap
        Seed features (from detectors module).
    config : KLTConfig, optional
        Tracking parameters. Default KLTConfig().
    start_frame : int or None
        Frame to start tracking from. None uses feature_map.ref_frame_index.
    end_frame : int or None
        Frame to stop tracking at (exclusive). None tracks to end of video.
    progress_callback : callable or None
        Called with (current_frame, total_frames) for progress reporting.
    """

    def __init__(
            self,
            source,  # VideoSource — no type hint to avoid circular import
            feature_map,  # FeatureMap
            config: Optional[KLTConfig] = None,
            start_frame: Optional[int] = None,
            end_frame: Optional[int] = None,
            progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        self.source = source
        self.feature_map = feature_map
        self.config = config or KLTConfig()
        self.start_frame = (
            start_frame if start_frame is not None else feature_map.ref_frame_index
        )
        self.end_frame = end_frame
        self.progress_callback = progress_callback

    def run(self) -> TrackingResult:
        """Execute tracking across the video.

        Returns
        -------
        TrackingResult
            Positions, status, and metadata for all tracked features.
        """
        src = self.source
        fmap = self.feature_map
        cfg = self.config

        # Resolve frame range
        start = self.start_frame
        end = self.end_frame if self.end_frame is not None else src.frame_count
        if end <= start:
            raise ValueError(f"end_frame ({end}) must be > start_frame ({start}).")

        total_frames = end - start
        n_features = fmap.n_features

        if n_features == 0:
            raise ValueError("FeatureMap is empty — nothing to track.")

        # Pre-allocate output arrays
        positions = np.full((total_frames, n_features, 2), np.nan, dtype=np.float32)
        status = np.full((total_frames, n_features), TrackStatus.LOST, dtype=np.uint8)

        # Seed with reference keypoints
        positions[0] = fmap.keypoints.copy()
        status[0] = TrackStatus.REFERENCE

        # Read reference frame
        src.seek(start)
        prev_frame = src.read()
        if prev_frame is None:
            raise IOError(f"Failed to read start frame {start}.")
        prev_gray = self._to_gray(prev_frame)

        h, w = prev_gray.shape[:2]

        # Current tracked points — (N, 1, 2) for OpenCV
        prev_pts = fmap.keypoints.reshape(-1, 1, 2).astype(np.float32)

        # Which features are still alive
        alive = np.ones(n_features, dtype=bool)

        lk_params = cfg.lk_params

        for t in range(1, total_frames):
            frame = src.read()
            if frame is None:
                # Video ended early — mark remaining as LOST
                status[t:] = TrackStatus.LOST
                break

            curr_gray = self._to_gray(frame)

            # ---- Forward flow ----
            next_pts, st_fwd, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_pts, None, **lk_params
            )

            if next_pts is None:
                status[t, alive] = TrackStatus.LOST
                alive[:] = False
                break

            st_fwd = st_fwd.ravel().astype(bool)

            # ---- Forward-backward consistency check ----
            if cfg.fb_threshold is not None:
                back_pts, st_bwd, _ = cv2.calcOpticalFlowPyrLK(
                    curr_gray, prev_gray, next_pts, None, **lk_params
                )
                st_bwd = st_bwd.ravel().astype(bool) if st_bwd is not None else np.zeros(n_features, dtype=bool)

                if back_pts is not None:
                    fb_err = np.linalg.norm(
                        prev_pts.reshape(-1, 2) - back_pts.reshape(-1, 2), axis=1
                    )
                    drift_mask = fb_err > cfg.fb_threshold
                else:
                    drift_mask = np.ones(n_features, dtype=bool)
            else:
                drift_mask = np.zeros(n_features, dtype=bool)
                st_bwd = np.ones(n_features, dtype=bool)

            # ---- Bounds check ----
            pts_2d = next_pts.reshape(-1, 2)
            oob_mask = (
                    (pts_2d[:, 0] < 0)
                    | (pts_2d[:, 0] >= w)
                    | (pts_2d[:, 1] < 0)
                    | (pts_2d[:, 1] >= h)
            )

            # ---- Assign status ----
            frame_status = np.full(n_features, TrackStatus.LOST, dtype=np.uint8)

            # Successfully tracked
            good = alive & st_fwd & ~oob_mask & ~drift_mask
            frame_status[good] = TrackStatus.TRACKED

            # Classify failure modes for alive features
            frame_status[alive & ~st_fwd] = TrackStatus.LOST
            frame_status[alive & st_fwd & oob_mask] = TrackStatus.OOB
            frame_status[alive & st_fwd & ~oob_mask & drift_mask] = TrackStatus.DRIFT

            # Dead features stay LOST
            frame_status[~alive] = TrackStatus.LOST

            # Store results
            positions[t, good] = pts_2d[good]
            status[t] = frame_status

            # Update alive mask — once lost, stay lost
            alive = alive & (frame_status == TrackStatus.TRACKED)

            # Advance for next iteration
            prev_gray = curr_gray
            prev_pts = next_pts  # Keep shape (N, 1, 2)

            # Progress
            if self.progress_callback is not None:
                self.progress_callback(t, total_frames)

            # Early exit if all features lost
            if not np.any(alive):
                status[t + 1 :] = TrackStatus.LOST
                break

        # Build timestamps
        timestamps = np.arange(total_frames, dtype=np.float64) / src.fps

        return TrackingResult(
            positions=positions,
            status=status,
            timestamps=timestamps,
            fps=src.fps,
            feature_map=fmap,
            tracker_config=cfg.to_dict(),
            frame_range=(start, start + total_frames),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        """Convert to grayscale if needed."""
        if frame.ndim == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def __repr__(self) -> str:
        return (
            f"Tracker(features={self.feature_map.n_features}, "
            f"start={self.start_frame}, "
            f"end={self.end_frame or 'eof'}, "
            f"fb={self.config.fb_threshold})"
        )