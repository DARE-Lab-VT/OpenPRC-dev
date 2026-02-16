"""
openprc.vision.calibration
==========================

Camera calibration and coordinate transformation.

Handles the mapping from pixel coordinates to physical units.
When no camera calibration is available, provides a normalized
fallback that maps pixel coordinates to [0, 1] based on frame
dimensions — ensuring signals are always consistent and comparable.

Dependencies:
    - numpy
    - opencv-python (cv2) — only for undistortion

Location: openprc/vision/calibration.py

Example
-------
>>> from openprc.vision.calibration import CameraCalibration, NormalizedCalibration
>>>
>>> # With a real camera matrix
>>> cal = CameraCalibration.from_matrix(K, dist_coeffs=D, frame_shape=(1080, 1920))
>>> physical_pts = cal.transform(pixel_pts)  # (N, 2) in mm or physical units
>>>
>>> # Without calibration — normalized fallback
>>> cal = NormalizedCalibration(frame_shape=(1080, 1920))
>>> norm_pts = cal.transform(pixel_pts)  # (N, 2) in [0, 1]
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import cv2
import numpy as np


# ======================================================================
# Base interface
# ======================================================================


class BaseCalibration(ABC):
    """Abstract calibration interface.

    All calibration objects transform (N, 2) pixel coordinates
    into some output coordinate system and provide serialization
    for HDF5 storage.
    """

    @abstractmethod
    def transform(self, points: np.ndarray) -> np.ndarray:
        """Transform pixel coordinates to calibrated coordinates.

        Parameters
        ----------
        points : np.ndarray
            (N, 2) or (T, N, 2) array of (x, y) pixel coordinates.

        Returns
        -------
        np.ndarray
            Same shape, in calibrated coordinate system.
        """
        ...

    @abstractmethod
    def inverse(self, points: np.ndarray) -> np.ndarray:
        """Transform calibrated coordinates back to pixel space.

        Parameters
        ----------
        points : np.ndarray
            (N, 2) or (T, N, 2) array in calibrated coordinates.

        Returns
        -------
        np.ndarray
            Same shape, in pixel coordinates.
        """
        ...

    @property
    @abstractmethod
    def kind(self) -> str:
        """Calibration type identifier string."""
        ...

    @property
    @abstractmethod
    def units(self) -> str:
        """Unit label for the output coordinate system."""
        ...

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize to a dict for HDF5 storage."""
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict) -> BaseCalibration:
        """Reconstruct from a serialized dict."""
        ...


# ======================================================================
# Normalized calibration (default fallback)
# ======================================================================


class NormalizedCalibration(BaseCalibration):
    """Maps pixel coordinates to [0, 1] range based on frame dimensions.

    This is the default when no camera calibration is available.
    X is normalized by frame width, Y by frame height, giving
    consistent, resolution-independent signals.

    Parameters
    ----------
    frame_shape : tuple of (int, int)
        (height, width) of the video frames.
    """

    def __init__(self, frame_shape: Tuple[int, int]):
        self.frame_shape = frame_shape
        self._h = frame_shape[0]
        self._w = frame_shape[1]

        if self._h <= 0 or self._w <= 0:
            raise ValueError(f"Invalid frame shape: {frame_shape}")

    @property
    def kind(self) -> str:
        return "normalized"

    @property
    def units(self) -> str:
        return "normalized"

    def transform(self, points: np.ndarray) -> np.ndarray:
        """Normalize pixel coordinates to [0, 1].

        Parameters
        ----------
        points : np.ndarray
            (..., 2) array where last dimension is (x, y) in pixels.

        Returns
        -------
        np.ndarray
            Same shape with x / width and y / height.
        """
        out = np.empty_like(points, dtype=np.float64)
        out[..., 0] = points[..., 0] / self._w
        out[..., 1] = points[..., 1] / self._h
        return out

    def inverse(self, points: np.ndarray) -> np.ndarray:
        """Convert normalized coordinates back to pixels."""
        out = np.empty_like(points, dtype=np.float64)
        out[..., 0] = points[..., 0] * self._w
        out[..., 1] = points[..., 1] * self._h
        return out

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "frame_shape": list(self.frame_shape),
        }

    @classmethod
    def from_dict(cls, d: dict) -> NormalizedCalibration:
        return cls(frame_shape=tuple(d["frame_shape"]))

    def __repr__(self) -> str:
        return f"NormalizedCalibration(frame={self._h}x{self._w})"


# ======================================================================
# Camera matrix calibration
# ======================================================================


class CameraCalibration(BaseCalibration):
    """Pinhole camera model calibration.

    Transforms pixel coordinates to physical units using the camera
    intrinsic matrix. Optionally undistorts points using lens
    distortion coefficients.

    The physical coordinate is computed as::

        X_phys = (x_px - cx) / fx
        Y_phys = (y_px - cy) / fy

    This gives coordinates in camera-normalized units. If a
    ``scale_factor`` is provided (e.g., mm/unit from a known
    calibration target), the output is in physical units.

    Parameters
    ----------
    camera_matrix : np.ndarray
        (3, 3) intrinsic camera matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
    dist_coeffs : np.ndarray or None
        Distortion coefficients (k1, k2, p1, p2[, k3[, ...]]).
        None means no distortion correction.
    frame_shape : tuple of (int, int)
        (height, width) of the frames.
    scale_factor : float
        Multiplier to convert from camera-normalized units to physical
        units (e.g., working distance in mm). Default 1.0.
    physical_units : str
        Label for the output units. Default "camera_normalized".
    """

    def __init__(
            self,
            camera_matrix: np.ndarray,
            dist_coeffs: Optional[np.ndarray] = None,
            frame_shape: Tuple[int, int] = (0, 0),
            scale_factor: float = 1.0,
            physical_units: str = "camera_normalized",
    ):
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        if self.camera_matrix.shape != (3, 3):
            raise ValueError(f"Camera matrix must be (3,3), got {self.camera_matrix.shape}")

        self.dist_coeffs = (
            np.asarray(dist_coeffs, dtype=np.float64) if dist_coeffs is not None else None
        )
        self.frame_shape = frame_shape
        self.scale_factor = scale_factor
        self.physical_units = physical_units

        # Extract intrinsics
        self._fx = self.camera_matrix[0, 0]
        self._fy = self.camera_matrix[1, 1]
        self._cx = self.camera_matrix[0, 2]
        self._cy = self.camera_matrix[1, 2]

        if self._fx == 0 or self._fy == 0:
            raise ValueError("Focal lengths (fx, fy) must be nonzero.")

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_matrix(
            cls,
            K: np.ndarray,
            dist_coeffs: Optional[np.ndarray] = None,
            frame_shape: Tuple[int, int] = (0, 0),
            scale_factor: float = 1.0,
            physical_units: str = "camera_normalized",
    ) -> CameraCalibration:
        """Create from a (3, 3) intrinsic matrix.

        Parameters
        ----------
        K : np.ndarray
            (3, 3) intrinsic camera matrix.
        dist_coeffs : np.ndarray or None
            Lens distortion coefficients.
        frame_shape : tuple of (int, int)
            (height, width).
        scale_factor : float
            Physical scale multiplier.
        physical_units : str
            Output unit label.
        """
        return cls(K, dist_coeffs, frame_shape, scale_factor, physical_units)

    @classmethod
    def from_intrinsics(
            cls,
            fx: float,
            fy: float,
            cx: float,
            cy: float,
            dist_coeffs: Optional[np.ndarray] = None,
            frame_shape: Tuple[int, int] = (0, 0),
            scale_factor: float = 1.0,
            physical_units: str = "camera_normalized",
    ) -> CameraCalibration:
        """Create from individual intrinsic parameters.

        Parameters
        ----------
        fx, fy : float
            Focal lengths in pixels.
        cx, cy : float
            Principal point in pixels.
        """
        K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        return cls(K, dist_coeffs, frame_shape, scale_factor, physical_units)

    @classmethod
    def from_opencv_yaml(cls, path: str) -> CameraCalibration:
        """Load calibration from an OpenCV YAML/XML file.

        Expects keys: camera_matrix, distortion_coefficients,
        and optionally image_width, image_height.

        Parameters
        ----------
        path : str
            Path to the calibration file.
        """
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        K = fs.getNode("camera_matrix").mat()
        D = fs.getNode("distortion_coefficients").mat()
        w = int(fs.getNode("image_width").real()) if fs.getNode("image_width").isReal() else 0
        h = int(fs.getNode("image_height").real()) if fs.getNode("image_height").isReal() else 0
        fs.release()
        return cls(K, D, frame_shape=(h, w))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def kind(self) -> str:
        return "camera"

    @property
    def units(self) -> str:
        return self.physical_units

    @property
    def has_distortion(self) -> bool:
        return self.dist_coeffs is not None

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        """Remove lens distortion from pixel coordinates.

        Parameters
        ----------
        points : np.ndarray
            (..., 2) pixel coordinates.

        Returns
        -------
        np.ndarray
            Undistorted pixel coordinates (same shape).
        """
        if self.dist_coeffs is None:
            return points

        orig_shape = points.shape
        pts_flat = points.reshape(-1, 1, 2).astype(np.float64)

        undistorted = cv2.undistortPoints(
            pts_flat, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix
        )
        return undistorted.reshape(orig_shape)

    def transform(self, points: np.ndarray) -> np.ndarray:
        """Transform pixel coordinates to physical/normalized coordinates.

        Applies undistortion (if available), then converts from pixel
        space to camera-normalized coordinates scaled by scale_factor.

        Parameters
        ----------
        points : np.ndarray
            (..., 2) array of (x, y) pixel coordinates.

        Returns
        -------
        np.ndarray
            Same shape, in physical units.
        """
        if self.dist_coeffs is not None:
            pts = self.undistort_points(points)
        else:
            pts = np.asarray(points, dtype=np.float64)

        out = np.empty_like(pts, dtype=np.float64)
        out[..., 0] = (pts[..., 0] - self._cx) / self._fx * self.scale_factor
        out[..., 1] = (pts[..., 1] - self._cy) / self._fy * self.scale_factor
        return out

    def inverse(self, points: np.ndarray) -> np.ndarray:
        """Transform physical coordinates back to pixel space.

        Note: Does NOT re-apply distortion.

        Parameters
        ----------
        points : np.ndarray
            (..., 2) in physical units.

        Returns
        -------
        np.ndarray
            Pixel coordinates.
        """
        pts = np.asarray(points, dtype=np.float64)
        out = np.empty_like(pts, dtype=np.float64)
        out[..., 0] = pts[..., 0] / self.scale_factor * self._fx + self._cx
        out[..., 1] = pts[..., 1] / self.scale_factor * self._fy + self._cy
        return out

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        d = {
            "kind": self.kind,
            "camera_matrix": self.camera_matrix.tolist(),
            "frame_shape": list(self.frame_shape),
            "scale_factor": self.scale_factor,
            "physical_units": self.physical_units,
        }
        if self.dist_coeffs is not None:
            d["dist_coeffs"] = self.dist_coeffs.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> CameraCalibration:
        K = np.array(d["camera_matrix"], dtype=np.float64)
        D = np.array(d["dist_coeffs"], dtype=np.float64) if "dist_coeffs" in d else None
        return cls(
            camera_matrix=K,
            dist_coeffs=D,
            frame_shape=tuple(d.get("frame_shape", (0, 0))),
            scale_factor=d.get("scale_factor", 1.0),
            physical_units=d.get("physical_units", "camera_normalized"),
        )

    def __repr__(self) -> str:
        dist_str = "yes" if self.has_distortion else "no"
        return (
            f"CameraCalibration(fx={self._fx:.1f}, fy={self._fy:.1f}, "
            f"cx={self._cx:.1f}, cy={self._cy:.1f}, "
            f"distortion={dist_str}, "
            f"scale={self.scale_factor}, units={self.physical_units!r})"
        )


# ======================================================================
# Factory function
# ======================================================================


def auto_calibration(
        frame_shape: Tuple[int, int],
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        scale_factor: float = 1.0,
        physical_units: str = "camera_normalized",
) -> BaseCalibration:
    """Automatically choose the right calibration.

    Returns a CameraCalibration if a camera matrix is provided,
    otherwise falls back to NormalizedCalibration.

    Parameters
    ----------
    frame_shape : tuple of (int, int)
        (height, width) of video frames.
    camera_matrix : np.ndarray or None
        (3, 3) intrinsic matrix. None triggers normalized fallback.
    dist_coeffs : np.ndarray or None
        Lens distortion coefficients.
    scale_factor : float
        Physical scale multiplier (only used with camera_matrix).
    physical_units : str
        Output unit label (only used with camera_matrix).

    Returns
    -------
    BaseCalibration
        CameraCalibration or NormalizedCalibration.
    """
    if camera_matrix is not None:
        return CameraCalibration(
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            frame_shape=frame_shape,
            scale_factor=scale_factor,
            physical_units=physical_units,
        )
    return NormalizedCalibration(frame_shape=frame_shape)


def load_calibration(d: dict) -> BaseCalibration:
    """Reconstruct a calibration object from a serialized dict.

    Parameters
    ----------
    d : dict
        Must contain a 'kind' key ('normalized' or 'camera').

    Returns
    -------
    BaseCalibration
    """
    kind = d.get("kind", "")
    if kind == "normalized":
        return NormalizedCalibration.from_dict(d)
    elif kind == "camera":
        return CameraCalibration.from_dict(d)
    else:
        raise ValueError(f"Unknown calibration kind: {kind!r}")