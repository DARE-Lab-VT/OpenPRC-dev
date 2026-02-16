"""
openprc.vision.detectors
========================

Feature detection backends with a unified interface.

Wraps OpenCV's feature detector implementations (SIFT, ORB, AKAZE)
behind a common API so downstream stages (tracking, caching) remain
detector-agnostic.

Dependencies:
    - opencv-python (cv2) >= 4.4  [SIFT is free since 4.4]
    - numpy

Location: openprc/vision/detectors.py

Example
-------
>>> from openprc.vision.source import VideoSource
>>> from openprc.vision.detectors import detect, SIFT, ORB
>>>
>>> src = VideoSource("experiment.mp4")
>>> ref_frame = src.read_frame(0)
>>>
>>> # Quick API — auto-selects SIFT
>>> fmap = detect(ref_frame, method="sift")
>>>
>>> # Explicit detector with custom params
>>> detector = SIFT(n_features=500, contrast_threshold=0.03)
>>> fmap = detector(ref_frame)
>>>
>>> print(fmap)
>>> print(fmap.keypoints.shape, fmap.descriptors.shape)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


# ======================================================================
# FeatureMap — the output object
# ======================================================================


@dataclass
class FeatureMap:
    """Container for detected features on a reference frame.

    This is the central object that gets cached as a sidecar file
    and consumed by the Tracker.

    Attributes
    ----------
    keypoints : np.ndarray
        (N, 2) array of keypoint coordinates (x, y) in pixel space.
    descriptors : np.ndarray
        (N, D) array of feature descriptors. D depends on detector
        (SIFT=128, ORB=32, AKAZE=variable).
    sizes : np.ndarray
        (N,) keypoint diameters (from cv2.KeyPoint.size).
    angles : np.ndarray
        (N,) keypoint orientations in degrees (from cv2.KeyPoint.angle).
    responses : np.ndarray
        (N,) detector response strength (from cv2.KeyPoint.response).
    octaves : np.ndarray
        (N,) octave/pyramid level (from cv2.KeyPoint.octave).
    detector_name : str
        Name of the detector that produced this map.
    detector_config : dict
        Parameters used for detection (for cache validation).
    ref_frame_index : int
        Index of the reference frame in the source video.
    frame_shape : tuple of (int, int)
        (height, width) of the reference frame.
    """

    keypoints: np.ndarray
    descriptors: np.ndarray
    sizes: np.ndarray
    angles: np.ndarray
    responses: np.ndarray
    octaves: np.ndarray
    detector_name: str = ""
    detector_config: dict = field(default_factory=dict)
    ref_frame_index: int = 0
    frame_shape: Tuple[int, int] = (0, 0)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_features(self) -> int:
        """Number of detected features."""
        return self.keypoints.shape[0]

    @property
    def descriptor_dim(self) -> int:
        """Dimensionality of each descriptor vector."""
        return self.descriptors.shape[1] if self.descriptors.ndim == 2 else 0

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def top_n(self, n: int) -> FeatureMap:
        """Return a new FeatureMap with only the top-n features by response.

        Parameters
        ----------
        n : int
            Number of features to keep.

        Returns
        -------
        FeatureMap
            Filtered copy.
        """
        if n >= self.n_features:
            return self
        idx = np.argsort(self.responses)[::-1][:n]
        return self._subset(idx)

    def filter_by_roi(
            self, x_min: float, y_min: float, x_max: float, y_max: float
    ) -> FeatureMap:
        """Keep only features inside a rectangular region of interest.

        Parameters
        ----------
        x_min, y_min, x_max, y_max : float
            Bounding box in pixel coordinates.

        Returns
        -------
        FeatureMap
            Filtered copy.
        """
        kp = self.keypoints
        mask = (
                (kp[:, 0] >= x_min)
                & (kp[:, 0] <= x_max)
                & (kp[:, 1] >= y_min)
                & (kp[:, 1] <= y_max)
        )
        return self._subset(np.where(mask)[0])

    def _subset(self, indices: np.ndarray) -> FeatureMap:
        """Create a new FeatureMap from a subset of indices."""
        return FeatureMap(
            keypoints=self.keypoints[indices],
            descriptors=self.descriptors[indices],
            sizes=self.sizes[indices],
            angles=self.angles[indices],
            responses=self.responses[indices],
            octaves=self.octaves[indices],
            detector_name=self.detector_name,
            detector_config=self.detector_config.copy(),
            ref_frame_index=self.ref_frame_index,
            frame_shape=self.frame_shape,
        )

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def to_cv2_keypoints(self) -> List[cv2.KeyPoint]:
        """Convert back to a list of cv2.KeyPoint objects.

        Useful when feeding features into OpenCV matchers or drawKeypoints.
        """
        kps = []
        for i in range(self.n_features):
            kp = cv2.KeyPoint(
                x=float(self.keypoints[i, 0]),
                y=float(self.keypoints[i, 1]),
                size=float(self.sizes[i]),
                angle=float(self.angles[i]),
                response=float(self.responses[i]),
                octave=int(self.octaves[i]),
            )
            kps.append(kp)
        return kps

    # ------------------------------------------------------------------
    # Serialization helpers (consumed by io.py / cache.py)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a dict of numpy arrays + metadata.

        Used by the HDF5 I/O layer.
        """
        return {
            "keypoints": self.keypoints,
            "descriptors": self.descriptors,
            "sizes": self.sizes,
            "angles": self.angles,
            "responses": self.responses,
            "octaves": self.octaves,
            "detector_name": self.detector_name,
            "detector_config": json.dumps(self.detector_config),
            "ref_frame_index": self.ref_frame_index,
            "frame_shape": np.array(self.frame_shape),
        }

    @classmethod
    def from_dict(cls, d: dict) -> FeatureMap:
        """Reconstruct from a dict (inverse of to_dict)."""
        config = d.get("detector_config", "{}")
        if isinstance(config, str):
            config = json.loads(config)

        shape = d.get("frame_shape", (0, 0))
        if isinstance(shape, np.ndarray):
            shape = tuple(shape.tolist())

        return cls(
            keypoints=np.asarray(d["keypoints"]),
            descriptors=np.asarray(d["descriptors"]),
            sizes=np.asarray(d["sizes"]),
            angles=np.asarray(d["angles"]),
            responses=np.asarray(d["responses"]),
            octaves=np.asarray(d["octaves"]),
            detector_name=str(d.get("detector_name", "")),
            detector_config=config,
            ref_frame_index=int(d.get("ref_frame_index", 0)),
            frame_shape=shape,
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"FeatureMap(detector={self.detector_name!r}, "
            f"n_features={self.n_features}, "
            f"desc_dim={self.descriptor_dim}, "
            f"frame={self.frame_shape})"
        )


# ======================================================================
# Base detector interface
# ======================================================================


class BaseDetector(ABC):
    """Abstract base class for feature detectors.

    Subclasses must implement ``_create_detector`` and ``name``.
    The ``__call__`` method handles grayscale conversion and
    keypoint → numpy conversion automatically.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier string (e.g. 'sift', 'orb')."""
        ...

    @abstractmethod
    def _create_detector(self) -> cv2.Feature2D:
        """Return a configured cv2 feature detector instance."""
        ...

    @abstractmethod
    def _get_config(self) -> dict:
        """Return detector parameters as a plain dict."""
        ...

    def __call__(
            self,
            frame: np.ndarray,
            mask: Optional[np.ndarray] = None,
            ref_frame_index: int = 0,
    ) -> FeatureMap:
        """Detect features on a single frame.

        Parameters
        ----------
        frame : np.ndarray
            Input image (BGR or grayscale).
        mask : np.ndarray, optional
            Binary mask — features only detected where mask > 0.
        ref_frame_index : int, optional
            Frame index for metadata. Default 0.

        Returns
        -------
        FeatureMap
            Detected features.
        """
        # Ensure grayscale for detection
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        h, w = gray.shape[:2]

        detector = self._create_detector()
        cv_kps, descs = detector.detectAndCompute(gray, mask)

        if cv_kps is None or len(cv_kps) == 0:
            # No features found — return empty FeatureMap
            return FeatureMap(
                keypoints=np.empty((0, 2), dtype=np.float32),
                descriptors=np.empty((0, 0), dtype=np.float32),
                sizes=np.empty(0, dtype=np.float32),
                angles=np.empty(0, dtype=np.float32),
                responses=np.empty(0, dtype=np.float32),
                octaves=np.empty(0, dtype=np.int32),
                detector_name=self.name,
                detector_config=self._get_config(),
                ref_frame_index=ref_frame_index,
                frame_shape=(h, w),
            )

        if descs is None:
            descs = np.empty((len(cv_kps), 0), dtype=np.float32)

        # Unpack cv2.KeyPoint list into numpy arrays
        n = len(cv_kps)
        keypoints = np.array([kp.pt for kp in cv_kps], dtype=np.float32)  # (N, 2)
        sizes = np.array([kp.size for kp in cv_kps], dtype=np.float32)
        angles = np.array([kp.angle for kp in cv_kps], dtype=np.float32)
        responses = np.array([kp.response for kp in cv_kps], dtype=np.float32)
        octaves = np.array([kp.octave for kp in cv_kps], dtype=np.int32)

        return FeatureMap(
            keypoints=keypoints,
            descriptors=descs,
            sizes=sizes,
            angles=angles,
            responses=responses,
            octaves=octaves,
            detector_name=self.name,
            detector_config=self._get_config(),
            ref_frame_index=ref_frame_index,
            frame_shape=(h, w),
        )


# ======================================================================
# Concrete detectors
# ======================================================================


class SIFT(BaseDetector):
    """Scale-Invariant Feature Transform detector.

    Parameters
    ----------
    n_features : int
        Maximum features to retain (0 = no limit). Default 0.
    n_octave_layers : int
        Layers per octave in the DoG pyramid. Default 3.
    contrast_threshold : float
        Filter out weak features. Default 0.04.
    edge_threshold : float
        Filter out edge-like features. Default 10.
    sigma : float
        Gaussian sigma for the first octave. Default 1.6.
    """

    def __init__(
            self,
            n_features: int = 0,
            n_octave_layers: int = 3,
            contrast_threshold: float = 0.04,
            edge_threshold: float = 10.0,
            sigma: float = 1.6,
    ):
        self.n_features = n_features
        self.n_octave_layers = n_octave_layers
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.sigma = sigma

    @property
    def name(self) -> str:
        return "sift"

    def _create_detector(self) -> cv2.Feature2D:
        return cv2.SIFT_create(
            nfeatures=self.n_features,
            nOctaveLayers=self.n_octave_layers,
            contrastThreshold=self.contrast_threshold,
            edgeThreshold=self.edge_threshold,
            sigma=self.sigma,
        )

    def _get_config(self) -> dict:
        return {
            "n_features": self.n_features,
            "n_octave_layers": self.n_octave_layers,
            "contrast_threshold": self.contrast_threshold,
            "edge_threshold": self.edge_threshold,
            "sigma": self.sigma,
        }


class ORB(BaseDetector):
    """Oriented FAST and Rotated BRIEF detector.

    Parameters
    ----------
    n_features : int
        Maximum features. Default 500.
    scale_factor : float
        Pyramid decimation ratio. Default 1.2.
    n_levels : int
        Pyramid levels. Default 8.
    edge_threshold : int
        Border exclusion size. Default 31.
    fast_threshold : int
        FAST corner threshold. Default 20.
    """

    def __init__(
            self,
            n_features: int = 500,
            scale_factor: float = 1.2,
            n_levels: int = 8,
            edge_threshold: int = 31,
            fast_threshold: int = 20,
    ):
        self.n_features = n_features
        self.scale_factor = scale_factor
        self.n_levels = n_levels
        self.edge_threshold = edge_threshold
        self.fast_threshold = fast_threshold

    @property
    def name(self) -> str:
        return "orb"

    def _create_detector(self) -> cv2.Feature2D:
        return cv2.ORB_create(
            nfeatures=self.n_features,
            scaleFactor=self.scale_factor,
            nlevels=self.n_levels,
            edgeThreshold=self.edge_threshold,
            fastThreshold=self.fast_threshold,
        )

    def _get_config(self) -> dict:
        return {
            "n_features": self.n_features,
            "scale_factor": self.scale_factor,
            "n_levels": self.n_levels,
            "edge_threshold": self.edge_threshold,
            "fast_threshold": self.fast_threshold,
        }


class AKAZE(BaseDetector):
    """Accelerated-KAZE detector with nonlinear scale space.

    Parameters
    ----------
    descriptor_type : int
        cv2.AKAZE_DESCRIPTOR_MLDB (default) or cv2.AKAZE_DESCRIPTOR_KAZE.
    descriptor_size : int
        Descriptor size (0 = full). Default 0.
    descriptor_channels : int
        Number of channels (1, 2, or 3). Default 3.
    threshold : float
        Detector response threshold. Default 0.001.
    n_octaves : int
        Maximum octave evolution. Default 4.
    n_octave_layers : int
        Sublevels per octave. Default 4.
    """

    def __init__(
            self,
            descriptor_type: int = cv2.AKAZE_DESCRIPTOR_MLDB,
            descriptor_size: int = 0,
            descriptor_channels: int = 3,
            threshold: float = 0.001,
            n_octaves: int = 4,
            n_octave_layers: int = 4,
    ):
        self.descriptor_type = descriptor_type
        self.descriptor_size = descriptor_size
        self.descriptor_channels = descriptor_channels
        self.threshold = threshold
        self.n_octaves = n_octaves
        self.n_octave_layers = n_octave_layers

    @property
    def name(self) -> str:
        return "akaze"

    def _create_detector(self) -> cv2.Feature2D:
        return cv2.AKAZE_create(
            descriptor_type=self.descriptor_type,
            descriptor_size=self.descriptor_size,
            descriptor_channels=self.descriptor_channels,
            threshold=self.threshold,
            nOctaves=self.n_octaves,
            nOctaveLayers=self.n_octave_layers,
        )

    def _get_config(self) -> dict:
        return {
            "descriptor_type": self.descriptor_type,
            "descriptor_size": self.descriptor_size,
            "descriptor_channels": self.descriptor_channels,
            "threshold": self.threshold,
            "n_octaves": self.n_octaves,
            "n_octave_layers": self.n_octave_layers,
        }


# ======================================================================
# Registry & convenience function
# ======================================================================

_DETECTOR_REGISTRY: Dict[str, type] = {
    "sift": SIFT,
    "orb": ORB,
    "akaze": AKAZE,
}


def get_detector(method: str, **kwargs) -> BaseDetector:
    """Look up a detector by name.

    Parameters
    ----------
    method : str
        One of 'sift', 'orb', 'akaze' (case-insensitive).
    **kwargs
        Passed to the detector constructor.

    Returns
    -------
    BaseDetector
        Configured detector instance.

    Raises
    ------
    ValueError
        If method is not recognized.
    """
    key = method.strip().lower()
    if key not in _DETECTOR_REGISTRY:
        available = ", ".join(sorted(_DETECTOR_REGISTRY))
        raise ValueError(
            f"Unknown detector '{method}'. Available: {available}"
        )
    return _DETECTOR_REGISTRY[key](**kwargs)


def detect(
        frame: np.ndarray,
        method: str = "sift",
        mask: Optional[np.ndarray] = None,
        ref_frame_index: int = 0,
        **kwargs,
) -> FeatureMap:
    """One-liner detection: create detector + run on frame.

    Parameters
    ----------
    frame : np.ndarray
        Input image (BGR or grayscale).
    method : str
        Detector name ('sift', 'orb', 'akaze'). Default 'sift'.
    mask : np.ndarray, optional
        Detection mask.
    ref_frame_index : int, optional
        Frame index metadata.
    **kwargs
        Forwarded to the detector constructor.

    Returns
    -------
    FeatureMap
        Detected features.

    Example
    -------
    >>> fmap = detect(frame, method="orb", n_features=1000)
    """
    detector = get_detector(method, **kwargs)
    return detector(frame, mask=mask, ref_frame_index=ref_frame_index)


def register_detector(name: str, cls: type) -> None:
    """Register a custom detector class.

    Parameters
    ----------
    name : str
        Lookup key (lowercase).
    cls : type
        Must be a subclass of BaseDetector.

    Example
    -------
    >>> class SuperPointDetector(BaseDetector):
    ...     ...
    >>> register_detector("superpoint", SuperPointDetector)
    """
    if not (isinstance(cls, type) and issubclass(cls, BaseDetector)):
        raise TypeError(f"{cls} must be a subclass of BaseDetector.")
    _DETECTOR_REGISTRY[name.strip().lower()] = cls