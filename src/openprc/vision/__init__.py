"""
openprc.vision
==============

Video-based feature extraction for Physical Reservoir Computing.

Provides a complete pipeline from video input to calibrated timeseries
signals: detect robust features on a reference frame, track them
through the video via sparse optical flow, and export trajectories
as 1D/2D/3D signals compatible with the openprc analysis and
reservoir modules.

Modules
-------
source      : Video file / camera abstraction.
detectors   : Feature detection (SIFT, ORB, AKAZE).
tracker     : KLT sparse optical flow tracking.
calibration : Pixel-to-physical coordinate transforms.
trajectories: Signal extraction and trajectory processing.
cache       : Sidecar file caching for feature maps.
io          : HDF5 serialization for the full pipeline.

Quick Start
-----------
>>> import openprc.vision as vision
>>>
>>> # 1. Open video
>>> src = vision.VideoSource("experiment.mp4")
>>>
>>> # 2. Detect features (with sidecar caching)
>>> cache = vision.FeatureMapCache(src)
>>> ref_frame = src.read_frame(0)
>>> fmap = cache.get_or_detect(ref_frame, method="sift")
>>>
>>> # 3. Track through video
>>> src.reset()
>>> tracker = vision.Tracker(src, fmap)
>>> result = tracker.run()
>>>
>>> # 4. Extract calibrated signals
>>> tset = vision.TrajectorySet.from_tracking_result(result)
>>> tset = tset.fill_gaps("linear").smooth(window=5)
>>> signals = tset.to_signals(dim=2)  # (T, N, 2) normalized
>>>
>>> # 5. Save everything
>>> vision.save("experiment.h5", source=src, feature_map=fmap,
...             tracking_result=result, trajectory_set=tset,
...             signals=signals, signals_meta={"dim": 2, "units": "normalized"})
>>>
>>> src.release()

Location: openprc/vision/__init__.py
"""

# --- Core classes (most commonly used) ---
from .source import VideoSource
from .detectors import (
    FeatureMap,
    BaseDetector,
    SIFT,
    ORB,
    AKAZE,
    detect,
    get_detector,
    register_detector,
)
from .tracker import (
    Tracker,
    TrackingResult,
    TrackStatus,
    KLTConfig,
)
from .calibration import (
    BaseCalibration,
    NormalizedCalibration,
    CameraCalibration,
    auto_calibration,
    load_calibration,
)
from .trajectories import (
    TrajectorySet,
    GapStrategy,
    Axis1D,
)
from .cache import FeatureMapCache

# --- I/O functions ---
from .io import save, load, info

# --- Visualization (imported as submodule, not star-imported) ---
from . import viz

__all__ = [
    # Source
    "VideoSource",
    # Detectors
    "FeatureMap",
    "BaseDetector",
    "SIFT",
    "ORB",
    "AKAZE",
    "detect",
    "get_detector",
    "register_detector",
    # Tracker
    "Tracker",
    "TrackingResult",
    "TrackStatus",
    "KLTConfig",
    # Calibration
    "BaseCalibration",
    "NormalizedCalibration",
    "CameraCalibration",
    "auto_calibration",
    "load_calibration",
    # Trajectories
    "TrajectorySet",
    "GapStrategy",
    "Axis1D",
    # Cache
    "FeatureMapCache",
    # I/O
    "save",
    "load",
    "info",
    # Visualization
    "viz",
]

__version__ = "0.1.0"