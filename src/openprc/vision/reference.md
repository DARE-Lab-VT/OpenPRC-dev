# openprc.vision — API Reference

**Video-based feature extraction for Physical Reservoir Computing**

Version 0.1.0

---

## Installation

```bash
pip install opencv-python numpy h5py scipy
```

---

## Quick Start

```python
import openprc.vision as vision

# Open → Detect → Track → Extract → Save
src = vision.VideoSource("experiment.mp4")
ref = src.read_frame(0)

cache = vision.FeatureMapCache(src)
fmap = cache.get_or_detect(ref, method="sift")

src.reset()
result = vision.Tracker(src, fmap).run()

tset = vision.TrajectorySet.from_tracking_result(result)
tset = tset.fill_gaps("linear").smooth(window=5)
signals = tset.to_signals(dim=2)

vision.save("output.h5", source=src, feature_map=fmap,
            tracking_result=result, trajectory_set=tset,
            signals=signals)
src.release()
```

---

## Module: `source`

### `VideoSource`

Unified video source for files and live cameras.

```python
from openprc.vision import VideoSource
```

**Constructor**

| Parameter   | Type              | Default | Description                                      |
|-------------|-------------------|---------|--------------------------------------------------|
| `source`    | `str, Path, int`  | —       | File path to video, or camera index (e.g. `0`)   |
| `grayscale` | `bool`            | `False` | Convert frames to grayscale on read              |

**Attributes**

| Attribute     | Type           | Description                              |
|---------------|----------------|------------------------------------------|
| `fps`         | `float`        | Frames per second                        |
| `resolution`  | `(int, int)`   | `(width, height)` in pixels              |
| `frame_count` | `int`          | Total frames (`-1` for live cameras)     |
| `is_live`     | `bool`         | `True` if source is a camera             |
| `path`        | `Path or None` | File path (`None` for cameras)           |

**Properties**

| Property       | Returns  | Description                                           |
|----------------|----------|-------------------------------------------------------|
| `content_hash` | `str`    | SHA-256 hex digest of file (lazy, cached). Raises `RuntimeError` for live sources. |
| `position`     | `int`    | Current frame position (zero-based)                   |

**Methods**

```python
# Sequential read
frame = src.read()                 # → np.ndarray or None

# Random access (file only)
frame = src.read_frame(index=50)   # → np.ndarray or None

# Navigation
src.seek(100)                      # Jump to frame 100 without reading
src.reset()                        # Rewind to frame 0

# Metadata for HDF5
meta = src.metadata()              # → dict

# Cleanup
src.release()
```

**Iteration & Context Manager**

```python
# Iterate all frames
for frame in src:
    process(frame)

# Auto-release
with VideoSource("video.mp4") as src:
    frame = src.read()

# Length
n = len(src)    # Raises TypeError for live cameras
```

---

## Module: `detectors`

### Detector Classes

Three built-in detectors, all subclasses of `BaseDetector`.

```python
from openprc.vision import SIFT, ORB, AKAZE
```

#### `SIFT`

| Parameter             | Type    | Default | Description                     |
|-----------------------|---------|---------|---------------------------------|
| `n_features`          | `int`   | `0`     | Max features (0 = no limit)     |
| `n_octave_layers`     | `int`   | `3`     | Layers per DoG octave           |
| `contrast_threshold`  | `float` | `0.04`  | Weak feature filter             |
| `edge_threshold`      | `float` | `10.0`  | Edge-like feature filter        |
| `sigma`               | `float` | `1.6`   | Gaussian sigma (first octave)   |

Descriptor dimensionality: **128**

#### `ORB`

| Parameter        | Type    | Default | Description                 |
|------------------|---------|---------|-----------------------------|
| `n_features`     | `int`   | `500`   | Max features                |
| `scale_factor`   | `float` | `1.2`   | Pyramid decimation ratio    |
| `n_levels`       | `int`   | `8`     | Pyramid levels              |
| `edge_threshold` | `int`   | `31`    | Border exclusion size       |
| `fast_threshold` | `int`   | `20`    | FAST corner threshold       |

Descriptor dimensionality: **32**

#### `AKAZE`

| Parameter              | Type    | Default                    | Description                |
|------------------------|---------|----------------------------|----------------------------|
| `descriptor_type`      | `int`   | `cv2.AKAZE_DESCRIPTOR_MLDB`| MLDB or KAZE descriptor    |
| `descriptor_size`      | `int`   | `0`                        | Size (0 = full)            |
| `descriptor_channels`  | `int`   | `3`                        | Channels (1, 2, or 3)      |
| `threshold`            | `float` | `0.001`                    | Response threshold          |
| `n_octaves`            | `int`   | `4`                        | Maximum octave evolution    |
| `n_octave_layers`      | `int`   | `4`                        | Sublevels per octave        |

Descriptor dimensionality: **variable** (depends on config)

#### Calling a Detector

All detectors are callable:

```python
detector = SIFT(n_features=500)
fmap = detector(frame, mask=None, ref_frame_index=0)
```

### Convenience Functions

```python
from openprc.vision import detect, get_detector, register_detector

# One-liner detection
fmap = detect(frame, method="sift", contrast_threshold=0.03)

# Get a reusable detector instance
detector = get_detector("orb", n_features=1000)

# Register a custom detector
register_detector("superpoint", MyCustomDetector)
```

| Function            | Description                                                |
|---------------------|------------------------------------------------------------|
| `detect(frame, method, mask, ref_frame_index, **kwargs)` | Create detector + run on frame |
| `get_detector(method, **kwargs)` | Look up detector by name, return configured instance |
| `register_detector(name, cls)`   | Add custom detector to the registry                 |

**Available methods:** `"sift"`, `"orb"`, `"akaze"` (case-insensitive)

### `FeatureMap`

Container for detected features. Returned by all detectors.

```python
from openprc.vision import FeatureMap
```

**Attributes**

| Attribute        | Shape / Type  | Description                                |
|------------------|---------------|--------------------------------------------|
| `keypoints`      | `(N, 2)`      | `float32` — (x, y) pixel coordinates       |
| `descriptors`    | `(N, D)`      | Feature descriptors (D varies by detector)  |
| `sizes`          | `(N,)`        | `float32` — keypoint diameters              |
| `angles`         | `(N,)`        | `float32` — orientations in degrees         |
| `responses`      | `(N,)`        | `float32` — detector response strength      |
| `octaves`        | `(N,)`        | `int32` — pyramid level                     |
| `detector_name`  | `str`         | e.g. `"sift"`                              |
| `detector_config`| `dict`        | Parameters used for detection               |
| `ref_frame_index`| `int`         | Reference frame index in source video       |
| `frame_shape`    | `(int, int)`  | `(height, width)` of reference frame        |

**Properties**

| Property         | Returns | Description                       |
|------------------|---------|-----------------------------------|
| `n_features`     | `int`   | Number of detected features       |
| `descriptor_dim` | `int`   | Dimensionality of descriptors     |

**Methods**

```python
# Filter by response strength
fmap_top = fmap.top_n(200)

# Filter by region of interest
fmap_roi = fmap.filter_by_roi(x_min=100, y_min=50, x_max=500, y_max=400)

# Convert back to OpenCV keypoints
cv_kps = fmap.to_cv2_keypoints()

# Serialization
d = fmap.to_dict()                # → dict of arrays + metadata
fmap2 = FeatureMap.from_dict(d)   # Reconstruct
```

---

## Module: `tracker`

### `TrackStatus`

Integer codes for per-feature, per-frame state.

```python
from openprc.vision import TrackStatus
```

| Constant        | Value | Meaning                                   |
|-----------------|-------|-------------------------------------------|
| `TRACKED`       | `0`   | Successfully tracked                      |
| `LOST`          | `1`   | KLT lost the feature                      |
| `OOB`           | `2`   | Moved outside frame bounds                |
| `DRIFT`         | `3`   | Failed forward-backward consistency check |
| `INTERPOLATED`  | `4`   | Gap-filled by post-processing             |
| `REFERENCE`     | `5`   | Initial reference frame position          |

### `KLTConfig`

```python
from openprc.vision import KLTConfig
```

| Parameter              | Type                | Default    | Description                              |
|------------------------|---------------------|------------|------------------------------------------|
| `win_size`             | `(int, int)`        | `(21, 21)` | Search window per pyramid level          |
| `max_level`            | `int`               | `3`        | Pyramid levels (0 = none)                |
| `max_iter`             | `int`               | `30`       | Convergence iterations per level         |
| `epsilon`              | `float`             | `0.01`     | Convergence threshold                    |
| `fb_threshold`         | `float or None`     | `1.0`      | Forward-backward error limit (px). `None` disables. |
| `min_eigen_threshold`  | `float`             | `1e-4`     | Minimum eigenvalue for LK               |

**Tuning guide:**

- Fast motion → increase `win_size` (e.g. `(41, 41)`) and `max_level` (e.g. `5`)
- Noisy video → increase `fb_threshold` (e.g. `2.0–3.0`) or set `None`
- Sub-pixel precision needed → decrease `epsilon` (e.g. `0.001`)

### `Tracker`

```python
from openprc.vision import Tracker
```

**Constructor**

| Parameter           | Type                   | Default     | Description                          |
|---------------------|------------------------|-------------|--------------------------------------|
| `source`            | `VideoSource`          | —           | Video to track through               |
| `feature_map`       | `FeatureMap`           | —           | Seed features                        |
| `config`            | `KLTConfig or None`    | `KLTConfig()` | Tracking parameters                |
| `start_frame`       | `int or None`          | `None`      | Start frame (`None` = fmap reference)|
| `end_frame`         | `int or None`          | `None`      | End frame exclusive (`None` = EOF)   |
| `progress_callback` | `callable or None`     | `None`      | Called with `(current, total)`       |

**Methods**

```python
result = tracker.run()    # → TrackingResult
```

### `TrackingResult`

Returned by `Tracker.run()`.

**Attributes**

| Attribute        | Shape / Type   | Description                                   |
|------------------|----------------|-----------------------------------------------|
| `positions`      | `(T, N, 2)`   | `float32` — pixel coordinates. `NaN` where not tracked. |
| `status`         | `(T, N)`       | `uint8` — `TrackStatus` codes                 |
| `timestamps`     | `(T,)`         | `float64` — time in seconds                   |
| `fps`            | `float`        | Source framerate                               |
| `feature_map`    | `FeatureMap`   | Seed feature map                               |
| `tracker_config` | `dict`         | KLT parameters used                           |
| `frame_range`    | `(int, int)`   | `(start_frame, end_frame)` indices             |

**Properties & Methods**

```python
result.n_frames       # int — number of frames tracked
result.n_features     # int — number of features
result.duration       # float — duration in seconds

# Quality metrics
lengths = result.track_lengths()    # (N,) — frames tracked per feature
ratios  = result.track_ratios()     # (N,) — fraction of total frames

# Filter by quality
good = result.filter(min_track_ratio=0.5)   # → TrackingResult
```

---

## Module: `calibration`

### `NormalizedCalibration`

Default fallback — maps pixels to `[0, 1]`.

```python
from openprc.vision import NormalizedCalibration

cal = NormalizedCalibration(frame_shape=(1080, 1920))
physical = cal.transform(pixel_pts)     # x/width, y/height
pixels   = cal.inverse(physical)
```

| Property | Value          |
|----------|----------------|
| `kind`   | `"normalized"` |
| `units`  | `"normalized"` |

### `CameraCalibration`

Full pinhole model with optional lens undistortion.

```python
from openprc.vision import CameraCalibration
```

**Constructors**

```python
# From 3x3 intrinsic matrix
cal = CameraCalibration.from_matrix(
    K,                              # (3, 3) intrinsic matrix
    dist_coeffs=D,                  # optional distortion coefficients
    frame_shape=(1080, 1920),
    scale_factor=25.0,              # camera-normalized → physical units
    physical_units="mm",
)

# From individual parameters
cal = CameraCalibration.from_intrinsics(
    fx=800.0, fy=800.0,
    cx=960.0, cy=540.0,
    dist_coeffs=D,
    frame_shape=(1080, 1920),
)

# From OpenCV YAML/XML calibration file
cal = CameraCalibration.from_opencv_yaml("calibration.yaml")
```

**Methods**

```python
physical = cal.transform(pixel_pts)           # pixels → physical units
pixels   = cal.inverse(physical_pts)          # physical → pixels (no re-distortion)
undist   = cal.undistort_points(pixel_pts)    # remove lens distortion only
```

| Property         | Description                            |
|------------------|----------------------------------------|
| `kind`           | `"camera"`                             |
| `units`          | Value of `physical_units` parameter    |
| `has_distortion` | `True` if `dist_coeffs` was provided   |

### Factory Functions

```python
from openprc.vision import auto_calibration, load_calibration

# Auto-select: camera matrix → CameraCalibration, None → Normalized
cal = auto_calibration(
    frame_shape=(1080, 1920),
    camera_matrix=K,          # or None for normalized fallback
    dist_coeffs=D,
    scale_factor=25.0,
    physical_units="mm",
)

# Reconstruct from serialized dict (used by io.py)
cal = load_calibration({"kind": "normalized", "frame_shape": [1080, 1920]})
```

---

## Module: `trajectories`

### `TrajectorySet`

Calibrated feature trajectories with signal export.

```python
from openprc.vision import TrajectorySet
```

**Construction**

```python
# From tracking result (most common)
tset = TrajectorySet.from_tracking_result(
    result,                     # TrackingResult
    calibration=None,           # BaseCalibration or None (→ normalized)
)
```

**Attributes**

| Attribute      | Shape / Type       | Description                              |
|----------------|--------------------|------------------------------------------|
| `positions`    | `(T, N, 2)`       | `float64` — calibrated coordinates       |
| `positions_px` | `(T, N, 2)`       | `float32` — original pixel coordinates   |
| `status`       | `(T, N)`           | `uint8` — TrackStatus codes              |
| `timestamps`   | `(T,)`             | `float64` — time in seconds              |
| `fps`          | `float`            | Source framerate                          |
| `calibration`  | `BaseCalibration`  | Calibration used                         |
| `feature_ids`  | `(N,)`             | `int32` — feature identifiers            |

**Properties**

| Property      | Returns  | Description                |
|---------------|----------|----------------------------|
| `n_frames`    | `int`    | Number of frames           |
| `n_features`  | `int`    | Number of features         |
| `duration`    | `float`  | Total duration in seconds  |
| `units`       | `str`    | Coordinate unit label      |

### Gap Handling

```python
from openprc.vision import GapStrategy

tset_filled = tset.fill_gaps(
    strategy="linear",    # "nan", "linear", "zero", "hold"
    max_gap=10,           # max frames to interpolate (None = all)
)
```

| Strategy   | Behavior                                  |
|------------|-------------------------------------------|
| `"nan"`    | Leave NaN gaps untouched                  |
| `"linear"` | Linear interpolation across gaps         |
| `"zero"`   | Fill with zero                            |
| `"hold"`   | Forward-fill last known value             |

### Smoothing

```python
tset_smooth = tset.smooth(window=5)    # Moving average, NaN-aware
```

### Signal Extraction

```python
from openprc.vision import Axis1D

# 2D: (T, N, 2) — x, y coordinates
signals = tset.to_signals(dim=2)

# 1D: (T, N) — scalar per feature
signals = tset.to_signals(dim=1, axis="x")           # x-component
signals = tset.to_signals(dim=1, axis="y")           # y-component
signals = tset.to_signals(dim=1, axis="magnitude")   # displacement from start
signals = tset.to_signals(dim=1, axis="distance")    # cumulative path distance

# 3D: (T, N, 3) — x, y, displacement magnitude
signals = tset.to_signals(dim=3)
```

### Derived Signals

```python
vel = tset.velocities()    # (T, N, 2) — calibrated units/sec
spd = tset.speeds()        # (T, N) — scalar speed
```

### Filtering

```python
# Select specific features
tset_sub = tset.select_features([0, 3, 7, 12])

# Trim time window
tset_cut = tset.trim(start_time=2.0, end_time=8.5)
```

### Serialization

```python
d = tset.to_dict()    # → dict of arrays + calibration + metadata
```

---

## Module: `cache`

### `FeatureMapCache`

Sidecar HDF5 cache for detected feature maps.

Cache files are named `<video_name>.fmap.h5` and placed next to the video.

```python
from openprc.vision import FeatureMapCache
```

**Constructor**

| Parameter   | Type                | Default | Description                              |
|-------------|---------------------|---------|------------------------------------------|
| `source`    | `VideoSource`       | —       | Video source (file only, not live)       |
| `cache_dir` | `str, Path or None` | `None`  | Override directory. `None` = beside video.|

**Methods**

```python
# Save a feature map
path = cache.save(fmap)

# Load (returns None on miss or validation failure)
fmap = cache.load(
    detector_name="sift",
    detector_config=None,       # dict to match, or None to skip
    validate_hash=True,         # verify video hasn't changed
)

# One-call: load from cache or detect + cache
fmap = cache.get_or_detect(
    frame,
    method="sift",
    ref_frame_index=0,
    validate_hash=True,
    **detector_kwargs,
)

# Management
detectors = cache.list_cached()          # → ["sift", "orb"]
cache.invalidate("sift")                 # Remove one detector
cache.invalidate()                       # Delete entire cache file
```

**Properties**

| Property     | Returns | Description                    |
|--------------|---------|--------------------------------|
| `cache_path` | `Path`  | Path to the sidecar file       |
| `exists`     | `bool`  | Whether cache file exists      |

---

## Module: `io`

HDF5 serialization for the full pipeline.

```python
from openprc.vision import save, load, info
```

### `save()`

```python
save(
    path,                           # str or Path — output .h5 file
    source=None,                    # VideoSource
    feature_map=None,               # FeatureMap
    tracking_result=None,           # TrackingResult
    trajectory_set=None,            # TrajectorySet
    signals=None,                   # np.ndarray (T, N, d)
    signals_meta=None,              # dict — dim, units, description
    mode="a",                       # HDF5 mode: "a" (append) or "w" (overwrite)
)
```

Any combination of components can be provided. Existing subgroups are overwritten.

### `load()`

```python
data = load("experiment.h5")    # → dict
```

**Returned keys** (present only if stored):

| Key                    | Type             | Description                   |
|------------------------|------------------|-------------------------------|
| `source_meta`          | `dict`           | VideoSource metadata          |
| `feature_map`          | `FeatureMap`     | Detected features             |
| `tracking_positions`   | `(T, N, 2)`     | Raw tracking positions        |
| `tracking_status`      | `(T, N)`         | Raw tracking status           |
| `tracking_timestamps`  | `(T,)`           | Raw timestamps                |
| `tracking_fps`         | `float`          | Framerate                     |
| `tracking_config`      | `dict`           | Tracker config                |
| `trajectory_set`       | `TrajectorySet`  | Fully reconstructed object    |
| `signals`              | `np.ndarray`     | Exported signal array         |
| `signals_meta`         | `dict`           | Signal metadata               |

### `info()`

```python
summary = info("experiment.h5")    # → dict
```

Quick summary without loading arrays. Returns keys like `has_vision_data`, `n_features`, `detector`, `tracking_frames`, `traj_frames`, `signal_shape`.

### HDF5 Schema

```
<file>.h5
└── /vision/
    ├── /source/              attrs: source_type, path, fps, width, height, ...
    ├── /feature_map/
    │   ├── keypoints         (N, 2)     float32, gzip
    │   ├── descriptors       (N, D)     float32, gzip
    │   ├── sizes             (N,)       float32, gzip
    │   ├── angles            (N,)       float32, gzip
    │   ├── responses         (N,)       float32, gzip
    │   ├── octaves           (N,)       int32, gzip
    │   └── attrs: detector_name, detector_config, ref_frame_index, frame_shape
    ├── /tracking/
    │   ├── positions         (T, N, 2)  float32, gzip
    │   ├── status            (T, N)     uint8, gzip
    │   ├── timestamps        (T,)       float64
    │   └── attrs: fps, frame_range, tracker_config
    ├── /trajectories/
    │   ├── positions         (T, N, 2)  float64, gzip
    │   ├── positions_px      (T, N, 2)  float32, gzip
    │   ├── status            (T, N)     uint8, gzip
    │   ├── timestamps        (T,)       float64
    │   ├── feature_ids       (N,)       int32
    │   └── attrs: fps, calibration (JSON), metadata (JSON)
    └── /signals/
        ├── data              (T, N, d)  gzip
        └── attrs: dim, units, description, ...
```

---

## Module: `viz`

Visualization utilities for overlaying features and tracks on video.

```python
from openprc.vision import viz
```

### Frame Drawing

```python
# Draw detected features
annotated = viz.draw_features(
    frame,                      # BGR image (not modified)
    feature_map,                # FeatureMap
    top_n=None,                 # int or None — limit by response
    color=(0, 255, 0),          # BGR color
    radius=4,                   # circle radius (px)
    thickness=1,                # line thickness
    draw_orientation=True,      # show angle lines
)

# Draw tracked points with trajectory trails
annotated = viz.draw_tracks(
    frame,                      # BGR image
    positions,                  # (T, N, 2) pixel coords
    status,                     # (T, N) TrackStatus codes
    frame_idx,                  # current frame index
    trail=15,                   # past frames to draw as trail
    radius=5,                   # point radius
    trail_thickness=2,
    color_by="feature",         # "feature" (unique colors) or "status"
)

# Draw feature health bar
annotated = viz.draw_status_bar(
    frame,                      # BGR image
    status,                     # (T, N) status codes
    frame_idx,                  # current frame
    bar_height=20,              # pixels
)
```

**Status colors:**

| Status         | Color    |
|----------------|----------|
| `TRACKED`      | Green    |
| `LOST`         | Red      |
| `OOB`          | Orange   |
| `DRIFT`        | Yellow   |
| `INTERPOLATED` | Cyan     |
| `REFERENCE`    | Magenta  |

### Annotated Video Writer

```python
output_path = viz.write_annotated_video(
    "annotated.mp4",
    source,                         # VideoSource
    tracking_result=result,         # TrackingResult (or None)
    trajectory_set=None,            # TrajectorySet alternative
    feature_map=fmap,               # FeatureMap (drawn on ref frame)
    trail=15,
    color_by="feature",             # or "status"
    show_status_bar=True,
    codec="mp4v",                   # FourCC codec
    progress_callback=None,         # callable(current, total)
)
```

### Quick Preview

```python
viz.show(frame, title="openprc.vision", wait=0)    # OpenCV window
```

---

## Complete File Layout

```
openprc/vision/
├── __init__.py          Public API — all classes importable from here
├── source.py            VideoSource
├── detectors.py         SIFT, ORB, AKAZE, FeatureMap, BaseDetector
├── tracker.py           Tracker, TrackingResult, TrackStatus, KLTConfig
├── calibration.py       CameraCalibration, NormalizedCalibration
├── trajectories.py      TrajectorySet, GapStrategy, Axis1D
├── cache.py             FeatureMapCache
├── io.py                save(), load(), info()
└── viz.py               draw_features(), draw_tracks(), write_annotated_video()
```

## Dependencies

| Package          | Required | Used by                         |
|------------------|----------|---------------------------------|
| `opencv-python`  | Yes      | All modules                     |
| `numpy`          | Yes      | All modules                     |
| `h5py`           | Yes      | `io`, `cache`                   |
| `scipy`          | No       | `trajectories` (interpolation, smoothing). Falls back to numpy. |

---

## Extension: Adding Custom Detectors

```python
from openprc.vision.detectors import BaseDetector, register_detector

class SuperPointDetector(BaseDetector):

    @property
    def name(self) -> str:
        return "superpoint"

    def _create_detector(self):
        # Return any object with detectAndCompute(gray, mask) interface
        ...

    def _get_config(self) -> dict:
        return {"model": "superpoint_v1", "threshold": 0.015}

register_detector("superpoint", SuperPointDetector)

# Now usable everywhere
fmap = detect(frame, method="superpoint")
```