"""
openprc.vision.viz
==================

Visualization utilities for the vision pipeline.

Provides overlay drawing on video frames (features, trajectories,
status) and a simple annotated video writer. Uses OpenCV drawing
primitives — no heavy plotting dependencies required.

Dependencies:
    - opencv-python (cv2)
    - numpy

Location: openprc/vision/viz.py

Example
-------
>>> from openprc.vision import viz
>>>
>>> # Draw features on a frame
>>> annotated = viz.draw_features(frame, fmap, top_n=100)
>>>
>>> # Draw tracked points with trails
>>> annotated = viz.draw_tracks(frame, tset, frame_idx=50, trail=20)
>>>
>>> # Write full annotated video
>>> viz.write_annotated_video("annotated.mp4", src, tracking_result)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np


# ======================================================================
# Color palettes
# ======================================================================

def _generate_colors(n: int, seed: int = 42) -> np.ndarray:
    """Generate N visually distinct BGR colors via HSV spacing.

    Parameters
    ----------
    n : int
        Number of colors.
    seed : int
        Random seed for saturation/value jitter.

    Returns
    -------
    np.ndarray
        (N, 3) uint8 BGR colors.
    """
    rng = np.random.RandomState(seed)
    hues = np.linspace(0, 179, n, endpoint=False, dtype=np.uint8)
    sats = rng.randint(180, 255, size=n, dtype=np.uint8)
    vals = rng.randint(180, 255, size=n, dtype=np.uint8)

    hsv = np.stack([hues, sats, vals], axis=-1).reshape(1, n, 3).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr.reshape(n, 3)


# Default palette
_DEFAULT_COLORS = None


def _get_colors(n: int) -> np.ndarray:
    global _DEFAULT_COLORS
    if _DEFAULT_COLORS is None or len(_DEFAULT_COLORS) < n:
        _DEFAULT_COLORS = _generate_colors(max(n, 256))
    return _DEFAULT_COLORS[:n]


# ======================================================================
# Status colors
# ======================================================================

STATUS_COLORS = {
    0: (0, 255, 0),    # TRACKED — green
    1: (0, 0, 255),    # LOST — red
    2: (0, 165, 255),  # OOB — orange
    3: (0, 255, 255),  # DRIFT — yellow
    4: (255, 255, 0),  # INTERPOLATED — cyan
    5: (255, 0, 255),  # REFERENCE — magenta
}


# ======================================================================
# Frame-level drawing
# ======================================================================


def draw_features(
        frame: np.ndarray,
        feature_map,
        top_n: Optional[int] = None,
        color: Tuple[int, int, int] = (0, 255, 0),
        radius: int = 4,
        thickness: int = 1,
        draw_orientation: bool = True,
) -> np.ndarray:
    """Draw detected features on a frame.

    Parameters
    ----------
    frame : np.ndarray
        BGR image (will not be modified).
    feature_map : FeatureMap
        Detected features to draw.
    top_n : int or None
        Only draw top-N features by response. None draws all.
    color : tuple
        BGR color for keypoints.
    radius : int
        Circle radius in pixels.
    thickness : int
        Circle line thickness.
    draw_orientation : bool
        If True, draw orientation lines from keypoint center.

    Returns
    -------
    np.ndarray
        Annotated frame copy.
    """
    vis = frame.copy()
    fmap = feature_map.top_n(top_n) if top_n else feature_map

    for i in range(fmap.n_features):
        x, y = int(fmap.keypoints[i, 0]), int(fmap.keypoints[i, 1])
        cv2.circle(vis, (x, y), radius, color, thickness)

        if draw_orientation and fmap.angles[i] >= 0:
            angle_rad = np.deg2rad(fmap.angles[i])
            r = max(radius, int(fmap.sizes[i] * 0.5))
            ex = int(x + r * np.cos(angle_rad))
            ey = int(y + r * np.sin(angle_rad))
            cv2.line(vis, (x, y), (ex, ey), color, max(1, thickness))

    return vis


def draw_tracks(
        frame: np.ndarray,
        positions: np.ndarray,
        status: np.ndarray,
        frame_idx: int,
        trail: int = 15,
        radius: int = 5,
        trail_thickness: int = 2,
        color_by: str = "feature",
) -> np.ndarray:
    """Draw tracked points with trajectory trails on a frame.

    Parameters
    ----------
    frame : np.ndarray
        BGR image (will not be modified).
    positions : np.ndarray
        (T, N, 2) pixel coordinates.
    status : np.ndarray
        (T, N) TrackStatus codes.
    frame_idx : int
        Current frame index to draw.
    trail : int
        Number of past frames to show as trail. Default 15.
    radius : int
        Point radius.
    trail_thickness : int
        Trail line thickness.
    color_by : str
        'feature' — unique color per feature.
        'status' — color by tracking status.

    Returns
    -------
    np.ndarray
        Annotated frame copy.
    """
    vis = frame.copy()
    n_features = positions.shape[1]
    colors = _get_colors(n_features)

    trail_start = max(0, frame_idx - trail)

    for j in range(n_features):
        # Current point
        if frame_idx >= positions.shape[0]:
            continue

        st = status[frame_idx, j]
        x, y = positions[frame_idx, j]

        if np.isnan(x):
            continue

        # Pick color
        if color_by == "status":
            clr = STATUS_COLORS.get(int(st), (128, 128, 128))
        else:
            clr = tuple(int(c) for c in colors[j])

        # Draw trail
        trail_pts = []
        for t in range(trail_start, frame_idx + 1):
            tx, ty = positions[t, j]
            if not np.isnan(tx):
                trail_pts.append((int(tx), int(ty)))

        for k in range(1, len(trail_pts)):
            # Fade alpha via thickness (simple approximation)
            alpha = (k / len(trail_pts))
            t_clr = tuple(int(c * alpha) for c in clr)
            cv2.line(vis, trail_pts[k - 1], trail_pts[k], t_clr, trail_thickness)

        # Draw current point
        cv2.circle(vis, (int(x), int(y)), radius, clr, -1)

        # Feature ID label
        cv2.putText(
            vis,
            str(j),
            (int(x) + radius + 2, int(y) - radius),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            clr,
            1,
            cv2.LINE_AA,
        )

    return vis


def draw_status_bar(
        frame: np.ndarray,
        status: np.ndarray,
        frame_idx: int,
        bar_height: int = 20,
) -> np.ndarray:
    """Draw a horizontal status bar showing feature health.

    Each feature gets a colored pixel column — green if tracked,
    red if lost, etc.

    Parameters
    ----------
    frame : np.ndarray
        BGR image (will not be modified).
    status : np.ndarray
        (T, N) status codes.
    frame_idx : int
        Current frame index.
    bar_height : int
        Height of the bar in pixels.

    Returns
    -------
    np.ndarray
        Frame with status bar appended at bottom.
    """
    h, w = frame.shape[:2]
    n_features = status.shape[1]

    bar = np.zeros((bar_height, w, 3), dtype=np.uint8)
    col_width = max(1, w // max(n_features, 1))

    for j in range(n_features):
        st = int(status[frame_idx, j]) if frame_idx < status.shape[0] else 1
        clr = STATUS_COLORS.get(st, (128, 128, 128))
        x_start = j * col_width
        x_end = min(x_start + col_width, w)
        bar[:, x_start:x_end] = clr

    return np.vstack([frame, bar])


# ======================================================================
# Annotated video writer
# ======================================================================


def write_annotated_video(
        output_path: Union[str, Path],
        source,
        tracking_result=None,
        trajectory_set=None,
        feature_map=None,
        trail: int = 15,
        color_by: str = "feature",
        show_status_bar: bool = True,
        codec: str = "mp4v",
        progress_callback=None,
) -> Path:
    """Write a full annotated video with tracked features overlaid.

    Parameters
    ----------
    output_path : str or Path
        Output video file path.
    source : VideoSource
        Original video source.
    tracking_result : TrackingResult or None
        If provided, draws tracked points and trails.
    trajectory_set : TrajectorySet or None
        Alternative to tracking_result — uses pixel positions.
    feature_map : FeatureMap or None
        If provided (without tracking), draws static features on
        the reference frame only.
    trail : int
        Trail length in frames.
    color_by : str
        'feature' or 'status'.
    show_status_bar : bool
        Append a feature health bar at the bottom.
    codec : str
        FourCC codec string. Default 'mp4v'.
    progress_callback : callable or None
        Called with (current_frame, total_frames).

    Returns
    -------
    Path
        Output file path.
    """
    output_path = Path(output_path)

    # Determine positions and status arrays
    if tracking_result is not None:
        positions = tracking_result.positions
        statuses = tracking_result.status
        start_frame = tracking_result.frame_range[0]
        n_track_frames = tracking_result.n_frames
    elif trajectory_set is not None:
        positions = trajectory_set.positions_px
        statuses = trajectory_set.status
        start_frame = 0
        n_track_frames = trajectory_set.n_frames
    else:
        positions = None
        statuses = None
        start_frame = 0
        n_track_frames = source.frame_count

    # Setup writer
    w, h = source.resolution
    bar_h = 20 if show_status_bar and positions is not None else 0
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(
        str(output_path), fourcc, source.fps, (w, h + bar_h)
    )

    if not writer.isOpened():
        raise IOError(f"Failed to open video writer: {output_path}")

    source.reset()
    total = source.frame_count

    try:
        for i in range(total):
            frame = source.read()
            if frame is None:
                break

            # Feature map overlay (reference frame only)
            if feature_map is not None and i == feature_map.ref_frame_index:
                frame = draw_features(frame, feature_map, color=(0, 200, 255))

            # Tracking overlay
            track_idx = i - start_frame
            if positions is not None and 0 <= track_idx < n_track_frames:
                frame = draw_tracks(
                    frame, positions, statuses, track_idx,
                    trail=trail, color_by=color_by,
                )

                if show_status_bar:
                    frame = draw_status_bar(frame, statuses, track_idx, bar_h)
            elif bar_h > 0:
                # Pad with empty bar to keep resolution consistent
                pad = np.zeros((bar_h, w, 3), dtype=np.uint8)
                frame = np.vstack([frame, pad])

            writer.write(frame)

            if progress_callback is not None:
                progress_callback(i, total)

    finally:
        writer.release()

    return output_path


# ======================================================================
# Quick preview (display single frame — for notebooks / debugging)
# ======================================================================


def show(
        frame: np.ndarray,
        title: str = "openprc.vision",
        wait: int = 0,
) -> None:
    """Display a frame in an OpenCV window (blocking).

    Parameters
    ----------
    frame : np.ndarray
        BGR image.
    title : str
        Window title.
    wait : int
        Wait time in ms (0 = wait for keypress).
    """
    cv2.imshow(title, frame)
    cv2.waitKey(wait)
    cv2.destroyWindow(title)