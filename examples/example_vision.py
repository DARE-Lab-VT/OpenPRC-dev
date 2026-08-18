"""
example_pipeline.py
===================

Complete openprc.vision pipeline example.

Demonstrates the full workflow: video ingestion → feature detection
(with sidecar caching) → KLT tracking → trajectory extraction →
signal export → HDF5 serialization → annotated video output.

Usage:
    python example_pipeline.py experiment.mp4

Dependencies:
    pip install opencv-python numpy h5py scipy
"""

import sys
from pathlib import Path

import numpy as np

import openprc.vision as vision
from openprc.vision import viz


def main(video_path: str):

    print("=" * 60)
    print("  openprc.vision — Full Pipeline Example")
    print("=" * 60)

    output_dir = Path(video_path).parent / "vision_output"
    output_dir.mkdir(exist_ok=True)

    # ==================================================================
    # 1. Open video source
    # ==================================================================

    print("\n[1/7] Opening video source...")
    src = vision.VideoSource(video_path)
    print(f"  {src}")
    print(f"  Content hash: {src.content_hash[:16]}...")

    # ==================================================================
    # 2. Detect features (with sidecar caching)
    # ==================================================================

    print("\n[2/7] Detecting features...")
    cache = vision.FeatureMapCache(src)

    ref_frame = src.read_frame(0)
    if ref_frame is None:
        print("  ERROR: Could not read first frame.")
        return

    # First run detects + caches. Subsequent runs load from cache.
    fmap = cache.get_or_detect(
        ref_frame,
        method="sift",
        ref_frame_index=0,
        contrast_threshold=0.04,
    )
    print(f"  {fmap}")
    print(f"  Cache: {cache}")

    # Optional: keep only top 200 features by response strength
    if fmap.n_features > 200:
        fmap = fmap.top_n(200)
        print(f"  Filtered to top 200: {fmap.n_features} features")

    # Save annotated reference frame
    ref_annotated = viz.draw_features(ref_frame, fmap, radius=5)
    ref_path = output_dir / "reference_features.png"
    import cv2
    cv2.imwrite(str(ref_path), ref_annotated)
    print(f"  Saved reference frame → {ref_path}")

    # ==================================================================
    # 3. Track features through video
    # ==================================================================

    print("\n[3/7] Tracking features...")

    # Configure KLT tracker
    config = vision.KLTConfig(
        win_size=(21, 21),
        max_level=3,
        fb_threshold=1.0,   # Forward-backward consistency check
    )

    def progress(current, total):
        if current % 50 == 0 or current == total - 1:
            pct = (current + 1) / total * 100
            print(f"  Frame {current + 1}/{total} ({pct:.0f}%)")

    src.reset()
    tracker = vision.Tracker(
        source=src,
        feature_map=fmap,
        config=config,
        progress_callback=progress,
    )
    result = tracker.run()
    print(f"  {result}")

    # ==================================================================
    # 4. Build calibrated trajectory set
    # ==================================================================

    print("\n[4/7] Building trajectory set...")

    # No camera matrix → normalized coordinates [0, 1]
    tset = vision.TrajectorySet.from_tracking_result(result, calibration=None)
    print(f"  Raw: {tset}")

    # Filter: keep features tracked for at least 60% of frames
    good_mask = result.track_ratios() >= 0.6
    good_idx = np.where(good_mask)[0]
    if len(good_idx) < tset.n_features:
        tset = tset.select_features(good_idx)
        print(f"  After quality filter (>60%%): {tset.n_features} features")

    # Fill small gaps with linear interpolation (max 10 frame gaps)
    tset = tset.fill_gaps(strategy="linear", max_gap=10)
    print("  Gaps filled (linear, max_gap=10)")

    # Smooth with 5-frame moving average
    tset = tset.smooth(window=5)
    print("  Smoothed (window=5)")

    # ==================================================================
    # 5. Extract signals
    # ==================================================================

    print("\n[5/7] Extracting signals...")

    # 2D signals: (T, N, 2) — x, y trajectories
    signals_2d = tset.to_signals(dim=2)
    print(f"  2D signals: {signals_2d.shape}")

    # 1D signals: (T, N) — displacement magnitude from initial position
    signals_1d = tset.to_signals(dim=1, axis="magnitude")
    print(f"  1D displacement: {signals_1d.shape}")

    # 3D signals: (T, N, 3) — x, y, displacement magnitude
    signals_3d = tset.to_signals(dim=3)
    print(f"  3D signals: {signals_3d.shape}")

    # Derived: velocities and speeds
    velocities = tset.velocities()
    speeds = tset.speeds()
    print(f"  Velocities: {velocities.shape}, Speeds: {speeds.shape}")

    # Quick stats
    mean_speed = np.nanmean(speeds, axis=0)
    print(f"  Mean speed per feature: min={mean_speed.min():.4f}, "
          f"max={mean_speed.max():.4f}, median={np.median(mean_speed):.4f}")

    # ==================================================================
    # 6. Save to HDF5
    # ==================================================================

    print("\n[6/7] Saving to HDF5...")

    h5_path = output_dir / "experiment_vision.h5"
    vision.save(
        h5_path,
        source=src,
        feature_map=fmap,
        tracking_result=result,
        trajectory_set=tset,
        signals=signals_2d,
        signals_meta={
            "dim": 2,
            "units": tset.units,
            "description": "Calibrated 2D feature trajectories",
        },
    )
    print(f"  Saved → {h5_path}")

    # Print file summary
    file_info = vision.info(h5_path)
    print(f"  Contents: {file_info.get('subgroups', [])}")
    print(f"  Features: {file_info.get('n_features', '?')}")
    print(f"  Trajectory frames: {file_info.get('traj_frames', '?')}")
    print(f"  Signal shape: {file_info.get('signal_shape', '?')}")

    # Verify round-trip
    loaded = vision.load(h5_path)
    loaded_tset = loaded.get("trajectory_set")
    if loaded_tset is not None:
        print(f"  Round-trip verified: {loaded_tset}")

    # ==================================================================
    # 7. Write annotated video
    # ==================================================================

    print("\n[7/7] Writing annotated video...")

    src.reset()
    annotated_path = output_dir / "experiment_annotated.mp4"

    viz.write_annotated_video(
        annotated_path,
        source=src,
        tracking_result=result,
        feature_map=fmap,
        trail=20,
        color_by="feature",
        show_status_bar=True,
        progress_callback=progress,
    )
    print(f"  Saved → {annotated_path}")

    # ==================================================================
    # Done
    # ==================================================================

    src.release()

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print(f"  Outputs in: {output_dir}")
    print(f"    - reference_features.png  (detected features)")
    print(f"    - experiment_vision.h5    (all pipeline data)")
    print(f"    - experiment_annotated.mp4 (tracked video)")
    print("=" * 60)


# ==================================================================
# Alternative: Using camera calibration
# ==================================================================

def example_with_calibration(video_path: str):
    """Same pipeline but with a camera matrix for physical units."""

    src = vision.VideoSource(video_path)
    ref = src.read_frame(0)

    fmap = vision.detect(ref, method="sift")

    src.reset()
    result = vision.Tracker(src, fmap).run()

    # Provide camera intrinsics (from calibration procedure)
    K = np.array([
        [800.0,   0.0, 960.0],   # fx, 0, cx
        [  0.0, 800.0, 540.0],   # 0, fy, cy
        [  0.0,   0.0,   1.0],
    ])
    cal = vision.CameraCalibration.from_matrix(
        K,
        frame_shape=(1080, 1920),
        scale_factor=25.0,         # e.g., working distance → mm
        physical_units="mm",
    )

    tset = vision.TrajectorySet.from_tracking_result(result, calibration=cal)
    signals = tset.to_signals(dim=2)  # Now in mm!

    print(f"Physical signals: {signals.shape}, units={tset.units}")
    src.release()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python example_pipeline.py <video_path>")
        print("  e.g. python example_pipeline.py experiment.mp4")
        sys.exit(1)

    main(sys.argv[1])