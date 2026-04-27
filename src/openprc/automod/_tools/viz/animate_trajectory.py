"""
Interactive trajectory animator — Stage 3 deliverable.

Loads one trajectory HDF5 from the bundle, loads the robot's URDF, and
animates the robot following the recorded joint angles and base pose.

Usage:
    python _tools/viz/animate_trajectory.py \\
        --bundle-dir /path/to/robot_bundle \\
        --robot go1 \\
        --trajectory corridor_000

UI controls (on top of PiViz's built-in navigation):
    - Play/Pause toggle
    - Time scrubber (seek to any frame)
    - Speed slider (0.1x to 4x)
    - Current frame / total frames display

Recording to mp4 is handled by PiViz's built-in record button in the overlay.

Implementation notes:
  - FK is precomputed for the entire trajectory at startup so the render loop
    just looks up per-link 4x4 matrices by frame index. For a 1500-frame Go1
    clip with ~20 links, that's 30k FK evaluations up-front, ~1 second on a
    laptop. Trade-off: startup delay for smooth playback.
  - `playback_time` advances in wall-clock seconds scaled by `speed`; we then
    wrap to [0, duration] and look up the nearest frame index.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "kinematics"))

from piviz_adapter import draw_robot  # noqa: E402


def _load_trajectory(bundle_dir: str, robot_name: str, trajectory_id: str):
    """Load one trajectory from the bundle. Returns (qpos_seq, base_pose_seq_or_None, time, attrs)."""
    import h5py  # local import: only needed at runtime

    manifest_path = os.path.join(bundle_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"{manifest_path} not found. Run _tools/build_manifest.py first."
        )
    with open(manifest_path) as f:
        manifest = json.load(f)

    if robot_name not in manifest["robots"]:
        raise KeyError(
            f"Robot {robot_name!r} not in manifest. Available: "
            f"{list(manifest['robots'].keys())}"
        )
    robot_entry = manifest["robots"][robot_name]

    match = next(
        (t for t in robot_entry["trajectories"] if t["id"] == trajectory_id),
        None,
    )
    if match is None:
        ids = [t["id"] for t in robot_entry["trajectories"]]
        raise KeyError(
            f"Trajectory {trajectory_id!r} not found for {robot_name}. "
            f"Available ({len(ids)}): {ids[:10]}{'...' if len(ids) > 10 else ''}"
        )

    h5_path = os.path.join(bundle_dir, match["path"])
    with h5py.File(h5_path, "r") as f:
        attrs = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in f.attrs.items()}
        time = f["time"][:]
        qpos = f["qpos"][:]
        base_pose = f["base_pose"][:] if "base_pose" in f else None

    return qpos, base_pose, time, attrs


def run(bundle_dir: str, robot_name: str, trajectory_id: str):
    """Construct the PiViz app and run it. Blocks until the window is closed."""
    # Load bundle metadata
    metadata_path = os.path.join(bundle_dir, robot_name, "metadata.json")
    with open(metadata_path) as f:
        robot_metadata = json.load(f)
    urdf_path = os.path.join(bundle_dir, robot_name, robot_metadata["urdf_path"])

    qpos, base_pose, time, traj_attrs = _load_trajectory(
        bundle_dir, robot_name, trajectory_id
    )
    print(f"Loaded {trajectory_id}: {len(time)} frames, "
          f"{time[-1] - time[0]:.2f}s, {traj_attrs['frequency_hz']:.1f} Hz")

    # FK precomputation
    from urdf_fk import RobotFK
    print("Precomputing forward kinematics...")
    fk = RobotFK(
        urdf_path=urdf_path,
        joint_names=list(traj_attrs["joint_names"]),
        floating_base=robot_metadata["floating_base"],
    )
    print(f"  {len(fk.visuals)} visual meshes across {len(fk.link_names)} links")
    per_link_seq = fk.compute_sequence(qpos, base_pose)
    print(f"  FK done for {len(time)} frames")

    # Lazy import of PiViz so --help etc. don't require a display
    try:
        from piviz import PiVizStudio, PiVizFX, pgfx
        from piviz.ui import Label, Slider, Button, Checkbox
    except ImportError as e:
        raise ImportError(
            "piviz-3d is required. Install with: pip install piviz-3d"
        ) from e

    duration = float(time[-1] - time[0])

    class TrajectoryViewer(PiVizFX):
        def setup(self):
            self.playing = True
            self.speed = 1.0
            self.playback_time = 0.0  # seconds within the trajectory
            self.current_frame = 0
            self._scrubbing = False

            # Stash references to precomputed data
            self._visuals = fk.visuals
            self._per_link_seq = per_link_seq
            self._time = time
            self._duration = duration
            self._n_frames = len(time)
            self._trajectory_id = trajectory_id

            # Camera setup: iso view, far enough to see the whole robot
            if self.camera:
                self.camera.set_view('iso')
                self.camera.distance = 3.0

            self._setup_ui()

        def _setup_ui(self):
            if not self.ui_manager:
                return
            self.ui_manager.set_panel_title(f"{robot_name} / {trajectory_id}")

            self.lbl_frame = Label("Frame: 0 / 0", color=(1, 1, 1, 1))
            self.ui_manager.add_widget("frame", self.lbl_frame)

            self.lbl_time = Label("Time: 0.00 / 0.00 s", color=(0.7, 0.7, 0.7, 1))
            self.ui_manager.add_widget("time", self.lbl_time)

            def toggle_play(v):
                self.playing = v
            self.ui_manager.add_widget(
                "play",
                Checkbox("Playing", self.playing, toggle_play),
            )

            def set_speed(v):
                self.speed = float(v)
            self.ui_manager.add_widget(
                "speed",
                Slider("Speed", 0.1, 4.0, 1.0, set_speed),
            )

            def scrub(v):
                self._scrubbing = True
                self.playback_time = float(v)
                self._scrubbing = False
            self.ui_manager.add_widget(
                "scrub",
                Slider("Seek (s)", 0.0, max(self._duration, 1e-3), 0.0, scrub),
            )

            def restart():
                self.playback_time = 0.0
            self.ui_manager.add_widget("restart", Button("Restart", restart))

        def render(self, time_val, dt):
            # Advance playback time
            if self.playing and not self._scrubbing:
                self.playback_time += dt * self.speed
                if self.playback_time >= self._duration:
                    # Loop
                    self.playback_time = 0.0

            # Find the nearest frame index
            frame_idx = int(np.searchsorted(
                self._time - self._time[0], self.playback_time, side="left"
            ))
            frame_idx = min(max(frame_idx, 0), self._n_frames - 1)
            self.current_frame = frame_idx

            # Assemble per-link transforms for this frame
            link_transforms = {
                name: self._per_link_seq[name][frame_idx]
                for name in self._per_link_seq
            }

            # Ground plane reference
            pgfx.draw_plane(
                size=(4.0, 4.0),
                color=(0.20, 0.22, 0.25, 1.0),
                center=(0.0, 0.0, 0.0),
                normal=(0, 0, 1),
            )

            # Draw the robot
            draw_robot(pgfx, self._visuals, link_transforms)

            # Update UI labels
            if hasattr(self, 'lbl_frame'):
                self.lbl_frame.text = f"Frame: {frame_idx} / {self._n_frames - 1}"
                self.lbl_time.text = (
                    f"Time: {self.playback_time:5.2f} / {self._duration:5.2f} s"
                )

    print("Starting PiViz...")
    PiVizStudio(scene_fx=TrajectoryViewer()).run()


def main():
    parser = argparse.ArgumentParser(
        description="Animate a bundle trajectory in PiViz",
        # Let PiViz print its own --help for windowing options; ours covers data.
    )
    parser.add_argument("--bundle-dir", required=True,
                        help="Path to the robot_bundle root directory")
    parser.add_argument("--robot", required=True,
                        help="Robot name as in manifest.json (e.g. go1)")
    parser.add_argument("--trajectory", required=True,
                        help="Trajectory id as in manifest.json (e.g. corridor_000)")

    # PiVizStudio.run() invokes moderngl-window, which re-parses sys.argv for
    # its own flags (--backend, -fs, --size, etc.). parse_known_args peels off
    # our args and leaves the rest for PiViz to consume. We then rewrite
    # sys.argv to contain only the script name + PiViz's leftover args so that
    # when PiViz re-parses, it sees a clean, valid argv.
    args, piviz_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + piviz_args

    run(args.bundle_dir, args.robot, args.trajectory)


if __name__ == "__main__":
    main()
