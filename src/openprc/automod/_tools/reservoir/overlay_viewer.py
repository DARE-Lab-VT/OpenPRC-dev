"""
Trajectory + Reservoir overlay viewer.

Subclasses DEMLAT's ``DEMLATVisualizer`` (in ``openprc.demlat.utils.animator``)
so we inherit all of its UI: icon-aware playback controls, frame timeline,
speed slider, step controls, strain coloring, particle/cylinder switching,
camera, theme handling — everything.

We add one new layer on top: the robot's CAD mesh, animated by forward
kinematics at the same frame index DEMLAT is already tracking. Because
``timestep_idx`` is the master clock, mesh and reservoir stay frame-locked
automatically.

Three display modes:
  - reservoir_only : DEMLAT's normal rendering only (CAD hidden)
  - cad_only       : robot mesh only, reservoir layers hidden
  - overlay        : both, with mesh translucent so reservoir shows through

Run from the bundle root:
    python _tools/reservoir/overlay_viewer.py \\
        --bundle-dir . --robot go1 --trajectory corridor_000

Looks for the simulation at:
    <bundle>/<robot>/reservoir_sims/<trajectory>/

If you want extra DEMLAT flags (e.g. --particles, --no-strain), pass them
through and they'll forward to DEMLAT's loader.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                          # _tools/
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "kinematics"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "viz"))


# Display-mode constants
MODE_RESERVOIR = 0
MODE_CAD = 1
MODE_OVERLAY = 2
MODE_LABELS = {
    MODE_RESERVOIR: "RESERVOIR ONLY",
    MODE_CAD: "CAD ONLY",
    MODE_OVERLAY: "OVERLAY (CAD + RESERVOIR)",
}

# Default link tint when CAD mesh is rendered. Stays gray to keep the
# focus on the simulation/reservoir colors.
ROBOT_TINT_OPAQUE = (0.78, 0.78, 0.82, 1.0)
ROBOT_TINT_TRANSLUCENT_DEFAULT = 0.55  # mesh alpha when OVERLAY mode


def _quat_wxyz_to_rotation(q: np.ndarray) -> np.ndarray:
    """Convert a (4,) wxyz quaternion to a (3,3) rotation matrix."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),   2*(x*z + w*y)],
        [  2*(x*y + w*z), 1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [  2*(x*z - w*y),   2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _load_robot_assets(bundle_dir: str, robot_name: str, trajectory_id: str):
    """
    Read the robot's URDF, the trajectory HDF5, and run FK so we can
    animate the CAD mesh frame-by-frame.

    Returns: (fk, fk_seq, n_traj_frames, base_pose_or_None)
    """
    import h5py

    metadata_path = os.path.join(bundle_dir, robot_name, "metadata.json")
    with open(metadata_path) as f:
        robot_meta = json.load(f)
    urdf_path = os.path.join(bundle_dir, robot_name, robot_meta["urdf_path"])

    manifest_path = os.path.join(bundle_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    traj_entry = next(
        t for t in manifest["robots"][robot_name]["trajectories"]
        if t["id"] == trajectory_id
    )
    traj_h5_path = os.path.join(bundle_dir, traj_entry["path"])

    print(f"[overlay] loading trajectory: {traj_h5_path}")
    with h5py.File(traj_h5_path, "r") as f:
        joint_names = list(f.attrs["joint_names"])
        qpos = f["qpos"][:].astype(np.float32)
        base_pose = f["base_pose"][:].astype(np.float32) if "base_pose" in f else None

    print(f"[overlay] computing FK across {len(qpos)} frames...")
    from urdf_fk import RobotFK
    fk = RobotFK(
        urdf_path=urdf_path,
        joint_names=joint_names,
        floating_base=robot_meta["floating_base"],
    )
    fk_seq = fk.compute_sequence(qpos, base_pose)
    print(f"[overlay] FK ready: {len(fk.link_names)} links × {len(qpos)} frames")
    return fk, fk_seq, len(qpos), base_pose


# -----------------------------------------------------------------------------
# The viewer
# -----------------------------------------------------------------------------

def run(bundle_dir: str, robot_name: str, trajectory_id: str,
        demlat_config: Optional[dict] = None):
    # Locate the DEMLAT experiment directory
    exp_dir = os.path.join(bundle_dir, robot_name, "reservoir_sims", trajectory_id)
    if not os.path.exists(os.path.join(exp_dir, "output", "simulation.h5")):
        raise FileNotFoundError(
            f"Reservoir simulation not found at {exp_dir}/output/simulation.h5\n"
            f"Run reservoir_to_demlat.py first."
        )

    # Pull DEMLAT's player classes — we extend DEMLATVisualizer to add the CAD layer.
    try:
        from openprc.demlat.utils.animator import (
            DEMLATVisualizer, ExperimentData,
        )
    except ImportError as e:
        raise ImportError(
            "openprc.demlat.utils.animator is required. "
            "Verify your openprc install."
        ) from e

    from piviz import PiVizStudio, pgfx
    from piviz.ui import Slider, Label, Checkbox

    # Load FK + trajectory metadata for the CAD layer
    fk, fk_seq, n_traj_frames, base_pose = _load_robot_assets(
        bundle_dir, robot_name, trajectory_id
    )

    # Import the PiViz adapter that knows how to draw a robot from FK transforms
    from piviz_adapter import draw_robot

    # Load DEMLAT's experiment data
    print(f"[overlay] loading DEMLAT experiment: {exp_dir}")
    data = ExperimentData(exp_dir)

    # Verify alignment between trajectory and simulation
    n_sim_frames = data.n_frames
    if n_sim_frames != n_traj_frames:
        # Stretch or pad: we'll map sim frame i → traj frame i * (n_traj/n_sim).
        # In the typical case (save_dt = traj_dt) they're equal; otherwise we
        # build a lookup table.
        print(f"[overlay] note: sim has {n_sim_frames} frames, "
              f"trajectory has {n_traj_frames}; using nearest-frame mapping.")
        sim_to_traj_idx = np.round(
            np.arange(n_sim_frames) * (n_traj_frames - 1) / max(1, n_sim_frames - 1)
        ).astype(int)
    else:
        sim_to_traj_idx = np.arange(n_sim_frames)

    # ------------------------------------------------------------------
    # Subclass DEMLAT's visualizer so we get all of its UI for free
    # ------------------------------------------------------------------

    # Axis length for the body frame indicator (meters)
    BODY_FRAME_SCALE = 0.35

    # XYZ axis colors: red, green, blue (RGBA)
    BODY_FRAME_COLORS = [
        (1.0, 0.15, 0.15, 1.0),  # X  forward  — red
        (0.15, 1.0, 0.15, 1.0),  # Y  left     — green
        (0.15, 0.45, 1.0, 1.0),  # Z  up       — blue
    ]
    BODY_FRAME_AXES = np.eye(3)

    class OverlayVisualizer(DEMLATVisualizer):
        def __init__(self, data, config=None):
            super().__init__(data, config)

            # Overlay-specific state
            self.display_mode = MODE_OVERLAY
            self.cad_alpha = ROBOT_TINT_TRANSLUCENT_DEFAULT
            self._fk = fk
            self._fk_seq = fk_seq
            self._sim_to_traj_idx = sim_to_traj_idx
            self._n_traj_frames = n_traj_frames
            self._base_pose = base_pose   # (T, 7) xyz + wxyz, or None

            # Cached translucent color tuple (recomputed on slider change)
            self._cad_tint = (
                ROBOT_TINT_OPAQUE[0],
                ROBOT_TINT_OPAQUE[1],
                ROBOT_TINT_OPAQUE[2],
                self.cad_alpha,
            )

            # Used to suppress duplicate work when display mode changes
            self._cad_link_color_cache = None

            # Body frame indicator default state
            self._show_body_frame = base_pose is not None

        # ------------------------------------------------------------
        # UI: extend DEMLAT's panel with our overlay controls
        # ------------------------------------------------------------

        def _setup_ui(self):
            super()._setup_ui()
            if not self.ui_manager:
                return

            self.ui_manager.add_widget(
                "lbl_overlay_section",
                Label("— Overlay —", color=(0.6, 0.85, 1.0, 1.0)),
            )
            self.lbl_mode = Label(
                f"Mode: {MODE_LABELS[self.display_mode]}",
                color=(1.0, 1.0, 1.0, 1.0),
            )
            self.ui_manager.add_widget("lbl_overlay_mode", self.lbl_mode)

            # Discrete tri-state via slider (Checkbox is binary, this is cleaner
            # than three separate checkboxes that you'd have to keep in sync)
            self.ui_manager.add_widget(
                "sld_display_mode",
                Slider(
                    "Display (0=RES 1=CAD 2=BOTH)",
                    0, 2, self.display_mode,
                    self._on_display_mode,
                ),
            )

            self.ui_manager.add_widget(
                "sld_cad_alpha",
                Slider(
                    "CAD alpha", 0.05, 1.0, self.cad_alpha,
                    self._on_cad_alpha,
                ),
            )

            self.ui_manager.add_widget(
                "btn_cad_color_reset",
                Checkbox(
                    "Alpha=1 when CAD ONLY",
                    True,
                    lambda v: setattr(self, "_auto_alpha_in_cad_mode", v),
                ),
            )
            self._auto_alpha_in_cad_mode = True

            if self._base_pose is not None:
                self.ui_manager.add_widget(
                    "chk_body_frame",
                    Checkbox(
                        "Body frame  (X=fwd  Y=left  Z=up)",
                        self._show_body_frame,
                        lambda v: setattr(self, "_show_body_frame", v),
                    ),
                )

        def _on_display_mode(self, v):
            new_mode = int(round(float(v)))
            new_mode = max(MODE_RESERVOIR, min(MODE_OVERLAY, new_mode))
            if new_mode != self.display_mode:
                self.display_mode = new_mode
                if hasattr(self, "lbl_mode"):
                    self.lbl_mode.text = f"Mode: {MODE_LABELS[new_mode]}"

        def _on_cad_alpha(self, v):
            self.cad_alpha = float(v)
            self._cad_link_color_cache = None  # invalidate

        # ------------------------------------------------------------
        # Render: gate DEMLAT layers by display mode and inject CAD layer
        # ------------------------------------------------------------

        def render(self, time_val, dt):
            # Decide what's visible this frame based on the mode toggle.
            # We override DEMLAT's per-flag config: don't permanently modify
            # user-set flags; just gate the call sites locally.

            mode = self.display_mode

            # Save the user-tweaked DEMLAT flags so we can restore after frame
            saved = {
                "show_nodes": self.config["show_nodes"],
                "show_bars": self.config["show_bars"],
                "show_faces": self.config["show_faces"],
                "show_velocity": self.config["show_velocity"],
                "show_trails": self.config["show_trails"],
                "show_hinges": self.config["show_hinges"],
            }

            # In CAD-only mode, suppress reservoir layers
            if mode == MODE_CAD:
                self.config["show_nodes"] = False
                self.config["show_bars"] = False
                self.config["show_faces"] = False
                self.config["show_velocity"] = False
                self.config["show_trails"] = False
                self.config["show_hinges"] = False

            # Run DEMLAT's normal render path. It updates the playback clock,
            # slider, and labels — we don't want to duplicate any of that.
            super().render(time_val, dt)

            # Restore the user's toggles
            self.config.update(saved)

            # Now overlay the CAD mesh, if requested
            if mode in (MODE_CAD, MODE_OVERLAY):
                self._render_cad()

            # Body frame indicator (independent of display mode)
            if self._base_pose is not None and self._show_body_frame:
                self._render_base_frame()

        def _render_cad(self):
            # Map current sim frame to trajectory frame
            sim_idx = int(self.timestep_idx)
            sim_idx = max(0, min(len(self._sim_to_traj_idx) - 1, sim_idx))
            traj_idx = int(self._sim_to_traj_idx[sim_idx])

            # Determine effective alpha for this mode
            if self.display_mode == MODE_CAD and self._auto_alpha_in_cad_mode:
                alpha = 1.0
            else:
                alpha = self.cad_alpha

            tint = (
                ROBOT_TINT_OPAQUE[0],
                ROBOT_TINT_OPAQUE[1],
                ROBOT_TINT_OPAQUE[2],
                alpha,
            )

            # Build per-link transform dict for THIS frame
            link_transforms = {
                name: self._fk_seq[name][traj_idx]
                for name in self._fk_seq
            }
            link_colors = {n: tint for n in self._fk.link_names}

            draw_robot(pgfx, self._fk.visuals, link_transforms,
                       link_colors=link_colors)

        def _render_base_frame(self):
            """Draw the robot body-frame XYZ axes at the base position."""
            sim_idx = int(self.timestep_idx)
            sim_idx = max(0, min(len(self._sim_to_traj_idx) - 1, sim_idx))
            traj_idx = int(self._sim_to_traj_idx[sim_idx])

            bp = self._base_pose[traj_idx]          # (7,)  xyz + wxyz
            origin = bp[:3].astype(np.float64)
            R = _quat_wxyz_to_rotation(bp[3:7])     # (3,3)

            for axis_vec, color in zip(BODY_FRAME_AXES, BODY_FRAME_COLORS):
                tip = origin + R @ axis_vec * BODY_FRAME_SCALE
                pgfx.draw_line(origin.tolist(), tip.tolist(), color, width=3.0)

    # ------------------------------------------------------------------
    # Run the studio with our subclass
    # ------------------------------------------------------------------
    print(f"[overlay] starting PiViz (mode: OVERLAY by default)...")

    # DEMLATVisualizer's __init__ takes (data, config); piviz arg-stripping
    # handled by ShowSimulation in viz_player.py — we replicate it here.
    argv_backup = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        viz = OverlayVisualizer(data, demlat_config or {})
        studio = PiVizStudio(scene_fx=viz)
        studio.run()
    finally:
        sys.argv = argv_backup


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Overlay viewer: trajectory CAD mesh + DEMLAT reservoir simulation"
    )
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--robot", required=True)
    parser.add_argument("--trajectory", required=True)

    # DEMLAT pass-throughs (mirror the original viz_player flags)
    parser.add_argument("--no-strain", action="store_true",
                        help="Disable bar strain coloring")
    parser.add_argument("--velocity", action="store_true",
                        help="Show velocity arrows")
    parser.add_argument("--trails", action="store_true",
                        help="Show motion trails")
    parser.add_argument("--particles", action="store_true",
                        help="Force particle rendering for nodes")
    parser.add_argument("--lines", action="store_true",
                        help="Use lines instead of cylinders for bars")

    args, piviz_args = parser.parse_known_args()

    demlat_config: Dict = {}
    if args.no_strain:
        demlat_config["show_strain"] = False
    if args.velocity:
        demlat_config["show_velocity"] = True
    if args.trails:
        demlat_config["show_trails"] = True
    if args.particles:
        demlat_config["use_particles_for_nodes"] = True
    if args.lines:
        demlat_config["use_lines_for_bars"] = True

    # Forward any unrecognized flags to PiViz
    sys.argv = [sys.argv[0]] + piviz_args
    run(args.bundle_dir, args.robot, args.trajectory, demlat_config)


if __name__ == "__main__":
    main()
