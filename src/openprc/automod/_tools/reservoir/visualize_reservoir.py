"""
Reservoir visualizer — eye-test tool for spring-mass reservoirs produced
by mesh_to_reservoir.py. Renders in PiViz with the same batching pattern
as the native DEMLAT animator (piviz.pgfx.draw_spheres_batch / draw_lines_batch).

What you see:
  - Red cubes:      anchor nodes (driven by position actuators at runtime)
  - Blue spheres:   interior (free) nodes
  - Gray lines:     springs connecting nodes
  - Optional mesh:  translucent overlay of the original link mesh

Controls (UI panel, not key bindings — avoiding conflicts with PiViz's
built-in G/A/T/H/M/0-3 shortcuts):
  - Checkboxes:     toggle anchors / interior / springs / mesh
  - Sliders:        sphere radius, line width
  - Button:         reset view

Node positions are in link-local frame at rest pose. Static view; dynamic
reservoir playback is a separate tool built later.

Usage:
    python visualize_reservoir.py /path/to/reservoir.npz
    python visualize_reservoir.py /path/to/reservoir.npz --mesh /path/to/original.stl
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from mesh_to_reservoir import load_reservoir  # noqa: E402


# Visual defaults — match the DEMLAT animator palette for consistency
ANCHOR_COLOR = (0.90, 0.25, 0.20)    # warm red, like position-actuator cubes
INTERIOR_COLOR = (0.30, 0.55, 0.85)  # cool blue, like floating nodes
SPRING_COLOR = (0.50, 0.50, 0.55)    # neutral gray bars
MESH_COLOR = (0.80, 0.80, 0.85, 0.25)  # faint white-ish, translucent


def _characteristic_length(positions: np.ndarray, edges: np.ndarray) -> float:
    """
    Same heuristic the DEMLAT animator uses: median spring rest length is
    the best scale for sphere/line sizing. Fall back to bbox median if no
    springs.
    """
    if len(edges) > 0:
        rl = np.linalg.norm(
            positions[edges[:, 1]] - positions[edges[:, 0]], axis=1
        )
        return float(np.median(rl))
    extent = positions.max(axis=0) - positions.min(axis=0)
    return float(np.median(extent)) if np.any(extent > 0) else 1.0


def run(npz_path: str, mesh_path: Optional[str] = None):
    reservoir = load_reservoir(npz_path)
    node_pos = reservoir["node_positions"].astype("f4")
    n_anchors = reservoir["n_anchors"]
    edges = reservoir["edges"]

    n_total = len(node_pos)
    n_interior = n_total - n_anchors
    n_springs = len(edges)

    print(f"Loaded reservoir: {reservoir['link_name'] or os.path.basename(npz_path)}")
    print(f"  nodes: {n_total} ({n_anchors} anchors + {n_interior} interior)")
    print(f"  springs: {n_springs}")

    L_char = _characteristic_length(node_pos, edges)
    node_radius = max(L_char * 0.15, 0.0005)
    print(f"  characteristic length: {L_char*1000:.2f} mm")
    print(f"  default sphere radius: {node_radius*1000:.2f} mm")

    # Pre-slice and pre-allocate arrays that don't change across frames
    anchor_pos = node_pos[:n_anchors]
    interior_pos = node_pos[n_anchors:]
    anchor_colors = np.tile(np.array(ANCHOR_COLOR, dtype="f4"), (n_anchors, 1))
    interior_colors = np.tile(np.array(INTERIOR_COLOR, dtype="f4"), (n_interior, 1))
    spring_colors = np.tile(np.array(SPRING_COLOR, dtype="f4"), (n_springs, 1))
    spring_starts = node_pos[edges[:, 0]]
    spring_ends = node_pos[edges[:, 1]]

    # Camera bounds
    bbox_min = node_pos.min(axis=0)
    bbox_max = node_pos.max(axis=0)
    center = (bbox_min + bbox_max) * 0.5
    scene_diag = float(np.linalg.norm(bbox_max - bbox_min))
    camera_distance = max(scene_diag * 2.5, 0.1)

    try:
        from piviz import PiVizStudio, PiVizFX, pgfx
        from piviz.ui import Label, Slider, Checkbox, Button
    except ImportError as e:
        raise ImportError(
            "piviz-3d is required. Install with: pip install piviz-3d"
        ) from e

    class ReservoirViewer(PiVizFX):
        def setup(self):
            # Visibility state — all exposed via UI checkboxes, no keys
            self.show_anchors = True
            self.show_interior = True
            self.show_springs = True
            self.show_mesh = mesh_path is not None

            # Sizing (live-tuneable via sliders)
            self.node_radius = node_radius
            self.anchor_size = node_radius * 2.5  # anchors render as cubes
            self.line_width = 1.5

            # Pre-allocate per-instance radii array (refilled on slider change)
            self._interior_radii = np.full(n_interior, self.node_radius, dtype="f4")

            if self.camera:
                self.camera.set_view("iso")
                self.camera.distance = camera_distance
                self.camera.target = tuple(center)

            self._setup_ui()

        def _setup_ui(self):
            if not self.ui_manager:
                return
            self.ui_manager.set_panel_title("Reservoir Viewer")

            # Stats header
            link_name = reservoir["link_name"] or os.path.basename(npz_path)
            self.ui_manager.add_widget(
                "lbl_link",
                Label(f"link: {link_name}", color=(0.85, 0.85, 0.85, 1.0)),
            )
            self.ui_manager.add_widget(
                "lbl_counts",
                Label(
                    f"{n_anchors} anchors | {n_interior} interior | {n_springs} springs",
                    color=(0.65, 0.65, 0.65, 1.0),
                ),
            )

            # Visibility toggles — checkboxes, not keys (avoid PiViz conflicts)
            self.ui_manager.add_widget(
                "chk_anchors",
                Checkbox(
                    "Anchors",
                    self.show_anchors,
                    lambda v: setattr(self, "show_anchors", v),
                ),
            )
            self.ui_manager.add_widget(
                "chk_interior",
                Checkbox(
                    "Interior",
                    self.show_interior,
                    lambda v: setattr(self, "show_interior", v),
                ),
            )
            self.ui_manager.add_widget(
                "chk_springs",
                Checkbox(
                    "Springs",
                    self.show_springs,
                    lambda v: setattr(self, "show_springs", v),
                ),
            )
            if mesh_path is not None:
                self.ui_manager.add_widget(
                    "chk_mesh",
                    Checkbox(
                        "Mesh overlay",
                        self.show_mesh,
                        lambda v: setattr(self, "show_mesh", v),
                    ),
                )

            # Sizing sliders — in mm for readability
            radius_mm = node_radius * 1000.0
            self.ui_manager.add_widget(
                "sld_node_radius",
                Slider(
                    "Node size (mm)",
                    radius_mm * 0.25,
                    radius_mm * 4.0,
                    radius_mm,
                    self._on_radius_change,
                    ),
            )
            self.ui_manager.add_widget(
                "sld_line_width",
                Slider(
                    "Line width",
                    0.5,
                    5.0,
                    self.line_width,
                    lambda v: setattr(self, "line_width", float(v)),
                ),
            )

            # Reset button
            self.ui_manager.add_widget(
                "btn_reset",
                Button("Reset view", self._reset_view),
            )

        def _on_radius_change(self, v_mm: float):
            r = float(v_mm) / 1000.0
            self.node_radius = r
            self.anchor_size = r * 2.5
            self._interior_radii[:] = r

        def _reset_view(self):
            if self.camera:
                self.camera.set_view("iso")
                self.camera.distance = camera_distance
                self.camera.target = tuple(center)

        def render(self, time_val, dt):
            # 1. Mesh overlay — translucent, drawn first so other geometry sits on top
            if self.show_mesh and mesh_path is not None:
                pgfx.draw_mesh(
                    mesh_path,
                    position=(0.0, 0.0, 0.0),
                    scale=(1.0, 1.0, 1.0),
                    rotation=(0.0, 0.0, 0.0),
                    color=MESH_COLOR,
                    mtl="",
                )

            # 2. Springs (lines) — drawn before spheres so nodes render on top
            if self.show_springs and n_springs > 0:
                pgfx.draw_lines_batch(
                    spring_starts,
                    spring_ends,
                    spring_colors,
                    width=self.line_width,
                )

            # 3. Interior (spheres)
            if self.show_interior and n_interior > 0:
                pgfx.draw_spheres_batch(
                    centers=interior_pos,
                    radii=self._interior_radii,
                    colors=interior_colors,
                    detail=8,
                )

            # 4. Anchors (cubes) — matches DEMLAT animator's convention for
            #    position-actuated nodes. Small count (typically <200), so
            #    the per-cube loop is fine.
            if self.show_anchors and n_anchors > 0:
                s = self.anchor_size
                size_tuple = (s, s, s)
                for i in range(n_anchors):
                    pgfx.draw_cube(
                        center=tuple(anchor_pos[i]),
                        size=size_tuple,
                        color=ANCHOR_COLOR,
                    )

    # Strip our args so PiViz's own argparse doesn't choke on them
    argv_backup = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        PiVizStudio(scene_fx=ReservoirViewer()).run()
    finally:
        sys.argv = argv_backup


def main():
    p = argparse.ArgumentParser(description="Visualize a reservoir .npz in PiViz")
    p.add_argument("npz_path", help="Path to reservoir .npz file")
    p.add_argument(
        "--mesh",
        default=None,
        help="Optional path to original mesh (.stl/.obj) for overlay",
    )
    args, piviz_args = p.parse_known_args()

    if args.mesh and not os.path.exists(args.mesh):
        print(f"warning: mesh not found at {args.mesh}; continuing without overlay")
        args.mesh = None

    sys.argv = [sys.argv[0]] + piviz_args
    run(args.npz_path, args.mesh)


if __name__ == "__main__":
    main()