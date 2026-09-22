"""
Reservoir visualizer — eye-test tool for spring-mass reservoirs produced
by mesh_to_reservoir.py. Renders in PiViz with the same batching pattern
as the native DEMLAT animator (piviz.pgfx.draw_spheres_batch /
draw_cylinders_batch).

What you see:
  - Dark-red cubes:   anchor nodes (driven by position actuators at runtime)
  - Royal-blue sph.:  interior (free) nodes
  - Steel-blue cyl.:  spring cylinders
  - Optional mesh:    translucent overlay of the original link mesh

Colors are chosen for white/light backgrounds (high contrast, no neon).

Controls (UI panel — avoids PiViz's built-in G/A/T/H/M/0-3 key bindings):
  Visibility:
    Anchors / Interior / Spring network / CAD mesh overlay (checkboxes)
  CAD mesh:
    Mesh alpha  (0 → 1 slider)
  Spring network:
    Spring density  (2 % → 100 % slider + live count label)
    Spring radius (mm)
  Sizing:
    Node size (mm)
    Reset view button

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


# ── Palette for white / light backgrounds ─────────────────────────────────────
# Vibrant but non-neon: each hue is saturated and dark enough to pop on white.
ANCHOR_COLOR   = (0.78, 0.06, 0.12)   # deep crimson  — anchor cubes
INTERIOR_COLOR = (0.10, 0.28, 0.82)   # royal blue    — free-node spheres
SPRING_COLOR   = (0.18, 0.52, 0.80)   # steel blue    — spring cylinders
MESH_BASE_RGB  = (0.45, 0.55, 0.75)   # slate blue    — mesh (alpha via slider)
MESH_ALPHA_DEFAULT = 0.18

# Spring cylinder radius as fraction of characteristic spring length
_SPRING_RADIUS_FRAC = 0.030
# Cylinder tessellation — matches DEMLAT animator default
_CYL_DETAIL = 8


def _characteristic_length(positions: np.ndarray, edges: np.ndarray) -> float:
    """Median spring rest-length, or bbox-median fallback when no springs."""
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

    link_name = reservoir["link_name"] or os.path.basename(npz_path)
    print(f"Loaded reservoir: {link_name}")
    print(f"  nodes  : {n_total}  ({n_anchors} anchors + {n_interior} interior)")
    print(f"  springs: {n_springs}")

    L_char = _characteristic_length(node_pos, edges)
    node_radius_default  = max(L_char * 0.15, 0.0005)
    spring_radius_default = max(L_char * _SPRING_RADIUS_FRAC, 0.0001)
    print(f"  char. length   : {L_char*1000:.2f} mm")
    print(f"  node radius    : {node_radius_default*1000:.2f} mm")
    print(f"  spring radius  : {spring_radius_default*1000:.2f} mm")

    # Pre-slice — never re-computed
    anchor_pos    = node_pos[:n_anchors]
    interior_pos  = node_pos[n_anchors:]
    anchor_colors   = np.tile(np.array(ANCHOR_COLOR,   dtype="f4"), (n_anchors,  1))
    interior_colors = np.tile(np.array(INTERIOR_COLOR, dtype="f4"), (n_interior, 1))
    spring_colors_all = np.tile(np.array(SPRING_COLOR, dtype="f4"), (n_springs,  1))
    spring_starts_all = node_pos[edges[:, 0]] if n_springs > 0 else np.empty((0, 3), dtype="f4")
    spring_ends_all   = node_pos[edges[:, 1]] if n_springs > 0 else np.empty((0, 3), dtype="f4")

    # Fixed random permutation for density subsampling (seed=42 → stable under dragging)
    rng = np.random.default_rng(42)
    spring_perm = rng.permutation(n_springs).astype(np.int32) if n_springs > 0 \
                  else np.array([], dtype=np.int32)

    # Camera
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
            # Visibility
            self.show_anchors  = True
            self.show_interior = True
            self.show_springs  = True
            self.show_mesh     = mesh_path is not None

            # Sizes
            self.node_radius   = node_radius_default
            self.anchor_size   = node_radius_default * 2.5
            self.spring_radius = spring_radius_default

            # Mesh alpha
            self.mesh_alpha = MESH_ALPHA_DEFAULT

            # Spring density
            self.spring_density = 1.0
            self._visible_spring_count = n_springs

            # Per-instance radii for interior spheres
            self._interior_radii = np.full(n_interior, self.node_radius, dtype="f4")

            if self.camera:
                self.camera.set_view("iso")
                self.camera.distance = camera_distance
                self.camera.target = tuple(center)

            self._setup_ui()

        # ── UI ──────────────────────────────────────────────────────────────

        def _setup_ui(self):
            if not self.ui_manager:
                return
            self.ui_manager.set_panel_title("Reservoir Viewer")

            self.ui_manager.add_widget("lbl_link",
                Label(f"link: {link_name}", color=(0.20, 0.20, 0.20, 1.0)))
            self.ui_manager.add_widget("lbl_counts",
                Label(f"{n_anchors} anchors | {n_interior} interior | {n_springs} springs",
                      color=(0.45, 0.45, 0.45, 1.0)))

            # ── Visibility ────────────────────────────────────────────
            self.ui_manager.add_widget("lbl_vis",
                Label("── Visibility ──", color=(0.18, 0.35, 0.70, 1.0)))
            self.ui_manager.add_widget("chk_anchors",
                Checkbox("Anchors  (dark red)",
                         self.show_anchors,
                         lambda v: setattr(self, "show_anchors", v)))
            self.ui_manager.add_widget("chk_interior",
                Checkbox("Interior  (royal blue)",
                         self.show_interior,
                         lambda v: setattr(self, "show_interior", v)))
            self.ui_manager.add_widget("chk_springs",
                Checkbox("Spring network  (steel blue)",
                         self.show_springs,
                         lambda v: setattr(self, "show_springs", v)))
            if mesh_path is not None:
                self.ui_manager.add_widget("chk_mesh",
                    Checkbox("CAD mesh overlay",
                             self.show_mesh,
                             lambda v: setattr(self, "show_mesh", v)))

            # ── CAD mesh ──────────────────────────────────────────────
            if mesh_path is not None:
                self.ui_manager.add_widget("lbl_mesh",
                    Label("── CAD mesh ──", color=(0.18, 0.35, 0.70, 1.0)))
                self.ui_manager.add_widget("sld_mesh_alpha",
                    Slider("Mesh alpha", 0.0, 1.0, self.mesh_alpha,
                           lambda v: setattr(self, "mesh_alpha", float(v))))

            # ── Spring network ────────────────────────────────────────
            if n_springs > 0:
                self.ui_manager.add_widget("lbl_net",
                    Label("── Spring network ──", color=(0.18, 0.35, 0.70, 1.0)))
                self.ui_manager.add_widget("sld_density",
                    Slider("Spring density", 0.02, 1.0, self.spring_density,
                           self._on_density_change))
                self.ui_manager.add_widget("lbl_density_count",
                    Label(f"showing {n_springs} / {n_springs} springs",
                          color=(0.50, 0.50, 0.50, 1.0)))
                spr_mm = spring_radius_default * 1000.0
                self.ui_manager.add_widget("sld_spring_radius",
                    Slider("Spring radius (mm)",
                           spr_mm * 0.2, spr_mm * 5.0, spr_mm,
                           self._on_spring_radius_change))

            # ── Node sizing ───────────────────────────────────────────
            self.ui_manager.add_widget("lbl_sizing",
                Label("── Sizing ──", color=(0.18, 0.35, 0.70, 1.0)))
            rad_mm = node_radius_default * 1000.0
            self.ui_manager.add_widget("sld_node_radius",
                Slider("Node size (mm)",
                       rad_mm * 0.25, rad_mm * 4.0, rad_mm,
                       self._on_radius_change))

            self.ui_manager.add_widget("btn_reset",
                Button("Reset view", self._reset_view))

        # ── Callbacks ───────────────────────────────────────────────────────

        def _on_radius_change(self, v_mm: float):
            r = float(v_mm) / 1000.0
            self.node_radius = r
            self.anchor_size = r * 2.5
            self._interior_radii[:] = r

        def _on_spring_radius_change(self, v_mm: float):
            self.spring_radius = float(v_mm) / 1000.0

        def _on_density_change(self, v: float):
            self.spring_density = float(v)
            count = max(1, int(n_springs * self.spring_density))
            self._visible_spring_count = count
            lbl = self.ui_manager.get_widget("lbl_density_count") if self.ui_manager else None
            if lbl is not None:
                lbl.text = f"showing {count} / {n_springs} springs"

        def _reset_view(self):
            if self.camera:
                self.camera.set_view("iso")
                self.camera.distance = camera_distance
                self.camera.target = tuple(center)

        # ── Render ──────────────────────────────────────────────────────────

        def render(self, time_val, dt):
            # 1. Mesh overlay — translucent, drawn first
            if self.show_mesh and mesh_path is not None:
                mesh_color = MESH_BASE_RGB + (self.mesh_alpha,)
                pgfx.draw_mesh(
                    mesh_path,
                    position=(0.0, 0.0, 0.0),
                    scale=(1.0, 1.0, 1.0),
                    rotation=(0.0, 0.0, 0.0),
                    color=mesh_color,
                    mtl="",
                )

            # 2. Spring cylinders — drawn before spheres so nodes sit on top
            if self.show_springs and n_springs > 0:
                idx = spring_perm[:self._visible_spring_count]
                pgfx.draw_cylinders_batch(
                    starts=spring_starts_all[idx],
                    ends=spring_ends_all[idx],
                    radii=self.spring_radius,
                    colors=spring_colors_all[idx],
                    detail=_CYL_DETAIL,
                )

            # 3. Interior spheres (royal blue)
            if self.show_interior and n_interior > 0:
                pgfx.draw_spheres_batch(
                    centers=interior_pos,
                    radii=self._interior_radii,
                    colors=interior_colors,
                    detail=8,
                )

            # 4. Anchor cubes (dark crimson)
            if self.show_anchors and n_anchors > 0:
                s = self.anchor_size
                size_tuple = (s, s, s)
                for i in range(n_anchors):
                    pgfx.draw_cube(
                        center=tuple(anchor_pos[i]),
                        size=size_tuple,
                        color=ANCHOR_COLOR,
                    )

    argv_backup = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        PiVizStudio(scene_fx=ReservoirViewer()).run()
    finally:
        sys.argv = argv_backup


def main():
    p = argparse.ArgumentParser(description="Visualize a reservoir .npz in PiViz")
    p.add_argument("npz_path", help="Path to reservoir .npz file")
    p.add_argument("--mesh", default=None,
                   help="Optional path to original mesh (.stl/.obj) for overlay")
    args, piviz_args = p.parse_known_args()

    if args.mesh and not os.path.exists(args.mesh):
        print(f"warning: mesh not found at {args.mesh}; continuing without overlay")
        args.mesh = None

    sys.argv = [sys.argv[0]] + piviz_args
    run(args.npz_path, args.mesh)


if __name__ == "__main__":
    main()
