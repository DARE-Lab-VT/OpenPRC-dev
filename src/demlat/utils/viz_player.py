"""
DEMLAT Minimal Visualization Player (PhalcoPulse)
===================================================

A stripped-down twin of the full viz_player using PhalcoPulse.
Covers the essentials: bars, hinges, nodes, strain coloring, and playback.

Usage:
    python viz_player_minimal.py <experiment_path>
"""

import numpy as np
import h5py
import json
import sys
from pathlib import Path

from phalcopulse import PhalcoPulseStudio, PhalcoPulseFX, pgfx
from phalcopulse.ui import Label, Button, Slider, Checkbox


# =============================================================================
# DATA LOADER (minimal)
# =============================================================================

class ExperimentData:
    """Load geometry + optional simulation data from DEMLAT experiment folder."""

    def __init__(self, experiment_path: str):
        self.exp_path = Path(experiment_path)
        self.geometry_path = self.exp_path / "input" / "geometry.h5"
        self.simulation_path = self.exp_path / "output" / "simulation.h5"

        # Data
        self.positions = None        # (n_nodes, 3) initial
        self.attributes = None       # (n_nodes,) node type flags
        self.bar_indices = None      # (n_bars, 2)
        self.bar_rest_lengths = None # (n_bars,)
        self.hinge_indices = None    # (n_hinges, 4) [edge1, edge2, wing1, wing2]
        self.hinge_rest_angles = None

        # Time series (if simulation exists)
        self.time = None             # (n_frames,)
        self.ts_positions = None     # (n_frames, n_nodes, 3)
        self.ts_bar_strain = None    # (n_frames, n_bars)
        self.ts_hinge_angle = None   # (n_frames, n_hinges)

        self.n_nodes = 0
        self.n_bars = 0
        self.n_hinges = 0
        self.n_frames = 1
        self.mode = "geometry"

        self._load()

    def _load(self):
        if not self.geometry_path.exists():
            raise FileNotFoundError(f"Missing: {self.geometry_path}")

        # --- Geometry ---
        with h5py.File(self.geometry_path, "r") as f:
            self.positions = f["nodes/positions"][:]
            self.n_nodes = len(self.positions)

            if "nodes/attributes" in f:
                self.attributes = f["nodes/attributes"][:]
            else:
                self.attributes = np.zeros(self.n_nodes, dtype="u1")

            if "elements/bars/indices" in f:
                self.bar_indices = f["elements/bars/indices"][:]
                self.n_bars = len(self.bar_indices)
                if "elements/bars/rest_length" in f:
                    self.bar_rest_lengths = f["elements/bars/rest_length"][:]
                else:
                    p = self.positions
                    idx = self.bar_indices
                    self.bar_rest_lengths = np.linalg.norm(
                        p[idx[:, 0]] - p[idx[:, 1]], axis=1
                    ).astype("f4")

            if "elements/hinges/indices" in f:
                self.hinge_indices = f["elements/hinges/indices"][:]
                self.n_hinges = len(self.hinge_indices)
                if "elements/hinges/rest_angle" in f:
                    self.hinge_rest_angles = f["elements/hinges/rest_angle"][:]

        print(f"[Geometry] {self.n_nodes} nodes, {self.n_bars} bars, {self.n_hinges} hinges")

        # --- Simulation (optional) ---
        if self.simulation_path.exists():
            with h5py.File(self.simulation_path, "r") as f:
                if "time_series/time" in f and "time_series/nodes/positions" in f:
                    self.time = f["time_series/time"][:]
                    self.ts_positions = f["time_series/nodes/positions"][:]
                    self.n_frames = len(self.time)
                    self.mode = "simulation"
                    print(f"[Simulation] {self.n_frames} frames, duration {self.time[-1]:.2f}s")

                    if "time_series/elements/bars/strain" in f:
                        self.ts_bar_strain = f["time_series/elements/bars/strain"][:]
                        print(f"  Bar strains loaded")

                    if "time_series/elements/hinges/angle" in f:
                        self.ts_hinge_angle = f["time_series/elements/hinges/angle"][:]
                        print(f"  Hinge angles loaded")


# =============================================================================
# COLORMAP UTILITY
# =============================================================================

def jet_color(t):
    """Simple jet colormap: t in [0, 1] -> (r, g, b)."""
    t = max(0.0, min(1.0, t))
    r = max(0.0, min(1.0, 1.5 - abs(t - 0.75) * 4))
    g = max(0.0, min(1.0, 1.5 - abs(t - 0.50) * 4))
    b = max(0.0, min(1.0, 1.5 - abs(t - 0.25) * 4))
    return (r, g, b)


def strain_to_color(strain, limit):
    """Map strain value to jet color. Negative=blue, zero=green, positive=red."""
    t = (strain + limit) / (2.0 * limit)
    return jet_color(t)


# =============================================================================
# VISUALIZER
# =============================================================================

class DEMLATMinimalViz(PhalcoPulseFX):
    """Minimal DEMLAT visualizer using PhalcoPulse."""

    def __init__(self, data: ExperimentData):
        super().__init__()
        self.data = data

        # Playback
        self.frame_idx = 0
        self.float_frame = 0.0
        self.paused = False
        self.speed = 1.0

        # Display toggles
        self.show_bars = True
        self.show_hinges = (data.n_hinges > 0)
        self.show_nodes = True
        self.show_strain = (data.ts_bar_strain is not None)
        self.use_lines = (data.n_bars > 300)

        # Auto-scale
        extent = np.ptp(data.positions, axis=0)
        self.center = np.mean(data.positions, axis=0)
        self.L_char = float(np.median(extent)) if np.any(extent > 0) else 1.0
        if data.n_bars > 0 and data.bar_rest_lengths is not None:
            med_len = float(np.median(data.bar_rest_lengths))
            if med_len > 0:
                self.L_char = med_len

        self.node_radius = self.L_char * 0.05
        self.bar_radius = self.L_char * 0.03

        # Strain limit (95th percentile)
        self.strain_limit = 0.3
        if data.ts_bar_strain is not None:
            self.strain_limit = float(np.percentile(np.abs(data.ts_bar_strain), 95))
            if self.strain_limit < 1e-10:
                self.strain_limit = 0.3

        # Pre-allocate work arrays
        self.cur_pos = np.copy(data.positions).astype("f4")

        # Categorize nodes
        attrs = data.attributes
        self.fixed_idx = np.where((attrs & 1) != 0)[0]
        self.actuator_idx = np.where((attrs & 2) != 0)[0]
        self.floating_idx = np.where(attrs == 0)[0]

        print(f"[Scales] L_char={self.L_char:.4f}, strain_limit=±{self.strain_limit:.4f}")

    def setup(self, ui_manager):
        # --- Info labels ---
        y = 220
        ui_manager.add_widget("lbl_title", Label((10, y, 300, 20), "DEMLAT Minimal Player", align="left"))
        y -= 25
        info = f"Nodes: {self.data.n_nodes}  Bars: {self.data.n_bars}  Hinges: {self.data.n_hinges}"
        ui_manager.add_widget("lbl_info", Label((10, y, 300, 20), info, align="left"))

        if self.data.mode == "simulation":
            y -= 25
            self.lbl_time = ui_manager.add_widget(
                "lbl_time", Label((10, y, 300, 20), "Frame: 0 / 0", align="left")
            )

            # --- Playback controls ---
            y -= 30
            ui_manager.add_widget("sld_frame", Slider(
                (10, y, 250, 20), "Frame", 0, self.data.n_frames - 1, 0, self._on_frame_slider
            ))

            y -= 30
            ui_manager.add_widget("btn_start", Button((10, y, 40, 25), "|<", self._goto_start))
            ui_manager.add_widget("btn_prev", Button((55, y, 40, 25), "<", self._step_back))
            ui_manager.add_widget("btn_pause", Button((100, y, 40, 25), "||", self._toggle_pause))
            ui_manager.add_widget("btn_next", Button((145, y, 40, 25), ">", self._step_fwd))
            ui_manager.add_widget("btn_end", Button((190, y, 40, 25), ">|", self._goto_end))

            y -= 30
            ui_manager.add_widget("sld_speed", Slider(
                (10, y, 250, 20), "Speed", 10, 500, 100,
                lambda v: setattr(self, "speed", v / 100.0)
            ))

        # --- Toggles ---
        y -= 30
        ui_manager.add_widget("chk_bars", Checkbox(
            (10, y, 200, 20), "Show Bars", self.show_bars,
            lambda v: setattr(self, "show_bars", v)
        ))

        if self.data.n_hinges > 0:
            y -= 25
            ui_manager.add_widget("chk_hinges", Checkbox(
                (10, y, 200, 20), "Show Hinges", self.show_hinges,
                lambda v: setattr(self, "show_hinges", v)
            ))

        y -= 25
        ui_manager.add_widget("chk_nodes", Checkbox(
            (10, y, 200, 20), "Show Nodes", self.show_nodes,
            lambda v: setattr(self, "show_nodes", v)
        ))

        if self.data.ts_bar_strain is not None:
            y -= 25
            ui_manager.add_widget("chk_strain", Checkbox(
                (10, y, 200, 20), "Strain Color", self.show_strain,
                lambda v: setattr(self, "show_strain", v)
            ))

        y -= 25
        ui_manager.add_widget("chk_lines", Checkbox(
            (10, y, 200, 20), "Fast Lines", self.use_lines,
            lambda v: setattr(self, "use_lines", v)
        ))

        self.ui_mgr = ui_manager

    # ---- Playback callbacks ----

    def _on_frame_slider(self, val):
        self.frame_idx = int(val)
        self.float_frame = float(self.frame_idx)
        self.paused = True

    def _toggle_pause(self):
        self.paused = not self.paused

    def _goto_start(self):
        self.frame_idx = 0
        self.float_frame = 0.0

    def _goto_end(self):
        self.frame_idx = self.data.n_frames - 1
        self.float_frame = float(self.frame_idx)

    def _step_fwd(self):
        self.paused = True
        self.frame_idx = min(self.frame_idx + 1, self.data.n_frames - 1)
        self.float_frame = float(self.frame_idx)

    def _step_back(self):
        self.paused = True
        self.frame_idx = max(self.frame_idx - 1, 0)
        self.float_frame = float(self.frame_idx)

    # ---- Main render loop ----

    def loop(self, delta_time):
        # Advance playback
        if self.data.mode == "simulation" and not self.paused:
            self.float_frame += delta_time * 60.0 * self.speed
            if self.float_frame >= self.data.n_frames:
                self.float_frame = 0.0
            self.frame_idx = int(self.float_frame)

        # Get current positions
        if self.data.mode == "simulation":
            self.cur_pos[:] = self.data.ts_positions[self.frame_idx]
        else:
            self.cur_pos[:] = self.data.positions

        # Draw layers
        if self.show_bars:
            self._draw_bars()

        if self.show_hinges and self.data.n_hinges > 0:
            self._draw_hinges()

        if self.show_nodes:
            self._draw_nodes()

        # Update UI
        if self.data.mode == "simulation" and hasattr(self, "lbl_time"):
            t = self.data.time[self.frame_idx]
            status = "PAUSED" if self.paused else "PLAYING"
            self.lbl_time.text = f"t={t:.3f}s  Frame {self.frame_idx}/{self.data.n_frames-1}  {status}"

    # ---- Drawing methods ----

    def _draw_bars(self):
        """Render bars, optionally colored by strain."""
        idx = self.data.bar_indices
        if idx is None:
            return

        # Get strain data for this frame
        strains = None
        if self.show_strain and self.data.ts_bar_strain is not None:
            strains = self.data.ts_bar_strain[self.frame_idx]

        for i in range(self.data.n_bars):
            p0 = tuple(self.cur_pos[idx[i, 0]])
            p1 = tuple(self.cur_pos[idx[i, 1]])

            if strains is not None:
                color = strain_to_color(strains[i], self.strain_limit)
            else:
                color = (0.5, 0.5, 0.5)

            if self.use_lines:
                pgfx.draw_cylinder(start=p0, end=p1, radius=self.bar_radius * 0.3, color=color, detail=4)
            else:
                pgfx.draw_cylinder(
                    start=p0, end=p1,
                    radius=self.bar_radius,
                    color=color, detail=6
                )

    def _draw_hinges(self):
        """
        Render hinges as colored triangular faces.
        Color encodes angular deviation from rest angle (jet colormap).
        """
        hidx = self.data.hinge_indices
        if hidx is None:
            return

        pos = self.cur_pos

        # Vectorized dihedral angle computation
        p_j = pos[hidx[:, 0]]  # edge node 1
        p_k = pos[hidx[:, 1]]  # edge node 2
        p_i = pos[hidx[:, 2]]  # wing 1
        p_l = pos[hidx[:, 3]]  # wing 2

        r_ij = p_i - p_j
        r_kj = p_k - p_j
        r_kl = p_k - p_l

        m = np.cross(r_ij, r_kj)
        n = np.cross(r_kj, r_kl)

        len_m = np.linalg.norm(m, axis=1)
        len_n = np.linalg.norm(n, axis=1)
        valid = (len_m > 1e-9) & (len_n > 1e-9)

        if not np.any(valid):
            return

        m_hat = np.zeros_like(m)
        n_hat = np.zeros_like(n)
        m_hat[valid] = m[valid] / len_m[valid, None]
        n_hat[valid] = n[valid] / len_n[valid, None]

        cos_phi = np.clip(np.einsum("ij,ij->i", m_hat, n_hat), -1.0, 1.0)
        phi = np.arccos(cos_phi)

        # Sign convention
        dot_m_rkl = np.einsum("ij,ij->i", m, r_kl)
        phi[dot_m_rkl < 0] *= -1

        # Angular deviation from rest
        if self.data.hinge_rest_angles is not None:
            delta = phi - self.data.hinge_rest_angles
            delta = (delta + np.pi) % (2 * np.pi) - np.pi
        else:
            delta = np.zeros_like(phi)

        limit = np.deg2rad(45)
        geom_scale = 0.4

        for i in np.where(valid)[0]:
            d = delta[i]
            t = float(np.clip((d + limit) / (2 * limit), 0.0, 1.0))
            color = jet_color(t)
            face_color = (color[0], color[1], color[2])

            mid = (p_j[i] + p_k[i]) * 0.5
            pj_s = mid + (p_j[i] - mid) * geom_scale
            pk_s = mid + (p_k[i] - mid) * geom_scale
            pi_s = mid + (p_i[i] - mid) * geom_scale
            pl_s = mid + (p_l[i] - mid) * geom_scale

            tj = tuple(map(float, pj_s))
            tk = tuple(map(float, pk_s))
            ti = tuple(map(float, pi_s))
            tl = tuple(map(float, pl_s))

            # Two wing triangles
            pgfx.draw_triangle(tj, tk, ti, color=face_color)
            pgfx.draw_triangle(tj, tk, tl, color=face_color)

            # Spine (edge) as thin white cylinder
            pgfx.draw_cylinder(start=tj, end=tk, radius=self.L_char * 0.008, color=(1.0, 1.0, 1.0), detail=4)

    def _draw_nodes(self):
        """Render nodes: spheres for floating, cubes for fixed/actuators."""
        r = self.node_radius

        # Floating nodes
        for i in self.floating_idx:
            pgfx.draw_sphere(
                center=tuple(self.cur_pos[i]),
                radius=r, color=(0.35, 0.35, 0.35)
            )

        # Fixed nodes (red cubes)
        s = r * 2.0
        for i in self.fixed_idx:
            pgfx.draw_cube(
                center=tuple(self.cur_pos[i]),
                size=s, color=(0.9, 0.2, 0.2)
            )

        # Actuator nodes (green cubes)
        for i in self.actuator_idx:
            pgfx.draw_cube(
                center=tuple(self.cur_pos[i]),
                size=s, color=(0.2, 0.8, 0.3)
            )


# =============================================================================
# PUBLIC API — drop-in replacement for piviz visualize_experiment
# =============================================================================

def visualize_experiment(experiment_path: str, config: dict = None):
    """
    Main entry point for visualizing DEMLAT experiments.
    Drop-in compatible with demlat.utils.viz_player.visualize_experiment.

    Usage:
        from demlat.utils.viz_player_minimal import visualize_experiment
        visualize_experiment("path/to/experiment")
        visualize_experiment("path/to/experiment", config={'show_strain': False})
    """
    data = ExperimentData(experiment_path)
    viz = DEMLATMinimalViz(data)

    # Apply config overrides
    if config:
        for key, val in config.items():
            if hasattr(viz, key):
                setattr(viz, key, val)

    # PhalcoPulseStudio (via moderngl_window) parses sys.argv,
    # so strip custom args to avoid "unrecognized arguments" errors.
    argv_backup = sys.argv
    sys.argv = [sys.argv[0]]

    try:
        studio = PhalcoPulseStudio(scene_fx=viz)
        studio.run()
    finally:
        sys.argv = argv_backup


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python viz_player_minimal.py <experiment_path> [--no-strain] [--lines]")
        sys.exit(1)

    exp_path = sys.argv[1]

    config = {}
    if "--no-strain" in sys.argv:
        config["show_strain"] = False
    if "--lines" in sys.argv:
        config["use_lines"] = True

    visualize_experiment(exp_path, config)


if __name__ == "__main__":
    main()