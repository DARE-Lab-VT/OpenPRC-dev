import h5py
import numpy as np
from pathlib import Path
import sys
from piviz import PiVizStudio, PiVizFX, pgfx
from piviz.ui import Slider, Label, Button, Checkbox


class SimulationPlayer(PiVizFX):
    """
    Universal Player for the Demlat architecture.
    """

    def __init__(self, experiment_path, config=None):
        self.exp_path = Path(experiment_path)
        self.input_dir = self.exp_path / "input"
        self.output_dir = self.exp_path / "output"

        # --- 1. Load Data ---
        self._load_geometry()
        self._load_visualization()
        self._load_simulation()

        # --- 2. Configuration ---
        self.config = {
            'show_faces': True,
            'show_edges': True,
            'show_nodes': True,
            'show_strain': True,
            'show_velocity': False,
            'show_trails': False,
            'trail_length': 40,
            'strain_limit': 0.15,
            'node_radius': self.L_char * 0.02,
            'edge_width': 2.0,
            'face_color': (0.2, 0.6, 1.0, 0.8),
            'rigid_color': (0.4, 0.4, 0.4),
            'velocity_scale': 1.0
        }
        if config:
            self.config.update(config)

        # --- 3. State Management ---
        self.paused = True
        self.speed = 1.0
        self.frame_idx = 0
        self.float_frame = 0.0
        self.trails = {i: [] for i in range(self.n_nodes)}

        self._preallocate_buffers()

    def _load_geometry(self):
        geo_path = self.input_dir / "geometry.h5"
        if not geo_path.exists():
            raise FileNotFoundError(f"Geometry file not found: {geo_path}")

        with h5py.File(geo_path, 'r') as f:
            self.ref_positions = f['nodes/positions'][:]
            self.n_nodes = self.ref_positions.shape[0]

            if 'elements/bars' in f:
                self.bar_indices = f['elements/bars/indices'][:]
                self.bar_rest_lengths = f['elements/bars/rest_length'][:]
                self.n_bars = self.bar_indices.shape[0]
            else:
                self.bar_indices = np.empty((0, 2), dtype=int)
                self.bar_rest_lengths = np.empty((0,), dtype=float)
                self.n_bars = 0

        # Calculate bounding box
        bbox_min = np.min(self.ref_positions, axis=0)
        bbox_max = np.max(self.ref_positions, axis=0)
        self.L_char = np.linalg.norm(bbox_max - bbox_min)
        if self.L_char == 0: self.L_char = 1.0

    def _load_visualization(self):
        viz_path = self.input_dir / "visualization.h5"
        self.has_viz = viz_path.exists()

        if self.has_viz:
            with h5py.File(viz_path, 'r') as f:
                self.faces = f['faces'][:]
            
            # Calculate rest areas for strain computation
            p0 = self.ref_positions[self.faces[:, 0]]
            p1 = self.ref_positions[self.faces[:, 1]]
            p2 = self.ref_positions[self.faces[:, 2]]
            self.face_rest_areas = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
        else:
            self.faces = None
            print("[Player] No visualization.h5 found. Face rendering disabled.")

    def _load_simulation(self):
        sim_path = self.output_dir / "simulation.h5"
        self.has_sim = sim_path.exists()

        if self.has_sim:
            self.sim_file = h5py.File(sim_path, 'r')
            self.ts_pos = self.sim_file['time_series/nodes/positions']
            self.ts_time = self.sim_file['time_series/time']
            self.n_frames = self.ts_pos.shape[0]

            if 'time_series/nodes/velocities' in self.sim_file:
                self.ts_vel = self.sim_file['time_series/nodes/velocities']
            else:
                self.ts_vel = None

            print(f"[Player] Simulation loaded: {self.n_frames} frames.")
        else:
            self.n_frames = 1
            self.ts_pos = None
            self.ts_time = None
            print("[Player] No simulation.h5 found. Running in Geometry Mode.")

    def _preallocate_buffers(self):
        self.curr_pos = np.zeros((self.n_nodes, 3), dtype='f4')
        self.bar_colors = np.zeros((self.n_bars, 3), dtype='f4')
        self.bar_colors[:] = self.config['rigid_color']

        if self.faces is not None:
            self.face_colors = np.zeros((len(self.faces), 4), dtype='f4')
            self.face_colors[:] = self.config['face_color']

        self.node_colors = np.zeros((self.n_nodes, 3), dtype='f4')
        self.node_colors[:] = (0.2, 0.2, 0.2)
        self.node_sizes = np.ones(self.n_nodes, dtype='f4') * 5.0

    def setup(self):
        if self.camera:
            self.camera.set_view('iso')
            self.camera.distance = self.L_char * 2.0

        if self.ui_manager:
            title = "Simulation Player" if self.has_sim else "Geometry Inspector"
            self.ui_manager.set_panel_title(title)

            self.lbl_status = Label((0, 0, 200, 20), f"Nodes: {self.n_nodes} | Bars: {self.n_bars}")
            self.ui_manager.add_widget("lbl_stat", self.lbl_status)

            if self.has_sim:
                self.lbl_time = Label((0, 0, 200, 20), "Time: 0.00s")
                self.ui_manager.add_widget("lbl_time", self.lbl_time)

                self.ui_manager.add_widget("btn_play",
                                           Button((0, 0, 80, 25), "Play/Pause", lambda: self._toggle_pause()))

                self.ui_manager.add_widget("sld_time",
                                           Slider((0, 0, 200, 20), "Frame", 0, self.n_frames - 1, 0, self._seek_frame))

                self.ui_manager.add_widget("sld_speed",
                                           Slider((0, 0, 150, 20), "Speed", 10, 300, 100,
                                                  lambda v: setattr(self, 'speed', v / 100.0)))

            if self.has_viz:
                self.ui_manager.add_widget("chk_faces",
                                           Checkbox((0, 0, 150, 20), "Show Faces", True,
                                                    lambda v: self.config.update({'show_faces': v})))

            self.ui_manager.add_widget("chk_strain",
                                       Checkbox((0, 0, 150, 20), "Strain Color", True,
                                                lambda v: self.config.update({'show_strain': v})))

            self.ui_manager.add_widget("chk_nodes",
                                       Checkbox((0, 0, 150, 20), "Show Nodes", True,
                                                lambda v: self.config.update({'show_nodes': v})))

            if self.has_sim:
                self.ui_manager.add_widget("chk_vel",
                                           Checkbox((0, 0, 150, 20), "Velocity", False,
                                                    lambda v: self.config.update({'show_velocity': v})))

                self.ui_manager.add_widget("chk_trail",
                                           Checkbox((0, 0, 150, 20), "Trails", False,
                                                    lambda v: self.config.update({'show_trails': v})))

    def _toggle_pause(self):
        self.paused = not self.paused

    def _seek_frame(self, frame_idx):
        self.float_frame = float(frame_idx)
        self.frame_idx = int(frame_idx)
        for k in self.trails: self.trails[k].clear()

    def _update_physics_state(self, dt):
        if self.has_sim:
            if not self.paused:
                self.float_frame += dt * 60 * self.speed
                if self.float_frame >= self.n_frames:
                    self.float_frame = 0
                    for k in self.trails: self.trails[k].clear()
                self.frame_idx = int(self.float_frame)

            self.curr_pos[:] = self.ts_pos[self.frame_idx]
        else:
            self.curr_pos[:] = self.ref_positions

    def _compute_strain_colors(self):
        limit = self.config['strain_limit']

        # 1. Bars
        if self.n_bars > 0:
            p1 = self.curr_pos[self.bar_indices[:, 0]]
            p2 = self.curr_pos[self.bar_indices[:, 1]]

            curr_lengths = np.linalg.norm(p1 - p2, axis=1)
            strains = (curr_lengths - self.bar_rest_lengths) / (self.bar_rest_lengths + 1e-9)

            t = np.clip((strains + limit) / (2 * limit), 0.0, 1.0)

            self.bar_colors[:, 0] = np.clip(1.5 - np.abs(t - 0.75) * 4, 0, 1)  # R
            self.bar_colors[:, 1] = np.clip(1.5 - np.abs(t - 0.5) * 4, 0, 1)  # G
            self.bar_colors[:, 2] = np.clip(1.5 - np.abs(t - 0.25) * 4, 0, 1)  # B

        # 2. Faces (Averaged Nodal Strain)
        if self.faces is not None:
            # Calculate per-face strain first
            p0 = self.curr_pos[self.faces[:, 0]]
            p1 = self.curr_pos[self.faces[:, 1]]
            p2 = self.curr_pos[self.faces[:, 2]]
            
            curr_areas = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
            face_strains = (curr_areas - self.face_rest_areas) / (self.face_rest_areas + 1e-9)
            
            # Accumulate strain at nodes
            node_strain_sum = np.zeros(self.n_nodes, dtype=float)
            node_strain_count = np.zeros(self.n_nodes, dtype=int)
            
            for i in range(len(self.faces)):
                strain = face_strains[i]
                for node_idx in self.faces[i]:
                    node_strain_sum[node_idx] += strain
                    node_strain_count[node_idx] += 1
            
            # Average strain at nodes
            # Avoid division by zero for isolated nodes
            mask = node_strain_count > 0
            node_avg_strain = np.zeros(self.n_nodes, dtype=float)
            node_avg_strain[mask] = node_strain_sum[mask] / node_strain_count[mask]
            
            # Re-assign smoothed strain to faces (average of its 3 nodes)
            smoothed_face_strains = (
                node_avg_strain[self.faces[:, 0]] +
                node_avg_strain[self.faces[:, 1]] +
                node_avg_strain[self.faces[:, 2]]
            ) / 3.0
            
            # Map smoothed strain to color
            t = np.clip((smoothed_face_strains + limit) / (2 * limit), 0.0, 1.0)
            
            self.face_colors[:, 0] = np.clip(1.5 - np.abs(t - 0.75) * 4, 0, 1)  # R
            self.face_colors[:, 1] = np.clip(1.5 - np.abs(t - 0.5) * 4, 0, 1)  # G
            self.face_colors[:, 2] = np.clip(1.5 - np.abs(t - 0.25) * 4, 0, 1)  # B
            self.face_colors[:, 3] = self.config['face_color'][3]

    def render(self, time_val, dt):
        self._update_physics_state(dt)

        if self.has_sim:
            t_curr = self.ts_time[self.frame_idx]
            self.lbl_time.text = f"Time: {t_curr:.3f}s | Frame: {self.frame_idx}"
            if not self.paused and self.ui_manager:
                self.ui_manager.widgets['sld_time'].value = self.frame_idx

        if self.config['show_strain']:
            self._compute_strain_colors()

        # 1. Faces
        if self.has_viz and self.config['show_faces'] and self.faces is not None:
            default_color = self.config['face_color']
            for i in range(len(self.faces)):
                idx0, idx1, idx2 = self.faces[i]
                v1 = tuple(self.curr_pos[idx0])
                v2 = tuple(self.curr_pos[idx1])
                v3 = tuple(self.curr_pos[idx2])
                
                if self.config['show_strain']:
                    color = tuple(self.face_colors[i])
                else:
                    color = default_color
                
                pgfx.draw_triangle(v1, v2, v3, color)

        # 2. Bars
        if self.config['show_edges'] and self.n_bars > 0:
            width = self.config['edge_width']
            rigid_c = self.config['rigid_color']

            for i in range(self.n_bars):
                idx0, idx1 = self.bar_indices[i]
                start = tuple(self.curr_pos[idx0])
                end = tuple(self.curr_pos[idx1])

                color = tuple(self.bar_colors[i]) if self.config['show_strain'] else rigid_c
                pgfx.draw_line(start, end, color=color, width=width)

        # 3. Nodes
        if self.config['show_nodes']:
            pgfx.draw_particles(
                positions=self.curr_pos,
                colors=self.node_colors,
                sizes=self.node_sizes
            )

        # 4. Extras
        if self.has_sim:
            if self.config['show_trails']:
                self._render_trails()
            if self.config['show_velocity'] and self.ts_vel is not None:
                self._render_velocity()

    def _render_trails(self):
        limit = self.config['trail_length']
        step = 1 if self.n_nodes < 1000 else 5
        for i in range(0, self.n_nodes, step):
            pos = tuple(self.curr_pos[i])
            self.trails[i].append(pos)
            if len(self.trails[i]) > limit:
                self.trails[i].pop(0)
            if len(self.trails[i]) > 1:
                pgfx.draw_path(self.trails[i], color=(1, 0.5, 0), width=1.0)

    def _render_velocity(self):
        vels = self.ts_vel[self.frame_idx]
        scale = self.config['velocity_scale']
        norms = np.linalg.norm(vels, axis=1)
        mask_indices = np.where(norms > 1e-4)[0]
        arrow_color = (0, 1, 1)

        for idx in mask_indices:
            start = self.curr_pos[idx]
            vec = vels[idx] * scale
            end = start + vec
            pgfx.draw_arrow(
                start=tuple(start),
                end=tuple(end),
                color=arrow_color,
                head_size=self.L_char * 0.05,
                width_radius=self.L_char * 0.005
            )

    def cleanup(self):
        if hasattr(self, 'sim_file'):
            self.sim_file.close()


def play_experiment(experiment_name):
    path = Path(experiment_name)
    player = SimulationPlayer(path)
    studio = PiVizStudio(scene_fx=player)
    studio.run()


if __name__ == "__main__":
    # --- ARGUMENT HANDLING FIX ---
    # PiViz (moderngl_window) will parse sys.argv automatically.
    # We must extract our experiment path and remove it from sys.argv
    # so PiViz doesn't crash on "unrecognized arguments".

    experiment_arg = None

    # Filter arguments
    new_argv = [sys.argv[0]]
    for arg in sys.argv[1:]:
        if arg.startswith("-"):
            # This looks like a flag (e.g., -wnd, --fullscreen), keep it for PiViz
            new_argv.append(arg)
        else:
            # This looks like our positional path argument
            experiment_arg = arg

    # Replace sys.argv with the "clean" version that PiViz expects
    sys.argv = new_argv

    # Run logic
    if experiment_arg:
        play_experiment(experiment_arg)
    else:
        print("Usage: python viz_player.py <path_to_experiment> [piviz_options]")
        print("Example: python viz_player.py experiments/test_01 -wnd glfw")
