"""
DEMLAT Visualization Player
============================

A robust, high-performance visualization tool for DEMLAT simulation results.
Automatically adapts to available data and provides comprehensive visual feedback.

Features:
---------
- Automatic mode detection (geometry-only vs. full simulation)
- Robust handling of missing/optional data
- Auto-scaling for optimal visualization
- Strain/stress visualization with colormaps
- Interactive UI with real-time controls
- Support for both bar-hinge and origami models
- Efficient batch rendering for large systems

Usage:
------
    from demlat.utils.viz_player import ShowSimulation
    
    # Auto-detect and visualize
    ShowSimulation("path/to/experiment")
    
    # Custom configuration
    ShowSimulation("path/to/experiment", config={
        'show_strain': True,
        'show_stress': False,
        'colormap': 'jet',
        'auto_scale': True
    })

Author: Yogesh Phalak
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import json
import sys

from piviz import PiVizStudio, PiVizFX, pgfx, Palette
from piviz.ui import Slider, Label, Button, Checkbox
from piviz import Colors, Colormap


# =============================================================================
# DATA LOADER WITH ROBUST ERROR HANDLING
# =============================================================================

class ExperimentData:
    """
    Loads and validates experiment data with comprehensive error handling.
    """

    def __init__(self, experiment_path: str):
        self.exp_path = Path(experiment_path)
        self.mode = None  # 'geometry' or 'simulation'

        # File paths
        self.geometry_path = self.exp_path / "input" / "geometry.h5"
        self.simulation_path = self.exp_path / "output" / "simulation.h5"
        self.visualization_path = self.exp_path / "input" / "visualization.h5"
        self.config_path = self.exp_path / "input" / "config.json"

        # Data containers
        self.nodes = {}
        self.elements = {}
        self.visualization = {}
        self.time_series = {}
        self.config = {}
        self.metadata = {}

        # Computed properties
        self.n_nodes = 0
        self.n_bars = 0
        self.n_hinges = 0
        self.n_frames = 1
        self.has_faces = False

        # Load data
        self._load_data()
        self._validate_data()
        self._compute_properties()

    def _load_data(self):
        """Load all available data files."""
        print("\n" + "=" * 60)
        print("DEMLAT Visualization Player - Loading Data")
        print("=" * 60)

        if not self.geometry_path.exists():
            raise FileNotFoundError(f"Geometry file not found: {self.geometry_path}")

        self._load_geometry()

        if self.simulation_path.exists():
            self.mode = 'simulation'
            self._load_simulation()
            print(f"[Mode] Full Simulation Playback")
        else:
            self.mode = 'geometry'
            print(f"[Mode] Geometry Viewer (no simulation data)")

        if self.visualization_path.exists():
            self._load_visualization()

        if self.config_path.exists():
            self._load_config()

    def _load_geometry(self):
        """Load geometry.h5 with robust handling of optional datasets."""
        print(f"\n[Loading] {self.geometry_path.name}")

        with h5py.File(self.geometry_path, 'r') as f:
            if 'nodes' not in f:
                raise ValueError("geometry.h5 missing 'nodes' group")

            nodes_grp = f['nodes']

            if 'positions' not in nodes_grp:
                raise ValueError("geometry.h5 missing 'nodes/positions'")
            self.nodes['positions'] = nodes_grp['positions'][:]
            self.n_nodes = self.nodes['positions'].shape[0]

            if 'masses' in nodes_grp:
                self.nodes['masses'] = nodes_grp['masses'][:]
            else:
                self.nodes['masses'] = np.ones(self.n_nodes, dtype='f4')

            if 'attributes' in nodes_grp:
                self.nodes['attributes'] = nodes_grp['attributes'][:]
            else:
                self.nodes['attributes'] = np.zeros(self.n_nodes, dtype='u1')

            if 'radius' in nodes_grp:
                self.nodes['radius'] = nodes_grp['radius'][:]
            else:
                self.nodes['radius'] = None

            if 'elements' not in f:
                print("  [Warning] No elements found in geometry")
                return

            elem_grp = f['elements']

            if 'bars' in elem_grp:
                bars_grp = elem_grp['bars']
                self.elements['bars'] = {
                    'indices': bars_grp['indices'][:] if 'indices' in bars_grp else None,
                    'stiffness': bars_grp['stiffness'][:] if 'stiffness' in bars_grp else None,
                    'rest_length': bars_grp['rest_length'][:] if 'rest_length' in bars_grp else None,
                    'damping': bars_grp['damping'][:] if 'damping' in bars_grp else None,
                    'prestress': bars_grp['prestress'][:] if 'prestress' in bars_grp else None,
                }
                if self.elements['bars']['indices'] is not None:
                    self.n_bars = self.elements['bars']['indices'].shape[0]
                    print(f"  [Loaded] {self.n_bars} bars")

            if 'hinges' in elem_grp:
                hinges_grp = elem_grp['hinges']
                self.elements['hinges'] = {
                    'indices': hinges_grp['indices'][:] if 'indices' in hinges_grp else None,
                    'stiffness': hinges_grp['stiffness'][:] if 'stiffness' in hinges_grp else None,
                    'rest_angle': hinges_grp['rest_angle'][:] if 'rest_angle' in hinges_grp else None,
                }
                if self.elements['hinges']['indices'] is not None:
                    self.n_hinges = self.elements['hinges']['indices'].shape[0]
                    print(f"  [Loaded] {self.n_hinges} hinges")

        if self.elements.get('hinges') and self.elements['hinges']['indices'] is not None:
            hinge_indices = self.elements['hinges']['indices']
            positions = self.nodes['positions']
            self.elements['hinges']['edge_midpoints'] = (
                                                                positions[hinge_indices[:, 0]] + positions[hinge_indices[:, 1]]
                                                        ) / 2
            edge_vec = positions[hinge_indices[:, 1]] - positions[hinge_indices[:, 0]]
            wing1_vec = positions[hinge_indices[:, 2]] - self.elements['hinges']['edge_midpoints']
            wing2_vec = positions[hinge_indices[:, 3]] - self.elements['hinges']['edge_midpoints']
            normals = np.cross(wing1_vec, wing2_vec)
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1.0, norms)
            self.elements['hinges']['normals'] = normals / norms
            print(f"  [Computed] Hinge visualization data")

        print(f"  [Loaded] {self.n_nodes} nodes")

    def _load_simulation(self):
        """Load simulation.h5 with robust handling."""
        print(f"\n[Loading] {self.simulation_path.name}")

        with h5py.File(self.simulation_path, 'r') as f:
            self.metadata = dict(f.attrs)

            if 'time_series' not in f:
                print("  [Warning] No time_series group found")
                self.mode = 'geometry'
                return

            ts_grp = f['time_series']

            if 'time' in ts_grp:
                self.time_series['time'] = ts_grp['time'][:]
                self.n_frames = len(self.time_series['time'])
                print(f"  [Loaded] {self.n_frames} frames")
            else:
                print("  [Warning] No time array found")
                self.mode = 'geometry'
                return

            if 'nodes' in ts_grp:
                nodes_grp = ts_grp['nodes']
                if 'positions' in nodes_grp:
                    self.time_series['positions'] = nodes_grp['positions'][:]
                    print(f"  [Loaded] Node positions [{self.time_series['positions'].shape}]")
                else:
                    print("  [Warning] No node positions in simulation")
                    self.mode = 'geometry'
                    return

                if 'velocities' in nodes_grp:
                    self.time_series['velocities'] = nodes_grp['velocities'][:]
                    print(f"  [Loaded] Node velocities")
                if 'kinetic_energy' in nodes_grp:
                    self.time_series['node_kinetic'] = nodes_grp['kinetic_energy'][:]
                if 'potential_energy' in nodes_grp:
                    self.time_series['node_potential'] = nodes_grp['potential_energy'][:]

            if 'elements' in ts_grp and 'bars' in ts_grp['elements']:
                bars_grp = ts_grp['elements']['bars']
                if 'strain' in bars_grp:
                    self.time_series['bar_strain'] = bars_grp['strain'][:]
                    print(f"  [Loaded] Bar strains")
                if 'stress' in bars_grp:
                    self.time_series['bar_stress'] = bars_grp['stress'][:]
                    print(f"  [Loaded] Bar stresses")
                if 'potential_energy' in bars_grp:
                    self.time_series['bar_energy'] = bars_grp['potential_energy'][:]

            if 'elements' in ts_grp and 'hinges' in ts_grp['elements']:
                hinges_grp = ts_grp['elements']['hinges']
                if 'angle' in hinges_grp:
                    self.time_series['hinge_angle'] = hinges_grp['angle'][:]
                    print(f"  [Loaded] Hinge angles")
                if 'torsional_strain' in hinges_grp:
                    self.time_series['hinge_strain'] = hinges_grp['torsional_strain'][:]
                    print(f"  [Loaded] Hinge strains")

    def _load_visualization(self):
        """Load visualization.h5 with backward compatibility."""
        print(f"\n[Loading] {self.visualization_path.name}")

        with h5py.File(self.visualization_path, 'r') as f:
            if 'faces' in f:
                if isinstance(f['faces'], h5py.Group) and 'triangles' in f['faces']:
                    self.visualization['faces'] = f['faces']['triangles'][:]
                    self.has_faces = True
                    print(f"  [Loaded] {len(self.visualization['faces'])} triangular faces")
                elif isinstance(f['faces'], h5py.Dataset):
                    self.visualization['faces'] = f['faces'][:]
                    self.has_faces = True
                    print(f"  [Loaded] {len(self.visualization['faces'])} faces (legacy format)")

            if 'appearance' in f:
                app_grp = f['appearance']
                if 'face_colors' in app_grp:
                    self.visualization['face_colors'] = app_grp['face_colors'][:]
                if 'face_opacity' in app_grp:
                    self.visualization['face_opacity'] = app_grp['face_opacity'][:]

    def _load_config(self):
        """Load config.json (optional)."""
        print(f"\n[Loading] {self.config_path.name}")
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            print(f"  [Loaded] Configuration")
        except Exception as e:
            print(f"  [Warning] Could not load config: {e}")

    def _validate_data(self):
        """Validate loaded data for consistency."""
        if self.mode == 'simulation':
            expected_shape = (self.n_frames, self.n_nodes, 3)
            actual_shape = self.time_series['positions'].shape
            if actual_shape != expected_shape:
                raise ValueError(
                    f"Position data shape mismatch: expected {expected_shape}, got {actual_shape}"
                )

    def _compute_properties(self):
        """Compute derived properties for visualization."""
        if self.n_bars > 0 and self.elements['bars']['rest_length'] is None:
            indices = self.elements['bars']['indices']
            pos = self.nodes['positions']
            rest_lengths = np.linalg.norm(
                pos[indices[:, 0]] - pos[indices[:, 1]], axis=1
            )
            self.elements['bars']['rest_length'] = rest_lengths.astype('f4')
            print("[Computed] Bar rest lengths from initial geometry")


# =============================================================================
# VISUALIZATION SCALE CALCULATOR
# =============================================================================

class VisualizationScales:
    """Automatically computes optimal visualization scales."""

    def __init__(self, data: ExperimentData):
        self.data = data
        self.L_char = 1.0
        self.extent = np.array([1.0, 1.0, 1.0])
        self.center = np.array([0.0, 0.0, 0.0])
        self.node_radius = 0.1
        self.bar_radius = 0.05
        self.velocity_scale = 1.0
        self.strain_limit = 0.3
        self.stress_limit = None
        self._compute_scales()

    def _compute_scales(self):
        """Compute all visualization scales."""
        print("\n" + "=" * 60)
        print("Computing Visualization Scales")
        print("=" * 60)

        positions = self.data.nodes['positions']
        self.extent = np.max(positions, axis=0) - np.min(positions, axis=0)
        self.center = np.mean(positions, axis=0)
        self.L_char = np.median(self.extent) if np.any(self.extent > 0) else 1.0

        if self.data.n_bars > 0:
            rest_lengths = self.data.elements['bars']['rest_length']
            if rest_lengths is not None:
                median_length = np.median(rest_lengths)
                if median_length > 0:
                    self.L_char = median_length

        print(f"Characteristic Length: {self.L_char:.4f}")
        print(f"Extent: [{self.extent[0]:.2f}, {self.extent[1]:.2f}, {self.extent[2]:.2f}]")
        print(f"Center: [{self.center[0]:.2f}, {self.center[1]:.2f}, {self.center[2]:.2f}]")

        if self.data.nodes['radius'] is not None:
            self.node_radius = np.median(self.data.nodes['radius'])
        else:
            self.node_radius = self.L_char * 0.05
        print(f"Node Radius: {self.node_radius:.4f}")

        self.bar_radius = self.L_char * 0.03
        print(f"Bar Radius: {self.bar_radius:.4f}")

        if self.data.mode == 'simulation' and 'velocities' in self.data.time_series:
            self._compute_velocity_scale()

        if 'bar_strain' in self.data.time_series:
            strains = self.data.time_series['bar_strain']
            self.strain_limit = np.percentile(np.abs(strains), 95)
            print(f"Strain Limit (95th percentile): ±{self.strain_limit:.4f}")

        if 'bar_stress' in self.data.time_series:
            stresses = self.data.time_series['bar_stress']
            self.stress_limit = np.percentile(np.abs(stresses), 95)
            print(f"Stress Limit (95th percentile): ±{self.stress_limit:.2e}")

    def _compute_velocity_scale(self):
        """Auto-tune velocity arrow scaling."""
        vels = self.data.time_series['velocities']
        sample_indices = np.linspace(0, len(vels) - 1, min(10, len(vels)), dtype=int)
        max_vel = 0.0
        for idx in sample_indices:
            v_mag = np.linalg.norm(vels[idx], axis=1)
            max_vel = max(max_vel, np.max(v_mag))
        if max_vel > 1e-9:
            self.velocity_scale = (1.5 * self.L_char) / max_vel
        else:
            self.velocity_scale = 1.0
        print(f"Velocity Scale: {self.velocity_scale:.4f}")


# =============================================================================
# VECTORIZED COLORMAP HELPERS
# =============================================================================

def _jet_colormap_batch(t):
    """
    Vectorized jet colormap: t in [0, 1] -> (N, 3) float32 RGB.
    Matches Colormap.jet() behavior but operates on entire arrays.
    """
    t = np.clip(t, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(t - 0.75) * 4, 0, 1)
    g = np.clip(1.5 - np.abs(t - 0.5) * 4, 0, 1)
    b = np.clip(1.5 - np.abs(t - 0.25) * 4, 0, 1)
    return np.column_stack([r, g, b]).astype('f4')


def _jet_colormap_batch_rgba(t, alpha=1.0):
    """Vectorized jet colormap returning (N, 4) RGBA."""
    rgb = _jet_colormap_batch(t)
    rgba = np.empty((len(t), 4), dtype='f4')
    rgba[:, :3] = rgb
    rgba[:, 3] = alpha
    return rgba


# =============================================================================
# MAIN VISUALIZATION CLASS
# =============================================================================

class DEMLATVisualizer(PiVizFX):
    """
    High-performance visualizer for DEMLAT simulations.
    All rendering paths fully vectorized using bulk batch APIs.
    """

    def __init__(self, data: ExperimentData, config: Optional[Dict] = None):
        super().__init__()

        self.data = data
        self.scales = VisualizationScales(data)

        # Playback state
        self.timestep_idx = 0
        self.float_timestep = 0.0
        self.paused = False
        self.speed = 1.0
        self.frame_step = 1
        self._updating_slider = False

        # Configuration
        self.config = {
            'show_nodes': True,
            'show_bars': True,
            'show_hinges': False,
            'show_faces': True,
            'show_velocity': False,
            'show_trails': False,
            'trail_length': 50,
            'show_strain': True,
            'show_stress': False,
            'use_particles_for_nodes': False,
            'particle_threshold': 300,
            'use_lines_for_bars': False,
            'sphere_detail': 12,
            'cylinder_detail': 8,
            'strain_limit': None,
            'base_node_radius': None,
            'velocity_scale': None,
            'hinge_color_mountain': (0.2, 0.6, 1.0),
            'hinge_color_valley': (1.0, 0.4, 0.2),
            'hinge_color_flat': (0.5, 0.5, 0.5),
            'hinge_indicator_scale': 0.15,
            'show_hinge_rest_angle': True,
            'hinge_force_scale': 0.5,
            'hinge_geom_scale': 0.4,
        }

        if config:
            self.config.update(config)

        # Auto-adjust based on system size
        if self.data.n_nodes > 500:
            self.config['sphere_detail'] = 8
            self.config['cylinder_detail'] = 6
            self.config['use_particles_for_nodes'] = True
        if self.data.n_bars > 500:
            self.config['use_lines_for_bars'] = True

        if self.config['strain_limit'] is None:
            self.config['strain_limit'] = self.scales.strain_limit
        if self.config['base_node_radius'] is None:
            self.config['base_node_radius'] = self.scales.node_radius
        if self.config['velocity_scale'] is None:
            self.config['velocity_scale'] = self.scales.velocity_scale

        self._categorize_nodes()
        self.trails = {i: [] for i in range(self.data.n_nodes)}
        self._preallocate_batch_arrays()

    def _categorize_nodes(self):
        """Categorize nodes by type."""
        attrs = self.data.nodes['attributes']
        self.fixed_nodes_idx = np.where((attrs & 1) != 0)[0]
        self.position_actuators_idx = np.where((attrs & 2) != 0)[0]
        self.force_actuators_idx = np.where((attrs & 4) != 0)[0]
        self.floating_nodes_idx = np.where(attrs == 0)[0]

        print(f"\n[Node Categories]")
        print(f"  Fixed: {len(self.fixed_nodes_idx)}")
        print(f"  Position Actuators: {len(self.position_actuators_idx)}")
        print(f"  Force Actuators: {len(self.force_actuators_idx)}")
        print(f"  Floating: {len(self.floating_nodes_idx)}")

    def _preallocate_batch_arrays(self):
        """Pre-allocate all arrays for rendering — no per-frame allocation."""
        n = self.data.n_nodes
        n_bars = self.data.n_bars
        n_floating = len(self.floating_nodes_idx)

        self.all_positions = np.zeros((n, 3), dtype='f4')

        # Bars — pre-allocated for vectorized rendering
        if n_bars > 0:
            self.bar_indices = self.data.elements['bars']['indices']
            self.bar_rest_lengths = self.data.elements['bars']['rest_length']
            self.bar_colors = np.zeros((n_bars, 3), dtype='f4')
            self.bar_start_pos = np.zeros((n_bars, 3), dtype='f4')
            self.bar_end_pos = np.zeros((n_bars, 3), dtype='f4')

        # Particles
        if n_floating > 0:
            self.particle_positions = np.zeros((n_floating, 3), dtype='f4')
            self.particle_colors = np.full((n_floating, 3), (0.3, 0.3, 0.3), dtype='f4')
            self.particle_sizes = np.full(n_floating, self.scales.node_radius * 80, dtype='f4')

        # Faces — pre-extract vertex index arrays for fancy indexing
        if self.data.has_faces:
            n_faces = len(self.data.visualization['faces'])
            self.face_indices = self.data.visualization['faces']
            self.face_idx0 = self.face_indices[:, 0].astype('i4')
            self.face_idx1 = self.face_indices[:, 1].astype('i4')
            self.face_idx2 = self.face_indices[:, 2].astype('i4')
            self.face_default_opacity = 0.7
            self.vertex_strains = np.zeros(n, dtype='f4')
            self.vertex_counts = np.zeros(n, dtype='f4')

        # Velocities
        if 'velocities' in self.data.time_series:
            self.current_velocities = np.zeros((n, 3), dtype='f4')

        # Hinges
        if self.data.n_hinges > 0 and self.data.elements.get('hinges'):
            hinges = self.data.elements['hinges']
            self.hinge_indices = hinges['indices']
            self.hinge_rest_angles = hinges.get('rest_angle')
            self.hinge_edge_midpoints = hinges.get('edge_midpoints')
            self.hinge_normals = hinges.get('normals')
            self.hinge_colors = np.zeros((self.data.n_hinges, 3), dtype='f4')
            print(f"[Preallocated] Hinge visualization arrays ({self.data.n_hinges} hinges)")

        print("[Preallocated] All rendering arrays")

    def setup(self):
        """Initialize visualization."""
        if self.camera:
            self.camera.set_view('iso')
            max_extent = np.max(self.scales.extent)
            self.camera.distance = max_extent * 3.0
            self.camera.target = tuple(self.scales.center)

        if self.ui_manager:
            self._setup_ui()

        print("\n" + "=" * 60)
        print("Visualization Ready")
        print("=" * 60)
        print(f"Mode: {self.data.mode.upper()}")
        print(f"Nodes: {self.data.n_nodes} | Bars: {self.data.n_bars}")
        if self.data.mode == 'simulation':
            print(f"Frames: {self.data.n_frames}")
            print(f"Duration: {self.data.time_series['time'][-1]:.2f}s")
        print("=" * 60 + "\n")

    # ============ FRAME-BY-FRAME CONTROL METHODS ============

    def _on_timeline_change(self, value):
        if self._updating_slider:
            return
        self.timestep_idx = int(value)
        self.float_timestep = float(self.timestep_idx)
        self.paused = True
        self._update_pause_button()

    def _on_step_change(self, value):
        self.frame_step = max(1, int(value))

    def _goto_start(self):
        self.timestep_idx = 0
        self.float_timestep = 0.0
        self._clear_trails()
        self._sync_timeline_slider()

    def _goto_end(self):
        self.timestep_idx = self.data.n_frames - 1
        self.float_timestep = float(self.timestep_idx)
        self._sync_timeline_slider()

    def _step_forward(self):
        self.paused = True
        self._update_pause_button()
        step = max(1, int(self.frame_step))
        self.timestep_idx = min(self.data.n_frames - 1, self.timestep_idx + step)
        self.float_timestep = float(self.timestep_idx)
        self._sync_timeline_slider()

    def _step_backward(self):
        self.paused = True
        self._update_pause_button()
        step = max(1, int(self.frame_step))
        self.timestep_idx = max(0, self.timestep_idx - step)
        self.float_timestep = float(self.timestep_idx)
        self._sync_timeline_slider()

    def _toggle_pause(self):
        self.paused = not self.paused
        self._update_pause_button()

    def _update_pause_button(self):
        if hasattr(self, 'ui_manager') and self.ui_manager:
            btn = self.ui_manager.get_widget("btn_pause")
            if btn:
                btn.text = ">" if self.paused else "||"

    def _sync_timeline_slider(self):
        if hasattr(self, 'ui_manager') and self.ui_manager:
            slider = self.ui_manager.get_widget("sld_timeline")
            if slider:
                self._updating_slider = True
                slider.value = self.timestep_idx
                self._updating_slider = False

    def _clear_trails(self):
        for k in self.trails:
            self.trails[k].clear()

    def _reset_sim(self):
        self._goto_start()
        self.paused = False
        self.timestep_idx = 0
        self.float_timestep = 0.0
        for k in self.trails:
            self.trails[k].clear()
        self._update_pause_button()

    # ============ UI SETUP ============

    def _setup_ui(self):
        """Setup UI controls."""
        self.ui_manager.set_panel_title("DEMLAT Player")

        if self.data.mode == 'simulation':
            self.lbl_time = Label("Time: 0.00s | Frame: 0")
            self.ui_manager.add_widget("lbl_time", self.lbl_time)
        else:
            self.lbl_mode = Label("Geometry Viewer", color=(0.7, 0.9, 1.0))
            self.ui_manager.add_widget("lbl_mode", self.lbl_mode)

        self.lbl_perf = Label(
            f"Nodes: {self.data.n_nodes} | Bars: {self.data.n_bars}",
            color=(0.7, 0.7, 0.7)
        )
        self.ui_manager.add_widget("lbl_perf", self.lbl_perf)

        if self.data.mode == 'simulation':
            self.ui_manager.add_widget("sld_timeline",
                                       Slider("Frame", 0, self.data.n_frames - 1, 0, self._on_timeline_change))
            self.ui_manager.add_widget("btn_start", Button("<<", self._goto_start))
            self.ui_manager.add_widget("btn_prev", Button("<-", self._step_backward))
            self.ui_manager.add_widget("btn_pause", Button("||", self._toggle_pause))
            self.ui_manager.add_widget("btn_next", Button("->", self._step_forward))
            self.ui_manager.add_widget("btn_end", Button(">>", self._goto_end))
            self.ui_manager.add_widget("sld_speed",
                                       Slider("Speed", 10, 500, int(self.speed * 100),
                                              lambda v: setattr(self, 'speed', v / 100.0)))
            self.frame_step = 1
            self.ui_manager.add_widget("sld_step",
                                       Slider("Step", 1, 50, 1, self._on_step_change))

        if self.data.n_hinges > 0:
            self.ui_manager.add_widget("chk_hinges",
                                       Checkbox("Show Hinges", self.config['show_hinges'],
                                                lambda v: self.config.update({'show_hinges': v})))

        self.ui_manager.add_widget("chk_hinge_rest",
                                   Checkbox("Show Rest Angle", self.config['show_hinge_rest_angle'],
                                            lambda v: self.config.update({'show_hinge_rest_angle': v})))
        self.ui_manager.add_widget("sld_hinge_force",
                                   Slider("Hinge Force Scale", 0.1, 5.0, self.config['hinge_force_scale'],
                                          lambda v: self.config.update({'hinge_force_scale': v})))
        self.ui_manager.add_widget("sld_hinge_geom",
                                   Slider("Hinge Size", 0.1, 1.0, self.config['hinge_geom_scale'],
                                          lambda v: self.config.update({'hinge_geom_scale': v})))
        self.ui_manager.add_widget("chk_nodes",
                                   Checkbox("Show Nodes", self.config['show_nodes'],
                                            lambda v: self.config.update({'show_nodes': v})))
        self.ui_manager.add_widget("chk_bars",
                                   Checkbox("Show Bars", self.config['show_bars'],
                                            lambda v: self.config.update({'show_bars': v})))

        if self.data.has_faces:
            self.ui_manager.add_widget("chk_faces",
                                       Checkbox("Show Faces", self.config['show_faces'],
                                                lambda v: self.config.update({'show_faces': v})))
        if 'bar_strain' in self.data.time_series:
            self.ui_manager.add_widget("chk_strain",
                                       Checkbox("Strain Color", self.config['show_strain'],
                                                lambda v: self.config.update({'show_strain': v})))
        if 'velocities' in self.data.time_series:
            self.ui_manager.add_widget("chk_vel",
                                       Checkbox("Show Velocity", self.config['show_velocity'],
                                                lambda v: self.config.update({'show_velocity': v})))

        self.ui_manager.add_widget("chk_trails",
                                   Checkbox("Show Trails", self.config['show_trails'],
                                            lambda v: self.config.update({'show_trails': v})))
        self.ui_manager.add_widget("chk_lines",
                                   Checkbox("Fast Lines", self.config['use_lines_for_bars'],
                                            lambda v: self.config.update({'use_lines_for_bars': v})))
        self.ui_manager.add_widget("chk_particles",
                                   Checkbox("Particle Nodes", self.config['use_particles_for_nodes'],
                                            lambda v: self.config.update({'use_particles_for_nodes': v})))

    # ====================================================================
    # RENDER — fully vectorized, zero per-element Python loops
    # ====================================================================

    def render(self, time_val, dt):
        """Main render loop."""
        # Update playback
        if self.data.mode == 'simulation' and not self.paused:
            self.float_timestep += dt * 60 * self.speed
            if self.float_timestep >= self.data.n_frames:
                self.float_timestep = 0.0
                self._clear_trails()
            self.timestep_idx = int(self.float_timestep)
            self._sync_timeline_slider()

        idx = self.timestep_idx

        # Get current frame
        if self.data.mode == 'simulation':
            x_curr = self.data.time_series['positions'][idx]
        else:
            x_curr = self.data.nodes['positions']
        self.all_positions[:] = x_curr

        # Velocities
        vels = None
        if self.config['show_velocity'] and idx > 0 and 'velocities' in self.data.time_series:
            self.current_velocities[:] = self.data.time_series['velocities'][idx]
            vels = self.current_velocities

        # --- DRAWING LAYERS ---
        if self.config['show_trails']:
            self._render_trails(x_curr)
        if self.config['show_faces']:
            self._render_faces()
        if self.config['show_bars']:
            self._render_bars()
        if self.config['show_hinges'] and self.data.n_hinges > 0:
            self._render_hinges()
        if self.config['show_nodes']:
            self._render_nodes()
        if vels is not None:
            self._render_velocity_arrows(vels)

        # UI Updates
        if self.data.mode == 'simulation' and hasattr(self, 'lbl_time'):
            t_display = self.data.time_series['time'][idx]
            status = "PAUSED" if self.paused else "PLAYING"
            self.lbl_time.text = f"Time: {t_display:.3f}s | Frame: {idx}/{self.data.n_frames - 1} | {status}"

    # ====================================================================
    # VECTORIZED RENDER METHODS
    # ====================================================================

    def _render_bars(self):
        """Render bars — fully vectorized with bulk batch API."""
        if self.data.n_bars == 0:
            return

        # Vectorized position gather
        self.bar_start_pos[:] = self.all_positions[self.bar_indices[:, 0]]
        self.bar_end_pos[:] = self.all_positions[self.bar_indices[:, 1]]

        # Vectorized strain coloring
        if self.config['show_strain'] and 'bar_strain' in self.data.time_series:
            strains = self.data.time_series['bar_strain'][self.timestep_idx]
            limit = self.config['strain_limit']
            t = np.clip((strains + limit) / (2 * limit), 0.0, 1.0)
            self.bar_colors[:] = _jet_colormap_batch(t)
        else:
            self.bar_colors[:] = (0.5, 0.5, 0.5)

        # Bulk render — single call, zero per-bar loop
        if self.config['use_lines_for_bars']:
            pgfx.draw_lines_batch(
                self.bar_start_pos, self.bar_end_pos,
                self.bar_colors, width=2.0
            )
        else:
            pgfx.draw_cylinders_batch(
                starts=self.bar_start_pos,
                ends=self.bar_end_pos,
                radii=self.scales.bar_radius,
                colors=self.bar_colors,
                detail=self.config['cylinder_detail']
            )

    def _render_faces(self):
        """Render faces — fully vectorized with bulk batch API."""
        if not self.data.has_faces:
            return

        # Gather vertex positions via fancy indexing (vectorized)
        v0 = self.all_positions[self.face_idx0]  # (n_faces, 3)
        v1 = self.all_positions[self.face_idx1]
        v2 = self.all_positions[self.face_idx2]

        if self.config['show_strain'] and 'bar_strain' in self.data.time_series:
            strains = self.data.time_series['bar_strain'][self.timestep_idx]

            # Vectorized per-vertex strain accumulation
            self.vertex_strains[:] = 0
            self.vertex_counts[:] = 0
            np.add.at(self.vertex_strains, self.bar_indices[:, 0], strains)
            np.add.at(self.vertex_strains, self.bar_indices[:, 1], strains)
            np.add.at(self.vertex_counts, self.bar_indices[:, 0], 1)
            np.add.at(self.vertex_counts, self.bar_indices[:, 1], 1)
            mask = self.vertex_counts > 0
            self.vertex_strains[mask] /= self.vertex_counts[mask]

            # Vectorized colormap for all vertices at once
            limit = self.config['strain_limit']
            t_all = np.clip((self.vertex_strains + limit) / (2 * limit), 0.0, 1.0)
            all_colors_rgba = _jet_colormap_batch_rgba(t_all, self.face_default_opacity)

            # Gather per-face vertex colors
            c0 = all_colors_rgba[self.face_idx0]  # (n_faces, 4)
            c1 = all_colors_rgba[self.face_idx1]
            c2 = all_colors_rgba[self.face_idx2]

            pgfx.draw_faces_batch(v0, v1, v2, c0, c1, c2)
        else:
            # Static color — single flat color for all faces
            n_faces = len(self.face_idx0)
            static_color = np.full((n_faces, 4), (0.85, 0.85, 0.85, self.face_default_opacity), dtype='f4')
            pgfx.draw_triangles_batch(v0, v1, v2, static_color)

    def _render_nodes(self):
        """Render nodes — vectorized with bulk batch APIs."""
        n_floating = len(self.floating_nodes_idx)
        use_particles = (self.config['use_particles_for_nodes'] or
                         n_floating > self.config['particle_threshold'])

        # Floating nodes
        if n_floating > 0:
            self.particle_positions[:] = self.all_positions[self.floating_nodes_idx]

            if use_particles:
                pgfx.draw_particles(
                    positions=self.particle_positions,
                    colors=self.particle_colors,
                    sizes=self.particle_sizes
                )
            else:
                # Bulk sphere rendering
                radii = np.full(n_floating, self.scales.node_radius, dtype='f4')
                pgfx.draw_spheres_batch(
                    centers=self.particle_positions,
                    radii=radii,
                    colors=self.particle_colors,
                    detail=self.config['sphere_detail']
                )

        # Fixed nodes (cubes) — small count, loop is fine
        s_fixed = self.scales.node_radius * 2.0
        for i in self.fixed_nodes_idx:
            pgfx.draw_cube(
                center=tuple(self.all_positions[i]),
                size=(s_fixed, s_fixed, s_fixed),
                color=Palette.Standard10[3]
            )

        # Position actuators
        s_act = self.scales.node_radius * 2.0
        for i in self.position_actuators_idx:
            pgfx.draw_cube(
                center=tuple(self.all_positions[i]),
                size=(s_act, s_act, s_act),
                color=Palette.Standard10[2]
            )

        # Force actuators
        s_force = self.scales.node_radius * 1.5
        for i in self.force_actuators_idx:
            pgfx.draw_cube(
                center=tuple(self.all_positions[i]),
                size=(s_force, s_force, s_force),
                color=Palette.Standard10[1]
            )

    def _render_velocity_arrows(self, vels):
        """Render velocity arrows — vectorized for floating nodes."""
        idx = self.floating_nodes_idx
        if len(idx) == 0:
            return

        positions = self.all_positions[idx]      # (n, 3)
        velocities = vels[idx]                   # (n, 3)
        v_mag = np.linalg.norm(velocities, axis=1)

        # Filter: only draw arrows with significant velocity
        valid = v_mag > 1e-6
        if not np.any(valid):
            return

        starts = positions[valid]
        scale = self.config['velocity_scale']
        ends = starts + velocities[valid] * scale

        # Use cylinders for arrow shafts + cones for heads
        # For simplicity and performance, use lines for many arrows
        n_arrows = len(starts)
        arrow_colors = np.full((n_arrows, 3), (0.0, 1.0, 1.0), dtype='f4')

        if n_arrows > 200:
            # Many arrows: use fast lines
            pgfx.draw_lines_batch(starts, ends, arrow_colors, width=2.0)
        else:
            # Fewer arrows: use proper arrow geometry
            head_size = self.scales.L_char * 0.1
            width_radius = self.scales.L_char * 0.02
            for i in range(n_arrows):
                pgfx.draw_arrow(
                    start=tuple(starts[i]),
                    end=tuple(ends[i]),
                    color=(0.0, 1.0, 1.0),
                    head_size=head_size,
                    width_radius=width_radius
                )

    def _render_trails(self, x_curr):
        """Render motion trails — uses batch line API for accumulated trails."""
        limit = self.config['trail_length']

        # Update trail buffers (this must be per-node, but it's just list.append)
        for i in self.floating_nodes_idx:
            pos = tuple(x_curr[i])
            trail = self.trails[i]
            trail.append(pos)
            if len(trail) > limit:
                trail.pop(0)

        # Collect all trail segments into a single batch
        all_starts = []
        all_ends = []
        for i in self.floating_nodes_idx:
            trail = self.trails[i]
            if len(trail) > 1:
                for j in range(len(trail) - 1):
                    all_starts.append(trail[j])
                    all_ends.append(trail[j + 1])

        if all_starts:
            starts = np.array(all_starts, dtype='f4')
            ends = np.array(all_ends, dtype='f4')
            colors = np.full((len(starts), 4), (1.0, 0.5, 0.0, 0.6), dtype='f4')
            pgfx.draw_lines_batch(starts, ends, colors, width=2.0)

    def _render_hinges(self):
        """
        Render hinges — vectorized geometry computation, batched drawing.
        Colored faces based on angular deviation + force vectors.
        """
        if self.data.n_hinges == 0 or not hasattr(self, 'hinge_indices'):
            return

        idx = self.hinge_indices
        p_j = self.all_positions[idx[:, 0]]
        p_k = self.all_positions[idx[:, 1]]
        p_i = self.all_positions[idx[:, 2]]
        p_l = self.all_positions[idx[:, 3]]

        # Vectorized angle computation
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

        dot_mn = np.einsum('ij,ij->i', m_hat, n_hat)
        cos_phi = np.clip(dot_mn, -1.0, 1.0)
        phi = np.arccos(cos_phi)

        dot_m_rkl = np.einsum('ij,ij->i', m, r_kl)
        mask_neg = dot_m_rkl < 0
        phi[mask_neg] = -phi[mask_neg]

        if self.hinge_rest_angles is not None:
            phi0 = self.hinge_rest_angles
            delta = phi - phi0
            delta = (delta + np.pi) % (2 * np.pi) - np.pi
        else:
            delta = np.zeros_like(phi)

        # Vectorized color computation
        limit_rad = np.deg2rad(45)
        t = np.clip((delta + limit_rad) / (2 * limit_rad), 0.0, 1.0)
        hinge_rgb = _jet_colormap_batch(t)  # (n_hinges, 3)

        # Vectorized geometry scaling
        geom_scale = self.config.get('hinge_geom_scale', 0.4)
        mid = (p_j + p_k) * 0.5  # (n_hinges, 3)

        pj_s = mid + (p_j - mid) * geom_scale
        pk_s = mid + (p_k - mid) * geom_scale
        pi_s = mid + (p_i - mid) * geom_scale
        pl_s = mid + (p_l - mid) * geom_scale

        # Filter to valid hinges
        v_idx = np.where(valid)[0]
        n_valid = len(v_idx)

        # --- Batch triangles: 2 per hinge (wing1 + wing2) ---
        tri_v1 = np.empty((n_valid * 2, 3), dtype='f4')
        tri_v2 = np.empty((n_valid * 2, 3), dtype='f4')
        tri_v3 = np.empty((n_valid * 2, 3), dtype='f4')
        tri_colors = np.empty((n_valid * 2, 4), dtype='f4')

        tri_v1[0::2] = pj_s[v_idx]
        tri_v2[0::2] = pk_s[v_idx]
        tri_v3[0::2] = pi_s[v_idx]
        tri_v1[1::2] = pj_s[v_idx]
        tri_v2[1::2] = pk_s[v_idx]
        tri_v3[1::2] = pl_s[v_idx]

        tri_colors[0::2, :3] = hinge_rgb[v_idx]
        tri_colors[0::2, 3] = 0.4
        tri_colors[1::2, :3] = hinge_rgb[v_idx]
        tri_colors[1::2, 3] = 0.4

        pgfx.draw_triangles_batch(tri_v1, tri_v2, tri_v3, tri_colors)

        # --- Batch lines: spine + 4 wing edges per hinge = 5 lines ---
        n_lines = n_valid * 5
        line_starts = np.empty((n_lines, 3), dtype='f4')
        line_ends = np.empty((n_lines, 3), dtype='f4')
        line_colors = np.empty((n_lines, 4), dtype='f4')

        # Spine (white)
        line_starts[0::5] = pj_s[v_idx]
        line_ends[0::5] = pk_s[v_idx]
        line_colors[0::5] = (1.0, 1.0, 1.0, 1.0)

        # Wing edges (face color)
        wing_c = np.empty((n_valid, 4), dtype='f4')
        wing_c[:, :3] = hinge_rgb[v_idx]
        wing_c[:, 3] = 0.4

        line_starts[1::5] = pj_s[v_idx]
        line_ends[1::5] = pi_s[v_idx]
        line_colors[1::5] = wing_c

        line_starts[2::5] = pk_s[v_idx]
        line_ends[2::5] = pi_s[v_idx]
        line_colors[2::5] = wing_c

        line_starts[3::5] = pj_s[v_idx]
        line_ends[3::5] = pl_s[v_idx]
        line_colors[3::5] = wing_c

        line_starts[4::5] = pk_s[v_idx]
        line_ends[4::5] = pl_s[v_idx]
        line_colors[4::5] = wing_c

        pgfx.draw_lines_batch(line_starts, line_ends, line_colors, width=2.0)

        # --- Force vectors (arrows) — only for significant deviations ---
        scale = self.config.get('hinge_force_scale', 0.5) * self.scales.L_char
        mag = np.abs(delta[v_idx]) / limit_rad
        mag = np.minimum(mag, 2.0)
        significant = mag > 0.01
        sig_idx = np.where(significant)[0]

        if len(sig_idx) > 0:
            # For significant hinges, use bulk lines for force vectors
            g_idx = v_idx[sig_idx]  # global hinge indices

            sign_d = np.sign(delta[g_idx])
            arrow_mag = mag[sig_idx]

            # Wing i force: -sign(d) * m_hat * scale * mag
            dir_i = -sign_d[:, None] * m_hat[g_idx]
            arrow_start_i = pi_s[g_idx]
            arrow_end_i = pi_s[g_idx] + dir_i * scale * arrow_mag[:, None]

            # Wing l force: +sign(d) * n_hat * scale * mag
            dir_l = sign_d[:, None] * n_hat[g_idx]
            arrow_start_l = pl_s[g_idx]
            arrow_end_l = pl_s[g_idx] + dir_l * scale * arrow_mag[:, None]

            # Combine both wings
            all_arrow_starts = np.concatenate([arrow_start_i, arrow_start_l], axis=0)
            all_arrow_ends = np.concatenate([arrow_end_i, arrow_end_l], axis=0)
            n_arrows = len(all_arrow_starts)
            arrow_color = np.full((n_arrows, 3), (1.0, 1.0, 0.0), dtype='f4')

            if n_arrows > 100:
                # Many arrows: use lines for speed
                pgfx.draw_lines_batch(all_arrow_starts, all_arrow_ends, arrow_color, width=2.0)
            else:
                # Fewer: use proper arrows
                head_size = 0.2 * self.scales.L_char
                width_radius = 0.04 * self.scales.L_char
                for i in range(n_arrows):
                    pgfx.draw_arrow(
                        tuple(all_arrow_starts[i]),
                        tuple(all_arrow_ends[i]),
                        color=(1.0, 1.0, 0.0),
                        head_size=head_size,
                        width_radius=width_radius
                    )


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def ShowSimulation(experiment_path: str, config: Optional[Dict] = None):
    """Main entry point for visualizing DEMLAT experiments."""
    data = ExperimentData(experiment_path)
    viz = DEMLATVisualizer(data, config)

    argv_backup = sys.argv
    sys.argv = [sys.argv[0]]

    try:
        studio = PiVizStudio(scene_fx=viz)
        studio.run()
    finally:
        sys.argv = argv_backup


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    experiment_path = None
    user_flags = []
    piviz_args = []

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in ['--no-strain', '--velocity', '--trails', '--particles', '--lines']:
            user_flags.append(arg)
            i += 1
        elif arg in ['-wnd', '-vs', '-hd', '-s', '-c', '--size', '--size_mult', '--backend']:
            piviz_args.append(arg)
            if i + 1 < len(sys.argv):
                piviz_args.append(sys.argv[i + 1])
                i += 2
            else:
                i += 1
        elif arg in ['-fs', '-r', '-h', '--help']:
            piviz_args.append(arg)
            i += 1
        elif not arg.startswith('-') and experiment_path is None:
            experiment_path = arg
            i += 1
        else:
            piviz_args.append(arg)
            i += 1

    sys.argv = [sys.argv[0]] + piviz_args

    if experiment_path is None:
        print("Usage: python -m demlat.utils.viz_player <experiment_path> [options]")
        print("\nExample:")
        print("  python -m demlat.utils.viz_player experiments/miura_test")
        print("\nOptions:")
        print("  --no-strain    Disable strain coloring")
        print("  --velocity     Show velocity arrows")
        print("  --trails       Show motion trails")
        print("  --particles    Force particle rendering")
        print("  --lines        Use lines instead of cylinders")
        sys.exit(1)

    config = {}
    if '--no-strain' in user_flags:
        config['show_strain'] = False
    if '--velocity' in user_flags:
        config['show_velocity'] = True
    if '--trails' in user_flags:
        config['show_trails'] = True
    if '--particles' in user_flags:
        config['use_particles_for_nodes'] = True
    if '--lines' in user_flags:
        config['use_lines_for_bars'] = True

    ShowSimulation(experiment_path, config)