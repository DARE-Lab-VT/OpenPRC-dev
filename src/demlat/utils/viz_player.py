# demlat/utils/viz_player.py
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
    from demlat.utils.viz_player import visualize_experiment
    
    # Auto-detect and visualize
    visualize_experiment("path/to/experiment")
    
    # Custom configuration
    visualize_experiment("path/to/experiment", config={
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

        # Load geometry (required)
        if not self.geometry_path.exists():
            raise FileNotFoundError(f"Geometry file not found: {self.geometry_path}")

        self._load_geometry()

        # Load simulation (optional)
        if self.simulation_path.exists():
            self.mode = 'simulation'
            self._load_simulation()
            print(f"[Mode] Full Simulation Playback")
        else:
            self.mode = 'geometry'
            print(f"[Mode] Geometry Viewer (no simulation data)")

        # Load visualization (optional)
        if self.visualization_path.exists():
            self._load_visualization()

        # Load config (optional)
        if self.config_path.exists():
            self._load_config()

    def _load_geometry(self):
        """Load geometry.h5 with robust handling of optional datasets."""
        print(f"\n[Loading] {self.geometry_path.name}")

        with h5py.File(self.geometry_path, 'r') as f:
            # Nodes (required)
            if 'nodes' not in f:
                raise ValueError("geometry.h5 missing 'nodes' group")

            nodes_grp = f['nodes']

            # Positions (required)
            if 'positions' not in nodes_grp:
                raise ValueError("geometry.h5 missing 'nodes/positions'")
            self.nodes['positions'] = nodes_grp['positions'][:]
            self.n_nodes = self.nodes['positions'].shape[0]

            # Masses (optional, default to 1.0)
            if 'masses' in nodes_grp:
                self.nodes['masses'] = nodes_grp['masses'][:]
            else:
                self.nodes['masses'] = np.ones(self.n_nodes, dtype='f4')

            # Attributes (optional, default to 0 = free floating)
            if 'attributes' in nodes_grp:
                self.nodes['attributes'] = nodes_grp['attributes'][:]
            else:
                self.nodes['attributes'] = np.zeros(self.n_nodes, dtype='u1')

            # Radius (optional, for visualization)
            if 'radius' in nodes_grp:
                self.nodes['radius'] = nodes_grp['radius'][:]
            else:
                self.nodes['radius'] = None

            # Elements
            if 'elements' not in f:
                print("  [Warning] No elements found in geometry")
                return

            elem_grp = f['elements']

            # Bars (optional)
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

            # Hinges (optional)
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

            # Inside _load_geometry, after loading hinges, add:
        if self.elements.get('hinges') and self.elements['hinges']['indices'] is not None:
            # Compute hinge edge midpoints and normals for visualization
            hinge_indices = self.elements['hinges']['indices']  # Shape: (n_hinges, 4)
            positions = self.nodes['positions']

            # For each hinge: [edge_node1, edge_node2, wing1, wing2]
            self.elements['hinges']['edge_midpoints'] = (positions[hinge_indices[:, 0]] + positions[
                hinge_indices[:, 1]]) / 2

            # Compute fold angle direction (cross product of wing vectors)
            edge_vec = positions[hinge_indices[:, 1]] - positions[hinge_indices[:, 0]]
            wing1_vec = positions[hinge_indices[:, 2]] - self.elements['hinges']['edge_midpoints']
            wing2_vec = positions[hinge_indices[:, 3]] - self.elements['hinges']['edge_midpoints']

            # Normal to the hinge plane
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
            # Metadata
            self.metadata = dict(f.attrs)

            if 'time_series' not in f:
                print("  [Warning] No time_series group found")
                self.mode = 'geometry'
                return

            ts_grp = f['time_series']

            # Time array (required for simulation)
            if 'time' in ts_grp:
                self.time_series['time'] = ts_grp['time'][:]
                self.n_frames = len(self.time_series['time'])
                print(f"  [Loaded] {self.n_frames} frames")
            else:
                print("  [Warning] No time array found")
                self.mode = 'geometry'
                return

            # Node data
            if 'nodes' in ts_grp:
                nodes_grp = ts_grp['nodes']

                # Positions (required for animation)
                if 'positions' in nodes_grp:
                    self.time_series['positions'] = nodes_grp['positions'][:]
                    print(f"  [Loaded] Node positions [{self.time_series['positions'].shape}]")
                else:
                    print("  [Warning] No node positions in simulation")
                    self.mode = 'geometry'
                    return

                # Velocities (optional)
                if 'velocities' in nodes_grp:
                    self.time_series['velocities'] = nodes_grp['velocities'][:]
                    print(f"  [Loaded] Node velocities")

                # Energies (optional)
                if 'kinetic_energy' in nodes_grp:
                    self.time_series['node_kinetic'] = nodes_grp['kinetic_energy'][:]
                if 'potential_energy' in nodes_grp:
                    self.time_series['node_potential'] = nodes_grp['potential_energy'][:]

            # Element data (bars)
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

            # Element data (hinges)
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
            # Check if 'faces' exists
            if 'faces' in f:
                # Check if it's a group (new schema) or dataset (legacy)
                if isinstance(f['faces'], h5py.Group) and 'triangles' in f['faces']:
                    self.visualization['faces'] = f['faces']['triangles'][:]
                    self.has_faces = True
                    print(f"  [Loaded] {len(self.visualization['faces'])} triangular faces")

                elif isinstance(f['faces'], h5py.Dataset):
                    self.visualization['faces'] = f['faces'][:]
                    self.has_faces = True
                    print(f"  [Loaded] {len(self.visualization['faces'])} faces (legacy format)")

            # Appearance data (optional)
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
        # Check position dimensions
        if self.mode == 'simulation':
            expected_shape = (self.n_frames, self.n_nodes, 3)
            actual_shape = self.time_series['positions'].shape
            if actual_shape != expected_shape:
                raise ValueError(
                    f"Position data shape mismatch: expected {expected_shape}, got {actual_shape}"
                )

    def _compute_properties(self):
        """Compute derived properties for visualization."""
        # Compute rest lengths if not provided
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
    """
    Automatically computes optimal visualization scales.
    """

    def __init__(self, data: ExperimentData):
        self.data = data

        # Geometric scales
        self.L_char = 1.0  # Characteristic length
        self.extent = np.array([1.0, 1.0, 1.0])
        self.center = np.array([0.0, 0.0, 0.0])

        # Element scales
        self.node_radius = 0.1
        self.bar_radius = 0.05
        self.velocity_scale = 1.0

        # Colormap limits
        self.strain_limit = 0.3
        self.stress_limit = None

        self._compute_scales()

    def _compute_scales(self):
        """Compute all visualization scales."""
        print("\n" + "=" * 60)
        print("Computing Visualization Scales")
        print("=" * 60)

        # Geometric extent
        positions = self.data.nodes['positions']
        self.extent = np.max(positions, axis=0) - np.min(positions, axis=0)
        self.center = np.mean(positions, axis=0)
        self.L_char = np.median(self.extent) if np.any(self.extent > 0) else 1.0

        # Use bar rest lengths if available
        if self.data.n_bars > 0:
            rest_lengths = self.data.elements['bars']['rest_length']
            if rest_lengths is not None:
                median_length = np.median(rest_lengths)
                if median_length > 0:
                    self.L_char = median_length

        print(f"Characteristic Length: {self.L_char:.4f}")
        print(f"Extent: [{self.extent[0]:.2f}, {self.extent[1]:.2f}, {self.extent[2]:.2f}]")
        print(f"Center: [{self.center[0]:.2f}, {self.center[1]:.2f}, {self.center[2]:.2f}]")

        # Node radius
        if self.data.nodes['radius'] is not None:
            self.node_radius = np.median(self.data.nodes['radius'])
        else:
            self.node_radius = self.L_char * 0.05

        print(f"Node Radius: {self.node_radius:.4f}")

        # Bar radius
        self.bar_radius = self.L_char * 0.03
        print(f"Bar Radius: {self.bar_radius:.4f}")

        # Velocity scale (if simulation data available)
        if self.data.mode == 'simulation' and 'velocities' in self.data.time_series:
            self._compute_velocity_scale()

        # Strain/stress limits
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

        # Sample velocities across time
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
# MAIN VISUALIZATION CLASS
# =============================================================================

class DEMLATVisualizer(PiVizFX):
    """
    High-performance visualizer for DEMLAT simulations.
    Optimized following the ReservoirAnimator pattern.
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
        self._updating_slider = False  # Flag to prevent feedback loops

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
            'hinge_color_mountain': (0.2, 0.6, 1.0),  # Blue for mountain folds (angle > π)
            'hinge_color_valley': (1.0, 0.4, 0.2),  # Orange for valley folds (angle < π)
            'hinge_color_flat': (0.5, 0.5, 0.5),  # Gray for flat (angle ≈ π)
            'hinge_indicator_scale': 0.15,  # Scale for hinge direction indicators
            'show_hinge_rest_angle': True,  # NEW: toggle rest angle visualization
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

        # Set scales from computed values
        if self.config['strain_limit'] is None:
            self.config['strain_limit'] = self.scales.strain_limit

        if self.config['base_node_radius'] is None:
            self.config['base_node_radius'] = self.scales.node_radius

        if self.config['velocity_scale'] is None:
            self.config['velocity_scale'] = self.scales.velocity_scale

        # Categorize nodes
        self._categorize_nodes()

        # Trails
        self.trails = {i: [] for i in range(self.data.n_nodes)}

        # Pre-allocate batch arrays
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
        """Pre-allocate all arrays for rendering - no per-frame allocation!"""
        n = self.data.n_nodes
        n_bars = self.data.n_bars
        n_floating = len(self.floating_nodes_idx)

        # Current frame data
        self.all_positions = np.zeros((n, 3), dtype='f4')

        # Bars
        if n_bars > 0:
            self.bar_indices = self.data.elements['bars']['indices']
            self.bar_rest_lengths = self.data.elements['bars']['rest_length']
            self.bar_colors = np.zeros((n_bars, 3), dtype='f4')
            self.bar_start_pos = np.zeros((n_bars, 3), dtype='f4')
            self.bar_end_pos = np.zeros((n_bars, 3), dtype='f4')

        # Particles (for nodes)
        if n_floating > 0:
            self.particle_positions = np.zeros((n_floating, 3), dtype='f4')
            self.particle_colors = np.full((n_floating, 3), (0.3, 0.3, 0.3), dtype='f4')
            self.particle_sizes = np.full(n_floating, self.scales.node_radius * 80, dtype='f4')

        # Faces
        if self.data.has_faces:
            n_faces = len(self.data.visualization['faces'])
            self.face_indices = self.data.visualization['faces']
            self.face_default_opacity = 0.7

            # Pre-compute vertex strain buffers
            self.vertex_strains = np.zeros(n, dtype='f4')
            self.vertex_counts = np.zeros(n, dtype='f4')

        # Velocity arrows
        if 'velocities' in self.data.time_series:
            self.current_velocities = np.zeros((n, 3), dtype='f4')

        # Hinges visualization
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
        """Called when timeline slider is moved."""
        if self._updating_slider:
            return
            
        self.timestep_idx = int(value)
        self.float_timestep = float(self.timestep_idx)
        # Pause when manually scrubbing
        self.paused = True
        self._update_pause_button()

    def _on_step_change(self, value):
        """Called when step size slider changes."""
        self.frame_step = max(1, int(value))

    def _goto_start(self):
        """Jump to first frame."""
        self.timestep_idx = 0
        self.float_timestep = 0.0
        self._clear_trails()
        self._sync_timeline_slider()

    def _goto_end(self):
        """Jump to last frame."""
        self.timestep_idx = self.data.n_frames - 1
        self.float_timestep = float(self.timestep_idx)
        self._sync_timeline_slider()

    def _prev_frame(self):
        """Go to previous frame(s)."""
        self.paused = True
        self._update_pause_button()
        
        step = max(1, int(self.frame_step))
        self.timestep_idx = max(0, self.timestep_idx - step)
        self.float_timestep = float(self.timestep_idx)
        self._sync_timeline_slider()

    def _next_frame(self):
        """Go to next frame(s)."""
        print(f"DEBUG: Next frame. Current: {self.timestep_idx}")
        self.paused = True
        self._update_pause_button()
        
        step = max(1, int(self.frame_step))
        max_idx = self.data.n_frames - 1
        
        new_idx = min(max_idx, self.timestep_idx + step)
        
        self.timestep_idx = int(new_idx)
        self.float_timestep = float(self.timestep_idx)
        
        print(f"DEBUG: New index: {self.timestep_idx}")
        self._sync_timeline_slider()

    def _toggle_pause(self):
        """Toggle play/pause state."""
        self.paused = not self.paused
        self._update_pause_button()

    def _update_pause_button(self):
        """Update pause button text based on state."""
        if hasattr(self, 'ui_manager') and self.ui_manager:
            btn = self.ui_manager.get_widget("btn_pause")
            if btn:
                btn.text = ">" if self.paused else "||"

    def _sync_timeline_slider(self):
        """Sync timeline slider to current frame."""
        if hasattr(self, 'ui_manager') and self.ui_manager:
            slider = self.ui_manager.get_widget("sld_timeline")
            if slider:
                self._updating_slider = True
                slider.value = self.timestep_idx
                self._updating_slider = False

    def _clear_trails(self):
        """Clear all motion trails."""
        for k in self.trails:
            self.trails[k].clear()

    def _reset_sim(self):
        """Reset simulation to start."""
        self._goto_start()
        self.paused = False
        self.timestep_idx = 0
        self.float_timestep = 0.0
        for k in self.trails:
            self.trails[k].clear()
        self._update_pause_button()

        # ========================================================

    def _setup_ui(self):
        """Setup UI controls with explicit layout."""
        self.ui_manager.set_panel_title("DEMLAT Player")
        
        # Time display
        if self.data.mode == 'simulation':
            self.lbl_time = Label("Time: 0.00s | Frame: 0")
            self.ui_manager.add_widget("lbl_time", self.lbl_time)
        else:
            self.lbl_mode = Label("Geometry Viewer", color=(0.7, 0.9, 1.0))
            self.ui_manager.add_widget("lbl_mode", self.lbl_mode)

        # Performance info
        self.lbl_perf = Label(
            f"Nodes: {self.data.n_nodes} | Bars: {self.data.n_bars}",
            color=(0.7, 0.7, 0.7)
        )
        self.ui_manager.add_widget("lbl_perf", self.lbl_perf)

        # ============ NEW: PLAYBACK CONTROLS ============
        if self.data.mode == 'simulation':
            # Timeline slider
            self.ui_manager.add_widget("sld_timeline",
                                       Slider("Frame", 0, self.data.n_frames - 1, 0, self._on_timeline_change))

            # Playback buttons row
            # Note: PiViz widgets don't support explicit positioning in this version.
            # We rely on the layout manager.
            
            self.ui_manager.add_widget("btn_start",
                                       Button("|<", self._goto_start))
            self.ui_manager.add_widget("btn_prev",
                                       Button("<", self._prev_frame))
            self.ui_manager.add_widget("btn_pause",
                                       Button("||", self._toggle_pause))
            self.ui_manager.add_widget("btn_next",
                                       Button(">", self._next_frame))
            self.ui_manager.add_widget("btn_end",
                                       Button(">|", self._goto_end))

            # Speed control
            self.ui_manager.add_widget("sld_speed",
                                       Slider("Speed", 10, 500, int(self.speed * 100),
                                              lambda v: setattr(self, 'speed', v / 100.0)))

            # Frame step size (for prev/next buttons)
            self.frame_step = 1
            self.ui_manager.add_widget("sld_step",
                                       Slider("Step", 1, 50, 1, self._on_step_change))
        # ================================================

        # Toggles
        if self.data.n_hinges > 0:
            self.ui_manager.add_widget("chk_hinges",
                                       Checkbox("Show Hinges", self.config['show_hinges'],
                                                lambda v: self.config.update({'show_hinges': v})))

        self.ui_manager.add_widget("chk_hinge_rest",
                                   Checkbox("Show Rest Angle", self.config['show_hinge_rest_angle'],
                                            lambda v: self.config.update({'show_hinge_rest_angle': v})))
        
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
                                                self._toggle_velocity))

        self.ui_manager.add_widget("chk_trails",
                                   Checkbox("Show Trails", self.config['show_trails'],
                                            lambda v: self.config.update({'show_trails': v})))

        self.ui_manager.add_widget("chk_lines",
                                   Checkbox("Fast Lines", self.config['use_lines_for_bars'],
                                            lambda v: self.config.update({'use_lines_for_bars': v})))

        self.ui_manager.add_widget("chk_particles",
                                   Checkbox("Particle Nodes", self.config['use_particles_for_nodes'],
                                            lambda v: self.config.update({'use_particles_for_nodes': v})))

    def _toggle_velocity(self, v):
        self.config['show_velocity'] = v

    def render(self, time_val, dt):
        """Main render loop - optimized like ReservoirAnimator."""

        # Update playback (only when not paused)
        if self.data.mode == 'simulation' and not self.paused:
            self.float_timestep += dt * 60 * self.speed

            if self.float_timestep >= self.data.n_frames:
                self.float_timestep = 0.0
                self._clear_trails()

            self.timestep_idx = int(self.float_timestep)

            # Sync slider during playback
            self._sync_timeline_slider()

        idx = self.timestep_idx

        # Get current frame data
        if self.data.mode == 'simulation':
            x_curr = self.data.time_series['positions'][idx]
        else:
            x_curr = self.data.nodes['positions']

        self.all_positions[:] = x_curr

        # Calculate velocities (if needed)
        vels = None
        if self.config['show_velocity'] and idx > 0 and 'velocities' in self.data.time_series:
            self.current_velocities[:] = self.data.time_series['velocities'][idx]
            vels = self.current_velocities

        # --- DRAWING LAYERS ---

        # 1. Trails
        if self.config['show_trails']:
            self._render_trails(x_curr)

        # 2. Faces
        if self.config['show_faces']:
            self._render_faces()

        # 3. Bars
        if self.config['show_bars']:
            self._render_bars()

        # 3.5 Hinges (geometry mode or always)
        if self.config['show_hinges'] and self.data.n_hinges > 0:
            self._render_hinges()

        # 4. Nodes
        if self.config['show_nodes']:
            self._render_nodes()

        # 5. Velocity arrows
        if vels is not None:
            self._render_velocity_arrows(vels)

        # 6. UI Updates
        if self.data.mode == 'simulation' and hasattr(self, 'lbl_time'):
            t_display = self.data.time_series['time'][idx]
            self.lbl_time.text = f"Time: {t_display:.2f}s | Frame: {idx}/{self.data.n_frames - 1}"

        if self.data.mode == 'simulation' and hasattr(self, 'lbl_time'):
            t_display = self.data.time_series['time'][idx]
            status = "PAUSED" if self.paused else "PLAYING"
            self.lbl_time.text = f"Time: {t_display:.3f}s | Frame: {idx}/{self.data.n_frames - 1} | {status}"

    def _render_trails(self, x_curr):
        """Render motion trails for floating nodes."""
        limit = self.config['trail_length']
        for i in self.floating_nodes_idx:
            pos = tuple(x_curr[i])
            self.trails[i].append(pos)
            if len(self.trails[i]) > limit:
                self.trails[i].pop(0)

            if len(self.trails[i]) > 1:
                pgfx.draw_path(
                    points=self.trails[i],
                    color=(1.0, 0.5, 0.0, 0.6),
                    width=2.0
                )

    def _render_hinges(self):
        """
        Enhanced hinge visualization showing:
        - Current hinge configuration (solid)
        - Rest angle configuration (semi-transparent)
        - Color coding by fold type
        - Clear wing assignments
        """
        if self.data.n_hinges == 0 or not hasattr(self, 'hinge_indices'):
            return

        scale = self.config['hinge_indicator_scale'] * self.scales.L_char
        show_rest = self.config.get('show_hinge_rest_angle', True)

        for i in range(self.data.n_hinges):
            idx = self.hinge_indices[i]  # [edge1, edge2, wing1, wing2]

            # Get current positions
            p_edge1 = self.all_positions[idx[0]]
            p_edge2 = self.all_positions[idx[1]]
            p_wing1 = self.all_positions[idx[2]]
            p_wing2 = self.all_positions[idx[3]]

            edge_mid = (p_edge1 + p_edge2) / 2
            edge_vec = p_edge2 - p_edge1
            edge_length = np.linalg.norm(edge_vec)
            if edge_length < 1e-10:
                continue
            edge_vec = edge_vec / edge_length

            # Determine color based on rest angle
            if self.hinge_rest_angles is not None:
                rest_angle = self.hinge_rest_angles[i]

                if rest_angle > np.pi + 0.05:
                    color = self.config['hinge_color_mountain']
                    fold_type = "M"
                elif rest_angle < np.pi - 0.05:
                    color = self.config['hinge_color_valley']
                    fold_type = "V"
                else:
                    color = self.config['hinge_color_flat']
                    fold_type = "F"
            else:
                color = self.config['hinge_color_flat']
                fold_type = "F"
                rest_angle = np.pi

            # === Draw CURRENT configuration ===

            # 1. Hinge edge (thick line)
            pgfx.draw_line(
                tuple(p_edge1),
                tuple(p_edge2),
                color=color,
                width=5.0
            )

            # 2. Wing vectors (current configuration)
            wing1_vec = p_wing1 - edge_mid
            wing2_vec = p_wing2 - edge_mid

            w1_len = np.linalg.norm(wing1_vec)
            w2_len = np.linalg.norm(wing2_vec)

            if w1_len < 1e-10 or w2_len < 1e-10:
                continue

            wing1_dir = wing1_vec / w1_len
            wing2_dir = wing2_vec / w2_len

            # Draw wing indicators (solid, current position)
            wing_length = scale * 1.5

            # Wing 1 - darker shade of hinge color
            pgfx.draw_line(
                tuple(edge_mid),
                tuple(edge_mid + wing1_dir * wing_length),
                color=(color[0] * 0.8, color[1] * 0.8, color[2] * 0.8),
                width=3.0
            )

            # Wing 2 - lighter shade (CURRENT position)
            pgfx.draw_line(
                tuple(edge_mid),
                tuple(edge_mid + wing2_dir * wing_length),
                color=(min(color[0] * 1.2, 1.0), min(color[1] * 1.2, 1.0), min(color[2] * 1.2, 1.0)),
                width=3.0
            )

            # Small spheres at wing tips (current)
            pgfx.draw_sphere(
                center=tuple(edge_mid + wing1_dir * wing_length),
                radius=self.scales.node_radius * 0.4,
                color=(color[0] * 0.8, color[1] * 0.8, color[2] * 0.8),
                detail=6
            )

            pgfx.draw_sphere(
                center=tuple(edge_mid + wing2_dir * wing_length),
                radius=self.scales.node_radius * 0.4,
                color=(min(color[0] * 1.2, 1.0), min(color[1] * 1.2, 1.0), min(color[2] * 1.2, 1.0)),
                detail=6
            )

            # === Draw REST configuration (ghosted) ===
            if show_rest and self.hinge_rest_angles is not None:
                # Project wings onto plane perpendicular to edge
                w1_perp = wing1_vec - np.dot(wing1_vec, edge_vec) * edge_vec
                w2_perp = wing2_vec - np.dot(wing2_vec, edge_vec) * edge_vec

                w1_perp_len = np.linalg.norm(w1_perp)
                w2_perp_len = np.linalg.norm(w2_perp)

                if w1_perp_len > 1e-10 and w2_perp_len > 1e-10:
                    w1_perp_norm = w1_perp / w1_perp_len
                    w2_perp_norm = w2_perp / w2_perp_len

                    # Calculate current dihedral angle
                    cos_current = np.clip(np.dot(w1_perp_norm, w2_perp_norm), -1.0, 1.0)
                    cross_prod = np.cross(w1_perp_norm, w2_perp_norm)
                    sign = np.sign(np.dot(cross_prod, edge_vec))
                    if sign == 0:
                        sign = 1.0
                    current_angle = sign * np.arccos(cos_current)

                    # Compute rotation needed to go from current to rest
                    # Rest angle is the dihedral angle we want
                    rotation_angle = rest_angle - np.pi - current_angle

                    # Only draw if there's a significant difference
                    if abs(rotation_angle) > 0.05:  # ~3 degrees
                        # Compute rest position of wing2 by rotating current w2_perp
                        cos_theta = np.cos(rotation_angle)
                        sin_theta = np.sin(rotation_angle)

                        # Rodrigues rotation formula
                        w2_rest_perp = (w2_perp_norm * cos_theta +
                                        np.cross(edge_vec, w2_perp_norm) * sin_theta +
                                        edge_vec * np.dot(edge_vec, w2_perp_norm) * (1 - cos_theta))

                        # Scale back to original length
                        w2_rest_perp = w2_rest_perp * w2_perp_len

                        # Add back the component along edge
                        w2_rest = w2_rest_perp + np.dot(wing2_vec, edge_vec) * edge_vec

                        w2_rest_len = np.linalg.norm(w2_rest)
                        if w2_rest_len > 1e-10:
                            w2_rest_dir = w2_rest / w2_rest_len

                            # Draw rest configuration (dashed)
                            rest_wing2_end = edge_mid + w2_rest_dir * wing_length

                            # Draw as dashed line
                            self._draw_dashed_line(
                                tuple(edge_mid),
                                tuple(rest_wing2_end),
                                color=(1.0, 1.0, 0.0, 0.8),  # Yellow for rest
                                width=2.5,
                                dash_length=0.03 * self.scales.L_char
                            )

                            # Sphere at rest wing tip
                            pgfx.draw_sphere(
                                center=tuple(rest_wing2_end),
                                radius=self.scales.node_radius * 0.3,
                                color=(1.0, 1.0, 0.0, 0.6),
                                detail=6
                            )

                            # Draw arc showing the rotation angle
                            self._draw_angle_arc(
                                center=edge_mid,
                                axis=edge_vec,
                                start_vec=w2_perp_norm,
                                angle=rotation_angle,
                                radius=wing_length * 0.6,
                                color=(1.0, 1.0, 0.0, 0.7),
                                segments=16
                            )

            # 3. Central sphere at hinge midpoint
            pgfx.draw_sphere(
                center=tuple(edge_mid),
                radius=self.scales.node_radius * 0.7,
                color=color,
                detail=8
            )

    def _draw_dashed_line(self, start, end, color, width, dash_length):
        """Draw a dashed line."""
        start = np.array(start)
        end = np.array(end)
        vec = end - start
        length = np.linalg.norm(vec)

        if length < 1e-10:
            return

        direction = vec / length
        num_dashes = int(length / (2 * dash_length))

        for i in range(num_dashes):
            t0 = (2 * i * dash_length) / length
            t1 = ((2 * i + 1) * dash_length) / length
            if t1 > 1.0:
                t1 = 1.0

            p0 = start + t0 * vec
            p1 = start + t1 * vec

            pgfx.draw_line(
                tuple(p0),
                tuple(p1),
                color=color,
                width=width
            )

    def _draw_angle_arc(self, center, axis, start_vec, angle, radius, color, segments=16):
        """Draw an arc to visualize an angle around an axis."""
        center = np.array(center)
        axis = np.array(axis)
        start_vec = np.array(start_vec)

        # Normalize
        axis = axis / (np.linalg.norm(axis) + 1e-10)
        start_vec = start_vec / (np.linalg.norm(start_vec) + 1e-10)

        # Generate arc points using Rodrigues rotation
        points = []
        for i in range(segments + 1):
            theta = angle * i / segments
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            # Rodrigues rotation formula
            rotated = (start_vec * cos_t +
                       np.cross(axis, start_vec) * sin_t +
                       axis * np.dot(axis, start_vec) * (1 - cos_t))

            point = center + rotated * radius
            points.append(tuple(point))

        # Draw arc as connected line segments
        if len(points) > 1:
            pgfx.draw_path(
                points=points,
                color=color,
                width=2.0
            )

    def _render_hinge_angles_overlay(self):
        """
        Optional: Render current vs rest angle as arc indicators.
        Useful for debugging hinge behavior during simulation.
        """
        if self.data.n_hinges == 0 or self.hinge_rest_angles is None:
            return

        for i in range(min(self.data.n_hinges, 50)):  # Limit for performance
            idx = self.hinge_indices[i]

            p_edge1 = self.all_positions[idx[0]]
            p_edge2 = self.all_positions[idx[1]]
            p_wing1 = self.all_positions[idx[2]]
            p_wing2 = self.all_positions[idx[3]]

            edge_mid = (p_edge1 + p_edge2) / 2

            # Compute current dihedral angle
            edge_vec = p_edge2 - p_edge1
            edge_vec = edge_vec / (np.linalg.norm(edge_vec) + 1e-10)

            v1 = p_wing1 - edge_mid
            v2 = p_wing2 - edge_mid

            # Project onto plane perpendicular to edge
            v1_perp = v1 - np.dot(v1, edge_vec) * edge_vec
            v2_perp = v2 - np.dot(v2, edge_vec) * edge_vec

            n1 = np.linalg.norm(v1_perp)
            n2 = np.linalg.norm(v2_perp)

            if n1 > 1e-10 and n2 > 1e-10:
                v1_perp /= n1
                v2_perp /= n2

                cos_angle = np.clip(np.dot(v1_perp, v2_perp), -1, 1)
                current_angle = np.arccos(cos_angle)

                # Color based on deviation from rest
                rest_angle = self.hinge_rest_angles[i]
                deviation = abs(current_angle - (2 * np.pi - rest_angle))

                # Green = at rest, Red = strained
                strain_color = min(deviation / 0.5, 1.0)
                color = (strain_color, 1.0 - strain_color, 0.0)

                # Draw small arc indicator
                arc_radius = self.scales.L_char * 0.1
                pgfx.draw_sphere(
                    center=tuple(edge_mid + edge_vec * arc_radius * 0.5),
                    radius=arc_radius * 0.3,
                    color=color,
                    detail=4
                )

    def _render_bars(self):
        """Render bars with strain coloring - vectorized."""
        if self.data.n_bars == 0:
            return

        # Update positions (vectorized)
        self.bar_start_pos[:] = self.all_positions[self.bar_indices[:, 0]]
        self.bar_end_pos[:] = self.all_positions[self.bar_indices[:, 1]]

        # Compute strain colors (vectorized)
        if self.config['show_strain'] and 'bar_strain' in self.data.time_series:
            strains = self.data.time_series['bar_strain'][self.timestep_idx]
            limit = self.config['strain_limit']
            t = np.clip((strains + limit) / (2 * limit), 0.0, 1.0)

            self.bar_colors[:, 0] = np.clip(1.5 - np.abs(t - 0.75) * 4, 0, 1)
            self.bar_colors[:, 1] = np.clip(1.5 - np.abs(t - 0.5) * 4, 0, 1)
            self.bar_colors[:, 2] = np.clip(1.5 - np.abs(t - 0.25) * 4, 0, 1)
        else:
            self.bar_colors[:] = (0.5, 0.5, 0.5)

        # Render
        use_lines = self.config['use_lines_for_bars']

        if use_lines:
            for i in range(self.data.n_bars):
                pgfx.draw_line(
                    tuple(self.bar_start_pos[i]),
                    tuple(self.bar_end_pos[i]),
                    color=tuple(self.bar_colors[i]),
                    width=2.0
                )
        else:
            radius = self.scales.bar_radius
            detail = self.config['cylinder_detail']
            for i in range(self.data.n_bars):
                pgfx.draw_cylinder(
                    start=tuple(self.bar_start_pos[i]),
                    end=tuple(self.bar_end_pos[i]),
                    radius=radius,
                    color=tuple(self.bar_colors[i]),
                    detail=detail
                )

    def _render_faces(self):
        """Render faces with smooth per-vertex strain coloring."""
        if not self.data.has_faces:
            return

        # Compute per-vertex strains (vectorized)
        if self.config['show_strain'] and 'bar_strain' in self.data.time_series:
            strains = self.data.time_series['bar_strain'][self.timestep_idx]

            # Reset buffers
            self.vertex_strains[:] = 0
            self.vertex_counts[:] = 0

            # Accumulate strains (vectorized)
            np.add.at(self.vertex_strains, self.bar_indices[:, 0], strains)
            np.add.at(self.vertex_strains, self.bar_indices[:, 1], strains)
            np.add.at(self.vertex_counts, self.bar_indices[:, 0], 1)
            np.add.at(self.vertex_counts, self.bar_indices[:, 1], 1)

            # Average
            mask = self.vertex_counts > 0
            self.vertex_strains[mask] /= self.vertex_counts[mask]

            # Render with per-vertex colors
            limit = self.config['strain_limit']
            for face in self.face_indices:
                v0_pos = tuple(self.all_positions[face[0]])
                v1_pos = tuple(self.all_positions[face[1]])
                v2_pos = tuple(self.all_positions[face[2]])

                # Compute colors
                s0 = self.vertex_strains[face[0]]
                s1 = self.vertex_strains[face[1]]
                s2 = self.vertex_strains[face[2]]

                c0 = self._strain_to_color(s0, limit)
                c1 = self._strain_to_color(s1, limit)
                c2 = self._strain_to_color(s2, limit)

                pgfx.draw_face(v0_pos, v1_pos, v2_pos, c0, c1, c2)
        else:
            # Static color
            color = (0.85, 0.85, 0.85, self.face_default_opacity)
            for face in self.face_indices:
                v0 = tuple(self.all_positions[face[0]])
                v1 = tuple(self.all_positions[face[1]])
                v2 = tuple(self.all_positions[face[2]])
                pgfx.draw_triangle(v0, v1, v2, color)

    def _strain_to_color(self, strain, limit):
        """Convert strain to color tuple with opacity."""
        t = np.clip((strain + limit) / (2 * limit), 0.0, 1.0)
        # r = float(np.clip(1.5 - np.abs(t - 0.75) * 4, 0, 1))
        # g = float(np.clip(1.5 - np.abs(t - 0.5) * 4, 0, 1))
        # b = float(np.clip(1.5 - np.abs(t - 0.25) * 4, 0, 1))
        clr = Colormap.jet(t, self.face_default_opacity)
        # return (r, g, b, self.face_default_opacity)
        return clr

    def _render_nodes(self):
        """Render nodes - spheres or particles."""
        n_floating = len(self.floating_nodes_idx)
        use_particles = self.config['use_particles_for_nodes'] or n_floating > self.config['particle_threshold']

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
                for local_idx, global_idx in enumerate(self.floating_nodes_idx):
                    pgfx.draw_sphere(
                        center=tuple(self.particle_positions[local_idx]),
                        radius=self.scales.node_radius,
                        color=(0.3, 0.3, 0.3),
                        detail=self.config['sphere_detail']
                    )

        # Fixed nodes (cubes)
        s_fixed = self.scales.node_radius * 2.0
        for i in self.fixed_nodes_idx:
            pgfx.draw_cube(
                center=tuple(self.all_positions[i]),
                size=(s_fixed, s_fixed, s_fixed),
                color=Palette.Standard10[3]
            )

        # Actuators
        s_act = self.scales.node_radius * 2.0
        for i in self.position_actuators_idx:
            pgfx.draw_cube(
                center=tuple(self.all_positions[i]),
                size=(s_act, s_act, s_act),
                color=Palette.Standard10[2]
            )

        s_force = self.scales.node_radius * 1.5
        for i in self.force_actuators_idx:
            pgfx.draw_cube(
                center=tuple(self.all_positions[i]),
                size=(s_force, s_force, s_force),
                color=Palette.Standard10[1]
            )

    def _render_velocity_arrows(self, vels):
        """Render velocity arrows for floating nodes."""
        scale = self.config['velocity_scale']
        head_size = self.scales.L_char * 0.1
        width_radius = self.scales.L_char * 0.02
        min_vel_threshold = 1e-6

        for idx in self.floating_nodes_idx:
            pos = self.all_positions[idx]
            vel = vels[idx]
            v_mag = np.linalg.norm(vel)

            if v_mag > min_vel_threshold:
                pgfx.draw_arrow(
                    start=tuple(pos),
                    end=tuple(pos + vel * scale),
                    color=(0.0, 1.0, 1.0),
                    head_size=head_size,
                    width_radius=width_radius
                )


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def visualize_experiment(experiment_path: str, config: Optional[Dict] = None):
    """
    Main entry point for visualizing DEMLAT experiments.
    """
    # Load data
    data = ExperimentData(experiment_path)

    # Create visualizer
    viz = DEMLATVisualizer(data, config)

    # Launch studio
    # PiVizStudio (via moderngl_window) parses sys.argv, so we need to clear it
    # to avoid "unrecognized arguments" errors for our custom args.
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
    # Separate user args from PiViz args BEFORE any processing
    experiment_path = None
    user_flags = []
    piviz_args = []

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]

        # User-specific flags
        if arg in ['--no-strain', '--velocity', '--trails', '--particles', '--lines']:
            user_flags.append(arg)
            i += 1
        # PiViz flags that take arguments
        elif arg in ['-wnd', '-vs', '-hd', '-s', '-c', '--size', '--size_mult', '--backend']:
            piviz_args.append(arg)
            if i + 1 < len(sys.argv):
                piviz_args.append(sys.argv[i + 1])
                i += 2
            else:
                i += 1
        # PiViz boolean flags
        elif arg in ['-fs', '-r', '-h', '--help']:
            piviz_args.append(arg)
            i += 1
        # Experiment path (first non-flag argument)
        elif not arg.startswith('-') and experiment_path is None:
            experiment_path = arg
            i += 1
        else:
            # Unknown arg, assume it's for PiViz
            piviz_args.append(arg)
            i += 1

    # Update sys.argv for PiViz (remove user args)
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

    # Parse config options
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

    visualize_experiment(experiment_path, config)
