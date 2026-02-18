"""
demlat IO Manager
=================
High-level API for creating, editing, and validating experiment files.
Designed to make user scripts simple and robust.
"""

import h5py
import json
import numpy as np
import shutil
from pathlib import Path
from typing import List, Optional, Union

from openprc.schemas.logging import get_logger


class NumpyJSONEncoder(json.JSONEncoder):
    """
    A JSON encoder that can handle numpy data types.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)


class SimulationSetup:
    """
    A setup class to programmatically create and modify DEMLat experiments.
    
    This class abstracts away the complexity of HDF5 file structures and JSON configs,
    allowing users to define experiments using simple Python methods.
    """

    def __init__(self, experiment_dir: Union[str, Path], overwrite: bool = False):
        self.logger = get_logger("demlat.setup")
        self.root = Path(experiment_dir)
        self.input_dir = self.root / "input"
        self.output_dir = self.root / "output"

        if self.root.exists():
            if overwrite:
                self.logger.warning(f"Overwriting existing experiment at {self.root}")
                shutil.rmtree(self.root)
            else:
                self.logger.info(f"Loading existing experiment at {self.root}")

        # Ensure structure exists
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Internal state buffers
        self.config = {
            "simulation": {
                "duration": 1.0,
                "dt_base": 0.001,
                "dt_save": 0.01
            },
            "global_physics": {
                "gravity": -9.81,
                "global_damping": 0.1
            },
            "material": {
                "damping_coefficient": 0.1
            },
            "actuators": []
        }

        self.nodes = {'positions': [], 'masses': [], 'attributes': []}
        self.bars = {'indices': [], 'stiffness': [], 'rest_length': [], 'damping': [], 'prestress': []}
        self.hinges = {'indices': [], 'stiffness': [], 'angle': [], 'damping': []}
        self.signals = {'dt_base': 0.001, 'data': {}}

        # Add to the __init__ method's internal state buffers:
        self.visualization = {
            'faces': [],
            'face_colors': [],
            'face_groups': [],
            'group_names': []
        }
        self._face_group_map = {}  # name -> group_id mapping


    # --- Configuration Methods ---

    def set_simulation_params(self, duration: float, dt: float, save_interval: float):
        """Set time-stepping parameters."""
        self.config['simulation'] = {
            "duration": float(duration),
            "dt_base": float(dt),
            "dt_save": float(save_interval)
        }
        return self

    def set_physics(self, gravity: Union[float, List[float]] = -9.81, damping: float = 0.1,
                    enable_collision: bool = False, collision_radius: float = 0.01,
                    collision_restitution: float = 0.5):
        """Set global physics constants."""
        self.config['global_physics']['gravity'] = gravity
        self.config['global_physics']['global_damping'] = float(damping)
        self.config['global_physics']['enable_collision'] = enable_collision
        self.config['global_physics']['collision_radius'] = float(collision_radius)
        self.config['global_physics']['collision_restitution'] = float(collision_restitution)
        self.config['material']['damping_coefficient'] = float(damping)
        return self

    # --- Geometry Methods ---

    def add_node(self, position: List[float], mass: float = 1.0, fixed: bool = False,
                 collidable: bool = False) -> int:
        """
        Add a single node.
        Returns the index of the new node.

        Args:
            position: [x, y, z] coordinates
            mass: Node mass
            fixed: If True, node cannot move
            collidable: If True, node participates in collision detection
        """
        idx = len(self.nodes['positions'])
        self.nodes['positions'].append(position)
        self.nodes['masses'].append(mass)

        # Attribute bitmask:
        # Bit 0 (0x01): Fixed
        # Bit 1 (0x02): Position actuated (set by add_actuator)
        # Bit 2 (0x04): Force actuated (set by add_actuator)
        # Bit 3 (0x08): Collidable
        attr = 0
        if fixed:
            attr |= 0x01
        if collidable:
            attr |= 0x08

        self.nodes['attributes'].append(attr)
        return idx

    def add_nodes(self, positions: np.ndarray, masses: Union[float, np.ndarray] = 1.0,
                  fixed_mask: Optional[np.ndarray] = None, collidable_mask: Optional[np.ndarray] = None):
        """Batch add nodes for performance."""
        n = len(positions)
        self.nodes['positions'].extend(positions.tolist())

        if np.isscalar(masses):
            self.nodes['masses'].extend([masses] * n)
        else:
            self.nodes['masses'].extend(masses.tolist())

        # Build attributes
        attrs = np.zeros(n, dtype=np.uint8)

        if fixed_mask is not None:
            attrs[fixed_mask] |= 0x01  # Fixed bit

        if collidable_mask is not None:
            attrs[collidable_mask] |= 0x08  # Collidable bit

        self.nodes['attributes'].extend(attrs.tolist())
        return self

    def add_bar(self, node_a: int, node_b: int, stiffness: float = 1000.0, damping: float = 1.0, rest_length: Optional[float] = None):
        """Add a bar element between two nodes."""
        self.bars['indices'].append([node_a, node_b])
        self.bars['stiffness'].append(stiffness)
        self.bars['damping'].append(damping)
        self.bars['prestress'].append(0.0)

        if rest_length is None:
            # Auto-calculate from current positions
            pa = np.array(self.nodes['positions'][node_a])
            pb = np.array(self.nodes['positions'][node_b])
            l0 = np.linalg.norm(pb - pa)
            self.bars['rest_length'].append(l0)
        else:
            self.bars['rest_length'].append(rest_length)
        return self

    def add_hinge(self, nodes: List[int], stiffness: float = 100.0, damping: float = 0.1, rest_angle: Optional[float] = None):
        """
        Add a hinge element (4 nodes: j, k, i, l).
        Axis is j-k. Triangle 1 is j-k-i. Triangle 2 is j-k-l.
        """
        if len(nodes) != 4:
            raise ValueError("Hinge requires exactly 4 node indices [j, k, i, l]")

        self.hinges['indices'].append(nodes)
        self.hinges['stiffness'].append(stiffness)
        self.hinges['damping'].append(damping)

        if rest_angle is None:
            # We defer calculation or set a placeholder. 
            # Ideally, we calculate it here if positions are known, but let's assume 
            # the user wants the initial configuration to be the rest state.
            # For now, we'll store a sentinel or calculate it if possible.
            # Let's calculate it to be safe.
            try:
                pts = np.array([self.nodes['positions'][i] for i in nodes])
                # j, k, i, l = 0, 1, 2, 3
                xj, xk, xi, xl = pts[0], pts[1], pts[2], pts[3]

                r_ij = xi - xj
                r_kj = xk - xj
                r_kl = xk - xl

                m = np.cross(r_ij, r_kj)
                n = np.cross(r_kj, r_kl)

                len_m = np.linalg.norm(m)
                len_n = np.linalg.norm(n)

                if len_m < 1e-9 or len_n < 1e-9:
                    angle = 0.0
                else:
                    cos_phi = np.clip(np.dot(m, n) / (len_m * len_n), -1.0, 1.0)
                    angle = np.arccos(cos_phi)
                    if np.dot(m, r_kl) < 0:
                        angle = -angle

                self.hinges['angle'].append(angle)
            except IndexError:
                self.logger.warning("Cannot calculate rest angle: nodes not yet defined.")
                self.hinges['angle'].append(0.0)
        else:
            self.hinges['angle'].append(rest_angle)
        return self

    # --- Actuation & Signals ---

    def add_signal(self, name: str, data: Union[List[float], np.ndarray], dt: Optional[float] = None):
        """Add a time-series signal."""
        self.signals['data'][name] = np.array(data)
        if dt:
            self.signals['dt_base'] = dt
        return self

    def add_actuator(self, node_idx: int, signal_name: str, type: str = 'position', dof: List[int] = [1, 1, 1]):
        """
        Link a signal to a node.
        type: 'position', 'velocity', 'force'
        dof: [x, y, z] mask (1=active, 0=inactive)
        """
        self.config['actuators'].append({
            "node_idx": node_idx,
            "signal_name": signal_name,
            "type": type,
            "dof": dof
        })

        # Automatically mark node as driven based on type
        # 2 = Position Actuator (Driver)
        # 4 = Force Actuator (Thruster)
        if 0 <= node_idx < len(self.nodes['attributes']):
            if type == 'position':
                self.nodes['attributes'][node_idx] |= 2
            elif type == 'force':
                self.nodes['attributes'][node_idx] |= 4

        return self

    # Add these methods to the SimulationSetup class:

    # --- Visualization Methods ---

    def add_face(self, nodes: List[int], color: Optional[List[int]] = None, group: Optional[str] = None):
        """
        Add a triangular face for visualization.

        Parameters
        ----------
        nodes : List[int]
            Three node indices forming the triangle
        color : Optional[List[int]]
            RGBA color [0-255, 0-255, 0-255, 0-255]. If None, uses default.
        group : Optional[str]
            Group name for organizing faces (e.g., "top_surface", "hinge")

        Returns
        -------
        self
        """
        if len(nodes) != 3:
            raise ValueError("Face requires exactly 3 node indices for triangular mesh")

        # Validate node indices
        n_nodes = len(self.nodes['positions'])
        for node_idx in nodes:
            if not (0 <= node_idx < n_nodes):
                raise ValueError(f"Node index {node_idx} out of range [0, {n_nodes})")

        self.visualization['faces'].append(nodes)

        # Handle color
        if color is None:
            self.visualization['face_colors'].append([128, 128, 128, 255])  # Default gray
        else:
            if len(color) not in [3, 4]:
                raise ValueError("Color must be RGB [r,g,b] or RGBA [r,g,b,a]")
            rgba = list(color) + [255] if len(color) == 3 else list(color)
            # Validate color range
            for c in rgba:
                if not (0 <= c <= 255):
                    raise ValueError(f"Color values must be in range [0, 255], got {c}")
            self.visualization['face_colors'].append(rgba)

        # Handle group
        if group is None:
            self.visualization['face_groups'].append(0)  # Default group
        else:
            if group not in self._face_group_map:
                group_id = len(self._face_group_map) + 1
                self._face_group_map[group] = group_id
                self.visualization['group_names'].append(group)
            else:
                group_id = self._face_group_map[group]
            self.visualization['face_groups'].append(group_id)

        return self

    def add_faces(self, faces: np.ndarray, colors: Optional[np.ndarray] = None,
                  group: Optional[str] = None):
        """
        Batch add multiple faces for performance.

        Parameters
        ----------
        faces : np.ndarray
            [F, 3] array of node indices
        colors : Optional[np.ndarray]
            [F, 3] or [F, 4] array of RGB/RGBA colors
        group : Optional[str]
            Group name for all faces in this batch

        Returns
        -------
        self
        """
        n_faces = len(faces)
        n_nodes = len(self.nodes['positions'])

        # Validate all face indices
        if faces.shape[1] != 3:
            raise ValueError(f"Faces array must have shape [F, 3], got {faces.shape}")

        if np.any(faces < 0) or np.any(faces >= n_nodes):
            raise ValueError(f"Face indices must be in range [0, {n_nodes})")

        self.visualization['faces'].extend(faces.tolist())

        # Handle colors
        if colors is None:
            default_color = [128, 128, 128, 255]
            self.visualization['face_colors'].extend([default_color] * n_faces)
        else:
            if colors.shape[0] != n_faces:
                raise ValueError(f"Colors array length {colors.shape[0]} must match faces {n_faces}")

            if colors.shape[1] == 3:
                # Add alpha channel
                alpha = np.full((n_faces, 1), 255, dtype=np.uint8)
                colors_rgba = np.hstack([colors, alpha])
            elif colors.shape[1] == 4:
                colors_rgba = colors
            else:
                raise ValueError(f"Colors must have shape [F, 3] or [F, 4], got {colors.shape}")

            self.visualization['face_colors'].extend(colors_rgba.tolist())

        # Handle groups
        if group is None:
            self.visualization['face_groups'].extend([0] * n_faces)
        else:
            if group not in self._face_group_map:
                group_id = len(self._face_group_map) + 1
                self._face_group_map[group] = group_id
                self.visualization['group_names'].append(group)
            else:
                group_id = self._face_group_map[group]
            self.visualization['face_groups'].extend([group_id] * n_faces)

        return self

    def set_visualization_mode(self, topology: str = "triangle", shading: str = "flat"):
        """
        Set visualization rendering hints.

        Parameters
        ----------
        topology : str
            "triangle", "quad", or "polygon"
        shading : str
            "flat" or "smooth" (vertex normals)
        """
        if topology not in ["triangle", "quad", "polygon"]:
            raise ValueError(f"Invalid topology: {topology}")
        if shading not in ["flat", "smooth"]:
            raise ValueError(f"Invalid shading: {shading}")

        self.visualization['topology_type'] = topology
        self.visualization['shading_mode'] = shading
        return self

    # --- Build & Save ---

    def save(self):
        """Writes all buffers to disk in the correct format."""
        self.logger.info("Saving experiment files...")

        # 1. Save Config
        with open(self.input_dir / "config.json", 'w') as f:
            json.dump(self.config, f, indent=4)

        # 2. Save Geometry
        with h5py.File(self.input_dir / "geometry.h5", 'w') as f:
            # Nodes
            g_nodes = f.create_group('nodes')
            g_nodes.create_dataset('positions', data=np.array(self.nodes['positions'], dtype=np.float32))
            g_nodes.create_dataset('masses', data=np.array(self.nodes['masses'], dtype=np.float32))
            g_nodes.create_dataset('attributes', data=np.array(self.nodes['attributes'], dtype=np.uint8))

            # Bars
            if self.bars['indices']:
                g_bars = f.create_group('elements/bars')
                g_bars.create_dataset('indices', data=np.array(self.bars['indices'], dtype=np.int32))
                g_bars.create_dataset('stiffness', data=np.array(self.bars['stiffness'], dtype=np.float32))
                g_bars.create_dataset('rest_length', data=np.array(self.bars['rest_length'], dtype=np.float32))
                g_bars.create_dataset('damping', data=np.array(self.bars['damping'], dtype=np.float32))
                g_bars.create_dataset('prestress', data=np.array(self.bars['prestress'], dtype=np.float32))
            else:
                # Create empty datasets to satisfy validator
                g_bars = f.create_group('elements/bars')
                g_bars.create_dataset('indices', shape=(0, 2), dtype=np.int32)
                g_bars.create_dataset('stiffness', shape=(0,), dtype=np.float32)
                g_bars.create_dataset('rest_length', shape=(0,), dtype=np.float32)

            # Hinges
            if self.hinges['indices']:
                g_hinges = f.create_group('elements/hinges')
                g_hinges.create_dataset('indices', data=np.array(self.hinges['indices'], dtype=np.int32))
                g_hinges.create_dataset('stiffness', data=np.array(self.hinges['stiffness'], dtype=np.float32))
                g_hinges.create_dataset('angle', data=np.array(self.hinges['angle'], dtype=np.float32))
                g_hinges.create_dataset('damping', data=np.array(self.hinges['damping'], dtype=np.float32))

        # 3. Save Signals
        with h5py.File(self.input_dir / "signals.h5", 'w') as f:
            f.attrs['dt_base'] = self.signals['dt_base']
            for name, data in self.signals['data'].items():
                f.create_dataset(name, data=data)

        # 4. Save Visualization (only if faces exist)
        if self.visualization['faces']:
            with h5py.File(self.input_dir / "visualization.h5", 'w') as f:
                # Required: faces
                f.create_dataset("faces", data=np.array(self.visualization['faces'], dtype=np.int32))

                # Optional: face colors
                if self.visualization['face_colors']:
                    f.create_dataset("face_colors",
                                     data=np.array(self.visualization['face_colors'], dtype=np.uint8))

                # Optional: face groups
                if any(g != 0 for g in self.visualization['face_groups']):
                    f.create_dataset("face_groups",
                                     data=np.array(self.visualization['face_groups'], dtype=np.int32))

                    # Group names
                    if self.visualization['group_names']:
                        dt = h5py.special_dtype(vlen=str)
                        f.create_dataset("group_names",
                                         data=np.array(self.visualization['group_names'], dtype=object),
                                         dtype=dt)

                # Attributes
                f.attrs['topology_type'] = self.visualization.get('topology_type', 'triangle')
                f.attrs['shading_mode'] = self.visualization.get('shading_mode', 'flat')

            self.logger.info(f"Visualization file saved with {len(self.visualization['faces'])} faces")

        self.logger.info(f"Experiment saved to {self.root}")
        return self