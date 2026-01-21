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
from typing import Any, Dict, List, Optional, Union

from ..core.exceptions import DemlatError, SchemaValidationError, ConfigurationError
from ..utils.logging import get_logger


class ExperimentSetup:
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

    # --- Configuration Methods ---

    def set_simulation_params(self, duration: float, dt: float, save_interval: float):
        """Set time-stepping parameters."""
        self.config['simulation'] = {
            "duration": float(duration),
            "dt_base": float(dt),
            "dt_save": float(save_interval)
        }
        return self

    def set_physics(self, gravity: Union[float, List[float]] = -9.81, damping: float = 0.1):
        """Set global physics constants."""
        self.config['global_physics']['gravity'] = gravity
        self.config['global_physics']['global_damping'] = float(damping)
        self.config['material']['damping_coefficient'] = float(damping)
        return self

    # --- Geometry Methods ---

    def add_node(self, position: List[float], mass: float = 1.0, fixed: bool = False) -> int:
        """
        Add a single node.
        Returns the index of the new node.
        """
        idx = len(self.nodes['positions'])
        self.nodes['positions'].append(position)
        self.nodes['masses'].append(mass)
        
        # Attribute bitmask: 1 = Fixed, 0 = Free
        attr = 1 if fixed else 0
        self.nodes['attributes'].append(attr)
        return idx

    def add_nodes(self, positions: np.ndarray, masses: Union[float, np.ndarray] = 1.0, fixed_mask: Optional[np.ndarray] = None):
        """Batch add nodes for performance."""
        n = len(positions)
        self.nodes['positions'].extend(positions.tolist())
        
        if np.isscalar(masses):
            self.nodes['masses'].extend([masses] * n)
        else:
            self.nodes['masses'].extend(masses.tolist())
            
        if fixed_mask is None:
            self.nodes['attributes'].extend([0] * n)
        else:
            # Convert boolean mask to uint8 attributes (1=Fixed)
            attrs = fixed_mask.astype(np.uint8)
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

        self.logger.info(f"Experiment saved to {self.root}")
        return self
