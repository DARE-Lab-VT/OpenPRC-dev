"""
demlat Validator
================
Enforces schema correctness and logical integrity of the experiment.
Also provides inspection utilities for debugging HDF5 files.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import h5py
import json

from ..core.exceptions import (
    SchemaValidationError,
    DataIntegrityError,
    ConfigurationError
)
from ..utils.logging import get_logger

# Define expected HDF5 structures
REQUIRED_GEOMETRY_DATASETS = [
    '/nodes/positions',
    '/nodes/masses',
    '/nodes/attributes',
    '/elements/bars/indices',
    '/elements/bars/stiffness',
    '/elements/bars/rest_length',
]


class ExperimentValidator:
    def __init__(self, exp_path: Path, logger=None):
        self.root = exp_path
        self.logger = logger or get_logger("demlat.validator")

        self.input_dir = self.root / "input"
        self.files = {
            'config': self.input_dir / "config.json",
            'geometry': self.input_dir / "geometry.h5",
            'signals': self.input_dir / "signals.h5"
        }

    def validate_all(self):
        """Runs all validation checks sequentially."""
        self.logger.info(f"Validating experiment: {self.root.name}", exc_info=True)
        self._check_directory_structure()

        # We need to open files to check internals.
        # Using context managers ensures they close even if validation fails.
        with h5py.File(self.files['geometry'], 'r') as geo, \
                h5py.File(self.files['signals'], 'r') as sig:
            # 1. Check Hardware (Geometry)
            n_nodes = self._validate_geometry_schema(geo)

            # 2. Check Software (Signals)
            available_signals = self._validate_signals_schema(sig)

            # 3. Check Wiring (Config logic)
            # We must load config content for this
            with open(self.files['config'], 'r') as f:
                config_data = json.load(f)

            self._validate_config_logic(config_data, n_nodes, available_signals)

        self.logger.info("Validation successful. Simulation is robust.", exc_info=True)

    def inspect_file(self, file_type: str, show_data: bool = True, max_sample: int = 3):
        """
        Inspects a specific file type (geometry, signals, simulation) from the experiment.
        
        Args:
            file_type: 'geometry', 'signals', or 'simulation'
            show_data: Whether to print sample data values
            max_sample: Max number of samples to show
        """
        if file_type == 'simulation':
            path = self.root / "output" / "simulation.h5"
        else:
            path = self.files.get(file_type)

        if not path:
            self.logger.error(f"Unknown file type: {file_type}", exc_info=True)
            return

        self._inspect_h5(path, show_data, max_sample)

    def compare_geometry_simulation(self):
        """
        Compares the input geometry with the output simulation file to check for consistency.
        """
        sim_path = self.root / "output" / "simulation.h5"
        geom_path = self.files['geometry']

        if not sim_path.exists():
            self.logger.error(f"Simulation file not found: {sim_path}", exc_info=True)
            return

        self._compare_geometry_and_simulation(geom_path, sim_path)

    def _check_directory_structure(self):
        """Checks if files exist."""
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory missing: {self.input_dir}")
        for name, path in self.files.items():
            if not path.exists():
                raise FileNotFoundError(f"Required file missing: {name} ({path})")

    def _validate_geometry_schema(self, f: h5py.File) -> int:
        """Checks geometry.h5 structure and returns number of nodes."""
        # 1. Check existence of datasets
        for key in REQUIRED_GEOMETRY_DATASETS:
            if key not in f:
                raise SchemaValidationError(f"geometry.h5 missing dataset: {key}")

        # 2. Check Dimensions (Node consistency)
        pos = f['/nodes/positions']
        mass = f['/nodes/masses']
        attr = f['/nodes/attributes']

        n_nodes = pos.shape[0]

        if pos.shape[1] != 3:
            raise SchemaValidationError(f"Node positions must be (N, 3), got {pos.shape}")

        if mass.shape[0] != n_nodes:
            raise DataIntegrityError(f"Node count mismatch: positions={n_nodes}, masses={mass.shape[0]}")

        if attr.shape[0] != n_nodes:
            raise DataIntegrityError(f"Node count mismatch: positions={n_nodes}, attributes={attr.shape[0]}")

        # 3. Check Bars (Element consistency)
        bars_idx = f['/elements/bars/indices']
        if bars_idx.shape[0] > 0:
            if bars_idx.shape[1] != 2:
                raise SchemaValidationError("Bar indices must be (M, 2)")

            # Corner Case: Indices out of bounds
            # Note: Checking min/max can be slow on massive datasets,
            # but is crucial for "robustness".
            max_idx = np.max(bars_idx)
            min_idx = np.min(bars_idx)
            if max_idx >= n_nodes:
                raise DataIntegrityError(f"Bar refers to non-existent node index {max_idx} (Max nodes: {n_nodes})")
            if min_idx < 0:
                raise DataIntegrityError(f"Bar refers to negative node index {min_idx}")

        self.logger.debug(f"Geometry valid: {n_nodes} nodes, {bars_idx.shape[0]} bars.", exc_info=True)
        return n_nodes

    def _validate_signals_schema(self, f: h5py.File) -> List[str]:
        """Checks signals.h5 and returns list of available signal names."""
        if 'dt_base' not in f.attrs:
            self.logger.warning("signals.h5 missing 'dt_base' attribute. Assuming default 0.001s.", exc_info=True)

        signals = []

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                signals.append("/" + name if not name.startswith('/') else name)

        f.visititems(visitor)

        return signals

    def _validate_config_logic(self, config: Dict, n_nodes: int, available_signals: List[str]):
        """Ensures the config wiring makes sense physically."""

        # 1. Check Actuators
        if 'actuators' in config:
            for i, act in enumerate(config['actuators']):
                # Node Bounds
                node_idx = act.get('node_idx')
                if node_idx is None:
                    raise ConfigurationError(f"Actuator {i} missing 'node_idx'")
                if not (0 <= node_idx < n_nodes):
                    raise ConfigurationError(
                        f"Actuator {i} targets invalid node {node_idx}. Valid range: 0-{n_nodes - 1}")

                # Signal Existence
                sig_name = act.get('signal_name')
                # Normalize signal name for check (handle leading slashes)
                normalized_sigs = [s.strip('/') for s in available_signals]
                normalized_req = sig_name.strip('/')

                if normalized_req not in normalized_sigs:
                    raise ConfigurationError(
                        f"Actuator {i} requests missing signal '{sig_name}'. Available: {normalized_sigs}")

        self.logger.debug("Configuration logic (Wiring) is valid.", exc_info=True)

    def _inspect_h5(self, filepath: Path, show_data_samples: bool = True, max_sample: int = 3):
        """
        Thoroughly inspect an HDF5 file structure.
        """
        if not filepath.exists():
            print(f"❌ File not found: {filepath}")
            return

        print(f"\n{'=' * 60}")
        print(f"📁 HDF5 FILE: {filepath}")
        print(f"   Size: {filepath.stat().st_size / 1024:.2f} KB")
        print(f"{'=' * 60}\n")

        with h5py.File(filepath, 'r') as f:

            # Root Attributes
            print("📋 ROOT ATTRIBUTES:")
            print("-" * 40)
            if len(f.attrs) == 0:
                print("   ⚠️  No root attributes found!")
            else:
                for key, value in f.attrs.items():
                    print(f"   {key}: {value} ({type(value).__name__})")
            print()

            # Walk entire structure
            print("📂 STRUCTURE:")
            print("-" * 40)

            def visit_item(name, obj):
                indent = "   " * (name.count('/') + 1)

                if isinstance(obj, h5py.Group):
                    n_children = len(obj.keys())
                    print(f"{indent}📁 {name}/ ({n_children} items)")

                    # Group attributes
                    if len(obj.attrs) > 0:
                        for attr_key, attr_val in obj.attrs.items():
                            print(f"{indent}   @{attr_key}: {attr_val}")

                elif isinstance(obj, h5py.Dataset):
                    shape = obj.shape
                    dtype = obj.dtype
                    size_kb = obj.nbytes / 1024 if obj.nbytes else 0

                    # Check if empty
                    is_empty = any(s == 0 for s in shape)
                    empty_marker = " ⚠️ EMPTY" if is_empty else ""

                    print(f"{indent}📊 {name}: {dtype} {shape} ({size_kb:.2f} KB){empty_marker}")

                    # Show sample data
                    if show_data_samples and not is_empty and len(shape) > 0:
                        try:
                            data = obj[:]
                            if len(data) > 0:
                                # Show first few values
                                if data.ndim == 1:
                                    samples = data[:max_sample]
                                    print(f"{indent}   └─ First {len(samples)}: {samples}")
                                elif data.ndim == 2:
                                    print(f"{indent}   └─ First row: {data[0][:min(5, data.shape[1])]}")
                                elif data.ndim == 3:
                                    print(f"{indent}   └─ [0,0,:]: {data[0, 0, :min(5, data.shape[2])]}")

                                # Stats for numeric data
                                if np.issubdtype(dtype, np.number):
                                    print(f"{indent}   └─ Range: [{np.min(data):.4g}, {np.max(data):.4g}]")
                        except Exception as e:
                            print(f"{indent}   └─ Error reading: {e}")

            f.visititems(visit_item)

            # Summary
            print()
            print("📊 SUMMARY:")
            print("-" * 40)

            datasets = []
            groups = []

            def collect(name, obj):
                if isinstance(obj, h5py.Dataset):
                    datasets.append((name, obj.shape, obj.dtype))
                elif isinstance(obj, h5py.Group):
                    groups.append(name)

            f.visititems(collect)

            print(f"   Groups: {len(groups)}")
            print(f"   Datasets: {len(datasets)}")

            # Check for expected schema
            print()
            print("🔍 SCHEMA CHECK:")
            print("-" * 40)

            expected = [
                'time_series/time',
                'time_series/nodes/positions',
                'time_series/nodes/velocities',
                'time_series/nodes/kinetic_energy',
                'time_series/nodes/potential_energy',
                'time_series/elements/bars/strain',
                'time_series/elements/bars/stress',
                'time_series/elements/bars/potential_energy',
                'time_series/elements/hinges/angle',
                'time_series/elements/hinges/torsional_strain',
                'time_series/elements/hinges/potential_energy',
                'time_series/system/kinetic_energy',
                'time_series/system/potential_energy',
                'time_series/system/damping_loss',
            ]

            for path in expected:
                if path in f:
                    shape = f[path].shape
                    is_empty = any(s == 0 for s in shape)
                    status = "⚠️ EMPTY" if is_empty else "✅"
                    print(f"   {status} {path}: {shape}")
                else:
                    print(f"   ❌ {path}: MISSING")

    def _compare_geometry_and_simulation(self, geom_path: Path, sim_path: Path):
        """
        Compare geometry and simulation files to diagnose issues.
        """
        print(f"\n{'=' * 60}")
        print("🔗 GEOMETRY ↔ SIMULATION COMPARISON")
        print(f"{'=' * 60}\n")

        with h5py.File(geom_path, 'r') as geom, h5py.File(sim_path, 'r') as sim:

            # Node count
            if 'nodes/positions' in geom:
                n_nodes = geom['nodes/positions'].shape[0]
                print(f"📍 Geometry nodes: {n_nodes}")
            else:
                print("❌ Geometry missing nodes/positions")
                n_nodes = None

            if 'time_series/nodes/positions' in sim:
                sim_shape = sim['time_series/nodes/positions'].shape
                print(f"📍 Simulation node positions: {sim_shape}")
                if n_nodes and sim_shape[1] != n_nodes:
                    print(f"   ⚠️ MISMATCH: Expected {n_nodes} nodes, got {sim_shape[1]}")

            # Bar count
            if 'elements/bars/indices' in geom:
                n_bars = geom['elements/bars/indices'].shape[0]
                print(f"📍 Geometry bars: {n_bars}")
            else:
                print("⚠️ Geometry missing elements/bars/indices")
                n_bars = None

            if 'time_series/elements/bars/strain' in sim:
                sim_bars = sim['time_series/elements/bars/strain'].shape
                print(f"📍 Simulation bar strain: {sim_bars}")
                if n_bars and len(sim_bars) > 1 and sim_bars[1] != n_bars:
                    print(f"   ⚠️ MISMATCH: Expected {n_bars} bars, got {sim_bars[1]}")
            else:
                print("❌ Simulation missing bar strain")

            # Hinge count
            if 'elements/hinges/indices' in geom:
                n_hinges = geom['elements/hinges/indices'].shape[0]
                print(f"📍 Geometry hinges: {n_hinges}")
            else:
                print("⚠️ Geometry missing elements/hinges/indices")
                n_hinges = None

            if 'time_series/elements/hinges/angle' in sim:
                sim_hinges = sim['time_series/elements/hinges/angle'].shape
                print(f"📍 Simulation hinge angles: {sim_hinges}")
            else:
                print("❌ Simulation missing hinge angles")
