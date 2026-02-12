"""
DEMLAT Engine
=============
High-performance simulation driver.
Production ready: Robust error handling, crash recovery, and auto-analytics.
"""

import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, List, Optional

from openprc.schemas.logging import get_logger
from .post_processor import PostProcessor
from openprc.schemas.demlat_sim_validator import DemlatSimValidator


class SimulationError(Exception): pass


class Engine:
    SCHEMA_VERSION = "2.1.0"

    def __init__(self, model_class, backend='auto', buffer_size=50):
        self.logger = get_logger("demlat.engine")
        self.model_class = model_class
        self.backend = backend
        self.buffer_size = buffer_size
        self._interrupted = False

    def run(self, simulation, auto_process: bool = True) -> Dict[str, Any]:
        """
        Execute the simulation pipeline.
        Phase 1: Validation
        Phase 2: Fast Physics Loop (Positions/Velocities only).
        Phase 3: Auto Post-Processing (Default: True)
        """
        self.logger.info(f"Starting simulation pipeline: {simulation.root.name}")

        # --- 1. Pre-Run Validation ---
        try:
            # Pass the already configured logger to the validator
            validator = DemlatSimValidator(simulation.root, logger=self.logger)
            validator.validate_all()
        except Exception as e:
            self.logger.critical(f"simulation validation failed: {e}", exc_info=True)
            raise SimulationError(f"Validation failed: {e}") from e

        self.logger.info("Validation passed. Proceeding to physics loop.")

        try:
            # --- 2. Setup Configuration ---
            sim_cfg = simulation.config['simulation']

            # Fallback for dt_base if not present (using dt)
            dt = sim_cfg.get('dt_base')
            if dt is None:
                dt = simulation.config['physics'].get('dt', 0.001)

            dt_save = sim_cfg.get('save_interval', 0.01)  # Use save_interval if dt_save missing
            duration = sim_cfg['duration']

            save_interval = max(1, int(dt_save / dt))
            n_steps = int(duration / dt)

            # --- 3. Initialize Model & Signals ---
            model = self.model_class(simulation, backend=self.backend)
            signals = self._load_signals(simulation)
            actuators = simulation.config.get('actuators', [])

            # --- 4. Prepare Output File ---
            save_path = Path(simulation.paths['simulation'])
            save_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure folder exists
            simulation.reset_output()

            t_curr = 0.0
            frames_written = 0
            buffer = {'time': [], 'positions': [], 'velocities': []}

            with h5py.File(save_path, 'w') as f:
                # Initialize Storage
                self._init_minimal_storage(f, model.n_nodes)

                # --- Write Metadata ---
                f.attrs['schema_version'] = self.SCHEMA_VERSION
                f.attrs['frame_rate'] = 1.0 / dt_save
                f.attrs['completed'] = 0
                f.attrs['total_frames'] = 0

                # [FIX] Source Geometry Path
                f.attrs['source_geometry'] = "../input/geometry.h5"

                # [NEW] Check for Visualization File
                # We assume standard structure: ../input/visualization.h5 relative to output/
                viz_path = Path(simulation.root) / "input" / "visualization.h5"
                if viz_path.exists():
                    f.attrs['source_visualization'] = "../input/visualization.h5"
                    self.logger.info("Linked source_visualization.")

                pbar = tqdm(total=n_steps, desc="Physics Loop", unit="step")

                # --- 5. Main Physics Loop ---
                for step in range(n_steps + 1):
                    if self._interrupted: break

                    # Actuation & Step
                    actuation = self._get_actuation(t_curr, signals, actuators)

                    try:
                        raw_state = model.step(t_curr, dt, actuation)
                    except Exception as e:
                        self.logger.error(f"Physics failed at t={t_curr:.3f}: {e}", exc_info=True)
                        # Emergency Flush
                        if buffer['time']:
                            self._flush_buffer(f, buffer)
                            f.attrs['total_frames'] = frames_written + len(buffer['time'])
                        raise e

                    # Buffer Data
                    if step % save_interval == 0:
                        buffer['time'].append(t_curr)
                        buffer['positions'].append(raw_state['positions'])
                        buffer['velocities'].append(raw_state['velocities'])

                        if len(buffer['time']) >= self.buffer_size:
                            self._flush_buffer(f, buffer)
                            frames_written += len(buffer['time'])
                            f.attrs['total_frames'] = frames_written
                            buffer = {'time': [], 'positions': [], 'velocities': []}

                    t_curr += dt
                    pbar.update(1)

                pbar.close()  # Ensure progress bar is closed

                # Final Flush
                if buffer['time']:
                    self._flush_buffer(f, buffer)
                    frames_written += len(buffer['time'])
                    f.attrs['total_frames'] = frames_written

                f.attrs['completed'] = int(not self._interrupted)

        except KeyboardInterrupt:
            self.logger.warning("Simulation interrupted by user.")
            self._interrupted = True
        except Exception as e:
            self.logger.critical(f"Engine Failure: {e}", exc_info=True)
            raise SimulationError(f"Engine Failure: {e}") from e

        # --- 6. Auto Post-Processing ---
        if auto_process and not self._interrupted:
            try:
                self.post_process(simulation)
            except Exception as e:
                self.logger.error(f"Post-processing failed: {e}", exc_info=True)
                # Don't crash the whole run if analytics fail, just log it.

        return {
            'status': 'simulated',
            'frames': frames_written,
            'path': str(save_path)
        }

    def post_process(self, simulation):
        """
        Phase 2: Analytics Loop.
        Reads the simulation file, calculates energy/stress, and adds new datasets.
        """
        path = simulation.paths['simulation']
        self.logger.info(f"Starting Post-Processing: {path}")

        try:
            sc = self._create_state_computer(simulation)
            if not sc:
                self.logger.error("Could not initialize PostProcessor. Skipping analytics.")
                return

            with h5py.File(path, 'r+') as f:
                # 1. Get Simulation Metadata
                n_frames = int(f.attrs.get('total_frames', 0))
                if n_frames == 0:
                    self.logger.warning("No frames to process.")
                    return

                # Determine timestep for integration
                if 'frame_rate' in f.attrs:
                    dt_frame = 1.0 / f.attrs['frame_rate']
                else:
                    dt_frame = 0.01  # Fallback default

                # 2. Initialize Cumulative Variables [CRITICAL FIX]
                # Must be defined OUTSIDE the loop so they persist across chunks
                cumulative_loss = 0.0

                # 3. Process in Chunks
                chunk_size = 100
                pbar = tqdm(total=n_frames, desc="Calculating Analytics", unit="frame")

                for i in range(0, n_frames, chunk_size):
                    end = min(i + chunk_size, n_frames)

                    # Read Batch
                    pos_batch = f['time_series/nodes/positions'][i:end]
                    vel_batch = f['time_series/nodes/velocities'][i:end]

                    batch_results = {}

                    # Compute Analytics for this batch
                    for j in range(len(pos_batch)):
                        frame_res = sc.compute_frame(pos_batch[j], vel_batch[j])

                        # [FIX] Extract & Remove temporary 'Power' value
                        # We use .pop() so this key is NOT sent to the file writer
                        power = frame_res.pop('system_damping_power', 0.0)

                        # [FIX] Integrate Damping Loss
                        cumulative_loss += power * dt_frame

                        # Store result
                        frame_res['time_series/system/damping_loss'] = np.array([cumulative_loss], dtype=np.float32)

                        # Group results for writing
                        for key, val in frame_res.items():
                            if key not in batch_results:
                                batch_results[key] = []
                            batch_results[key].append(val)

                    # Write Batch
                    self._write_analytics_batch(f, batch_results, start_idx=i)
                    pbar.update(end - i)

                pbar.close()  # Ensure progress bar is closed

                self.logger.info("Post-processing complete.")
        except Exception as e:
            self.logger.error(f"Error during post-processing: {e}", exc_info=True)
            raise

    # --- Internal Helpers ---

    def _init_minimal_storage(self, f, n_nodes):
        """Create only the essential datasets."""
        try:
            g = f.create_group('time_series/nodes')
            # Time: Expandable 1D
            f.create_dataset('time_series/time', shape=(0,), maxshape=(None,), dtype='f4', chunks=(100,))
            # Pos/Vel: Expandable (T, N, 3)
            g.create_dataset('positions', shape=(0, n_nodes, 3), maxshape=(None, n_nodes, 3), dtype='f4',
                             chunks=(1, n_nodes, 3))
            g.create_dataset('velocities', shape=(0, n_nodes, 3), maxshape=(None, n_nodes, 3), dtype='f4',
                             chunks=(1, n_nodes, 3))
        except Exception as e:
            self.logger.error(f"Failed to initialize HDF5 storage: {e}", exc_info=True)
            raise

    def _flush_buffer(self, f, buffer):
        """Append buffer to HDF5."""
        try:
            n = len(buffer['time'])
            if n == 0: return

            d_time = f['time_series/time']
            d_pos = f['time_series/nodes/positions']
            d_vel = f['time_series/nodes/velocities']

            old_n = d_time.shape[0]
            new_n = old_n + n

            d_time.resize(new_n, axis=0)
            d_pos.resize(new_n, axis=0)
            d_vel.resize(new_n, axis=0)

            d_time[old_n:] = buffer['time']
            d_pos[old_n:] = np.stack(buffer['positions'])
            d_vel[old_n:] = np.stack(buffer['velocities'])
        except Exception as e:
            self.logger.error(f"Failed to flush buffer to disk: {e}", exc_info=True)
            raise

    def _write_analytics_batch(self, f, batch_data, start_idx):
        """Write computed analytics to HDF5."""
        if not batch_data: return

        try:
            n_new = len(next(iter(batch_data.values())))

            for path, data_list in batch_data.items():
                arr = np.array(data_list)

                # Create dataset if missing
                if path not in f:
                    self._recursive_create_dataset(f, path, arr.shape[1:])

                dset = f[path]
                target_size = start_idx + n_new

                # Expand dataset if needed
                if target_size > dset.shape[0]:
                    dset.resize(target_size, axis=0)

                dset[start_idx:target_size] = arr
        except Exception as e:
            self.logger.error(f"Failed to write analytics batch: {e}", exc_info=True)
            raise

    def _recursive_create_dataset(self, f, path, shape):
        """
        Robustly create a dataset and its parent groups.
        Handles cases where the dataset might already exist.
        """
        # 1. Sanity Check: If it exists, stop immediately.
        if path in f:
            return

        # 2. Ensure parent groups exist
        group_name = path.rsplit('/', 1)[0]
        curr = f
        for part in group_name.split('/'):
            if part:  # Skip empty strings from leading/trailing slashes
                if part not in curr:
                    curr.create_group(part)
                curr = curr[part]

        # 3. Try to create the dataset
        try:
            f.create_dataset(
                path,
                shape=(0,) + shape,
                maxshape=(None,) + shape,
                dtype='f4',
                chunks=True
            )
        except ValueError:
            # Race condition or 'Zombie' dataset found.
            # If HDF5 says it exists, we trust it and move on.
            self.logger.warning(f"Dataset '{path}' already exists. Skipping creation.")
            pass

    def _create_state_computer(self, simulation) -> Optional[PostProcessor]:
        """Load geometry and material to initialize PostProcessor."""
        geom_path = simulation.paths.get('geometry')
        if not geom_path or not Path(geom_path).exists():
            return None

        try:
            geometry = {}
            with h5py.File(geom_path, 'r') as f:
                # Nodes
                if 'nodes/positions' in f:
                    geometry['nodes'] = f['nodes/positions'][:]
                if 'nodes/masses' in f:
                    geometry['masses'] = f['nodes/masses'][:]

                # Bars
                if 'elements/bars/indices' in f:
                    geometry['bars'] = {
                        'indices': f['elements/bars/indices'][:].astype(np.int32),
                    }
                    for attr in ['stiffness', 'rest_length', 'prestress']:
                        key = f'elements/bars/{attr}'
                        if key in f:
                            geometry['bars'][attr] = f[key][:]

                # Hinges
                if 'elements/hinges/indices' in f:
                    geometry['hinges'] = {
                        'indices': f['elements/hinges/indices'][:].astype(np.int32),
                    }
                    for attr in ['stiffness', 'angle']:
                        key = f'elements/hinges/{attr}'
                        if key in f:
                            geometry['hinges'][attr] = f[key][:]

            material = simulation.config.get('material', {})
            return PostProcessor(geometry, material)

        except Exception as e:
            self.logger.error(f"Failed to load geometry for analytics: {e}", exc_info=True)
            return None

    def _load_signals(self, simulation) -> Dict[str, np.ndarray]:
        """Load input signals."""
        path = simulation.paths.get('signals')
        signals = {}
        if not path or not Path(path).exists():
            return signals

        try:
            with h5py.File(path, 'r') as f:
                signals['dt_base'] = f.attrs.get('dt_base', 0.001)
                for k in f.keys():
                    if isinstance(f[k], h5py.Dataset):
                        signals[k] = f[k][:]
        except Exception as e:
            self.logger.error(f"Signal load error: {e}", exc_info=True)

        return signals

    def _get_actuation(self, t: float, signals: Dict[str, np.ndarray], config: List[Dict]) -> Dict[int, Any]:
        """
        Resolve input signals into per-node actuation commands.
        Supports JSON keys: 'node_idx'/'node_id' and 'signal_name'/'signal'.
        """
        if not signals or not config:
            return {}

        try:
            dt_base = signals.get('dt_base', 0.001)
            idx = int(round(t / dt_base))

            actuation_map = {}

            for actuator_def in config:
                # [FIX 1] Handle 'node_idx' (from your JSON) vs 'node_id'
                node_id = actuator_def.get('node_idx')
                if node_id is None:
                    node_id = actuator_def.get('node_id')

                if node_id is None:
                    continue

                # Copy config to pass through 'type', 'dof', etc.
                command = actuator_def.copy()

                sig_key = actuator_def.get('signal_name')
                if not sig_key:
                    sig_key = actuator_def.get('signal')

                if sig_key and sig_key in signals:
                    signal_data = signals[sig_key]

                    # Clamp to signal bounds
                    safe_idx = min(max(0, idx), len(signal_data) - 1)

                    # Attach value
                    command['value'] = signal_data[safe_idx]
                    actuation_map[node_id] = command

            return actuation_map
        except Exception as e:
            self.logger.error(f"Error resolving actuation at t={t}: {e}", exc_info=True)
            return {}
