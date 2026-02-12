"""
demlat Simulation
=================
"""

import json
import h5py
import shutil
from pathlib import Path
from datetime import datetime
from openprc.schemas.logging import get_logger, setup_file_logging


class Simulation:
    """
    Manages the file system interface for a Demlat simulation, including input/output paths,
    configuration loading, and automatic documentation generation.
    """

    def __init__(self, simulation_dir):
        """Initializes an Simulation object.

        This sets up the simulation's directory structure, loads the configuration,
        and generates a README.md file.

        """
        self.root = Path(simulation_dir)
        self.input_dir = self.root / "input"
        self.output_dir = self.root / "output"
        self.log_dir = self.output_dir / "logs"

        # 1. Define Standard Paths
        self.paths = {
            "config": self.input_dir / "config.json",
            "geometry": self.input_dir / "geometry.h5",
            "signals": self.input_dir / "signals.h5",
            "visualization": self.input_dir / "visualization.h5",
            "simulation": self.output_dir / "simulation.h5",
            "readme": self.root / "README.md",
            "logs": self.log_dir,
            "log": self.log_dir / "simulation.log"
        }

        # 2. Validate Input Existence
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory missing: {self.input_dir}")
        if not self.paths["config"].exists():
            raise FileNotFoundError(f"Config missing: {self.paths['config']}")
        if not self.paths["geometry"].exists():
            raise FileNotFoundError(f"Geometry missing: {self.paths['geometry']}")

        # 3. Setup Output Structure & Logging
        self._setup_outputs()
        setup_file_logging(self.log_dir)

        # 4. Initialize Logger (Now that log_dir exists and handler is set)
        self.logger = get_logger("demlat.simulation")
        self.logger.info(f"Initialized simulation at: {self.root}")

        # 5. Load Configuration
        with open(self.paths["config"], 'r') as f:
            self.config = json.load(f)

        # 6. Auto-Generate Documentation
        self._generate_readme()

    def _setup_outputs(self):
        """
        Ensures that the output and log directories exist.

        Creates them if they do not already exist.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def reset_output(self):
        """
        Clears previous simulation results from the output directory.

        Removes all files and subdirectories within the output directory, then recreates the necessary structure.
        """
        # No logger here, as it might not be initialized yet if called early

        if self.output_dir.exists():
            # Remove all files in output but keep directory structure
            for item in self.output_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        self._setup_outputs()

        # Re-initialize logger since the log file was deleted
        setup_file_logging(self.log_dir)
        # Get a new logger instance after setup
        self.logger = get_logger("demlat.simulation")
        self.logger.info("Output directory reset complete.")

    def _generate_readme(self):
        """
        Auto-generates a README.md summary of the simulation.

        This method gathers information from the configuration and geometry files to create a human-readable summary.
        """

        # Load Stats
        n_nodes, n_bars, n_hinges = 0, 0, 0
        with h5py.File(self.paths["geometry"], 'r') as f:
            n_nodes = f['nodes/positions'].shape[0]
            if 'elements/bars/indices' in f:
                n_bars = f['elements/bars/indices'].shape[0]
            if 'elements/hinges/indices' in f:
                n_hinges = f['elements/hinges/indices'].shape[0]

        # Config Stats
        sim_cfg = self.config.get('simulation', {})
        phys_cfg = self.config.get('global_physics', {})
        actuators = self.config.get('actuators', [])

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        content = f"""# Simulation: {self.root.name}
                **Generated:** {timestamp}
                
                ## 1. Overview
                This is an auto-generated summary of the **{self.root.name}** simulation.
                
                ## 2. Geometry (Hardware)
                * **Nodes:** {n_nodes}
                * **Bars:** {n_bars}
                * **Hinges:** {n_hinges}
                
                ## 3. Configuration (Wiring)
                ### Simulation Settings
                * **Duration:** {sim_cfg.get('duration', 'N/A')} s
                * **Time Step (dt):** {sim_cfg.get('dt_base', 'N/A')} s
                * **Save Interval:** {sim_cfg.get('dt_save', 'N/A')} s
                
                ### Global Physics
                * **Gravity:** {phys_cfg.get('gravity', 'N/A')}
                * **Damping:** {phys_cfg.get('global_damping', 'N/A')}
                
                ### Actuation
                * **Active Actuators:** {len(actuators)}
                """

        # Write File
        with open(self.paths["readme"], 'w') as f:
            f.write(content)

    def get_writer(self):
        """
        Returns a configured HDF5 writer for the output simulation file.

        The file is opened in write mode ('w').
        """
        return h5py.File(self.paths["simulation"], 'w')

    def get_reader(self):
        """
        Returns a configured HDF5 reader for the output simulation file.

        The file is opened in read mode ('r').
        """
        return h5py.File(self.paths["simulation"], 'r')

    def get_geometry_reader(self):
        """
        Returns a configured HDF5 reader for the geometry file.

        The file is opened in read mode ('r').
        """
        return h5py.File(self.paths["geometry"], 'r')

    def get_signals_reader(self):
        """
        Returns a configured HDF5 reader for the signals file.

        The file is opened in read mode ('r').
        """
        return h5py.File(self.paths["signals"], 'r')

    def get_visualization_reader(self):
        """
        Returns a configured HDF5 reader for the visualization file.

        The file is opened in read mode ('r').
        """
        return h5py.File(self.paths["visualization"], 'r')

    def get_log_writer(self, log_name):
        """
        Returns a file object for writing logs within the simulation's log directory.

        :param log_name: The name of the log file (e.g., "simulation.log").
        :return: A file object opened in write mode ('w').
        """
        return open(self.log_dir / log_name, 'a')

    def get_log_reader(self, log_name):
        """
        Returns a file object for reading logs from the simulation's log directory.

        :param log_name: The name of the log file (e.g., "simulation.log").
        :return: A file object opened in read mode ('r').
        """
        return open(self.log_dir / log_name, 'r')

    def show(self, config=None):
        from ..utils.animator import ShowSimulation
        ShowSimulation(str(self.root), config)
