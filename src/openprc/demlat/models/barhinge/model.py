from ...core.base_model import BaseModel
from ...core.exceptions import ConfigurationError
from ...core.scaler import SimulationScaler
from ...utils.logging import get_logger
import numpy as np

# Import Solvers
try:
    from .solver_cuda import CudaSolver
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    CudaSolver = None

try:
    from .solver_cpu import CpuSolver
    CPU_AVAILABLE = True
except ImportError:
    CPU_AVAILABLE = False
    CpuSolver = None


class BarHingeModel(BaseModel):
    """
    A physics model simulating a system of particles connected by bars and hinges.

    This model supports both CPU and CUDA backends for simulation. It handles
    loading geometry and physical properties from HDF5 files, non-dimensionalization
    of physical quantities, and stepping the simulation forward in time.
    """

    def __init__(self, experiment, backend='auto', precision='float64'):
        """
        Initialize the BarHingeModel.

        Args:
            experiment: The experiment object containing configuration and paths.
            backend (str): The computation backend ('auto', 'cpu', 'cuda').
            precision (str): Floating point precision ('float32' or 'float64').
        """
        super().__init__(experiment, backend, precision)
        self.logger = get_logger("demlat.model.barhinge")

        # Backend Logic
        if backend == 'cuda':
            if not CUDA_AVAILABLE:
                self.logger.error("Backend 'cuda' requested but solver_cuda module could not be imported.")
                raise ConfigurationError("Backend 'cuda' requested but solver_cuda not available.")
            self.use_cuda = True
        elif backend == 'cpu':
            if not CPU_AVAILABLE:
                self.logger.error("Backend 'cpu' requested but solver_cpu module could not be imported.")
                raise ConfigurationError("Backend 'cpu' requested but solver_cpu not available.")
            self.use_cuda = False
        else:  # auto
            if CUDA_AVAILABLE:
                self.use_cuda = True
                self.logger.info("Auto-backend: Selected CUDA.")
            elif CPU_AVAILABLE:
                self.use_cuda = False
                self.logger.warning("Auto-backend: CUDA not available. Falling back to CPU.")
            else:
                self.logger.critical("Auto-backend failed: Neither CUDA nor CPU solvers are available.")
                raise ConfigurationError("Backend 'auto' requested but neither solver_cuda nor solver_cpu are available.")

        # 1. Load Data
        import h5py
        with h5py.File(experiment.paths['geometry'], 'r') as f:
            self.x0 = f['nodes/positions'][:]
            self.m = f['nodes/masses'][:]
            self.attrs = f['nodes/attributes'][:]

            # Load Bars
            self.bars = self._load_dict(f, 'elements/bars',
                                        ['indices', 'stiffness', 'rest_length', 'damping', 'prestress'])

            # Load Hinges
            self.hinges = self._load_dict(f, 'elements/hinges', ['indices', 'stiffness', 'phi0', 'angle', 'damping'])
            if 'phi0' in self.hinges and 'angle' not in self.hinges: self.hinges['angle'] = self.hinges.pop('phi0')
            if 'angle' not in self.hinges: self.hinges['angle'] = np.array([], dtype=np.float32)

        # 2. NON-DIMENSIONALIZATION (SCALING)
        all_stiff = np.concatenate([
            self.bars.get('stiffness', []),
            self.hinges.get('stiffness', [])
        ])
        self.scaler = SimulationScaler(self.m, self.x0, all_stiff)

        # Scale Geometry
        x0_sim = self.scaler.to_sim(self.x0, 'length')

        # Store scaled mass in Model for diagnostics
        self.m_sim = self.scaler.to_sim(self.m, 'mass')

        # Scale Bars
        if 'stiffness' in self.bars: self.bars['stiffness'] = self.scaler.to_sim(self.bars['stiffness'], 'stiffness')
        if 'rest_length' in self.bars: self.bars['rest_length'] = self.scaler.to_sim(self.bars['rest_length'], 'length')
        if 'damping' in self.bars: self.bars['damping'] = self.scaler.to_sim(self.bars['damping'], 'damping')

        # Scale Hinges
        if 'stiffness' in self.hinges: self.hinges['stiffness'] = self.scaler.to_sim(self.hinges['stiffness'],
                                                                                     'torque_k')
        if 'damping' in self.hinges: self.hinges['damping'] = self.scaler.to_sim(self.hinges['damping'], 'damping')

        # Scale Physics Options
        self.phys_opts = experiment.config.get('global_physics', {}).copy()

        # Default Gravity: -9.81 m/s^2
        grav_real = self.phys_opts.get('gravity', -9.81)
        self.phys_opts['gravity'] = self.scaler.to_sim(grav_real, 'accel')

        # Default Damping: 0.1
        damp_real = self.phys_opts.get('global_damping', 0.1)
        self.phys_opts['global_damping'] = self.scaler.to_sim(damp_real, 'damping')

        # 3. Initialize Solver
        SolverClass = CudaSolver if self.use_cuda else CpuSolver
        self.logger.info(f"Initializing solver: {SolverClass.__name__}")

        self.solver = SolverClass(
            n_nodes=self.n_nodes,
            mass=self.m_sim,
            attributes=self.attrs,
            bars=self.bars,
            hinges=self.hinges,
            options=self.phys_opts
        )

        self.solver.upload_state(x0_sim, np.zeros_like(x0_sim))

    def _load_dict(self, f, group, keys):
        """
        Helper to load a dictionary of arrays from an HDF5 group.

        Args:
            f (h5py.File): The HDF5 file handle.
            group (str): The group path within the HDF5 file.
            keys (list): List of dataset names to load from the group.

        Returns:
            dict: A dictionary containing the loaded arrays.
        """
        d = {}
        if group in f:
            g = f[group]
            for k in keys:
                if k in g: d[k] = g[k][:]
        if 'indices' not in d: d['indices'] = np.array([], dtype=np.int32)
        return d

    def step(self, t, dt, actuation_state):
        """
        Advance the simulation by one time step.

        Args:
            t (float): Current simulation time (physical units).
            dt (float): Time step size (physical units).
            actuation_state (dict): Dictionary of actuation commands for specific nodes.

        Returns:
            dict: A dictionary containing the new state (positions, velocities) and
                  diagnostic information (energies, strains) in physical units.
        """
        # 1. Scale Input
        t_sim = self.scaler.to_sim(t, 'time')
        dt_sim = self.scaler.to_sim(dt, 'time')

        # 2. Scale Actuation
        act_sim = {}
        for node, data in actuation_state.items():
            act_sim[node] = data.copy()
            if data['type'] == 'position':
                act_sim[node]['value'] = self.scaler.to_sim(data['value'], 'length')
            elif data['type'] == 'velocity':
                act_sim[node]['value'] = self.scaler.to_sim(data['value'], 'velocity')
            elif data['type'] == 'force':
                act_sim[node]['value'] = self.scaler.to_sim(data['value'], 'force')

        # 3. Run Solver
        self.solver.step(t_sim, dt_sim, act_sim)

        # 4. Download & Unscale
        x_sim, v_sim = self.solver.download_state()

        # Convert State to Real Units
        x_real = self.scaler.from_sim(x_sim, 'length')
        v_real = self.scaler.from_sim(v_sim, 'velocity')

        # Pack Result
        result = {
            'positions': x_real,
            'velocities': v_real,
        }
        return result

    def reset(self):
        """
        Reset the simulation state to the initial configuration.
        """
        x0_sim = self.scaler.to_sim(self.x0, 'length')
        self.solver.upload_state(x0_sim, np.zeros_like(x0_sim))
