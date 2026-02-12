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

try:
    from .solver_jax import JaxSolver

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    JaxSolver = None


class BarHingeModel(BaseModel):
    """
    A physics model simulating a system of particles connected by bars and hinges.

    This model supports CPU, CUDA, and JAX backends for simulation. It handles
    loading geometry and physical properties from HDF5 files, non-dimensionalization
    of physical quantities, and stepping the simulation forward in time.

    Backends:
        - 'cuda': GPU-accelerated using PyCUDA
        - 'cpu': CPU-based NumPy implementation
        - 'jax': JAX-accelerated (auto-selects GPU/TPU/CPU)
        - 'auto': Automatic selection (tries CUDA, then JAX, then CPU)
    """

    def __init__(self, experiment, backend='auto', precision='float64'):
        """
        Initialize the BarHingeModel.

        Args:
            experiment: The experiment object containing configuration and paths.
            backend (str): The computation backend ('auto', 'cpu', 'cuda', 'jax').
            precision (str): Floating point precision ('float32' or 'float64').
        """
        super().__init__(experiment, backend, precision)
        self.logger = get_logger("demlat.model.barhinge")

        # Backend Selection Logic
        backend_lower = backend.lower()

        if backend_lower == 'cuda':
            if not CUDA_AVAILABLE:
                self.logger.error("Backend 'cuda' requested but solver_cuda module could not be imported.")
                raise ConfigurationError("Backend 'cuda' requested but solver_cuda not available.")
            SolverClass = CudaSolver
            self.backend_name = 'CUDA'

        elif backend_lower == 'cpu':
            if not CPU_AVAILABLE:
                self.logger.error("Backend 'cpu' requested but solver_cpu module could not be imported.")
                raise ConfigurationError("Backend 'cpu' requested but solver_cpu not available.")
            SolverClass = CpuSolver
            self.backend_name = 'CPU'

        elif backend_lower == 'jax':
            if not JAX_AVAILABLE:
                self.logger.error("Backend 'jax' requested but solver_jax module could not be imported.")
                # raise ConfigurationError(
                #     "Backend 'jax' requested but solver_jax not available. Install with: pip install jax")
                # Fallback to CPU if JAX is not available
                self.logger.warning("Backend 'jax' requested but solver_jax not available. Falling back to CPU.")
                SolverClass = CpuSolver
                self.backend_name = 'CPU'
            else:
                SolverClass = JaxSolver
                self.backend_name = 'JAX'

        elif backend_lower == 'auto':
            # Auto-selection priority: CUDA > JAX > CPU
            if CUDA_AVAILABLE:
                SolverClass = CudaSolver
                self.backend_name = 'CUDA'
                self.logger.info("Auto-backend: Selected CUDA.")
            elif JAX_AVAILABLE:
                SolverClass = JaxSolver
                self.backend_name = 'JAX'
                self.logger.info("Auto-backend: Selected JAX.")
            elif CPU_AVAILABLE:
                SolverClass = CpuSolver
                self.backend_name = 'CPU'
                self.logger.warning("Auto-backend: CUDA and JAX not available. Falling back to CPU.")
            else:
                self.logger.critical("Auto-backend failed: No solvers available.")
                raise ConfigurationError("Backend 'auto' requested but no solvers (CUDA/JAX/CPU) are available.")
        else:
            raise ConfigurationError(f"Unknown backend: '{backend}'. Choose 'auto', 'cpu', 'cuda', or 'jax'.")

        # Store for compatibility
        self.use_cuda = (self.backend_name == 'CUDA')

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
            if 'phi0' in self.hinges and 'angle' not in self.hinges:
                self.hinges['angle'] = self.hinges.pop('phi0')
            if 'angle' not in self.hinges:
                self.hinges['angle'] = np.array([], dtype=np.float32)

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
        if 'stiffness' in self.bars:
            self.bars['stiffness'] = self.scaler.to_sim(self.bars['stiffness'], 'stiffness')
        if 'rest_length' in self.bars:
            self.bars['rest_length'] = self.scaler.to_sim(self.bars['rest_length'], 'length')
        if 'damping' in self.bars:
            self.bars['damping'] = self.scaler.to_sim(self.bars['damping'], 'damping')

        # Scale Hinges
        if 'stiffness' in self.hinges:
            self.hinges['stiffness'] = self.scaler.to_sim(self.hinges['stiffness'], 'torque_k')
        if 'damping' in self.hinges:
            self.hinges['damping'] = self.scaler.to_sim(self.hinges['damping'], 'damping')

        # Scale Physics Options
        self.phys_opts = experiment.config.get('global_physics', {}).copy()

        # Default Gravity: -9.81 m/s^2
        grav_real = self.phys_opts.get('gravity', -9.81)
        self.phys_opts['gravity'] = self.scaler.to_sim(grav_real, 'accel')

        # Default Damping: 0.1
        damp_real = self.phys_opts.get('global_damping', 0.1)
        self.phys_opts['global_damping'] = self.scaler.to_sim(damp_real, 'damping')

        # Collision parameters (already dimensionless or will be scaled in solver)
        if 'enable_collision' in self.phys_opts:
            # Radius needs scaling
            if 'collision_radius' in self.phys_opts:
                radius_real = self.phys_opts['collision_radius']
                self.phys_opts['collision_radius'] = self.scaler.to_sim(radius_real, 'length')
            # Restitution is dimensionless, keep as-is

        # 3. Initialize Solver
        self.logger.info(f"Initializing solver: {SolverClass.__name__} ({self.backend_name} backend)")

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
                if k in g:
                    d[k] = g[k][:]
        if 'indices' not in d:
            d['indices'] = np.array([], dtype=np.int32)
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

        # 3. Run Solver (same interface for all backends!)
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

    def dynamics_step(self, positions, velocities, dt=0.001):
        """
        Pure dynamics step function for black-box analysis.
        
        Takes current state (positions, velocities) in physical units,
        advances the physics by dt (ignoring actuation), and returns
        the new state.

        Args:
            positions (np.ndarray): Current positions (N x 3) in physical units.
            velocities (np.ndarray): Current velocities (N x 3) in physical units.
            dt (float): Time step size in physical units (default: 0.001).

        Returns:
            tuple: (new_positions, new_velocities) in physical units.
        """
        # 1. Scale Input to Simulation Units
        x_sim = self.scaler.to_sim(positions, 'length')
        v_sim = self.scaler.to_sim(velocities, 'velocity')
        dt_sim = self.scaler.to_sim(dt, 'time')

        # 2. Upload State to Solver
        self.solver.upload_state(x_sim, v_sim)

        # 3. Step Solver (No Actuation)
        # We pass an empty actuation map and t=0 (time doesn't matter for autonomous dynamics)
        self.solver.step(0.0, dt_sim, {})

        # 4. Download & Unscale
        x_new_sim, v_new_sim = self.solver.download_state()
        
        x_new = self.scaler.from_sim(x_new_sim, 'length')
        v_new = self.scaler.from_sim(v_new_sim, 'velocity')

        return x_new, v_new

    def reset(self):
        """
        Reset the simulation state to the initial configuration.
        """
        x0_sim = self.scaler.to_sim(self.x0, 'length')
        self.solver.upload_state(x0_sim, np.zeros_like(x0_sim))
