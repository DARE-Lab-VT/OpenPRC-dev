import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import os

from ...utils.logging import get_logger


class CudaSolver:
    """
    CUDA-accelerated solver for the Bar-Hinge model.

    This solver uses PyCUDA to execute physics calculations on the GPU. It implements
    an RK4 integration scheme for time-stepping and supports both soft (spring-damper)
    and rigid (Position Based Dynamics - PBD) constraints for bars and hinges.
    """

    def __init__(self, n_nodes, mass, attributes, bars, hinges=None, options=None):
        """
        Initialize the CUDA solver.

        Args:
            n_nodes (int): Number of particles.
            mass (np.ndarray): Array of particle masses.
            attributes (np.ndarray): Array of particle attributes (e.g., fixed/free).
            bars (dict): Dictionary containing bar element data.
            hinges (dict, optional): Dictionary containing hinge element data.
            options (dict, optional): Physics options (gravity, damping, etc.).
        """
        self.logger = get_logger("demlat.model.cuda")
        self.n_nodes = n_nodes
        self.options = options or {}

        # Load Kernel
        kernel_path = os.path.join(os.path.dirname(__file__), 'kernels', 'barhinge.cu')
        with open(kernel_path, 'r') as f:
            src = f.read()
        self.mod = SourceModule(src, no_extern_c=True)

        # Kernels
        self.k_zero = self.mod.get_function("zero_forces")
        self.k_act_pos = self.mod.get_function("apply_position_actuation")
        self.k_bars = self.mod.get_function("compute_bar_forces")
        self.k_hinges = self.mod.get_function("compute_hinge_forces")
        self.k_glob = self.mod.get_function("apply_global_forces")
        self.k_rk4_step = self.mod.get_function("rk4_step_integrate")
        self.k_rk4_final = self.mod.get_function("rk4_final_update")
        # PBD
        self.k_proj_bars = self.mod.get_function("project_rigid_bars")
        self.k_proj_hinges = self.mod.get_function("project_rigid_hinges")  # NEW
        self.k_vel_rigid = self.mod.get_function("correct_rigid_velocity")
        self.k_vel_correction = self.mod.get_function("correct_all_rigid_velocities")

        # Dimensions
        self.block = (256, 1, 1)
        self.grid_nodes = ((self.n_nodes + 255) // 256, 1)

        # 1. State Allocation
        self.d_x = cuda.mem_alloc(n_nodes * 3 * 8)
        self.d_v = cuda.mem_alloc(n_nodes * 3 * 8)
        self.d_f = cuda.mem_alloc(n_nodes * 3 * 8)
        self.d_mass = cuda.mem_alloc(mass.nbytes)
        cuda.memcpy_htod(self.d_mass, mass)
        self.d_attrs = cuda.mem_alloc(attributes.nbytes)
        cuda.memcpy_htod(self.d_attrs, attributes)

        # 2. Bars (Soft vs Rigid)
        n_total_bars = len(bars['indices'])
        if n_total_bars > 0:
            is_rigid = bars['stiffness'] < 0

            # Soft
            soft_mask = ~is_rigid
            self.n_bars = np.sum(soft_mask)
            if self.n_bars > 0:
                s_idx = bars['indices'][soft_mask].flatten().astype(np.int32)
                s_params = np.zeros((self.n_bars, 4), dtype=np.float64)
                s_params[:, 0] = bars['stiffness'][soft_mask]
                s_params[:, 1] = bars['rest_length'][soft_mask]
                s_params[:, 2] = bars['damping'][soft_mask]
                if 'prestress' in bars: s_params[:, 3] = bars['prestress'][soft_mask]

                self.d_bar_idx = cuda.mem_alloc(s_idx.nbytes)
                cuda.memcpy_htod(self.d_bar_idx, s_idx)
                self.d_bar_params = cuda.mem_alloc(s_params.nbytes)
                cuda.memcpy_htod(self.d_bar_params, s_params)
                self.grid_bars = ((int(self.n_bars) + 255) // 256, 1)
            else:
                self.n_bars = 0

            # Rigid Bars
            self.n_rigid_bars = np.sum(is_rigid)
            if self.n_rigid_bars > 0:
                r_idx = bars['indices'][is_rigid].flatten().astype(np.int32)
                r_l0 = bars['rest_length'][is_rigid].astype(np.float64)
                self.d_rbar_idx = cuda.mem_alloc(r_idx.nbytes)
                cuda.memcpy_htod(self.d_rbar_idx, r_idx)
                self.d_rbar_l0 = cuda.mem_alloc(r_l0.nbytes)
                cuda.memcpy_htod(self.d_rbar_l0, r_l0)
                self.grid_rbars = ((int(self.n_rigid_bars) + 255) // 256, 1)
            else:
                self.n_rigid_bars = 0
        else:
            self.n_bars = 0
            self.n_rigid_bars = 0

        # 3. Hinges (Soft vs Rigid)
        self.n_hinges = 0
        self.n_rigid_hinges = 0

        if hinges and len(hinges['indices']) > 0:
            is_rigid_h = hinges['stiffness'] < 0

            # Soft Hinges
            soft_h_mask = ~is_rigid_h
            self.n_hinges = np.sum(soft_h_mask)
            if self.n_hinges > 0:
                h_idx = hinges['indices'][soft_h_mask].flatten().astype(np.int32)
                h_params = np.zeros((self.n_hinges, 4), dtype=np.float64)
                h_params[:, 0] = hinges['stiffness'][soft_h_mask]
                h_params[:, 1] = hinges['angle'][soft_h_mask]
                h_params[:, 2] = hinges['damping'][soft_h_mask]

                self.d_hinge_idx = cuda.mem_alloc(h_idx.nbytes)
                cuda.memcpy_htod(self.d_hinge_idx, h_idx)
                self.d_hinge_params = cuda.mem_alloc(h_params.nbytes)
                cuda.memcpy_htod(self.d_hinge_params, h_params)
                self.grid_hinges = ((int(self.n_hinges) + 255) // 256, 1)

            # Rigid Hinges
            self.n_rigid_hinges = np.sum(is_rigid_h)
            if self.n_rigid_hinges > 0:
                rh_idx = hinges['indices'][is_rigid_h].flatten().astype(np.int32)
                rh_phi0 = hinges['angle'][is_rigid_h].astype(np.float64)

                self.d_rhinge_idx = cuda.mem_alloc(rh_idx.nbytes)
                cuda.memcpy_htod(self.d_rhinge_idx, rh_idx)
                self.d_rhinge_phi = cuda.mem_alloc(rh_phi0.nbytes)
                cuda.memcpy_htod(self.d_rhinge_phi, rh_phi0)
                self.grid_rhinges = ((int(self.n_rigid_hinges) + 255) // 256, 1)

        # 4. RK4 Buffers
        sz = n_nodes * 3 * 8
        self.d_k1x = cuda.mem_alloc(sz)
        self.d_k1v = cuda.mem_alloc(sz)
        self.d_k2x = cuda.mem_alloc(sz)
        self.d_k2v = cuda.mem_alloc(sz)
        self.d_k3x = cuda.mem_alloc(sz)
        self.d_k3v = cuda.mem_alloc(sz)
        self.d_k4x = cuda.mem_alloc(sz)
        self.d_k4v = cuda.mem_alloc(sz)
        self.d_xt = cuda.mem_alloc(sz)
        self.d_vt = cuda.mem_alloc(sz)

        # 5. Actuation
        self.actuator_indices = np.where(attributes.astype(np.uint8) & 2)[0].astype(np.int32)
        self.n_actuators = len(self.actuator_indices)
        if self.n_actuators > 0:
            self.d_act_idx = cuda.mem_alloc(self.actuator_indices.nbytes)
            cuda.memcpy_htod(self.d_act_idx, self.actuator_indices)
            self.d_act_vals = cuda.mem_alloc(self.n_actuators * 3 * 8)
            self.h_act_vals = np.zeros(self.n_actuators * 3, dtype=np.float64)

    def upload_state(self, x, v):
        """
        Upload state vectors (position and velocity) to the GPU.

        Args:
            x (np.ndarray): Position array (N x 3).
            v (np.ndarray): Velocity array (N x 3).
        """
        cuda.memcpy_htod(self.d_x, x.astype(np.float64))
        cuda.memcpy_htod(self.d_v, v.astype(np.float64))

    def download_state(self):
        """
        Download state vectors from the GPU.

        Returns:
            tuple: (positions, velocities) as numpy arrays (float32).
        """
        x = np.zeros(self.n_nodes * 3, dtype=np.float64)
        v = np.zeros(self.n_nodes * 3, dtype=np.float64)
        cuda.memcpy_dtoh(x, self.d_x)
        cuda.memcpy_dtoh(v, self.d_v)
        return x.astype(np.float32).reshape(-1, 3), v.astype(np.float32).reshape(-1, 3)

    def step(self, t, dt, actuation_map):
        """
        Perform a single simulation step.

        This involves:
        1. Updating actuation values on the GPU.
        2. Performing RK4 integration (4 stages of force computation).
        3. Applying PBD constraints for rigid elements.

        Args:
            t (float): Current simulation time.
            dt (float): Time step size.
            actuation_map (dict): Dictionary of actuation commands.
        """
        # 1. Update Actuation
        if self.n_actuators > 0 and actuation_map:
            for i, node_idx in enumerate(self.actuator_indices):
                if node_idx in actuation_map and actuation_map[node_idx]['type'] == 'position':
                    val = actuation_map[node_idx]['value']
                    self.h_act_vals[i * 3:i * 3 + 3] = val
            cuda.memcpy_htod(self.d_act_vals, self.h_act_vals)
            self.k_act_pos(np.int32(self.n_actuators), self.d_act_idx, self.d_act_vals, self.d_x, self.d_v,
                           np.float64(dt), block=self.block,
                           grid=((self.n_actuators + 255) // 256, 1))

        # 2. Physics Params
        grav = self.options.get('gravity', -9.81)
        damp = self.options.get('global_damping', 0.1)

        def compute_forces(x_ptr, v_ptr, f_ptr):
            self.k_zero(np.int32(self.n_nodes), f_ptr, block=self.block, grid=self.grid_nodes)
            if self.n_bars > 0:
                self.k_bars(np.int32(self.n_bars), self.d_bar_idx, self.d_bar_params, x_ptr, v_ptr, self.d_attrs, f_ptr,
                            block=self.block, grid=self.grid_bars)
            if self.n_hinges > 0:
                self.k_hinges(np.int32(self.n_hinges), self.d_hinge_idx, self.d_hinge_params, x_ptr, v_ptr,
                              self.d_attrs, f_ptr, block=self.block, grid=self.grid_hinges)
            self.k_glob(np.int32(self.n_nodes), v_ptr, self.d_mass, self.d_attrs, np.float64(damp), np.float64(grav),
                        f_ptr, block=self.block, grid=self.grid_nodes)

        # 3. RK4 Loop (K1-K4)
        for stage, kx, kv, frac in [(1, self.d_k1x, self.d_k1v, 0.5), (2, self.d_k2x, self.d_k2v, 0.5),
                                    (3, self.d_k3x, self.d_k3v, 1.0), (4, self.d_k4x, self.d_k4v, 1.0)]:
            x_in = self.d_x if stage == 1 else self.d_xt
            v_in = self.d_v if stage == 1 else self.d_vt
            compute_forces(x_in, v_in, self.d_f)
            self.k_rk4_step(np.int32(self.n_nodes), np.float64(dt), self.d_x, self.d_v, self.d_f, self.d_mass,
                            self.d_attrs, kx, kv, self.d_xt, self.d_vt, np.float64(frac), block=self.block,
                            grid=self.grid_nodes)

        self.k_rk4_final(np.int32(self.n_nodes), np.float64(dt), self.d_x, self.d_v, self.d_k1x, self.d_k1v, self.d_k2x,
                         self.d_k2v, self.d_k3x, self.d_k3v, self.d_k4x, self.d_k4v, self.d_attrs, block=self.block,
                         grid=self.grid_nodes)

        # 4. PBD Loop (Rigid Constraints)
        if self.n_rigid_bars > 0 or self.n_rigid_hinges > 0:
            for _ in range(5):
                if self.n_rigid_bars > 0:
                    self.k_proj_bars(
                        np.int32(self.n_rigid_bars),
                        self.d_rbar_idx,
                        self.d_rbar_l0,
                        self.d_mass,
                        self.d_attrs,
                        self.d_x,
                        np.float64(0.8),
                        block=self.block,
                        grid=self.grid_rbars
                    )

                if self.n_rigid_hinges > 0:
                    self.k_proj_hinges(
                        np.int32(self.n_rigid_hinges),
                        self.d_rhinge_idx,
                        self.d_rhinge_phi,
                        self.d_mass,
                        self.d_attrs,
                        self.d_x,
                        np.float64(0.8),
                        block=self.block,
                        grid=self.grid_rhinges
                    )

            # Velocity Correction (only for rigid bars)
            if self.n_rigid_bars > 0:
                self.k_vel_rigid(
                    np.int32(self.n_rigid_bars),
                    self.d_rbar_idx,
                    self.d_mass,
                    self.d_attrs,
                    self.d_x,
                    self.d_v,
                    block=self.block,
                    grid=self.grid_rbars
                )
