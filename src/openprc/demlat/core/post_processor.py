"""
State Computer
==============
Post-processing module to calculate derived physics quantities.
"""

import numpy as np
from typing import Dict, Any, Tuple
from openprc.schemas.logging import get_logger

# Try to import JAX for accelerated analytics
try:
    import jax
    import jax.numpy as jnp
    from jax import jit

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

logger = get_logger("demlat.post_processor")


class PostProcessor:
    """
    Computes derived quantities (Energy, Stress, Strain) from raw simulation state.
    Supports JAX acceleration if available.
    """

    def __init__(self, geometry: Dict[str, Any], material: Dict[str, Any]):
        self.logger = get_logger("demlat.post_processor")
        self.geometry = geometry
        self.material = material
        self._bar_rest_lengths = None
        self._bar_rest_lengths_effective = None
        self._hinge_rest_angles = None

        # Pre-extract and cache material/geometry scalars used every frame
        self._masses = None
        self._gravity_vec = None
        self._gravity_mag = None
        self._damping_coefficient = self.material.get('damping_coefficient', 0.0)

        # JAX Acceleration
        self.use_jax = JAX_AVAILABLE
        if self.use_jax:
            self.logger.info("JAX available. Using accelerated analytics.")
            self._init_jax_functions()
        else:
            self.logger.info("JAX not available. Using NumPy analytics.")

        try:
            self._init_caches()
        except Exception as e:
            self.logger.error(f"Failed to initialize PostProcessor caches: {e}", exc_info=True)
            raise

    def _init_caches(self):
        """Pre-compute static quantities like rest lengths, masses, gravity."""
        try:
            # --- Masses (used by KE, PE, damping) ---
            masses = self.geometry.get('masses')
            if masses is not None:
                self._masses = np.asarray(masses, dtype=np.float32)
            # Note: _masses=None is handled at compute time by falling back to unit mass

            # --- Gravity vector (used by gravitational PE) ---
            # --- Gravity vector ---
            # Convention: scalar gravity is the Z-component of the gravity vector.
            #   gravity = -9.81  →  g = [0, 0, -9.81]  (standard: downward)
            #   gravity =  9.81  →  g = [0, 0,  9.81]  (upward — probably wrong)
            #   gravity = [0, -9.81, 0] → Y-axis gravity (explicit vector)
            #
            # IMPORTANT: This MUST match the convention in the physics kernel.
            # If the kernel always applies gravity as F = m * [0, 0, -|g|],
            # then pass gravity as a NEGATIVE scalar (e.g., -9.81).
            grav_param = self.material.get('gravity', 0.0)  # Default: no gravity

            if np.isscalar(grav_param):
                scalar_g = float(grav_param)
                if scalar_g > 0:
                    self.logger.warning(
                        f"Positive scalar gravity ({scalar_g}) interpreted as UPWARD force. "
                        f"If your physics kernel applies gravity downward, use gravity={-scalar_g} instead."
                    )
                self._gravity_vec = np.array([0.0, 0.0, scalar_g], dtype=np.float32)
            else:
                self._gravity_vec = np.asarray(grav_param, dtype=np.float32)

            self._gravity_mag = float(np.linalg.norm(self._gravity_vec))

            # --- Bars ---
            bars = self.geometry.get('bars')
            if bars and 'rest_length' in bars:
                self._bar_rest_lengths = np.asarray(bars['rest_length'], dtype=np.float32)

                # Handle Prestress for PE calculation
                prestress = bars.get('prestress')
                if prestress is not None:
                    prestress = np.asarray(prestress, dtype=np.float32)
                    self._bar_rest_lengths_effective = self._bar_rest_lengths * (1.0 + prestress)
                else:
                    self._bar_rest_lengths_effective = self._bar_rest_lengths

                # Pre-expand scalar stiffness to array
                k = bars.get('stiffness', 1000.0)
                if np.ndim(k) == 0:
                    self._bar_stiffness = np.full(len(self._bar_rest_lengths), k, dtype=np.float32)
                else:
                    self._bar_stiffness = np.asarray(k, dtype=np.float32)

                self.logger.debug(f"Initialized bar cache. Count: {len(self._bar_rest_lengths)}")

            # --- Hinges ---
            hinges = self.geometry.get('hinges')
            if hinges:
                if 'angle' in hinges:
                    self._hinge_rest_angles = np.asarray(hinges['angle'], dtype=np.float32)
                elif self.geometry.get('nodes') is not None:
                    self.logger.info("Computing hinge rest angles from initial geometry.")
                    self._hinge_rest_angles = self._compute_hinge_angles(
                        self.geometry['nodes'],
                        hinges['indices']
                    )

                if self._hinge_rest_angles is not None:
                    # Pre-expand scalar stiffness
                    k = hinges.get('stiffness', 1.0)
                    if np.ndim(k) == 0:
                        self._hinge_stiffness = np.full(len(self._hinge_rest_angles), k, dtype=np.float32)
                    else:
                        self._hinge_stiffness = np.asarray(k, dtype=np.float32)

                    self.logger.debug(f"Initialized hinge cache. Count: {len(self._hinge_rest_angles)}")

        except Exception as e:
            self.logger.error(f"Error in _init_caches: {e}", exc_info=True)
            raise

    def _init_jax_functions(self):
        """Compile JAX functions for analytics."""
        if not self.use_jax:
            return

        @jit
        def jax_bar_quantities(pos, indices, l0, l_eff, k):
            p0 = pos[indices[:, 0]]
            p1 = pos[indices[:, 1]]

            dist = jnp.linalg.norm(p1 - p0, axis=1)
            strain = jnp.where(l0 > 0, (dist - l0) / l0, 0.0)
            stress = k * strain
            pe = 0.5 * k * (dist - l_eff) ** 2

            return strain, stress, pe

        self._jax_bar_quantities = jax_bar_quantities

        @jit
        def jax_hinge_quantities(pos, indices, phi0, k):
            xj = pos[indices[:, 0]]
            xk = pos[indices[:, 1]]
            xi = pos[indices[:, 2]]
            xl = pos[indices[:, 3]]

            r_ij = xi - xj
            r_kj = xk - xj
            r_kl = xk - xl

            m = jnp.cross(r_ij, r_kj)
            n = jnp.cross(r_kj, r_kl)

            len_m = jnp.linalg.norm(m, axis=1) + 1e-12
            len_n = jnp.linalg.norm(n, axis=1) + 1e-12

            m_hat = m / len_m[:, None]
            n_hat = n / len_n[:, None]

            cos_phi = jnp.clip(jnp.sum(m_hat * n_hat, axis=1), -1.0, 1.0)
            phi = jnp.arccos(cos_phi)

            sign_check = jnp.sum(m * r_kl, axis=1)
            phi = jnp.where(sign_check < 0, -phi, phi)

            delta = phi - phi0
            delta = jnp.where(delta > jnp.pi, delta - 2 * jnp.pi, delta)
            delta = jnp.where(delta < -jnp.pi, delta + 2 * jnp.pi, delta)

            pe = 0.5 * k * delta ** 2
            t_strain = jnp.where(jnp.abs(phi0) > 1e-6, delta / phi0, delta)

            return phi, t_strain, pe

        self._jax_hinge_quantities = jax_hinge_quantities

    def _get_masses(self, n_nodes: int) -> np.ndarray:
        """Return cached masses or unit-mass fallback."""
        if self._masses is not None:
            return self._masses
        return np.ones(n_nodes, dtype=np.float32)

    def compute_frame(self, positions: np.ndarray, velocities: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute all derived data for a single frame."""
        state = {}

        try:
            masses = self._get_masses(len(positions))

            # ----------------------------------------------------------------
            # 1. Per-node energies
            # ----------------------------------------------------------------
            ke = self._compute_kinetic_energy(velocities, masses)
            pe_grav = self._compute_gravitational_pe(positions, masses)

            state['time_series/nodes/kinetic_energy'] = ke
            state['time_series/nodes/potential_energy'] = pe_grav

            # ----------------------------------------------------------------
            # 2. Bar analytics
            # ----------------------------------------------------------------
            if self.geometry.get('bars') and self._bar_rest_lengths is not None:
                if self.use_jax:
                    strain, stress, bar_pe = self._jax_bar_quantities(
                        positions,
                        self.geometry['bars']['indices'],
                        self._bar_rest_lengths,
                        self._bar_rest_lengths_effective,
                        self._bar_stiffness,
                    )
                    # Single conversion from JAX device arrays
                    strain, stress, bar_pe = (
                        np.asarray(strain),
                        np.asarray(stress),
                        np.asarray(bar_pe),
                    )
                else:
                    strain, stress, bar_pe = self._compute_bar_quantities(positions)

                state['time_series/elements/bars/strain'] = strain
                state['time_series/elements/bars/stress'] = stress
                state['time_series/elements/bars/potential_energy'] = bar_pe

            # ----------------------------------------------------------------
            # 3. Hinge analytics
            # ----------------------------------------------------------------
            if self.geometry.get('hinges') and self._hinge_rest_angles is not None:
                if self.use_jax:
                    angle, t_strain, hinge_pe = self._jax_hinge_quantities(
                        positions,
                        self.geometry['hinges']['indices'],
                        self._hinge_rest_angles,
                        self._hinge_stiffness,
                    )
                    angle, t_strain, hinge_pe = (
                        np.asarray(angle),
                        np.asarray(t_strain),
                        np.asarray(hinge_pe),
                    )
                else:
                    angle, t_strain, hinge_pe = self._compute_hinge_quantities(positions)

                state['time_series/elements/hinges/angle'] = angle
                state['time_series/elements/hinges/torsional_strain'] = t_strain
                state['time_series/elements/hinges/potential_energy'] = hinge_pe

            # ----------------------------------------------------------------
            # 4. Damping power (stored as internal key, consumed by Engine)
            #    NOT prefixed with 'time_series/' — Engine pops this before writing.
            # ----------------------------------------------------------------
            if self._damping_coefficient > 1e-9:
                state['_damping_power'] = self._compute_damping_power(velocities, masses)
            else:
                state['_damping_power'] = 0.0

            # NOTE: System-level totals (system/kinetic_energy, system/potential_energy,
            # system/total_energy) are NOT computed here. They are computed by the Engine
            # in a final pass by summing the per-node/per-element datasets written to HDF5.
            # This avoids scalar-vs-array shape bugs in the batch writer.

        except Exception as e:
            self.logger.error(f"Error computing frame state: {e}", exc_info=True)
            raise

        return state

    # ------------------------------------------------------------------
    # Read-time utilities (bypass any writer bugs)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_system_energies(plotter) -> Dict[str, np.ndarray]:
        """
        Compute system-level energies from per-node/per-element datasets.

        Usage:
            energies = PostProcessor.compute_system_energies(plotter)
            plt.plot(energies['time'], energies['kinetic_energy'], label='KE')
            plt.plot(energies['time'], energies['potential_energy'], label='PE')
            plt.plot(energies['time'], energies['total_energy'], label='KE + PE')
            plt.plot(energies['time'], energies['conserved_energy'], label='Conserved (should be const)')

        Args:
            plotter: Object with get_dataset(key) -> (data, attrs) method.

        Returns:
            Dict with 'time', 'kinetic_energy', 'potential_energy',
            'total_energy', and 'conserved_energy' as 1-D arrays.
        """
        time, _ = plotter.get_dataset("time")

        # Per-node KE: shape (n_frames, n_nodes)
        kes, _ = plotter.get_dataset("nodes/kinetic_energy")
        kinetic_energy = np.sum(kes, axis=1)

        # Gravitational PE from nodes
        node_pe, _ = plotter.get_dataset("nodes/potential_energy")
        potential_energy = np.sum(node_pe, axis=1)

        # Bar elastic PE (if present)
        try:
            bar_pe, _ = plotter.get_dataset("elements/bars/potential_energy")
            potential_energy += np.sum(bar_pe, axis=1)
        except (KeyError, Exception):
            pass

        # Hinge torsional PE (if present)
        try:
            hinge_pe, _ = plotter.get_dataset("elements/hinges/potential_energy")
            potential_energy += np.sum(hinge_pe, axis=1)
        except (KeyError, Exception):
            pass

        total_energy = kinetic_energy + potential_energy

        # Conserved = KE + PE + cumulative damping loss (should be constant)
        try:
            damping_loss, _ = plotter.get_dataset("system/damping_loss")
            conserved = total_energy + damping_loss.flatten()
        except (KeyError, Exception):
            conserved = total_energy.copy()

        return {
            'time': time.flatten(),
            'kinetic_energy': kinetic_energy.flatten(),
            'potential_energy': potential_energy.flatten(),
            'total_energy': total_energy.flatten(),
            'conserved_energy': conserved.flatten(),
        }

    # ------------------------------------------------------------------
    # Energy helpers (split from the old combined function)
    # ------------------------------------------------------------------

    def _compute_kinetic_energy(self, vel: np.ndarray, masses: np.ndarray) -> np.ndarray:
        """KE = 0.5 * m * |v|^2  per node."""
        v2 = np.einsum('ij,ij->i', vel, vel)  # row-wise dot product, avoids temp array
        return (0.5 * masses * v2).astype(np.float32)

    def _compute_gravitational_pe(self, pos: np.ndarray, masses: np.ndarray) -> np.ndarray:
        """
        PE = -m * (g · r)  per node.

        This is the standard definition where g is the gravity VECTOR
        (e.g. [0, 0, -9.81] for downward gravity).

        Consistency check: F = -∇PE = -∇(-m * g·r) = m*g
        So if the kernel does F = mass * gravity_vec, energy is conserved.
        """
        if self._gravity_mag < 1e-9:
            return np.zeros(len(pos), dtype=np.float32)

        pe = -masses * (pos @ self._gravity_vec)
        return pe.astype(np.float32)

    def _compute_damping_power(self, vel: np.ndarray, masses: np.ndarray) -> float:
        """
        Instantaneous power dissipated by global viscous damping.
        P = c * sum(m_i * |v_i|^2)
        """
        v2 = np.einsum('ij,ij->i', vel, vel)
        return float(self._damping_coefficient * np.dot(masses, v2))

    # ------------------------------------------------------------------
    # Bar / Hinge NumPy fallbacks
    # ------------------------------------------------------------------

    def _compute_bar_quantities(self, pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            indices = self.geometry['bars']['indices']
            diff = pos[indices[:, 1]] - pos[indices[:, 0]]
            dist = np.linalg.norm(diff, axis=1)

            l0 = self._bar_rest_lengths
            l_eff = self._bar_rest_lengths_effective
            k = self._bar_stiffness

            # Engineering strain
            with np.errstate(divide='ignore', invalid='ignore'):
                strain = np.where(l0 > 0, (dist - l0) / l0, 0.0)

            stress = k * strain
            pe = 0.5 * k * (dist - l_eff) ** 2

            return strain.astype(np.float32), stress.astype(np.float32), pe.astype(np.float32)
        except Exception as e:
            self.logger.error(f"Error in bar quantity calculation: {e}", exc_info=True)
            raise

    def _compute_hinge_quantities(self, pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            indices = self.geometry['hinges']['indices']
            angles = self._compute_hinge_angles(pos, indices)

            angle0 = self._hinge_rest_angles
            k = self._hinge_stiffness

            # Angular deviation wrapped to [-pi, pi]
            delta = (angles - angle0 + np.pi) % (2 * np.pi) - np.pi

            pe = 0.5 * k * delta ** 2
            t_strain = np.where(np.abs(angle0) > 1e-6, delta / angle0, delta)

            return angles, t_strain.astype(np.float32), pe.astype(np.float32)
        except Exception as e:
            self.logger.error(f"Error in hinge quantity calculation: {e}", exc_info=True)
            raise

    def _compute_hinge_angles(self, pos: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """
        Vectorized dihedral angles.
        Index columns: [j, k, i, l]  (spine j-k, faces j-k-i and j-k-l)
        """
        try:
            j, k, i, l = indices.T

            xj, xk, xi, xl = pos[j], pos[k], pos[i], pos[l]

            r_ij = xi - xj
            r_kj = xk - xj
            r_kl = xk - xl

            m = np.cross(r_ij, r_kj)
            n = np.cross(r_kj, r_kl)

            len_m = np.linalg.norm(m, axis=1)
            len_n = np.linalg.norm(n, axis=1)

            valid = (len_m > 1e-10) & (len_n > 1e-10)
            angles = np.zeros(len(indices), dtype=np.float32)

            if not np.any(valid):
                return angles

            denom = len_m[valid] * len_n[valid]
            cos_phi = np.clip(np.einsum('ij,ij->i', m[valid], n[valid]) / denom, -1.0, 1.0)
            phi = np.arccos(cos_phi)

            # Sign from orientation
            sign_neg = np.einsum('ij,ij->i', m[valid], r_kl[valid]) < 0
            phi[sign_neg] = -phi[sign_neg]

            angles[valid] = phi
            return angles
        except Exception as e:
            self.logger.error(f"Error in hinge angle calculation: {e}", exc_info=True)
            raise

    def compute_actuation_signal(self, time_array: np.ndarray, raw_signal_data: np.ndarray, signal_dt: float, actuator_type: str = 'force', node_idx: int = None) -> np.ndarray:
        """
        Interpolates a high-resolution signal to align with the simulation time steps.
        If actuator_type is 'position', it computes the displacement from the rest position.
        """
        from scipy.interpolate import interp1d

        try:
            u_high_res = raw_signal_data.astype(np.float32)
            t_high_res = np.arange(u_high_res.shape[0]) * signal_dt

            # Create an interpolator
            interpolator = interp1d(t_high_res, u_high_res, axis=0, kind='linear', fill_value="extrapolate")
            
            # Align to Simulation Time
            u_input = interpolator(time_array)

            if actuator_type == 'position':
                if node_idx is None:
                    self.logger.warning("Actuator type is 'position' but no node_idx was provided. Cannot subtract rest position.")
                elif self.geometry.get('nodes') is None or node_idx >= len(self.geometry['nodes']):
                    self.logger.warning(f"Node index {node_idx} is out of bounds. Cannot subtract rest position.")
                else:
                    rest_position = self.geometry['nodes'][node_idx]
                    u_input -= rest_position

            return u_input.astype(np.float32)
        except Exception as e:
            self.logger.error(f"Error computing actuation signal: {e}", exc_info=True)
            # Return a zero array of the correct shape on failure
            if raw_signal_data.ndim > 1:
                shape = (len(time_array), raw_signal_data.shape[1])
            else:
                shape = len(time_array)
            return np.zeros(shape, dtype=np.float32)