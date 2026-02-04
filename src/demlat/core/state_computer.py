"""
State Computer
==============
Post-processing module to calculate derived physics quantities.
"""

import numpy as np
from typing import Dict, Any, Tuple
from ..utils.logging import get_logger

logger = get_logger("demlat.state_computer")


class StateComputer:
    """
    Computes derived quantities (Energy, Stress, Strain) from raw simulation state.
    """

    def __init__(self, geometry: Dict[str, Any], material: Dict[str, Any]):
        self.logger = get_logger("demlat.state_computer")
        self.geometry = geometry
        self.material = material
        self._bar_rest_lengths = None
        self._bar_rest_lengths_effective = None
        self._hinge_rest_angles = None

        try:
            self._init_caches()
        except Exception as e:
            self.logger.error(f"Failed to initialize StateComputer caches: {e}", exc_info=True)
            raise

    def _init_caches(self):
        """Pre-compute static quantities like rest lengths."""
        try:
            # --- Bars ---
            bars = self.geometry.get('bars')
            if bars and 'rest_length' in bars:
                self._bar_rest_lengths = np.asarray(bars['rest_length'], dtype=np.float32)

                # Handle Prestress for PE calculation
                # If prestress is missing, effective length is just rest length
                prestress = bars.get('prestress')
                if prestress is not None:
                    prestress = np.asarray(prestress, dtype=np.float32)
                    self._bar_rest_lengths_effective = self._bar_rest_lengths * (1.0 + prestress)
                else:
                    self._bar_rest_lengths_effective = self._bar_rest_lengths

                self.logger.debug(f"Initialized bar cache. Count: {len(self._bar_rest_lengths)}")

            # --- Hinges ---
            hinges = self.geometry.get('hinges')
            if hinges:
                # Priority 1: Explicit 'angle' (matches geometry file structure)
                if 'angle' in hinges:
                    self._hinge_rest_angles = np.asarray(hinges['angle'], dtype=np.float32)
                # Priority 2: 'phi0' or 'rest_angle' (legacy/alternative names)
                elif 'phi0' in hinges:
                    self._hinge_rest_angles = np.asarray(hinges['phi0'], dtype=np.float32)
                elif 'rest_angle' in hinges:
                    self._hinge_rest_angles = np.asarray(hinges['rest_angle'], dtype=np.float32)
                elif self.geometry.get('nodes') is not None:
                    # Priority 3: Compute from initial geometry
                    self.logger.info("Computing hinge rest angles from initial geometry.")
                    self._hinge_rest_angles = self._compute_hinge_angles(
                        self.geometry['nodes'],
                        hinges['indices']
                    )
                self.logger.debug(f"Initialized hinge cache. Count: {len(self._hinge_rest_angles)}")
        except Exception as e:
            self.logger.error(f"Error in _init_caches: {e}", exc_info=True)
            raise

    def compute_frame(self, positions: np.ndarray, velocities: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute all derived data for a single frame."""
        state = {}

        try:
            # 1. Node Energies
            ke, pe = self._compute_node_energies(positions, velocities)
            state['time_series/nodes/kinetic_energy'] = ke
            state['time_series/nodes/potential_energy'] = pe

            # 2. Bar Analytics
            if self.geometry.get('bars') and self._bar_rest_lengths is not None:
                strain, stress, bar_pe, lengths = self._compute_bar_quantities(positions)
                state['time_series/elements/bars/strain'] = strain
                state['time_series/elements/bars/stress'] = stress
                state['time_series/elements/bars/potential_energy'] = bar_pe
                state['time_series/elements/bars/lengths'] = lengths

            # 3. Hinge Analytics
            if self.geometry.get('hinges') and self._hinge_rest_angles is not None:
                angle, t_strain, hinge_pe = self._compute_hinge_quantities(positions)
                state['time_series/elements/hinges/angle'] = angle
                state['time_series/elements/hinges/torsional_strain'] = t_strain
                state['time_series/elements/hinges/potential_energy'] = hinge_pe

            # 4. System Totals
            state['time_series/system/kinetic_energy'] = np.array([np.sum(ke)], dtype=np.float32)

            total_pe = np.sum(pe)
            if 'time_series/elements/bars/potential_energy' in state:
                total_pe += np.sum(state['time_series/elements/bars/potential_energy'])
            if 'time_series/elements/hinges/potential_energy' in state:
                total_pe += np.sum(state['time_series/elements/hinges/potential_energy'])

            state['time_series/system/potential_energy'] = np.array([total_pe], dtype=np.float32)

            # [NEW] Calculate Instantaneous Damping Power (Watts)
            # We return this to the Engine, which will integrate it over time
            damping_power = self._compute_damping_power(velocities)
            state['system_damping_power'] = damping_power  # Temporary key for Engine

        except Exception as e:
            self.logger.error(f"Error computing frame state: {e}", exc_info=True)
            # Depending on severity, we might want to re-raise or return partial state
            # For scientific data, partial state is dangerous, so we raise.
            raise

        return state

    def _compute_damping_power(self, vel: np.ndarray) -> float:
        """
        Estimate energy dissipation rate (Power).
        P = Force_damping * Velocity
        This is a simplified model for global damping.
        """
        try:
            power = 0.0

            # 1. Global Viscous Damping: F = -c * v  => P = c * v^2
            # Use 'damping_coefficient' from material or default to 0.0
            c_global = self.material.get('damping_coefficient', 0.0)
            if c_global > 1e-9:  # Only compute if damping is non-zero
                masses = self.geometry.get('masses')
                if masses is None:
                    masses = np.ones(len(vel), dtype=np.float32)

                power += c_global * np.sum(masses * np.sum(vel ** 2, axis=1))

            # 2. Bar Damping (Internal friction)
            # This is harder to do perfectly without bar connectivity, but we can approximate
            # if we assume bar damping is related to global motion for now.
            # For a strict calculation, we would need the relative velocity of every bar.

            return float(power)
        except Exception as e:
            self.logger.warning(f"Failed to compute damping power: {e}", exc_info=True)
            return 0.0

    # --- Math Implementations ---

    def _compute_node_energies(self, pos: np.ndarray, vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            masses = self.geometry.get('masses')
            if masses is None:
                masses = np.ones(len(pos), dtype=np.float32)

            # KE = 0.5 * m * v^2
            v2 = np.sum(vel ** 2, axis=1)
            ke = 0.5 * masses * v2

            # PE = m * g * h
            # Default to Z-axis gravity if not specified, matching Model's default
            grav_param = self.material.get('gravity', -9.81)

            if np.isscalar(grav_param):
                # Scalar gravity assumes Z-axis alignment (standard in this engine)
                g_vec = np.array([0.0, 0.0, float(grav_param)], dtype=np.float32)
            else:
                g_vec = np.asarray(grav_param, dtype=np.float32)

            g_mag = np.linalg.norm(g_vec)
            if g_mag > 1e-9:
                # Height is projection onto gravity vector (reversed)
                # PE = m * g * h
                # If g is [0, 0, -9.81], then -g is [0, 0, 9.81].
                # h = dot(pos, [0,0,1]). PE = m * 9.81 * z. Correct.
                h = np.dot(pos, -g_vec / g_mag)
                pe = masses * g_mag * h
            else:
                pe = np.zeros(len(pos), dtype=np.float32)

            return ke.astype(np.float32), pe.astype(np.float32)
        except Exception as e:
            self.logger.error(f"Error in node energy calculation: {e}", exc_info=True)
            raise

    def _compute_bar_quantities(self, pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        try:
            indices = self.geometry['bars']['indices']
            p0 = pos[indices[:, 0]]
            p1 = pos[indices[:, 1]]

            dist = np.linalg.norm(p1 - p0, axis=1)
            l0 = self._bar_rest_lengths
            l_eff = self._bar_rest_lengths_effective

            # Strain: (L_curr - L_rest) / L_rest
            # This matches standard engineering strain definition
            with np.errstate(divide='ignore', invalid='ignore'):
                strain = (dist - l0) / l0
                strain[l0 == 0] = 0.0

            # Stress & Energy
            k = self.geometry['bars'].get('stiffness', 1000.0)
            if np.ndim(k) == 0: k = np.full(len(dist), k)

            stress = k * strain

            # Potential Energy: 0.5 * k * (L_curr - L_effective)^2
            # Uses effective length (including prestress) to match physics model
            pe = 0.5 * k * (dist - l_eff) ** 2

            return strain.astype(np.float32), stress.astype(np.float32), pe.astype(np.float32), dist.astype(np.float32)
        except Exception as e:
            self.logger.error(f"Error in bar quantity calculation: {e}", exc_info=True)
            raise

    def _compute_hinge_quantities(self, pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            indices = self.geometry['hinges']['indices']
            angles = self._compute_hinge_angles(pos, indices)

            angle0 = self._hinge_rest_angles
            delta = angles - angle0

            # Wrap to [-pi, pi]
            delta = (delta + np.pi) % (2 * np.pi) - np.pi

            k = self.geometry['hinges'].get('stiffness', 1.0)
            if np.ndim(k) == 0: k = np.full(len(delta), k)

            pe = 0.5 * k * delta ** 2

            return angles, delta.astype(np.float32), pe.astype(np.float32)
        except Exception as e:
            self.logger.error(f"Error in hinge quantity calculation: {e}", exc_info=True)
            raise

    def _compute_hinge_angles(self, pos: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """
        Compute dihedral angles for N hinges vectorized.
        Indices structure: [j, k, i, l]
        - Spine: j-k
        - Face 1: j-k-i
        - Face 2: j-k-l
        """
        try:
            j, k, i, l = indices.T  # Unpack columns

            xj, xk, xi, xl = pos[j], pos[k], pos[i], pos[l]

            r_ij = xi - xj
            r_kj = xk - xj
            r_kl = xk - xl

            # Normals
            m = np.cross(r_ij, r_kj)
            n = np.cross(r_kj, r_kl)

            len_m = np.linalg.norm(m, axis=1)
            len_n = np.linalg.norm(n, axis=1)

            # Avoid div by zero
            valid = (len_m > 1e-10) & (len_n > 1e-10)
            angles = np.zeros(len(indices), dtype=np.float32)

            if not np.any(valid):
                return angles

            # Cosine rule
            dot_mn = np.einsum('ij,ij->i', m[valid], n[valid])
            cos_phi = np.clip(dot_mn / (len_m[valid] * len_n[valid]), -1.0, 1.0)
            phi = np.arccos(cos_phi)

            # Sign check (Orientation)
            # dot(m, r_kl) < 0 means convex vs concave
            dot_m_rkl = np.einsum('ij,ij->i', m[valid], r_kl[valid])
            mask_neg = dot_m_rkl < 0
            phi[mask_neg] = -phi[mask_neg]

            angles[valid] = phi
            return angles.astype(np.float32)
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