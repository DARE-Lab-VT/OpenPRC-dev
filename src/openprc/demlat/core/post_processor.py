"""
State Computer
==============
Post-processing module to calculate derived physics quantities.
"""

import numpy as np
from typing import Dict, Any, Tuple
from ..utils.logging import get_logger

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
        """Pre-compute static quantities like rest lengths."""
        try:
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

                self.logger.debug(f"Initialized bar cache. Count: {len(self._bar_rest_lengths)}")

            # --- Hinges ---
            hinges = self.geometry.get('hinges')
            if hinges:
                # Priority 1: Explicit 'angle'
                if 'angle' in hinges:
                    self._hinge_rest_angles = np.asarray(hinges['angle'], dtype=np.float32)
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

    def _init_jax_functions(self):
        """Compile JAX functions for analytics."""
        if not self.use_jax: return

        # --- JAX: Bar Quantities ---
        @jit
        def jax_bar_quantities(pos, indices, l0, l_eff, k):
            i = indices[:, 0]
            j = indices[:, 1]
            
            p0 = pos[i]
            p1 = pos[j]
            
            dist = jnp.linalg.norm(p1 - p0, axis=1)
            
            # Strain
            strain = jnp.where(l0 > 0, (dist - l0) / l0, 0.0)
            
            # Stress
            stress = k * strain
            
            # Potential Energy
            pe = 0.5 * k * (dist - l_eff)**2
            
            return strain, stress, pe
        
        self._jax_bar_quantities = jax_bar_quantities

        # --- JAX: Hinge Quantities ---
        @jit
        def jax_hinge_quantities(pos, indices, phi0, k):
            j_idx = indices[:, 0]
            k_idx = indices[:, 1]
            i_idx = indices[:, 2]
            l_idx = indices[:, 3]

            xj = pos[j_idx]
            xk = pos[k_idx]
            xi = pos[i_idx]
            xl = pos[l_idx]

            r_ij = xi - xj
            r_kj = xk - xj
            r_kl = xk - xl

            m = jnp.cross(r_ij, r_kj)
            n = jnp.cross(r_kj, r_kl)

            len_m = jnp.linalg.norm(m, axis=1) + 1e-12
            len_n = jnp.linalg.norm(n, axis=1) + 1e-12
            
            m_hat = m / len_m[:, None]
            n_hat = n / len_n[:, None]

            cos_phi = jnp.sum(m_hat * n_hat, axis=1)
            cos_phi = jnp.clip(cos_phi, -1.0, 1.0)
            phi = jnp.arccos(cos_phi)

            # Sign check
            sign_check = jnp.sum(m * r_kl, axis=1)
            phi = jnp.where(sign_check < 0, -phi, phi)

            # Delta
            delta = phi - phi0
            delta = jnp.where(delta > jnp.pi, delta - 2*jnp.pi, delta)
            delta = jnp.where(delta < -jnp.pi, delta + 2*jnp.pi, delta)

            # Energy
            pe = 0.5 * k * delta**2
            
            # Torsional Strain (normalized deviation)
            # Avoid div by zero if phi0 is 0 (unlikely for folds but possible)
            t_strain = jnp.where(jnp.abs(phi0) > 1e-6, delta / phi0, delta)

            return phi, t_strain, pe

        self._jax_hinge_quantities = jax_hinge_quantities

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
                if self.use_jax:
                    # JAX Path
                    indices = self.geometry['bars']['indices']
                    k = self.geometry['bars'].get('stiffness', 1000.0)
                    if np.ndim(k) == 0: k = np.full(len(indices), k)
                    
                    # Convert to JAX arrays if needed (though JAX handles numpy inputs)
                    strain, stress, bar_pe = self._jax_bar_quantities(
                        positions, indices, self._bar_rest_lengths, self._bar_rest_lengths_effective, k
                    )
                    # Convert back to numpy
                    strain = np.array(strain)
                    stress = np.array(stress)
                    bar_pe = np.array(bar_pe)
                else:
                    # NumPy Path
                    strain, stress, bar_pe = self._compute_bar_quantities(positions)
                
                state['time_series/elements/bars/strain'] = strain
                state['time_series/elements/bars/stress'] = stress
                state['time_series/elements/bars/potential_energy'] = bar_pe

            # 3. Hinge Analytics
            if self.geometry.get('hinges') and self._hinge_rest_angles is not None:
                if self.use_jax:
                    # JAX Path
                    indices = self.geometry['hinges']['indices']
                    k = self.geometry['hinges'].get('stiffness', 1.0)
                    if np.ndim(k) == 0: k = np.full(len(indices), k)
                    
                    angle, t_strain, hinge_pe = self._jax_hinge_quantities(
                        positions, indices, self._hinge_rest_angles, k
                    )
                    # Convert back to numpy
                    angle = np.array(angle)
                    t_strain = np.array(t_strain)
                    hinge_pe = np.array(hinge_pe)
                else:
                    # NumPy Path
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
            damping_power = self._compute_damping_power(velocities)
            state['system_damping_power'] = damping_power

        except Exception as e:
            self.logger.error(f"Error computing frame state: {e}", exc_info=True)
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

    # --- Math Implementations (NumPy Fallback) ---

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

    def _compute_bar_quantities(self, pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

            return strain.astype(np.float32), stress.astype(np.float32), pe.astype(np.float32)
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
