"""
Demlat Scaler
=============
Handles non-dimensionalization to improve numerical stability.

"""
import numpy as np
from openprc.schemas.logging import get_logger


class SimulationScaler:
    """
    Handles non-dimensionalization of physical quantities to improve numerical stability
    in simulations.

    This class calculates characteristic scales for mass, length, time, and other derived
    quantities based on the initial state of the system (masses, positions, stiffnesses).
    It provides methods to convert between physical units and dimensionless simulation units.
    """
    
    def __init__(self, masses, positions, stiffnesses):
        """
        Initialize the scaler by calculating characteristic scales.

        Args:
            masses (np.ndarray): Array of particle masses.
            positions (np.ndarray): Array of particle positions (N x D).
            stiffnesses (np.ndarray): Array of stiffness values.
        """
        self.logger = get_logger("demlat.scaler")
        # 1. Mass Scale (M*)
        # Mean of non-zero masses
        valid_m = masses[masses > 0]
        self.M_star = np.mean(valid_m) if len(valid_m) > 0 else 1.0

        # 2. Length Scale (L*)
        # Characteristic length of the domain
        if positions.shape[0] > 1:
            # Average distance from center is a robust metric
            center = np.mean(positions, axis=0)
            dists = np.linalg.norm(positions - center, axis=1)
            self.L_star = np.mean(dists) * 2.0
            if self.L_star < 1e-6: self.L_star = 1.0
        else:
            self.L_star = 1.0

        # 3. Stiffness Scale (K*)
        # Mean of positive stiffnesses
        valid_k = stiffnesses[stiffnesses > 0]
        self.K_star = np.mean(valid_k) if len(valid_k) > 0 else 1000.0

        # 4. Time Scale (T*)
        # Derived from natural frequency: T = sqrt(M/K)
        self.T_star = np.sqrt(self.M_star / self.K_star)

        # Derived scales
        self.V_star = self.L_star / self.T_star  # Velocity
        self.F_star = self.K_star * self.L_star  # Force
        self.Torque_star = self.F_star * self.L_star  # Torque (Force * Dist)
        self.Accel_star = self.L_star / (self.T_star ** 2)  # Acceleration (Gravity)

        self.logger.info(f"Characteristic Scales: L*={self.L_star:.2e}, M*={self.M_star:.2e}, T*={self.T_star:.2e}")

    def to_sim(self, val, type_str):
        """
        Convert a value from physical units to simulation (dimensionless) units.

        Args:
            val (float or np.ndarray): The value(s) in physical units.
            type_str (str): The type of physical quantity. Supported types:
                            'length', 'mass', 'time', 'stiffness', 'velocity',
                            'force', 'accel', 'torque_k', 'damping'.

        Returns:
            float or np.ndarray: The value(s) in simulation units.
        """
        if type_str == 'length': return val / self.L_star
        if type_str == 'mass': return val / self.M_star
        if type_str == 'time': return val / self.T_star
        if type_str == 'stiffness': return val / self.K_star
        if type_str == 'velocity': return val / self.V_star
        if type_str == 'force': return val / self.F_star
        if type_str == 'accel': return val / self.Accel_star
        if type_str == 'torque_k': return val / self.Torque_star
        if type_str == 'damping': return val / (self.M_star / self.T_star)
        return val

    def from_sim(self, val, type_str):
        """
        Convert a value from simulation (dimensionless) units to physical units.

        Args:
            val (float or np.ndarray): The value(s) in simulation units.
            type_str (str): The type of physical quantity. Supported types:
                            'length', 'velocity', 'time', 'force', 'accel',
                            'torque_k', 'damping', 'mass', 'stiffness',
                            'angular_velocity', 'angular_acceleration'.

        Returns:
            float or np.ndarray: The value(s) in physical units.
        """
        if type_str == 'length': return val * self.L_star
        if type_str == 'velocity': return val * self.V_star
        if type_str == 'time': return val * self.T_star
        if type_str == 'force': return val * self.F_star
        if type_str == 'accel': return val * self.Accel_star
        if type_str == 'torque_k': return val * self.Torque_star
        if type_str == 'damping': return val * (self.M_star / self.T_star)
        if type_str == 'mass': return val * self.M_star
        if type_str == 'stiffness': return val * self.K_star
        if type_str == 'angular_velocity': return val * (1 / self.T_star)
        if type_str == 'angular_acceleration': return val * (1 / (self.T_star ** 2))
        return val
