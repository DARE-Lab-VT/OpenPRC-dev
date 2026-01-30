from abc import ABC, abstractmethod
import numpy as np

class BaseTask(ABC):
    """
    Abstract interface for all task definitions.
    A task is responsible for generating a target signal 'y' that the reservoir
    computer will be trained to reproduce.
    """
    @abstractmethod
    def generate(self, u_input: np.ndarray) -> np.ndarray:
        """
        Generates the target signal based on an input signal.

        Args:
            u_input (np.ndarray): The input signal that drives the task dynamics.
                                  This is typically the same signal that drives
                                  the physical reservoir. The shape is [time, 1].

        Returns:
            np.ndarray: The generated target signal 'y'. Must be a 2D array
                        of shape [time, output_dim]. For a standard single-output
                        task, this will be [time, 1].
        """
        pass