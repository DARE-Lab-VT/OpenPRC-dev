from abc import ABC, abstractmethod

class BaseTask(ABC):
    """Abstract interface for all task definitions."""
    @abstractmethod
    def generate(self, length):
        """
        Generates the target signal.
        Returns:
            np.ndarray: Target signal 'y' of shape [length, output_dim]
        """
        pass