from abc import ABC, abstractmethod

class BaseFeature(ABC):
    """Abstract interface for all feature extractors."""
    @abstractmethod
    def transform(self, state_loader):
        """
        Args:
            state_loader: Instance of reservoir.io.StateLoader
        Returns:
            np.ndarray: Feature matrix [Time, Features]
        """
        pass