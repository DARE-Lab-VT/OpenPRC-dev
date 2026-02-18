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

    def get_feature_info(self, state_loader):
        """
        Returns a list of dictionaries, where each dictionary describes a feature.
        This default implementation returns None, indicating that detailed info
        is not available for this feature type.
        """
        return None