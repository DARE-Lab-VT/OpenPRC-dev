from abc import ABC, abstractmethod

class BaseReadout(ABC):
    """Abstract interface for all learning models."""
    @abstractmethod
    def fit(self, X, y):
        """Trains the model weights."""
        pass

    @abstractmethod
    def predict(self, X):
        """Generates predictions given input features."""
        pass