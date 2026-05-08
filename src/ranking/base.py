"""Abstract base class for ranking models."""
from abc import ABC, abstractmethod
import numpy as np


class BaseRanker(ABC):
    """Base class for ranking models."""

    @abstractmethod
    def train(self, X_train, y_train, X_val, y_val, **kwargs):
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Return prediction scores."""
        pass
