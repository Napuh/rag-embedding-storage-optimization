from abc import ABC, abstractmethod

import numpy as np


class DimensionalityReducer(ABC):
    @abstractmethod
    def fit(self, embeddings: np.ndarray) -> None:
        """Fit the reducer to the provided embeddings."""
        pass

    @abstractmethod
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform embeddings to the reduced dimension."""
        pass 