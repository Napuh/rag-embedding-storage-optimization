import numpy as np
from sklearn.random_projection import GaussianRandomProjection

from core.configs import RPConfig
from core.reduction.base import DimensionalityReducer


class RPReducer(DimensionalityReducer):
    def __init__(self, config: RPConfig):
        super().__init__()
        self.config = config
        self.reducer = None

    def fit(self, embeddings: np.ndarray) -> None:
        embeddings = embeddings.astype(np.float32)
        orig_dim = embeddings.shape[1]
        n_param = self.config.n_components
        if isinstance(n_param, float) and 0 < n_param <= 1:
            n_components = round(orig_dim * n_param)
        elif isinstance(n_param, int) and n_param > 0:
            n_components = n_param
        elif n_param == 'auto':
            n_components = 'auto'
        else:
            raise ValueError(f"Invalid n_components value for RP: {n_param}")
        self.reducer = GaussianRandomProjection(
            n_components=n_components,
            random_state=self.config.random_state,
        )
        self.reducer.fit(embeddings)

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        return self.reducer.transform(embeddings) 