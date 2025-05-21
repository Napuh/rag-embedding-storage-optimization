import numpy as np
from sklearn.decomposition import PCA

from core.configs import PCAConfig
from core.reduction.base import DimensionalityReducer


class PCAReducer(DimensionalityReducer):
    def __init__(self, config: PCAConfig):
        super().__init__()
        self.config = config
        self.reducer = None

    def fit(self, embeddings: np.ndarray) -> None:
        embeddings = embeddings.astype(np.float32)
        orig_dim = embeddings.shape[1]
        n_param = self.config.n_components
        if isinstance(n_param, float) and 0 < n_param <= 1:
            n_components = round(orig_dim * n_param)
        else:
            n_components = int(n_param)
        max_components = min(embeddings.shape[0], embeddings.shape[1])
        n_components = min(n_components, max_components)
        self.reducer = PCA(n_components=n_components, random_state=self.config.random_state)
        self.reducer.fit(embeddings)

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        return self.reducer.transform(embeddings) 