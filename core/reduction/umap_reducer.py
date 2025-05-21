import numpy as np
import umap

from core.configs import UMAPConfig
from core.reduction.base import DimensionalityReducer


class UMAPReducer(DimensionalityReducer):
    def __init__(self, config: UMAPConfig):
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
        else:
            raise ValueError(f"Invalid n_components value for UMAP: {self.config.n_components}")
        max_components = min(embeddings.shape[0] - 1, embeddings.shape[1])
        n_components = min(n_components, max_components)
        if n_components <= 0:
            raise ValueError(f"Calculated n_components for UMAP is not positive: {n_components}")
        self.reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=self.config.n_neighbors,
            min_dist=self.config.min_dist,
            metric=self.config.metric,
            random_state=self.config.random_state,
            n_jobs=8,
            transform_seed=self.config.random_state,
        )
        self.reducer.fit(embeddings)

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        return self.reducer.transform(embeddings) 