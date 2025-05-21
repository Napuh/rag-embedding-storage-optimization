import numpy as np
from sklearn.decomposition import KernelPCA

from core.configs import KPCAConfig
from core.reduction.base import DimensionalityReducer


class KPCAReducer(DimensionalityReducer):
    def __init__(self, config: KPCAConfig):
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
            raise ValueError(f"Invalid n_components value for KPCA: {self.config.n_components}")
        max_components = min(embeddings.shape[0], embeddings.shape[1])
        n_components = min(n_components, max_components)
        self.reducer = KernelPCA(
            n_components=n_components,
            kernel=self.config.kernel,
            random_state=self.config.random_state,
            n_jobs=8,
        )
        self.reducer.fit(embeddings)

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        return self.reducer.transform(embeddings) 