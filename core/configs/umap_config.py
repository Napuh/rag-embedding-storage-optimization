from dataclasses import dataclass
from typing import Optional


@dataclass
class UMAPConfig:
    """Configuration for UMAP dimensionality reduction.

    Args:
        n_components: The dimension of the space to embed into.
                      If float between 0 and 1, treated as percentage of original dim.
        n_neighbors: Controls how UMAP balances local versus global structure.
                     Larger values focus more on global structure. Defaults to 15.
        min_dist: Controls how tightly UMAP is allowed to pack points together.
                  Lower values mean tighter clusters. Defaults to 0.1.
        metric: The distance metric to use. Defaults to 'cosine'.
        random_state: Random seed for reproducibility. Defaults to 42.
    """

    n_components: int | float
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "cosine"
    random_state: Optional[int] = None
