from dataclasses import dataclass


@dataclass
class KPCAConfig:
    """Configuration for KPCA dimensionality reduction.

    Args:
        n_components: If None, no KPCA is applied. If float between 0 and 1,
            represents the percentage of components to keep. If int greater than 1,
            represents the total number of components to keep.
        random_state: Random seed for reproducibility. Defaults to 42.
    """

    n_components: int | float | None
    kernel: str = "linear"
    random_state: int = 42
