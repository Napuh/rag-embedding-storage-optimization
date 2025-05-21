from dataclasses import dataclass


@dataclass
class RPConfig:
    """Configuration for Random Projection dimensionality reduction.

    Args:
        n_components: If 'auto', uses the Johnson-Lindenstrauss lemma based
                      calculation. If float between 0 and 1, represents the
                      percentage of components to keep. If int greater than 1,
                      represents the total number of components to keep.
        random_state: Random seed for reproducibility. Defaults to 42.
        # eps: float = 0.1 # Alternative to n_components based on JL lemma
    """
    n_components: int | float | str # Allow 'auto'
    random_state: int = 42
