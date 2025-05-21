from dataclasses import dataclass, field
from typing import List


@dataclass
class AEConfig:
    """Configuration for Autoencoder dimensionality reduction.

    Args:
        latent_dim: The target dimension for the compressed representation.
                    If float (0-1), treated as percentage of original dim.
        hidden_dims: List of integers defining the size of hidden layers
                     in the encoder (decoder will be symmetric).
                     Example: [512, 256]
        activation: Activation function ('relu', 'tanh', etc.). Defaults to 'relu'.
        epochs: Number of training epochs. Defaults to 10.
        lr: Learning rate. Defaults to 1e-3.
        batch_size: Training batch size. Defaults to 64.
    """
    latent_dim: int | float
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    activation: str = "relu"
    epochs: int = 10
    lr: float = 1e-3
    batch_size: int = 64
