import hashlib
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from core.configs import AEConfig
from core.reduction.base import DimensionalityReducer


def get_activation(name: str) -> nn.Module:
    """Returns the activation function based on its name."""
    if name == "relu":
        return nn.ReLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unsupported activation function: {name}")


class Autoencoder(nn.Module):
    """Autoencoder model for dimensionality reduction."""
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int], activation: str = "relu"):
        super().__init__()
        act_fn = get_activation(activation)

        # Encoder
        encoder_layers = []
        last_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(last_dim, h_dim))
            encoder_layers.append(act_fn)
            last_dim = h_dim
        encoder_layers.append(nn.Linear(last_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        last_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(last_dim, h_dim))
            decoder_layers.append(act_fn)
            last_dim = h_dim
        decoder_layers.append(nn.Linear(last_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)


def get_ae_save_path(model_name: str, benchmark: str, config: AEConfig, latent_dim: int) -> Path:
    """Generates a unique path for saving the autoencoder model."""
    config_str = json.dumps({
        "latent": latent_dim,
        "hidden": config.hidden_dims,
        "act": config.activation,
        "epochs": config.epochs,
        "lr": config.lr,
    }, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    safe_model_name = model_name.replace("/", "_")
    filename = f"ae_encoder_{latent_dim}d_{config_hash}.pth"
    return Path("saved_models") / safe_model_name / benchmark / filename


def train_autoencoder(
    embeddings: np.ndarray,
    config: AEConfig,
    input_dim: int,
    latent_dim: int,
    device: str = "cuda",
    save_path: Optional[Path] = None
) -> nn.Module:
    """Trains an autoencoder and returns the trained encoder module."""
    
    print(f"\tStarting Autoencoder training for {config.epochs} epochs...")
    print(f"\tInput Dim: {input_dim}, Latent Dim: {latent_dim}, Hidden Dims: {config.hidden_dims}")
    print(f"\tLR: {config.lr}, Batch Size: {config.batch_size}, Activation: {config.activation}")

    # Prepare data
    dataset = TensorDataset(torch.tensor(embeddings, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Initialize model, loss, optimizer
    model = Autoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=config.hidden_dims,
        activation=config.activation
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # Training loop
    model.train()
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}", leave=False)
        for batch in progress_bar:
            inputs = batch[0].to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"\tEpoch [{epoch+1}/{config.epochs}], Average Loss: {avg_epoch_loss:.6f}")

    print("\tAutoencoder training finished.")

    # Save the encoder state dict if path provided
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.encoder.state_dict(), save_path)
        print(f"\tTrained encoder saved to {save_path}")

    return model.encoder.eval()


class AEReducer(DimensionalityReducer):
    """Dimensionality reducer using an autoencoder."""
    
    def __init__(self, config: AEConfig, model_name: str, benchmark: str, device: str):
        super().__init__()
        self.config = config
        self.model_name = model_name
        self.benchmark = benchmark
        self.device = device
        self.encoder = None

    def fit(self, embeddings: np.ndarray) -> None:
        original_dim = embeddings.shape[1]
        ld = self.config.latent_dim
        if isinstance(ld, float) and 0 < ld <= 1:
            latent_dim = round(original_dim * ld)
        elif isinstance(ld, int) and ld > 0:
            latent_dim = ld
        else:
            raise ValueError(f"Invalid latent_dim value for AE: {ld}")
        if latent_dim <= 0 or latent_dim >= original_dim:
            raise ValueError(f"Calculated latent_dim is invalid: {latent_dim} (original: {original_dim})")
        
        save_path = get_ae_save_path(
            self.model_name,
            self.benchmark,
            self.config,
            latent_dim,
        )
        
        if save_path.exists():
            temp_ae = Autoencoder(
                input_dim=original_dim,
                latent_dim=latent_dim,
                hidden_dims=self.config.hidden_dims,
                activation=self.config.activation,
            )
            temp_ae.encoder.load_state_dict(
                torch.load(save_path, map_location=self.device)
            )
            self.encoder = temp_ae.encoder.to(self.device).eval()
        else:
            self.encoder = train_autoencoder(
                embeddings=embeddings,
                config=self.config,
                input_dim=original_dim,
                latent_dim=latent_dim,
                device=self.device,
                save_path=save_path,
            )

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        if not self.encoder:
            raise RuntimeError("AE encoder is not trained or loaded.")
        with torch.no_grad():
            tensors = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
            latent = self.encoder(tensors)
            return latent.cpu().numpy() 