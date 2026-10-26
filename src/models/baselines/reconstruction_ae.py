"""
Baseline B: Spatio-Temporal Autoencoder (ST-AE) without Graph Modeling or SRM.

Standard deep reconstruction baseline for multivariate time-series anomaly detection.
Computes unweighted reconstruction error across all channels:
    S(t) = (1 / M) * sum_i (x_i(t) - x_hat_i(t))^2
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import numpy as np


class SpatioTemporalAE(nn.Module):
    """
    Encoder-Decoder Autoencoder for sequence reconstruction.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder: Bidirectional GRU
        self.encoder = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_latent = nn.Linear(hidden_dim * 2, latent_dim)

        # Decoder: GRU to reconstruct sequence
        self.decoder_gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_dim, in_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, in_channels)

        Returns:
            x_recon: Reconstructed tensor of shape (batch_size, seq_len, in_channels)
            latent: Latent representation (batch_size, latent_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Encode
        enc_out, hidden = self.encoder(x)
        # Latent from last time step
        latent = torch.relu(self.fc_latent(enc_out[:, -1, :]))

        # Repeat latent across sequence length for decoder input
        dec_in = latent.unsqueeze(1).repeat(1, seq_len, 1)
        dec_out, _ = self.decoder_gru(dec_in)
        x_recon = self.fc_out(dec_out)

        return x_recon, latent

    def compute_anomaly_score(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes unweighted per-sensor and system anomaly scores at latest time-step.

        Returns:
            per_sensor_error: (batch_size, in_channels) squared error at step t
            system_score: (batch_size,) unweighted mean error across channels
        """
        self.eval()
        with torch.no_grad():
            x_recon, _ = self.forward(x)
            # Evaluate at the current observation (last step of window)
            diff = x[:, -1, :] - x_recon[:, -1, :]
            per_sensor_error = diff ** 2
            system_score = torch.mean(per_sensor_error, dim=-1)
        return per_sensor_error, system_score
