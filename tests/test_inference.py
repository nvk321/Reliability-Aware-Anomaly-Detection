"""
Unit tests for model inference across all three architectures.

Validates:
- Forward pass execution on batch dimensions (B, L, M).
- Output tensor shapes and types.
- Numerical stability (no NaNs, Infs, or vanishing activations).
- Single-window inference and diagnostic extraction in R-STNet.
"""

import numpy as np
import torch
import pytest

from src.models.rstnet import RSTNet
from src.models.baselines.rstnet_no_srm import RSTNetNoSRM
from src.models.baselines.reconstruction_ae import SpatioTemporalAE


def test_rstnet_forward_pass():
    """Verify Proposed R-STNet forward pass tensor shapes and bounds."""
    B, L, M = 4, 30, 51
    x = torch.randn(B, L, M, dtype=torch.float32)

    model = RSTNet(in_channels=M, hidden_dim=64, node_dim=32, top_k=15)
    model.eval()

    with torch.no_grad():
        x_pred, adj = model(x)

    assert x_pred.shape == (B, M), f"Expected shape ({B}, {M}), got {x_pred.shape}"
    assert adj.shape == (M, M), f"Expected adjacency shape ({M}, {M}), got {adj.shape}"
    assert not torch.isnan(x_pred).any(), "NaN found in R-STNet predictions"
    assert not torch.isinf(x_pred).any(), "Inf found in R-STNet predictions"
    # Adjacency weights should be normalized probabilities / non-negative
    assert torch.all(adj >= 0.0), "Negative adjacency weights detected"


def test_rstnet_no_srm_forward_pass():
    """Verify Baseline C (RSTNetNoSRM) forward pass tensor shapes and bounds."""
    B, L, M = 4, 30, 51
    x = torch.randn(B, L, M, dtype=torch.float32)

    model = RSTNetNoSRM(in_channels=M, hidden_dim=64, node_dim=32, top_k=15)
    model.eval()

    with torch.no_grad():
        x_pred, adj = model(x)

    assert x_pred.shape == (B, M)
    assert adj.shape == (M, M)
    assert not torch.isnan(x_pred).any()
    assert not torch.isinf(x_pred).any()


def test_stae_forward_pass():
    """Verify Baseline B (ST-AE) sequence reconstruction output shapes."""
    B, L, M = 4, 30, 51
    x = torch.randn(B, L, M, dtype=torch.float32)

    model = SpatioTemporalAE(in_channels=M, hidden_dim=64, latent_dim=32, num_layers=2)
    model.eval()

    with torch.no_grad():
        x_recon, latent = model(x)

    assert x_recon.shape == (B, L, M), f"Expected recon shape ({B}, {L}, {M}), got {x_recon.shape}"
    assert latent.shape == (B, 32), f"Expected latent shape ({B}, 32), got {latent.shape}"
    assert not torch.isnan(x_recon).any()
    assert not torch.isinf(x_recon).any()


def test_rstnet_evaluate_step_diagnostics():
    """Verify single-window step evaluation returns all required diagnostic fields."""
    L, M = 30, 51
    window = np.random.normal(0, 1, (L, M)).astype(np.float32)

    model = RSTNet(in_channels=M, hidden_dim=64, node_dim=32, top_k=15)
    # Calibrate dummy SRM statistics
    model.srm.fit_calibration(np.random.normal(0, 1, (100, M)).astype(np.float32))

    res = model.evaluate_step(window)

    assert "system_score" in res
    assert "sensor_fault_score" in res
    assert "most_suspicious_sensor" in res
    assert "reliability_vector" in res
    assert "normalized_residuals" in res

    assert isinstance(res["system_score"], float)
    assert res["system_score"] >= 0.0
    assert res["reliability_vector"].shape == (M,)
    assert 0 <= res["most_suspicious_sensor"] < M
