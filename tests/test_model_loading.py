"""
Unit tests for model checkpoint integrity and weight loading.

Validates:
- Checkpoints exist at expected paths.
- PyTorch state_dict matches model architectures exactly without missing or unexpected keys.
- Calibrated decision thresholds tau are present and valid positive floats.
"""

import os
import torch
import pytest

from src.models.rstnet import RSTNet
from src.models.baselines.rstnet_no_srm import RSTNetNoSRM
from src.models.baselines.reconstruction_ae import SpatioTemporalAE


@pytest.fixture
def checkpoint_dir():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(base_dir, "results", "checkpoints")


def test_proposed_rstnet_checkpoint_loading(checkpoint_dir):
    """Verify Proposed R-STNet checkpoint loads into architecture."""
    ckpt_path = os.path.join(checkpoint_dir, "proposed_rstnet_swat.pt")
    assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt
    assert "threshold_tau" in ckpt

    tau = float(ckpt["threshold_tau"])
    assert tau > 0.0, f"Invalid threshold: {tau}"
    assert pytest.approx(tau, abs=1e-3) == 3.1401

    model = RSTNet(in_channels=51, hidden_dim=64, node_dim=32, top_k=15)
    load_res = model.load_state_dict(ckpt["model_state_dict"])
    assert len(load_res.missing_keys) == 0
    assert len(load_res.unexpected_keys) == 0


def test_baseline_rstnet_no_srm_checkpoint_loading(checkpoint_dir):
    """Verify Baseline C (RSTNetNoSRM) checkpoint loads into architecture."""
    ckpt_path = os.path.join(checkpoint_dir, "baseline_rstnet_no_srm_swat.pt")
    assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt
    assert "threshold_tau" in ckpt

    tau = float(ckpt["threshold_tau"])
    assert tau > 0.0, f"Invalid threshold: {tau}"
    assert pytest.approx(tau, abs=1e-3) == 12.0706

    model = RSTNetNoSRM(in_channels=51, hidden_dim=64, node_dim=32, top_k=15)
    load_res = model.load_state_dict(ckpt["model_state_dict"])
    assert len(load_res.missing_keys) == 0
    assert len(load_res.unexpected_keys) == 0


def test_baseline_stae_checkpoint_loading(checkpoint_dir):
    """Verify Baseline B (ST-AE) checkpoint loads into architecture."""
    ckpt_path = os.path.join(checkpoint_dir, "baseline_stae_swat.pt")
    assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt
    assert "threshold_tau" in ckpt

    tau = float(ckpt["threshold_tau"])
    assert tau > 0.0, f"Invalid threshold: {tau}"
    assert pytest.approx(tau, abs=1e-3) == 0.0892

    model = SpatioTemporalAE(in_channels=51, hidden_dim=64, latent_dim=32, num_layers=2)
    load_res = model.load_state_dict(ckpt["model_state_dict"])
    assert len(load_res.missing_keys) == 0
    assert len(load_res.unexpected_keys) == 0
