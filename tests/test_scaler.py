"""
Unit tests for RobustStandardScaler.
"""

import os
import tempfile
import numpy as np
import pytest

from src.preprocessing.scaler import RobustStandardScaler


def test_scaler_normal_data():
    """Test standard scaling on standard data without constants."""
    np.random.seed(42)
    X = np.random.normal(loc=10.0, scale=2.0, size=(1000, 5)).astype(np.float32)

    scaler = RobustStandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Scaled mean should be ~0, std should be ~1
    np.testing.assert_allclose(np.mean(X_scaled, axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(np.std(X_scaled, axis=0), 1.0, atol=1e-5)

    # Inverse transform should restore original data
    X_restored = scaler.inverse_transform(X_scaled)
    np.testing.assert_allclose(X_restored, X, atol=1e-5)


def test_scaler_constant_sensor():
    """Test scaling when a sensor has zero variance (constant reading)."""
    np.random.seed(42)
    X = np.random.normal(loc=5.0, scale=1.0, size=(500, 3)).astype(np.float32)
    # Channel 1 is constant 2.0 (e.g. backup pump OFF)
    X[:, 1] = 2.0

    scaler = RobustStandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Must NOT produce NaNs or Infs
    assert not np.isnan(X_scaled).any()
    assert not np.isinf(X_scaled).any()

    # Constant channel scaled value should be exactly 0.0 (x - mean = 2.0 - 2.0 = 0.0)
    np.testing.assert_allclose(X_scaled[:, 1], 0.0, atol=1e-6)

    # Inverse transform must recover 2.0 exactly
    X_restored = scaler.inverse_transform(X_scaled)
    np.testing.assert_allclose(X_restored[:, 1], 2.0, atol=1e-6)


def test_scaler_json_persistence():
    """Test JSON serialization and deserialization."""
    np.random.seed(42)
    X = np.random.normal(loc=0.0, scale=3.0, size=(100, 4)).astype(np.float32)
    X[:, 2] = 0.0  # Constant channel

    scaler = RobustStandardScaler()
    scaler.fit(X)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        scaler.save_json(tmp_path)
        loaded = RobustStandardScaler.load_json(tmp_path)

        np.testing.assert_allclose(loaded.mean_, scaler.mean_, atol=1e-6)
        np.testing.assert_allclose(loaded.std_, scaler.std_, atol=1e-6)
        np.testing.assert_array_equal(loaded.constant_mask_, scaler.constant_mask_)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
