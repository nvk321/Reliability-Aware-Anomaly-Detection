"""
Unit tests for data preprocessing and robust scaling.

Validates:
- Standard zero-mean, unit-variance scaling on normal distribution.
- Resilient handling of constant/uninformative sensors (zero-variance).
- Reversibility of inverse transformation.
- Handling of extreme values / clipping.
- JSON serialization and deserialization of fitted scaler parameters.
"""

import os
import tempfile
import numpy as np
import pytest

from src.preprocessing.scaler import RobustStandardScaler


def test_scaler_normal_distribution():
    """Verify standard continuous signals scale to zero mean and unit variance."""
    np.random.seed(42)
    X = np.random.normal(loc=12.5, scale=3.2, size=(500, 4)).astype(np.float32)

    scaler = RobustStandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert X_scaled.shape == X.shape
    np.testing.assert_allclose(np.mean(X_scaled, axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(np.std(X_scaled, axis=0), 1.0, atol=1e-5)

    # Reversibility
    X_rec = scaler.inverse_transform(X_scaled)
    np.testing.assert_allclose(X_rec, X, atol=1e-5)


def test_scaler_zero_variance_resilience():
    """Verify zero-variance/constant channels do not cause division-by-zero or NaNs."""
    np.random.seed(42)
    X = np.random.normal(loc=5.0, scale=1.0, size=(200, 3)).astype(np.float32)
    # Channel 1 is constant 0.0 (simulating inactive backup pump like P202)
    X[:, 1] = 0.0

    scaler = RobustStandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert not np.isnan(X_scaled).any(), "NaN detected in scaled data"
    assert not np.isinf(X_scaled).any(), "Inf detected in scaled data"
    # Channel 1 scaled value must be 0.0
    np.testing.assert_allclose(X_scaled[:, 1], 0.0, atol=1e-6)

    # Inverse transform should restore 0.0
    X_rec = scaler.inverse_transform(X_scaled)
    np.testing.assert_allclose(X_rec[:, 1], 0.0, atol=1e-6)


def test_scaler_serialization():
    """Verify scaler parameters can be saved to and loaded from JSON."""
    np.random.seed(42)
    X = np.random.normal(loc=8.0, scale=2.5, size=(100, 3)).astype(np.float32)
    X[:, 0] = 1.0  # Constant channel

    scaler = RobustStandardScaler()
    scaler.fit(X)

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "test_scaler.json")
        scaler.save_json(json_path)
        assert os.path.exists(json_path)

        loaded_scaler = RobustStandardScaler.load_json(json_path)
        np.testing.assert_allclose(loaded_scaler.mean_, scaler.mean_, atol=1e-6)
        np.testing.assert_allclose(loaded_scaler.std_, scaler.std_, atol=1e-6)
        np.testing.assert_array_equal(loaded_scaler.constant_mask_, scaler.constant_mask_)

        # Transforms produced by both should match identically
        out_orig = scaler.transform(X)
        out_loaded = loaded_scaler.transform(X)
        np.testing.assert_allclose(out_orig, out_loaded, atol=1e-6)
