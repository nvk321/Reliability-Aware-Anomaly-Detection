"""
Unit tests for SensorCorruptor.
"""

import numpy as np
import pytest

from src.corruption.corruptor import SensorCorruptor


def test_corruptor_modes():
    """Test all 9 corruption modes execute and produce valid masks."""
    corruptor = SensorCorruptor(seed=42)
    N, M = 200, 8
    clean_data = np.random.normal(0, 1, (N, M)).astype(np.float32)

    for c_type in SensorCorruptor.SUPPORTED_TYPES:
        corrupted, mask, meta = corruptor.apply_corruption(
            data=clean_data,
            corruption_type=c_type,
            severity=0.3,
            corruption_rate=0.1,
        )

        assert corrupted.shape == clean_data.shape
        assert mask.shape == clean_data.shape
        assert not np.isnan(corrupted).any()
        assert not np.isinf(corrupted).any()
        assert meta["corruption_type"] == c_type
        assert meta["random_seed"] == 42

        # Corrupted data should differ from clean data at mask positions
        if meta["total_corrupted_points"] > 0:
            assert not np.array_equal(corrupted, clean_data)


def test_corruptor_reproducibility():
    """Test deterministic reproduction with identical seeds."""
    data = np.random.normal(5, 2, (100, 6)).astype(np.float32)

    c1 = SensorCorruptor(seed=123)
    c2 = SensorCorruptor(seed=123)

    out1, mask1, _ = c1.apply_corruption(data, "gaussian_noise", severity=0.5)
    out2, mask2, _ = c2.apply_corruption(data, "gaussian_noise", severity=0.5)

    np.testing.assert_array_equal(out1, out2)
    np.testing.assert_array_equal(mask1, mask2)
