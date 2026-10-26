"""
Unit tests for SensorReliabilityModule (SRM).

Validates:
- Statistical perspective (z-score bounds, rate-of-change, variance checks).
- Output constraints: reliability scores r_i in [0.0, 1.0].
- Degradation penalties: stuck sensors, Gaussian noise, sensor drift.
- Empirical separation: r_clean >> r_corrupt.
- State dictionary export and restoration.
"""

import numpy as np
import pytest

from src.reliability.sensor_reliability import SensorReliabilityModule


def test_srm_clean_data_high_reliability():
    """Verify clean normal operational data achieves healthy reliability scores."""
    np.random.seed(42)
    N, M = 400, 6
    t = np.linspace(0, 20, N)
    clean_data = np.zeros((N, M), dtype=np.float32)
    for i in range(M):
        clean_data[:, i] = np.sin(t + i * 0.4) + np.random.normal(0, 0.05, N)

    srm = SensorReliabilityModule(num_sensors=M)
    srm.fit_calibration(clean_data)

    window = clean_data[100:130]
    r, comp = srm.estimate_reliability(window)

    assert r.shape == (M,)
    assert np.all(r >= 0.0) and np.all(r <= 1.0), "Reliability scores out of [0, 1] bounds"
    assert np.mean(r) >= 0.60, f"Clean data mean reliability too low: {np.mean(r)}"


def test_srm_stuck_sensor_penalization():
    """Verify that a sensor frozen at constant value is heavily penalized."""
    np.random.seed(42)
    N, M = 300, 5
    clean_data = np.random.normal(loc=0.0, scale=1.0, size=(N, M)).astype(np.float32)

    srm = SensorReliabilityModule(num_sensors=M)
    srm.fit_calibration(clean_data)

    # Test window with sensor 2 frozen
    test_window = np.random.normal(0, 1, (30, M)).astype(np.float32)
    test_window[:, 2] = 2.5  # Constant stuck value

    r, comp = srm.estimate_reliability(test_window)

    # Stuck channel should have lowest reliability
    assert r[2] < r[0]
    assert r[2] < r[1]
    assert r[2] < 0.40, f"Stuck sensor reliability not low enough: {r[2]}"


def test_srm_noise_and_drift_separation():
    """Verify additive Gaussian noise and linear drift reduce reliability below clean levels."""
    np.random.seed(42)
    N, M = 300, 4
    clean_data = np.random.normal(0, 1, (N, M)).astype(np.float32)

    srm = SensorReliabilityModule(num_sensors=M)
    srm.fit_calibration(clean_data)

    # Test window where channel 1 has severe noise, channel 3 has drift
    window = np.random.normal(0, 1, (30, M)).astype(np.float32)
    window[:, 1] += np.random.normal(0, 3.0, 30)  # Heavy noise
    window[:, 3] += np.linspace(0, 6.0, 30)       # Linear drift

    r, _ = srm.estimate_reliability(window)

    assert r[1] < r[0], "Noisy sensor reliability not depressed relative to clean"
    assert r[3] < r[0], "Drifting sensor reliability not depressed relative to clean"
    separation = r[0] - r[1]
    assert separation > 0.20, f"Insufficient noise separation: {separation}"


def test_srm_serialization():
    """Verify SRM state can be converted to dict and restored."""
    np.random.seed(42)
    N, M = 200, 4
    data = np.random.normal(1.0, 0.5, (N, M)).astype(np.float32)

    srm1 = SensorReliabilityModule(num_sensors=M)
    srm1.fit_calibration(data)

    state = srm1.to_dict()
    assert "num_sensors" in state
    assert "max_delta" in state
    assert "stuck_window" in state

    srm2 = SensorReliabilityModule.from_dict(state)
    assert srm2.num_sensors == srm1.num_sensors

    window = np.random.normal(1.0, 0.5, (30, M)).astype(np.float32)
    r1, _ = srm1.estimate_reliability(window)
    r2, _ = srm2.estimate_reliability(window)

    np.testing.assert_allclose(r1, r2, atol=1e-6)
