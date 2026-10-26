"""
Unit tests for SensorReliabilityModule.
"""

import numpy as np
import pytest

from src.reliability.sensor_reliability import SensorReliabilityModule


def test_srm_normal_clean_data():
    """Verify clean normal data achieves high reliability scores (> 0.70)."""
    np.random.seed(42)
    N, M = 300, 6
    # Simulated smooth normal time series
    t = np.linspace(0, 10, N)
    val_data = np.zeros((N, M), dtype=np.float32)
    for i in range(M):
        val_data[:, i] = np.sin(t + i * 0.5) + np.random.normal(0, 0.05, N)

    srm = SensorReliabilityModule(num_sensors=M)
    srm.fit_calibration(val_data)

    # Test on a normal window
    test_window = val_data[50:70]
    r, components = srm.estimate_reliability(test_window)

    assert r.shape == (M,)
    # All clean sensors should have high reliability
    assert np.all(r >= 0.60), f"Clean sensors got low reliability: {r}"


def test_srm_stuck_sensor_detection():
    """Verify a frozen/stuck sensor has its reliability heavily penalized."""
    np.random.seed(42)
    N, M = 200, 4
    val_data = np.random.normal(0, 1, (N, M)).astype(np.float32)

    srm = SensorReliabilityModule(num_sensors=M)
    srm.fit_calibration(val_data)

    # Create window where channel 2 is stuck at a constant value for 20 steps
    window = np.random.normal(0, 1, (25, M)).astype(np.float32)
    window[-20:, 2] = 1.5

    r, _ = srm.estimate_reliability(window)

    # Channel 2 should have significantly lower reliability than other channels
    assert r[2] < r[0]
    assert r[2] < r[1]
    assert r[2] < 0.40, f"Stuck sensor reliability not low enough: {r[2]}"


def test_srm_false_alarm_discounting():
    """
    Core scientific test:
    Verify that reliability weighting prevents a single noisy/corrupted sensor
    from triggering a false system anomaly alarm.
    """
    np.random.seed(42)
    M = 5
    srm = SensorReliabilityModule(num_sensors=M)
    srm.fit_calibration(np.random.normal(0, 1, (100, M)).astype(np.float32))

    # Clean residuals across 5 sensors: all low
    clean_residuals = np.array([0.2, 0.3, 0.25, 0.2, 0.15], dtype=np.float32)
    # Reliable weights
    high_rel = np.array([0.95, 0.95, 0.95, 0.95, 0.95], dtype=np.float32)

    # Now sensor 0 fails with a huge residual of 15.0 (corrupted sensor)
    faulty_residuals = np.array([15.0, 0.3, 0.25, 0.2, 0.15], dtype=np.float32)
    # Its reliability drops to 0.05
    faulty_rel = np.array([0.05, 0.95, 0.95, 0.95, 0.95], dtype=np.float32)

    # 1. Unweighted system anomaly score (conventional method)
    s_unweighted = np.mean(faulty_residuals)  # (15.0 + 0.9) / 5 = 3.18

    # 2. Reliability-weighted system anomaly score (our method)
    s_weighted = srm.compute_weighted_system_anomaly(
        residuals=faulty_residuals,
        reliability=faulty_rel,
    )

    # The reliability-weighted score should be substantially lower than unweighted
    assert s_weighted < s_unweighted * 0.4, (
        f"Weighted score ({s_weighted:.3f}) was not substantially lower than unweighted ({s_unweighted:.3f})"
    )
    # The weighted score remains close to normal baseline
    assert s_weighted < 0.8
