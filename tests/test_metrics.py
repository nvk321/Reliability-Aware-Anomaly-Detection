"""
Unit tests for evaluation metrics.

Validates:
- Threshold-dependent metrics: Precision, Recall, F1, FPR, Confusion Matrix.
- Threshold-independent metrics: PR-AUC, ROC-AUC.
- Point-adjusted evaluation (PA-F1) protocol integrity.
- Synthetic sensor fault localization metrics.
"""

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_anomaly_metrics,
    point_adjustment_eval,
    compute_sensor_fault_metrics,
)


def test_metrics_perfect_predictions():
    """Verify metrics when predictions match ground truth exactly."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 1], dtype=np.int32)
    y_scores = np.array([0.1, 0.2, 0.05, 0.3, 0.9, 0.95, 0.88, 0.2, 0.15, 0.92], dtype=np.float64)

    metrics = compute_anomaly_metrics(y_true, y_scores, threshold=0.5)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["fpr"] == 0.0
    assert metrics["tp"] == 4
    assert metrics["tn"] == 6
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0


def test_metrics_all_positive_predictions():
    """Verify metrics when model alerts on every single sample (saturation)."""
    y_true = np.array([0, 0, 0, 1, 1], dtype=np.int32)
    y_pred = np.array([1, 1, 1, 1, 1], dtype=np.int32)
    y_scores = np.array([0.9, 0.8, 0.85, 0.95, 0.99], dtype=np.float64)

    metrics = compute_anomaly_metrics(y_true, y_scores, y_pred=y_pred)

    assert metrics["recall"] == 1.0
    assert metrics["fpr"] == 1.0
    assert metrics["precision"] == pytest.approx(2 / 5, abs=1e-5)
    assert metrics["fp"] == 3
    assert metrics["tn"] == 0


def test_point_adjustment_protocol():
    """
    Verify point adjustment logic:
    If any single point inside an anomalous segment is flagged,
    the entire continuous segment is flagged as detected.
    """
    # 2 anomalous intervals: [3..5] and [8..9]
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1], dtype=np.int32)
    # Model only triggers on point 4 (inside first interval), misses second interval entirely
    y_pred = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.int32)

    raw_metrics = compute_anomaly_metrics(y_true, y_pred=y_pred, y_scores=y_pred.astype(float))
    pa_metrics = point_adjustment_eval(y_true, y_pred)

    # Raw recall should be 1/5 = 0.20
    assert raw_metrics["recall"] == pytest.approx(0.20, abs=1e-5)
    # Point-adjusted recall should be 3/5 = 0.60 (first interval fully credited)
    assert pa_metrics["recall_pa"] == pytest.approx(0.60, abs=1e-5)
    assert pa_metrics["precision_pa"] == 1.0


def test_sensor_fault_metrics():
    """Verify evaluation of synthetic sensor fault detection and localization."""
    N, M = 10, 4
    mask = np.zeros((N, M), dtype=np.int32)
    # Channel 2 is corrupted at steps 3, 4, 5
    mask[3:6, 2] = 1

    flags = np.zeros(N, dtype=np.int32)
    flags[3:6] = 1  # Successfully flagged fault
    flags[8] = 1    # False alarm on step 8

    indices = np.array([0, 0, 0, 2, 2, 2, 0, 0, 1, 0], dtype=np.int32)

    res = compute_sensor_fault_metrics(mask, indices, flags)

    # 3 out of 3 corrupted steps were flagged
    assert res["fault_detection_recall"] == 1.0
    # Channel 2 was correctly localized on all 3 flagged steps
    assert res["fault_localization_accuracy"] == 1.0
    # 1 false alarm out of 7 clean steps
    assert res["false_sensor_alarm_rate"] == pytest.approx(1 / 7, abs=1e-5)
