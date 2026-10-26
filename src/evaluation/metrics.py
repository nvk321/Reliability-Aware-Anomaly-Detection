"""
Comprehensive Evaluation Metrics for Reliability-Aware Anomaly Detection.

Enforces scientific reporting standards:
1. RAW POINT-WISE Precision, Recall, F1, and PR-AUC as the primary ground truth.
2. Avoids Point Adjustment (PA) as the primary metric (provides it only as optional labeled reference).
3. Sensor Fault Identification Accuracy: compares detected faulty sensor against ground-truth corruption mask.
4. False Alarm Suppression Rate (FASR) and False Positive Rate (FPR).
"""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
)


def compute_anomaly_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Computes threshold-dependent and threshold-independent metrics.

    Args:
        y_true: 1D binary ground truth (0 = normal, 1 = attack).
        y_scores: Continuous anomaly scores S(t).
        y_pred: Optional pre-thresholded binary predictions.
        threshold: Optional decision threshold applied to y_scores if y_pred is None.

    Returns:
        Dictionary of raw metrics (F1, Precision, Recall, PR-AUC, ROC-AUC, FPR).
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_scores = np.asarray(y_scores, dtype=np.float64)

    if y_pred is None:
        if threshold is None:
            # Default to 99th percentile if threshold not provided
            threshold = float(np.quantile(y_scores, 0.99))
        y_pred = (y_scores >= threshold).astype(np.int32)
    else:
        y_pred = np.asarray(y_pred, dtype=np.int32)

    # Raw point-wise metrics
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Confusion matrix & FPR
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # PR-AUC & ROC-AUC
    try:
        p_curve, r_curve, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = float(auc(r_curve, p_curve))
    except Exception:
        pr_auc = 0.0

    try:
        roc_auc = float(roc_auc_score(y_true, y_scores))
    except Exception:
        roc_auc = 0.0

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "fpr": fpr,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def compute_sensor_fault_metrics(
    sensor_fault_mask: np.ndarray,
    predicted_fault_indices: np.ndarray,
    sensor_fault_flags: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluates sensor fault localization against synthetic corruption ground-truth.

    Args:
        sensor_fault_mask: Binary matrix (N, M) indicating true corrupted sensors.
        predicted_fault_indices: (N,) indices of the most suspicious sensor at each step.
        sensor_fault_flags: (N,) binary indicator that a sensor fault is flagged.
    """
    N, M = sensor_fault_mask.shape
    corrupted_time_steps = np.where(np.any(sensor_fault_mask == 1, axis=1))[0]

    if len(corrupted_time_steps) == 0:
        return {
            "fault_detection_recall": 1.0,
            "fault_localization_accuracy": 1.0,
            "false_sensor_alarm_rate": 0.0,
        }

    # 1. Did the system flag a sensor fault when one was truly present?
    true_fault_detected = sensor_fault_flags[corrupted_time_steps]
    fault_recall = float(np.mean(true_fault_detected == 1))

    # 2. When a fault was flagged on a corrupted time step, was the correct channel identified?
    correct_localization = 0
    flagged_and_corrupted = 0
    for t in corrupted_time_steps:
        if sensor_fault_flags[t] == 1:
            flagged_and_corrupted += 1
            pred_ch = predicted_fault_indices[t]
            if sensor_fault_mask[t, pred_ch] == 1:
                correct_localization += 1

    loc_acc = float(correct_localization / flagged_and_corrupted) if flagged_and_corrupted > 0 else 0.0

    # 3. False sensor alarm rate: flagged sensor fault when sensors were completely clean
    clean_time_steps = np.where(~np.any(sensor_fault_mask == 1, axis=1))[0]
    false_sensor_alarms = float(np.mean(sensor_fault_flags[clean_time_steps] == 1)) if len(clean_time_steps) > 0 else 0.0

    return {
        "fault_detection_recall": fault_recall,
        "fault_localization_accuracy": loc_acc,
        "false_sensor_alarm_rate": false_sensor_alarms,
    }


def point_adjustment_eval(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Optional Point Adjustment (PA) computation for direct comparison with external papers.
    Explicitly labeled as 'PA' to maintain scientific integrity.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pa = np.copy(np.asarray(y_pred, dtype=np.int32))

    # Identify continuous anomalous segments
    in_anomaly = False
    start = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_anomaly:
            in_anomaly = True
            start = i
        elif y_true[i] == 0 and in_anomaly:
            in_anomaly = False
            # If any point was detected, adjust entire segment
            if np.any(y_pa[start:i] == 1):
                y_pa[start:i] = 1

    if in_anomaly and np.any(y_pa[start:] == 1):
        y_pa[start:] = 1

    return {
        "precision_pa": float(precision_score(y_true, y_pa, zero_division=0)),
        "recall_pa": float(recall_score(y_true, y_pa, zero_division=0)),
        "f1_pa": float(f1_score(y_true, y_pa, zero_division=0)),
    }
