"""
Driver script to train and evaluate Baseline A (Isolation Forest) on SWaT.

Strict Leak-Free Protocol:
- Model fits strictly on pure normal training split (y = 0).
- Decision threshold tau is calibrated strictly on the pure normal validation split (y = 0).
- Evaluation is performed blindly on the chronological mixed test set (y in {0, 1}).
- Raw point-wise F1, Precision, Recall, PR-AUC, and FPR are reported as primary metrics.
- Point-adjusted F1 (PA) is reported strictly as a secondary labeled reference.
"""

import argparse
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.baselines.isolation_forest import IsolationForestBaseline
from src.evaluation.metrics import compute_anomaly_metrics, point_adjustment_eval


def main():
    parser = argparse.ArgumentParser(description="Train Baseline A (Isolation Forest)")
    parser.add_argument("--data_dir", type=str, default="data/processed/SWaT")
    parser.add_argument("--output_dir", type=str, default="results/metrics")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--val_quantile", type=float, default=0.95, help="Quantile on normal validation split for threshold")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Baseline A] Loading processed SWaT data from {args.data_dir}...")
    train = np.load(os.path.join(args.data_dir, "train.npy"))
    val = np.load(os.path.join(args.data_dir, "val.npy"))
    test = np.load(os.path.join(args.data_dir, "test.npy"))
    test_labels = np.load(os.path.join(args.data_dir, "test_labels.npy"))

    print(f"  Train Split Shape:      {train.shape} (y = 0, 100% Normal)")
    print(f"  Validation Split Shape: {val.shape} (y = 0, 100% Normal)")
    print(f"  Test Split Shape:       {test.shape} (Attacks: {test_labels.sum()} = {test_labels.mean()*100:.2f}%)")

    # 1. Fit Model on Normal Training Data
    print(f"\n[Baseline A] Fitting Isolation Forest on normal training split (seed={args.seed})...")
    iso_forest = IsolationForestBaseline(
        n_estimators=args.n_estimators,
        contamination=1.0 - args.val_quantile,
        random_state=args.seed,
    )
    iso_forest.fit(train)

    # 2. Calibrate Threshold on Pure Normal Validation Split (NO TEST LEAKAGE)
    val_scores = iso_forest.score_samples(val)
    tau = float(np.quantile(val_scores, args.val_quantile))
    print(f"[Baseline A] Calibrated decision threshold on validation set (quantile {args.val_quantile}): tau = {tau:.5f}")

    # 3. Blind Evaluation on Mixed Chronological Test Set
    print("[Baseline A] Scoring chronological mixed test set...")
    test_scores = iso_forest.score_samples(test)
    test_preds = (test_scores >= tau).astype(np.int32)

    # 4. Compute Metrics
    raw_metrics = compute_anomaly_metrics(
        y_true=test_labels,
        y_scores=test_scores,
        y_pred=test_preds,
        threshold=tau,
    )
    pa_metrics = point_adjustment_eval(y_true=test_labels, y_pred=test_preds)

    result = {
        "model": "IsolationForest",
        "dataset": "SWaT_A1_A2_Dec2015",
        "experiment": "E1_clean_benchmark",
        "split_sizes": {
            "train": int(len(train)),
            "train_anomalies": 0,
            "val": int(len(val)),
            "val_anomalies": 0,
            "test": int(len(test)),
            "test_anomalies": int(test_labels.sum()),
            "test_attack_rate": float(test_labels.mean()),
        },
        "threshold_calibration": {
            "method": "validation_quantile",
            "val_quantile": args.val_quantile,
            "calibrated_threshold_tau": tau,
            "val_scores_mean": float(np.mean(val_scores)),
            "val_scores_std": float(np.std(val_scores)),
        },
        "raw_pointwise_metrics": raw_metrics,
        "labeled_point_adjusted_metrics": pa_metrics,
    }

    out_file = os.path.join(args.output_dir, "baseline_isolation_forest_swat.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 60)
    print("CORRECTED BASELINE A (ISOLATION FOREST) RESULTS:")
    print("=" * 60)
    print(f"  Raw Point-wise F1:        {raw_metrics['f1']:.4f}")
    print(f"  Raw Point-wise Precision: {raw_metrics['precision']:.4f}")
    print(f"  Raw Point-wise Recall:    {raw_metrics['recall']:.4f}")
    print(f"  PR-AUC (Average Prec):    {raw_metrics['pr_auc']:.4f}")
    print(f"  ROC-AUC:                  {raw_metrics['roc_auc']:.4f}")
    print(f"  False Positive Rate (FPR):{raw_metrics['fpr']:.4f}")
    print(f"  Confusion Matrix:         TP={raw_metrics['tp']}, FP={raw_metrics['fp']}, TN={raw_metrics['tn']}, FN={raw_metrics['fn']}")
    print(f"  Point-Adjusted F1 (PA):   {pa_metrics['f1_pa']:.4f} (Reference only)")
    print(f"  Saved to:                 {out_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
