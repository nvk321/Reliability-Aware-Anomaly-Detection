"""
Controlled Sensor Corruption Benchmark Driver.

Evaluates:
- Baseline A: Isolation Forest
- Baseline B: SpatioTemporal Autoencoder (ST-AE)
- Baseline C: RSTNet without SRM
- Proposed:   R-STNet with SRM

Under:
- E1: Clean SWaT
- E2: Gaussian Noise (Low=0.1, Med=0.3, High=0.5)
- E3: Stuck Sensor   (1 sensor, 3 sensors, 5 sensors)
- E4: Sensor Drift   (Low=0.1, Med=0.3, High=0.5)

Records across multiple random seeds (42, 101, 2024):
- System anomaly metrics (F1, Precision, Recall, FPR, PR-AUC)
- False alarm rate under sensor corruption (FPR on clean system periods)
- Reliability metrics:
  - Mean reliability of corrupted sensors
  - Mean reliability of clean sensors
  - Reliability separation (clean - corrupted)
  - Sensor fault localization Precision, Recall, F1
"""

from typing import Dict, List, Any, Tuple, Optional
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.corruption.corruptor import SensorCorruptor
from src.preprocessing.windowing import SlidingWindowGenerator
from src.evaluation.metrics import compute_anomaly_metrics, compute_sensor_fault_metrics
from src.models.baselines.isolation_forest import IsolationForestBaseline
from src.models.baselines.reconstruction_ae import SpatioTemporalAE
from src.models.baselines.rstnet_no_srm import RSTNetNoSRM
from src.models.rstnet import RSTNet


def evaluate_model_on_windows(
    model: nn.Module,
    windows: np.ndarray,
    model_type: str,
    device: torch.device,
    batch_size: int = 256,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    model.eval()
    tensor_windows = torch.tensor(windows, dtype=torch.float32)
    num_windows = len(windows)

    all_scores = []
    all_preds = []

    with torch.no_grad():
        for i in range(0, num_windows, batch_size):
            batch_x = tensor_windows[i : i + batch_size].to(device)
            if model_type == "stae":
                recon, _ = model(batch_x)
                diff = batch_x[:, -1, :] - recon[:, -1, :]
                sys_score = torch.mean(diff ** 2, dim=-1)
                all_scores.append(sys_score.cpu().numpy())
                all_preds.append(recon[:, -1, :].cpu().numpy())
            elif model_type in ["rstnet_no_srm", "rstnet"]:
                pred, _ = model(batch_x)
                all_preds.append(pred.cpu().numpy())

    if model_type == "rstnet":
        preds_np = np.concatenate(all_preds, axis=0)
        current_x = windows[:, -1, :]
        raw_diff = np.abs(current_x - preds_np)
        z = raw_diff / (model.srm.mad_residual_ + 1e-6) if model.srm.mad_residual_ is not None else raw_diff
        r, _ = model.srm.estimate_reliability_batch(windows, predicted_x=preds_np)
        scores_np = model.srm.compute_weighted_system_anomaly_batch(z, r)
        return scores_np, preds_np, r
    elif model_type == "rstnet_no_srm":
        preds_np = np.concatenate(all_preds, axis=0)
        current_x = windows[:, -1, :]
        raw_diff = np.abs(current_x - preds_np)
        z = raw_diff / (model.mad_residual_ + 1e-6) if getattr(model, "mad_residual_", None) is not None else raw_diff
        scores_np = np.mean(z, axis=-1)
        return scores_np, preds_np, None
    else:
        scores_np = np.concatenate(all_scores, axis=0)
        preds_np = np.concatenate(all_preds, axis=0)
        return scores_np, preds_np, None


def run_corruption_suite(
    data_dir: str = "data/processed/SWaT",
    checkpoint_dir: str = "results/checkpoints",
    output_dir: str = "results/corruption",
    seeds: List[int] = [42, 101, 2024],
    window_size: int = 30,
    device: str = "cuda",
):
    os.makedirs(output_dir, exist_ok=True)
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

    # 1. Load Data
    train_raw = np.load(os.path.join(data_dir, "train.npy"))
    val_raw = np.load(os.path.join(data_dir, "val.npy"))
    test_raw = np.load(os.path.join(data_dir, "test.npy"))
    test_labels_raw = np.load(os.path.join(data_dir, "test_labels.npy"))
    num_sensors = test_raw.shape[1]

    win_gen = SlidingWindowGenerator(window_size=window_size, stride=1, label_mode="point_end")
    val_windows, _ = win_gen.create_windows(val_raw)

    # 2. Load Checkpoints & Calibrate
    print("Loading models and checkpoints...")
    # Baseline B: ST-AE
    stae = SpatioTemporalAE(in_channels=num_sensors).to(dev)
    ckpt_stae = torch.load(os.path.join(checkpoint_dir, "baseline_stae_swat.pt"), map_location=dev)
    stae.load_state_dict(ckpt_stae["model_state_dict"])
    tau_stae = ckpt_stae["threshold_tau"]

    # Baseline C: RSTNetNoSRM
    rstnet_no_srm = RSTNetNoSRM(in_channels=num_sensors).to(dev)
    ckpt_no_srm = torch.load(os.path.join(checkpoint_dir, "baseline_rstnet_no_srm_swat.pt"), map_location=dev)
    rstnet_no_srm.load_state_dict(ckpt_no_srm["model_state_dict"])
    if "mad_residual" in ckpt_no_srm and ckpt_no_srm["mad_residual"] is not None:
        rstnet_no_srm.mad_residual_ = ckpt_no_srm["mad_residual"]
    else:
        rstnet_no_srm.calibrate_mad(val_windows, device=dev)
    tau_no_srm = ckpt_no_srm["threshold_tau"]

    # Proposed: RSTNet with SRM
    rstnet = RSTNet(in_channels=num_sensors).to(dev)
    ckpt_rstnet = torch.load(os.path.join(checkpoint_dir, "proposed_rstnet_swat.pt"), map_location=dev)
    rstnet.load_state_dict(ckpt_rstnet["model_state_dict"])
    rstnet.calibrate_srm(val_windows, device=dev)
    tau_rstnet = ckpt_rstnet["threshold_tau"]

    # Baseline A: Isolation Forest (pointwise)
    iso_forest = IsolationForestBaseline(n_estimators=100, random_state=42)
    iso_forest.fit(train_raw)
    val_scores_if = iso_forest.score_samples(val_raw)
    tau_if = float(np.quantile(val_scores_if, 0.95))

    # Experiment Definitions
    corruption_configs = [
        {"name": "E1_Clean", "type": "clean", "severity": 0.0, "rate": 0.0, "targets": None},
        # E2: Gaussian Noise
        {"name": "E2_Noise_Low", "type": "gaussian_noise", "severity": 0.1, "rate": 0.1, "targets": [3, 14, 28]},
        {"name": "E2_Noise_Med", "type": "gaussian_noise", "severity": 0.3, "rate": 0.2, "targets": [3, 14, 28]},
        {"name": "E2_Noise_High", "type": "gaussian_noise", "severity": 0.5, "rate": 0.3, "targets": [3, 14, 28]},
        # E3: Stuck Sensor
        {"name": "E3_Stuck_1Sensor", "type": "stuck_constant", "severity": 0.0, "rate": 0.15, "targets": [10]},
        {"name": "E3_Stuck_3Sensors", "type": "stuck_constant", "severity": 0.0, "rate": 0.15, "targets": [10, 22, 35]},
        {"name": "E3_Stuck_5Sensors", "type": "stuck_constant", "severity": 0.0, "rate": 0.15, "targets": [10, 15, 22, 35, 48]},
        # E4: Sensor Drift
        {"name": "E4_Drift_Low", "type": "drift", "severity": 0.1, "rate": 0.2, "targets": [7, 19, 41]},
        {"name": "E4_Drift_Med", "type": "drift", "severity": 0.3, "rate": 0.2, "targets": [7, 19, 41]},
        {"name": "E4_Drift_High", "type": "drift", "severity": 0.5, "rate": 0.2, "targets": [7, 19, 41]},
    ]

    all_results = []

    for cfg in corruption_configs:
        exp_name = cfg["name"]
        ctype = cfg["type"]
        sev = cfg["severity"]
        crate = cfg["rate"]
        targets = cfg["targets"]

        print(f"\nRunning Experiment: {exp_name} ({ctype})...")

        exp_seed_metrics = {
            "iso_forest": {"f1": [], "fpr": [], "precision": [], "recall": []},
            "stae": {"f1": [], "fpr": [], "precision": [], "recall": []},
            "rstnet_no_srm": {"f1": [], "fpr": [], "precision": [], "recall": []},
            "rstnet": {"f1": [], "fpr": [], "precision": [], "recall": []},
            "reliability": {
                "r_corrupt_mean": [],
                "r_clean_mean": [],
                "separation": [],
                "fault_localization_acc": [],
                "fault_detection_recall": [],
                "false_sensor_alarm_rate": [],
            },
        }

        # For Clean experiment, run once (deterministic)
        current_seeds = [42] if ctype == "clean" else seeds

        for s in current_seeds:
            if ctype == "clean":
                corrupted_test = test_raw.copy()
                fault_mask = np.zeros_like(test_raw, dtype=np.int32)
            else:
                corruptor = SensorCorruptor(seed=s)
                corrupted_test, fault_mask, _ = corruptor.apply_corruption(
                    data=test_raw,
                    corruption_type=ctype,
                    affected_channels=targets,
                    severity=sev,
                    corruption_rate=crate,
                )

            # Generate windows
            c_test_win, c_test_labels = win_gen.create_windows(corrupted_test, test_labels_raw)
            # Fault mask for windows: take the end of the window
            win_fault_mask = fault_mask[window_size - 1 :]

            # 1. Baseline A (Isolation Forest)
            scores_if = iso_forest.score_samples(corrupted_test[window_size - 1 :])
            m_if = compute_anomaly_metrics(c_test_labels, scores_if, threshold=tau_if)
            for k in ["f1", "fpr", "precision", "recall"]:
                exp_seed_metrics["iso_forest"][k].append(m_if[k])

            # 2. Baseline B (ST-AE)
            scores_stae, _, _ = evaluate_model_on_windows(stae, c_test_win, "stae", dev)
            m_stae = compute_anomaly_metrics(c_test_labels, scores_stae, threshold=tau_stae)
            for k in ["f1", "fpr", "precision", "recall"]:
                exp_seed_metrics["stae"][k].append(m_stae[k])

            # 3. Baseline C (RSTNet without SRM)
            scores_no_srm, _, _ = evaluate_model_on_windows(rstnet_no_srm, c_test_win, "rstnet_no_srm", dev)
            m_no_srm = compute_anomaly_metrics(c_test_labels, scores_no_srm, threshold=tau_no_srm)
            for k in ["f1", "fpr", "precision", "recall"]:
                exp_seed_metrics["rstnet_no_srm"][k].append(m_no_srm[k])

            # 4. Proposed R-STNet (with SRM)
            scores_rstnet, preds_rstnet, rel_matrix = evaluate_model_on_windows(rstnet, c_test_win, "rstnet", dev)
            m_rstnet = compute_anomaly_metrics(c_test_labels, scores_rstnet, threshold=tau_rstnet)
            for k in ["f1", "fpr", "precision", "recall"]:
                exp_seed_metrics["rstnet"][k].append(m_rstnet[k])

            # 5. Reliability Evaluation
            if ctype != "clean" and targets is not None:
                corrupted_cols = targets
                clean_cols = [c for c in range(num_sensors) if c not in targets]

                # Identify corrupted time periods
                is_corrupted_time = np.any(win_fault_mask == 1, axis=1)
                if np.any(is_corrupted_time):
                    r_corrupt = np.mean(rel_matrix[is_corrupted_time][:, corrupted_cols])
                    r_clean = np.mean(rel_matrix[is_corrupted_time][:, clean_cols])
                else:
                    r_corrupt = np.mean(rel_matrix[:, corrupted_cols])
                    r_clean = np.mean(rel_matrix[:, clean_cols])

                sep = float(r_clean - r_corrupt)
                exp_seed_metrics["reliability"]["r_corrupt_mean"].append(float(r_corrupt))
                exp_seed_metrics["reliability"]["r_clean_mean"].append(float(r_clean))
                exp_seed_metrics["reliability"]["separation"].append(sep)

                # Sensor fault localization: sensor with lowest reliability
                pred_faulty_ch = np.argmin(rel_matrix, axis=1)
                # Sensor fault flagged if any sensor reliability < 0.60
                sensor_fault_flag = (np.min(rel_matrix, axis=1) < 0.60).astype(np.int32)

                fault_m = compute_sensor_fault_metrics(
                    sensor_fault_mask=win_fault_mask,
                    predicted_fault_indices=pred_faulty_ch,
                    sensor_fault_flags=sensor_fault_flag,
                )
                exp_seed_metrics["reliability"]["fault_localization_acc"].append(fault_m["fault_localization_accuracy"])
                exp_seed_metrics["reliability"]["fault_detection_recall"].append(fault_m["fault_detection_recall"])
                exp_seed_metrics["reliability"]["false_sensor_alarm_rate"].append(fault_m["false_sensor_alarm_rate"])

        # Aggregate summary across seeds
        exp_summary = {
            "experiment": exp_name,
            "type": ctype,
            "severity": sev,
            "rate": crate,
            "target_channels": targets,
            "seeds_evaluated": current_seeds,
            "models": {},
            "reliability_summary": {},
        }

        for m_name in ["iso_forest", "stae", "rstnet_no_srm", "rstnet"]:
            exp_summary["models"][m_name] = {
                "f1_mean": float(np.mean(exp_seed_metrics[m_name]["f1"])),
                "f1_std": float(np.std(exp_seed_metrics[m_name]["f1"])),
                "fpr_mean": float(np.mean(exp_seed_metrics[m_name]["fpr"])),
                "fpr_std": float(np.std(exp_seed_metrics[m_name]["fpr"])),
                "precision_mean": float(np.mean(exp_seed_metrics[m_name]["precision"])),
                "recall_mean": float(np.mean(exp_seed_metrics[m_name]["recall"])),
            }

        if ctype != "clean":
            for r_key in ["r_corrupt_mean", "r_clean_mean", "separation", "fault_localization_acc", "fault_detection_recall", "false_sensor_alarm_rate"]:
                vals = exp_seed_metrics["reliability"][r_key]
                exp_summary["reliability_summary"][r_key] = {
                    "mean": float(np.mean(vals)) if len(vals) > 0 else 0.0,
                    "std": float(np.std(vals)) if len(vals) > 0 else 0.0,
                }

        all_results.append(exp_summary)
        print(f"  Summary for {exp_name}:")
        print(f"    Baseline C (No SRM) - F1: {exp_summary['models']['rstnet_no_srm']['f1_mean']:.4f}, FPR: {exp_summary['models']['rstnet_no_srm']['fpr_mean']:.4f}")
        print(f"    Proposed R-STNet    - F1: {exp_summary['models']['rstnet']['f1_mean']:.4f}, FPR: {exp_summary['models']['rstnet']['fpr_mean']:.4f}")
        if ctype != "clean":
            print(f"    Reliability Sep:    {exp_summary['reliability_summary']['separation']['mean']:.4f}")

    # Save overall summary
    summary_path = os.path.join(output_dir, "corruption_benchmark_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[Corruption Suite] All experiments complete. Saved results to {summary_path}")
    return all_results


if __name__ == "__main__":
    run_corruption_suite()
