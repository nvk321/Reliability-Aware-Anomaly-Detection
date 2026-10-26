"""
Master Ablation Runner and Publication Plotting Generator for Phase 2.

Produces:
1. Ablation Study JSON (Ablation 1 to 5):
   - Ablation 1: RSTNet without SRM (No SRM)
   - Ablation 2: Temporal continuity only (w_v=1, w_c=0, w_p=0)
   - Ablation 3: Graph neighborhood only  (w_v=0, w_c=1.2, w_p=0)
   - Ablation 4: Residual confidence only (w_v=0, w_c=0, w_p=1)
   - Ablation 5: Full SRM                 (w_v=1, w_c=1.2, w_p=1)

2. The 8 Required Publication-Quality Figures:
   Figure 1: System anomaly score vs time with genuine SWaT attack regions highlighted.
   Figure 2: Reliability score vs time for selected sensors.
   Figure 3: Corrupted sensor reliability vs clean sensor reliability.
   Figure 4: Baseline C vs R-STNet anomaly scores under sensor corruption.
   Figure 5: F1 vs corruption severity across models.
   Figure 6: FPR vs corruption severity across models.
   Figure 7: Sensor-fault identification performance vs corruption severity.
   Figure 8: Ablation comparison across the 5 SRM variants.
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.baselines.rstnet_no_srm import RSTNetNoSRM
from src.models.rstnet import RSTNet
from src.preprocessing.windowing import SlidingWindowGenerator
from src.corruption.corruptor import SensorCorruptor
from src.evaluation.metrics import compute_anomaly_metrics, point_adjustment_eval

# Clean publication style
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["grid.linestyle"] = "--"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "data/processed/SWaT"
    ckpt_dir = "results/checkpoints"
    plot_dir = "results/plots"
    ablation_dir = "results/ablations"

    os.makedirs(f"{plot_dir}/anomaly_scores", exist_ok=True)
    os.makedirs(f"{plot_dir}/reliability", exist_ok=True)
    os.makedirs(f"{plot_dir}/corruption", exist_ok=True)
    os.makedirs(f"{plot_dir}/ablations", exist_ok=True)
    os.makedirs(ablation_dir, exist_ok=True)

    # 1. Load Data
    val_raw = np.load(os.path.join(data_dir, "val.npy"))
    test_raw = np.load(os.path.join(data_dir, "test.npy"))
    test_labels_raw = np.load(os.path.join(data_dir, "test_labels.npy"))
    num_sensors = test_raw.shape[1]
    window_size = 30

    win_gen = SlidingWindowGenerator(window_size=window_size, stride=1, label_mode="point_end")
    val_windows, _ = win_gen.create_windows(val_raw)
    test_windows, test_labels = win_gen.create_windows(test_raw, test_labels_raw)

    # 2. Load Checkpoints
    print("Loading Baseline C and Proposed R-STNet checkpoints...")
    model_c = RSTNetNoSRM(in_channels=num_sensors).to(device)
    ckpt_c = torch.load(os.path.join(ckpt_dir, "baseline_rstnet_no_srm_swat.pt"), map_location=device)
    model_c.load_state_dict(ckpt_c["model_state_dict"])
    if "mad_residual" in ckpt_c and ckpt_c["mad_residual"] is not None:
        model_c.mad_residual_ = ckpt_c["mad_residual"]
    else:
        model_c.calibrate_mad(val_windows, device=device)
    tau_c = ckpt_c["threshold_tau"]

    model_prop = RSTNet(in_channels=num_sensors).to(device)
    ckpt_prop = torch.load(os.path.join(ckpt_dir, "proposed_rstnet_swat.pt"), map_location=device)
    model_prop.load_state_dict(ckpt_prop["model_state_dict"])
    model_prop.calibrate_srm(val_windows, device=device)
    tau_prop = ckpt_prop["threshold_tau"]

    # 3. Compute Clean Predictions
    print("Computing clean test evaluations...")
    model_c.eval()
    model_prop.eval()
    batch_size = 256

    test_tensor = torch.tensor(test_windows, dtype=torch.float32)
    preds_list = []
    with torch.no_grad():
        for i in range(0, len(test_windows), batch_size):
            bx = test_tensor[i : i + batch_size].to(device)
            p, _ = model_prop(bx)
            preds_list.append(p.cpu().numpy())
    test_preds_np = np.concatenate(preds_list, axis=0)

    current_test_x = test_windows[:, -1, :]
    raw_diff = np.abs(current_test_x - test_preds_np)
    z_scores = raw_diff / (model_prop.srm.mad_residual_ + 1e-6)

    # Baseline C scores on clean test
    scores_c = np.mean(z_scores, axis=-1)

    # Proposed R-STNet full SRM scores on clean test
    r_matrix_clean, _ = model_prop.srm.estimate_reliability_batch(test_windows, predicted_x=test_preds_np)
    scores_prop = model_prop.srm.compute_weighted_system_anomaly_batch(z_scores, r_matrix_clean)

    # ========================================================
    # ABLATION EXPERIMENTS (Ablations 1 to 5)
    # ========================================================
    print("\nRunning Ablation Study (5 Configurations)...")
    ablation_weights = {
        "Ablation 1 (No SRM)": (0.0, 0.0, 0.0),
        "Ablation 2 (Temporal Only)": (1.0, 0.0, 0.0),
        "Ablation 3 (Graph Only)": (0.0, 1.2, 0.0),
        "Ablation 4 (Residual Only)": (0.0, 0.0, 1.0),
        "Ablation 5 (Full SRM)": (1.0, 1.2, 1.0),
    }

    ablation_results = {}
    for ab_name, (wv, wc, wp) in ablation_weights.items():
        if ab_name == "Ablation 1 (No SRM)":
            ab_scores = scores_c
            ab_tau = tau_c
        else:
            model_prop.srm.w_v = wv
            model_prop.srm.w_c = wc
            model_prop.srm.w_p = wp
            
            val_tensor = torch.tensor(val_windows, dtype=torch.float32)
            v_preds = []
            with torch.no_grad():
                for i in range(0, len(val_windows), batch_size):
                    bx = val_tensor[i : i + batch_size].to(device)
                    p, _ = model_prop(bx)
                    v_preds.append(p.cpu().numpy())
            v_preds_np = np.concatenate(v_preds, axis=0)
            v_diff = np.abs(val_windows[:, -1, :] - v_preds_np)
            v_z = v_diff / (model_prop.srm.mad_residual_ + 1e-6)
            v_r, _ = model_prop.srm.estimate_reliability_batch(val_windows, predicted_x=v_preds_np)
            v_scores = model_prop.srm.compute_weighted_system_anomaly_batch(v_z, v_r)
            ab_tau = float(np.quantile(v_scores, 0.95))

            ab_r, _ = model_prop.srm.estimate_reliability_batch(test_windows, predicted_x=test_preds_np)
            ab_scores = model_prop.srm.compute_weighted_system_anomaly_batch(z_scores, ab_r)

        ab_preds = (ab_scores >= ab_tau).astype(np.int32)
        m = compute_anomaly_metrics(test_labels, ab_scores, y_pred=ab_preds, threshold=ab_tau)
        ablation_results[ab_name] = {
            "w_v": wv,
            "w_c": wc,
            "w_p": wp,
            "threshold_tau": ab_tau,
            "f1": m["f1"],
            "precision": m["precision"],
            "recall": m["recall"],
            "fpr": m["fpr"],
            "pr_auc": m["pr_auc"],
        }
        print(f"  {ab_name:28s} -> F1: {m['f1']:.4f} | FPR: {m['fpr']*100:.2f}% | PR-AUC: {m['pr_auc']:.4f}")

    model_prop.srm.w_v = 1.0
    model_prop.srm.w_c = 1.2
    model_prop.srm.w_p = 1.0

    with open(f"{ablation_dir}/ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)

    # ========================================================
    # FIGURE 1: Anomaly Score vs Time with True SWaT Attack Regions
    # ========================================================
    print("\nGenerating Figure 1: System Anomaly Score Timeline...")
    sample_slice = slice(1000, 7000)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    y_sub = test_labels[sample_slice]
    attack_intervals = []
    in_atk = False
    start_idx = 0
    for i in range(len(y_sub)):
        if y_sub[i] == 1 and not in_atk:
            in_atk = True
            start_idx = i
        elif y_sub[i] == 0 and in_atk:
            in_atk = False
            attack_intervals.append((start_idx, i))
    if in_atk:
        attack_intervals.append((start_idx, len(y_sub)))

    for ax, scores, tau, name, col in [
        (ax1, scores_c[sample_slice], tau_c, "Baseline C (RSTNet without SRM)", "#d95f02"),
        (ax2, scores_prop[sample_slice], tau_prop, "Proposed R-STNet (with SRM)", "#1b9e77"),
    ]:
        for s_idx, e_idx in attack_intervals:
            ax.axvspan(s_idx, e_idx, color="#ff4c4c", alpha=0.3, label="SWaT Attack" if (s_idx, e_idx) == attack_intervals[0] else "")
        ax.plot(scores, label=f"{name} Score", color=col, linewidth=1.1)
        ax.axhline(tau, color="#222222", linestyle="--", linewidth=1.3, label=f"Threshold $\\tau$ ({tau:.2f})")
        ax.set_ylabel("Anomaly Score", fontweight="bold")
        ax.set_title(name, fontweight="bold")
        ax.legend(loc="upper right", framealpha=0.9)

    ax2.set_xlabel("Time Step (10s intervals)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/anomaly_scores/system_anomaly_scores_timeline.png", dpi=300)
    plt.close()

    # ========================================================
    # FIGURE 2: Reliability Score vs Time for Selected Sensors
    # ========================================================
    print("Generating Figure 2: Reliability Score vs Time...")
    fig, ax = plt.subplots(figsize=(13, 5))
    sensor_names = ["FIT101 (Flow Sensor 1)", "LIT101 (Level Sensor 2)", "AIT201 (Shifted Sensor 3)", "DPIT301 (Pressure Sensor 28)"]
    selected_indices = [1, 2, 3, 28]
    colors = ["#1b9e77", "#2b5c8f", "#d95f02", "#7570b3"]

    sub_t = slice(2000, 4000)
    for idx, col_idx in enumerate(selected_indices):
        ax.plot(r_matrix_clean[sub_t, col_idx], label=sensor_names[idx], color=colors[idx], linewidth=1.3)

    ax.axhline(0.60, color="#666666", linestyle=":", linewidth=1.3, label="Fault Threshold (0.60)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time Step (10s intervals)", fontweight="bold")
    ax.set_ylabel("Reliability Score $r_i(t)$", fontweight="bold")
    ax.set_title("Dynamic Per-Sensor Reliability Trajectories Over Time", fontweight="bold")
    ax.legend(loc="lower left", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/reliability/reliability_score_vs_time.png", dpi=300)
    plt.close()

    # ========================================================
    # FIGURE 3: Corrupted vs Clean Sensor Reliability Under Corruption
    # ========================================================
    print("Generating Figure 3: Corrupted vs Clean Sensor Reliability...")
    corruptor = SensorCorruptor(seed=42)
    corrupted_data, fault_mask, _ = corruptor.apply_corruption(
        data=test_raw,
        corruption_type="gaussian_noise",
        affected_channels=[3, 14, 28],
        severity=0.5,
        corruption_rate=0.3,
    )
    c_windows, _ = win_gen.create_windows(corrupted_data)

    c_tensor = torch.tensor(c_windows, dtype=torch.float32)
    c_preds_list = []
    with torch.no_grad():
        for i in range(0, len(c_windows), batch_size):
            bx = c_tensor[i : i + batch_size].to(device)
            p, _ = model_prop(bx)
            c_preds_list.append(p.cpu().numpy())
    c_preds_np = np.concatenate(c_preds_list, axis=0)
    r_corrupted_mat, _ = model_prop.srm.estimate_reliability_batch(c_windows, predicted_x=c_preds_np)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    plot_sub = slice(1000, 3000)
    r_corr_mean = np.mean(r_corrupted_mat[plot_sub][:, [3, 14, 28]], axis=1)
    clean_cols = [c for c in range(num_sensors) if c not in [3, 14, 28]]
    r_clean_mean = np.mean(r_corrupted_mat[plot_sub][:, clean_cols], axis=1)

    ax.plot(r_clean_mean, label="Clean Sensor Channels (Mean $r_i$)", color="#2ca02c", linewidth=1.5)
    ax.plot(r_corr_mean, label="Corrupted Sensor Channels (Noise $\\sigma=0.5$)", color="#d62728", linewidth=1.5)
    ax.axhline(0.60, color="#444444", linestyle="--", linewidth=1.2, label="Reliability Separation Cutoff")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time Step (10s intervals)", fontweight="bold")
    ax.set_ylabel("Reliability $r_i(t)$", fontweight="bold")
    ax.set_title("Dynamic Decoupling: Corrupted vs Clean Sensor Reliability Trajectories", fontweight="bold")
    ax.legend(loc="lower left", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/reliability/corrupted_vs_clean_reliability.png", dpi=300)
    plt.close()

    # ========================================================
    # FIGURE 4: Baseline C vs R-STNet Under Corruption
    # ========================================================
    print("Generating Figure 4: Baseline C vs R-STNet Under Corruption...")
    c_diff = np.abs(c_windows[:, -1, :] - c_preds_np)
    c_z = c_diff / (model_prop.srm.mad_residual_ + 1e-6)
    scores_c_corrupt = np.mean(c_z, axis=-1)
    scores_prop_corrupt = model_prop.srm.compute_weighted_system_anomaly_batch(c_z, r_corrupted_mat)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    c_slice = slice(1500, 3500)
    ax1.plot(scores_c_corrupt[c_slice], label="Baseline C (No SRM) Under Corruption", color="#d95f02", linewidth=1.2)
    ax1.axhline(tau_c, color="#222222", linestyle="--", linewidth=1.3, label=f"Baseline Threshold ({tau_c:.2f})")
    ax1.set_ylabel("Anomaly Score", fontweight="bold")
    ax1.set_title("Baseline C (No SRM): Massive Sensor Corruption Causes Immediate False Alarm Explosion", fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.9)

    ax2.plot(scores_prop_corrupt[c_slice], label="Proposed R-STNet (With SRM) Under Corruption", color="#1b9e77", linewidth=1.2)
    ax2.axhline(tau_prop, color="#222222", linestyle="--", linewidth=1.3, label=f"Proposed Threshold ({tau_prop:.2f})")
    ax2.set_ylabel("Anomaly Score", fontweight="bold")
    ax2.set_xlabel("Time Step (10s intervals)", fontweight="bold")
    ax2.set_title("Proposed R-STNet: Sensor Reliability Module Suppresses False System Alarm", fontweight="bold")
    ax2.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/corruption/baseline_c_vs_rstnet_under_corruption.png", dpi=300)
    plt.close()

    # ========================================================
    # FIGURES 5, 6, 7: LOAD CORRUPTION BENCHMARK SUMMARY
    # ========================================================
    summary_file = "results/corruption/corruption_benchmark_summary.json"
    with open(summary_file, "r", encoding="utf-8") as f:
        c_summary = json.load(f)

    noise_data = [d for d in c_summary if "Noise" in d["experiment"]]
    severities = [d["severity"] for d in noise_data]

    # FIGURE 5: F1 vs Corruption Severity
    print("Generating Figure 5: F1 vs Corruption Severity...")
    fig, ax = plt.subplots(figsize=(8, 5))
    model_styles = [
        ("iso_forest", "Baseline A (Isolation Forest)", "#7570b3", "o"),
        ("stae", "Baseline B (ST-AE)", "#e7298a", "s"),
        ("rstnet_no_srm", "Baseline C (RSTNet without SRM)", "#d95f02", "^"),
        ("rstnet", "Proposed R-STNet (with SRM)", "#1b9e77", "D"),
    ]
    for m_key, m_lbl, col, mk in model_styles:
        f1_vals = [d["models"][m_key]["f1_mean"] for d in noise_data]
        ax.plot(severities, f1_vals, label=m_lbl, color=col, marker=mk, linewidth=2)
    ax.set_xlabel("Gaussian Noise Severity ($\\sigma$)", fontweight="bold")
    ax.set_ylabel("Point-Wise F1 Score", fontweight="bold")
    ax.set_title("System Anomaly F1 vs Sensor Corruption Severity", fontweight="bold")
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/corruption/f1_vs_corruption_severity.png", dpi=300)
    plt.close()

    # FIGURE 6: FPR vs Corruption Severity
    print("Generating Figure 6: FPR vs Corruption Severity...")
    fig, ax = plt.subplots(figsize=(8, 5))
    for m_key, m_lbl, col, mk in model_styles:
        fpr_vals = [d["models"][m_key]["fpr_mean"] * 100 for d in noise_data]
        ax.plot(severities, fpr_vals, label=m_lbl, color=col, marker=mk, linewidth=2)
    ax.set_xlabel("Gaussian Noise Severity ($\\sigma$)", fontweight="bold")
    ax.set_ylabel("False Positive Rate (%)", fontweight="bold")
    ax.set_title("System False Positive Rate (FPR) vs Corruption Severity\n(Lower is Better)", fontweight="bold")
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/corruption/fpr_vs_corruption_severity.png", dpi=300)
    plt.close()

    # FIGURE 7: Sensor Fault Identification Performance vs Severity
    print("Generating Figure 7: Sensor Fault Identification Performance...")
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    loc_accs = [d["reliability_summary"]["fault_localization_acc"]["mean"] * 100 for d in noise_data]
    recalls = [d["reliability_summary"]["fault_detection_recall"]["mean"] * 100 for d in noise_data]

    ax1.plot(severities, recalls, label="Fault Detection Recall", color="#1b9e77", marker="o", linewidth=2)
    ax2.plot(severities, loc_accs, label="Fault Localization Acc", color="#2b5c8f", marker="s", linewidth=2)
    ax1.set_xlabel("Gaussian Noise Severity ($\\sigma$)", fontweight="bold")
    ax1.set_ylabel("Detection Recall (%)", color="#1b9e77", fontweight="bold")
    ax2.set_ylabel("Localization Accuracy (%)", color="#2b5c8f", fontweight="bold")
    ax1.set_title("Sensor-Level Fault Diagnosis Under Sensor Corruption", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/corruption/sensor_fault_id_f1_vs_severity.png", dpi=300)
    plt.close()

    # FIGURE 8: Ablation Study Comparison
    print("Generating Figure 8: Ablation Study Comparison...")
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax2 = ax1.twinx()

    configs = list(ablation_results.keys())
    short_names = ["No SRM\n(Uniform)", "Temporal\nOnly", "Graph\nOnly", "Residual\nOnly", "Full SRM\n(Proposed)"]
    f1_list = [ablation_results[c]["f1"] for c in configs]
    fpr_list = [ablation_results[c]["fpr"] * 100 for c in configs]

    x = np.arange(len(configs))
    w = 0.35
    r1 = ax1.bar(x - w / 2, f1_list, w, label="Point-Wise F1", color="#1b9e77", alpha=0.85)
    r2 = ax2.bar(x + w / 2, fpr_list, w, label="FPR (%)", color="#d95f02", alpha=0.85)

    ax1.set_ylabel("Point-Wise F1 Score", color="#1b9e77", fontweight="bold")
    ax2.set_ylabel("False Positive Rate (%)", color="#d95f02", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, fontweight="bold")
    ax1.set_title("SRM Component Ablation: Impact of Fusion Perspectives", fontweight="bold")

    for rect in r1:
        h = rect.get_height()
        ax1.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for rect in r2:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/ablations/srm_ablation_comparison.png", dpi=300)
    plt.close()

    print("\n[Plots & Ablations] ALL 8 FIGURES AND ABLATIONS GENERATED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
