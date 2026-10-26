"""
Publication-Quality Plotting Suite for Reliability-Aware Anomaly Detection.

Generates the required scientific figures:
1. System Anomaly Score vs Time with Ground-Truth Attack Regions Highlighted.
2. Sensor Reliability Score r_i(t) vs Time for Clean vs Corrupted Sensors.
3. Baseline C (No SRM) vs Proposed R-STNet Anomaly Scores under Sensor Corruption.
4. System False Positive Rate (FPR) vs Sensor Corruption Severity.
5. System Anomaly F1 vs Sensor Corruption Severity.
6. Sensor-Fault Identification Performance vs Corruption Severity.
7. Ablation Study: Contribution of Individual SRM Components.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Apply clean scientific styling
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["grid.linestyle"] = "--"


def plot_anomaly_scores_timeline(
    timestamps: np.ndarray,
    scores_dict: dict,
    y_true: np.ndarray,
    thresholds_dict: dict,
    output_path: str = "results/plots/anomaly_scores_timeline.png",
    sample_slice: slice = slice(0, 5000),
):
    """Plot 1: System anomaly scores over time with true SWaT attack regions."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    t = timestamps[sample_slice]
    y = y_true[sample_slice]

    fig, axes = plt.subplots(len(scores_dict), 1, figsize=(14, 3.5 * len(scores_dict)), sharex=True)
    if len(scores_dict) == 1:
        axes = [axes]

    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]

    for idx, (model_name, scores) in enumerate(scores_dict.items()):
        ax = axes[idx]
        s = scores[sample_slice]
        tau = thresholds_dict.get(model_name, None)
        color = colors[idx % len(colors)]

        # Highlight attack regions
        attack_intervals = []
        in_atk = False
        start_idx = 0
        for i in range(len(y)):
            if y[i] == 1 and not in_atk:
                in_atk = True
                start_idx = i
            elif y[i] == 0 and in_atk:
                in_atk = False
                attack_intervals.append((start_idx, i))
        if in_atk:
            attack_intervals.append((start_idx, len(y)))

        for start, end in attack_intervals:
            ax.axvspan(start, end, color="#ff4c4c", alpha=0.25, label="SWaT Attack" if start == attack_intervals[0][0] and idx == 0 else "")

        ax.plot(range(len(s)), s, label=f"{model_name} Score", color=color, linewidth=1.2)
        if tau is not None:
            ax.axhline(tau, color="#333333", linestyle="--", linewidth=1.4, label=f"Threshold $\\tau$ ({tau:.3f})")

        ax.set_ylabel("Anomaly Score", fontweight="bold")
        ax.set_title(f"{model_name} Anomaly Scores vs Ground-Truth Attacks", fontweight="bold")
        ax.legend(loc="upper right", framealpha=0.9)

    axes[-1].set_xlabel("Time Step (10s intervals)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Plotting] Saved: {output_path}")


def plot_reliability_separation(
    r_corrupted_timeline: np.ndarray,
    r_clean_timeline: np.ndarray,
    corrupted_sensor_name: str = "FIT-101",
    clean_sensor_name: str = "LIT-101",
    output_path: str = "results/plots/reliability/reliability_separation.png",
):
    """Plot 2: Sensor reliability score r_i(t) over time for clean vs corrupted sensors."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4.5))

    steps = np.arange(len(r_corrupted_timeline))
    ax.plot(steps, r_clean_timeline, label=f"Clean Sensor ({clean_sensor_name})", color="#2ca02c", linewidth=1.5)
    ax.plot(steps, r_corrupted_timeline, label=f"Corrupted Sensor ({corrupted_sensor_name})", color="#d62728", linewidth=1.5)

    ax.axhline(0.60, color="#555555", linestyle=":", linewidth=1.3, label="Reliability Fault Cutoff (0.60)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time Step (10s intervals)", fontweight="bold")
    ax.set_ylabel("Reliability Score $r_i(t)$", fontweight="bold")
    ax.set_title(f"Dynamic Sensor Reliability Decoupling: {corrupted_sensor_name} vs {clean_sensor_name}", fontweight="bold")
    ax.legend(loc="lower left", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Plotting] Saved: {output_path}")


def plot_corruption_robustness_curves(
    benchmark_json_path: str = "results/corruption/corruption_benchmark_summary.json",
    output_dir: str = "results/plots/corruption",
):
    """Plots 4, 5, 6: FPR, F1, and Reliability Separation across corruption severities."""
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(benchmark_json_path):
        print(f"[Plotting] Benchmark JSON not found at {benchmark_json_path}. Skipping.")
        return

    with open(benchmark_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Noise Experiment Curves
    noise_exps = [d for d in data if "Noise" in d["experiment"]]
    if noise_exps:
        severities = [d["severity"] for d in noise_exps]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Subplot 1: False Positive Rate (FASR test)
        for m_key, m_label, color, marker in [
            ("iso_forest", "Baseline A (Isolation Forest)", "#7570b3", "o"),
            ("stae", "Baseline B (ST-AE)", "#d95f02", "s"),
            ("rstnet_no_srm", "Baseline C (R-STNet No SRM)", "#e7298a", "^"),
            ("rstnet", "Proposed (R-STNet With SRM)", "#1b9e77", "D"),
        ]:
            fpr_vals = [d["models"][m_key]["fpr_mean"] * 100 for d in noise_exps]
            ax1.plot(severities, fpr_vals, label=m_label, color=color, marker=marker, linewidth=2)

        ax1.set_xlabel("Gaussian Noise Severity ($\\sigma$)", fontweight="bold")
        ax1.set_ylabel("False Positive Rate (%)", fontweight="bold")
        ax1.set_title("System False Alarms Under Sensor Noise\n(Lower is Better)", fontweight="bold")
        ax1.legend(framealpha=0.9)

        # Subplot 2: System Anomaly F1
        for m_key, m_label, color, marker in [
            ("iso_forest", "Baseline A (Isolation Forest)", "#7570b3", "o"),
            ("stae", "Baseline B (ST-AE)", "#d95f02", "s"),
            ("rstnet_no_srm", "Baseline C (R-STNet No SRM)", "#e7298a", "^"),
            ("rstnet", "Proposed (R-STNet With SRM)", "#1b9e77", "D"),
        ]:
            f1_vals = [d["models"][m_key]["f1_mean"] for d in noise_exps]
            ax2.plot(severities, f1_vals, label=m_label, color=color, marker=marker, linewidth=2)

        ax2.set_xlabel("Gaussian Noise Severity ($\\sigma$)", fontweight="bold")
        ax2.set_ylabel("System Anomaly F1 Score", fontweight="bold")
        ax2.set_title("Detection Robustness Under Sensor Noise\n(Higher is Better)", fontweight="bold")
        ax2.legend(framealpha=0.9)

        plt.tight_layout()
        out_noise = os.path.join(output_dir, "robustness_under_gaussian_noise.png")
        plt.savefig(out_noise, dpi=300)
        plt.close()
        print(f"[Plotting] Saved: {out_noise}")

    # 2. Stuck Sensor Curves
    stuck_exps = [d for d in data if "Stuck" in d["experiment"]]
    if stuck_exps:
        k_sensors = [1, 3, 5]
        fig, ax = plt.subplots(figsize=(8, 5))
        for m_key, m_label, color, marker in [
            ("iso_forest", "Baseline A (Isolation Forest)", "#7570b3", "o"),
            ("stae", "Baseline B (ST-AE)", "#d95f02", "s"),
            ("rstnet_no_srm", "Baseline C (R-STNet No SRM)", "#e7298a", "^"),
            ("rstnet", "Proposed (R-STNet With SRM)", "#1b9e77", "D"),
        ]:
            fpr_vals = [d["models"][m_key]["fpr_mean"] * 100 for d in stuck_exps]
            ax.plot(k_sensors, fpr_vals, label=m_label, color=color, marker=marker, linewidth=2)

        ax.set_xticks(k_sensors)
        ax.set_xlabel("Number of Stuck Channels ($k$)", fontweight="bold")
        ax.set_ylabel("False Positive Rate (%)", fontweight="bold")
        ax.set_title("False Alarm Suppression Under Stuck Sensors\n(Lower is Better)", fontweight="bold")
        ax.legend(framealpha=0.9)

        plt.tight_layout()
        out_stuck = os.path.join(output_dir, "robustness_under_stuck_sensors.png")
        plt.savefig(out_stuck, dpi=300)
        plt.close()
        print(f"[Plotting] Saved: {out_stuck}")


def plot_ablation_summary(
    ablation_dict: dict,
    output_path: str = "results/plots/ablations/srm_ablation_comparison.png",
):
    """Plot 8: Ablation comparison of SRM components."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    configs = list(ablation_dict.keys())
    f1_scores = [ablation_dict[c]["f1"] for c in configs]
    fpr_scores = [ablation_dict[c]["fpr"] * 100 for c in configs]

    x = np.arange(len(configs))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax2 = ax1.twinx()

    rects1 = ax1.bar(x - width / 2, f1_scores, width, label="System F1", color="#1b9e77", alpha=0.85)
    rects2 = ax2.bar(x + width / 2, fpr_scores, width, label="FPR (%)", color="#d95f02", alpha=0.85)

    ax1.set_ylabel("System Anomaly F1", color="#1b9e77", fontweight="bold")
    ax2.set_ylabel("False Positive Rate (%)", color="#d95f02", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=25, ha="right", fontweight="bold")
    ax1.set_title("SRM Component Ablation Study Under Sensor Corruption", fontweight="bold")

    # Value labels
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        ax2.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Plotting] Saved: {output_path}")


if __name__ == "__main__":
    print("[Plotting Suite] Helper functions ready for generation upon experiment completion.")
