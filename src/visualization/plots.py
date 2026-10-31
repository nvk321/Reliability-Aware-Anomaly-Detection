"""
Reusable Publication-Quality Plotting Utilities for Reliability-Aware Anomaly Detection.

Provides standardized plotting functions for:
1. Time-series anomaly score timelines with ground-truth attack regions.
2. Per-sensor reliability distributions and multi-perspective SRM breakdowns.
3. Confusion matrices for binary detection and multi-state operational classification.
4. Robustness profiles (F1 and FPR) under controlled sensor corruption modes.
5. Component ablation bar charts.
"""

from typing import Dict, List, Optional, Tuple, Any
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def apply_scientific_style():
    """Applies standardized, clean matplotlib style for academic figures."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.linewidth"] = 1.1
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["grid.linestyle"] = "--"


def plot_anomaly_timeline(
    scores: np.ndarray,
    threshold: float,
    y_true: Optional[np.ndarray] = None,
    title: str = "System Anomaly Score Timeline",
    model_label: str = "Anomaly Score",
    start_step: int = 0,
    figsize: Tuple[int, int] = (12, 4),
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots continuous anomaly scores against calibrated decision threshold,
    with true attack intervals shaded in light red.
    """
    apply_scientific_style()
    fig, ax = plt.subplots(figsize=figsize)

    time_steps = np.arange(start_step, start_step + len(scores))

    # Highlight attack regions if labels provided
    if y_true is not None:
        y_slice = y_true[:len(scores)]
        in_atk = False
        start_idx = 0
        intervals = []
        for i in range(len(y_slice)):
            if y_slice[i] == 1 and not in_atk:
                in_atk = True
                start_idx = i
            elif y_slice[i] == 0 and in_atk:
                in_atk = False
                intervals.append((start_idx, i))
        if in_atk:
            intervals.append((start_idx, len(y_slice)))

        for idx, (s, e) in enumerate(intervals):
            ax.axvspan(
                time_steps[s],
                time_steps[min(e, len(time_steps) - 1)],
                color="#EF4444",
                alpha=0.22,
                label="Ground-Truth Attack" if idx == 0 else "",
            )

    ax.plot(time_steps, scores, color="#1D4ED8", linewidth=1.3, label=model_label)
    ax.axhline(threshold, color="#DC2626", linestyle="--", linewidth=1.5, label=f"Threshold $\\tau$ ({threshold:.3f})")

    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_xlabel("Time Step (10s intervals)", fontweight="bold")
    ax.set_ylabel("Anomaly Score $S(t)$", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300)

    return fig


def plot_reliability_breakdown(
    sensor_names: List[str],
    reliability_scores: np.ndarray,
    components: Optional[Dict[str, np.ndarray]] = None,
    threshold: float = 0.60,
    title: str = "Per-Sensor Reliability Scores",
    figsize: Tuple[int, int] = (14, 5),
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots per-channel reliability scores with color-coded confidence levels.
    """
    apply_scientific_style()
    fig, ax = plt.subplots(figsize=figsize)

    colors = [
        "#10B981" if r >= threshold else ("#F59E0B" if r >= 0.30 else "#EF4444")
        for r in reliability_scores
    ]

    x = np.arange(len(sensor_names))
    ax.bar(x, reliability_scores, color=colors, width=0.65, edgecolor="none")
    ax.axhline(threshold, color="#475569", linestyle="--", linewidth=1.2, label=f"Reliability Cutoff ($\\theta_{{rel}} = {threshold:.2f}$)")

    ax.set_xticks(x)
    ax.set_xticklabels(sensor_names, rotation=70, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_xlabel("Sensor Channel", fontweight="bold")
    ax.set_ylabel("Reliability $r_i(t) \\in [0, 1]$", fontweight="bold")

    green_patch = mpatches.Patch(color="#10B981", label="Reliable ($r_i \\ge 0.60$)")
    yellow_patch = mpatches.Patch(color="#F59E0B", label="Moderate ($0.30 \\le r_i < 0.60$)")
    red_patch = mpatches.Patch(color="#EF4444", label="Degraded / Downweighted ($r_i < 0.30$)")
    ax.legend(handles=[green_patch, yellow_patch, red_patch], loc="upper right")
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300)

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = ["Normal", "Attack"],
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (5, 4),
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots formatted confusion matrix with raw counts and normalized percentages.
    """
    from sklearn.metrics import confusion_matrix
    apply_scientific_style()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}\n({cm_norm[i, j]*100:.1f}%)",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold",
                fontsize=9,
            )

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300)

    return fig
