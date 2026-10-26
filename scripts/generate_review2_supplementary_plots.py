"""
Generate Supplementary Figures for Review 2 Chapter 4.

Produces:
- Fig 4.1: Overall Model Comparison Bar Chart (F1, Precision, Recall, PR-AUC)
- Fig 4.4: Confusion Matrix Comparison across all 4 Models
- Fig 4.5: Training Loss Convergence Curves across Epochs
- Fig 4.12: SWaT Channel Shift and Failure Case Diagnostics (AIT201 Distribution Shift and SRM Response)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["grid.linestyle"] = "--"

fig_dir = "report/REVIEW2/figures"
os.makedirs(f"{fig_dir}/generated", exist_ok=True)

# -------------------------------------------------------------
# FIGURE 4.1: Overall Model Comparison Bar Chart
# -------------------------------------------------------------
print("Generating Fig 4.1: Overall Model Comparison Bar Chart...")
models = ["Baseline A\n(Isolation Forest)", "Baseline B\n(ST-AE)", "Baseline C\n(RSTNet No SRM)", "Proposed\n(R-STNet With SRM)"]
f1_scores = [0.7182, 0.2169, 0.2171, 0.2740]
precision_scores = [0.8995, 0.1216, 0.1218, 0.1626]
recall_scores = [0.5977, 1.0000, 1.0000, 0.8713]
pr_auc_scores = [0.7367, 0.1216, 0.1380, 0.1169]

x = np.arange(len(models))
width = 0.20

fig, ax = plt.subplots(figsize=(11, 5.5))
r1 = ax.bar(x - 1.5 * width, f1_scores, width, label="Point-wise F1 (Primary)", color="#2b5c8f")
r2 = ax.bar(x - 0.5 * width, precision_scores, width, label="Precision", color="#41b6c4")
r3 = ax.bar(x + 0.5 * width, recall_scores, width, label="Recall", color="#fd8d3c")
r4 = ax.bar(x + 1.5 * width, pr_auc_scores, width, label="PR-AUC", color="#e7298a")

ax.set_ylabel("Metric Value", fontweight="bold")
ax.set_title("Benchmark Comparison Across All Four Models on SWaT Test Sequence", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models, fontweight="bold")
ax.set_ylim(0, 1.15)
ax.legend(loc="upper right", framealpha=0.9)

for r in [r1, r2, r3, r4]:
    for rect in r:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(f"{fig_dir}/generated/fig4_1_model_comparison_barchart.png", dpi=300)
plt.savefig(f"{fig_dir}/fig4_1_model_comparison_barchart.png", dpi=300)
plt.close()

# -------------------------------------------------------------
# FIGURE 4.4: Confusion Matrix Comparison Across Models
# -------------------------------------------------------------
print("Generating Fig 4.4: Confusion Matrix Subplots...")
cm_data = [
    ("Baseline A (Isolation Forest)", 3268, 365, 39159, 2200),
    ("Baseline B (ST-AE)", 5468, 39495, 0, 0),
    ("Baseline C (RSTNet No SRM)", 5468, 39439, 56, 0),
    ("Proposed R-STNet (With SRM)", 4764, 24541, 14954, 704),
]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for idx, (title, tp, fp, tn, fn) in enumerate(cm_data):
    ax = axes[idx]
    mat = np.array([[tn, fp], [fn, tp]])
    im = ax.imshow(mat, cmap="Blues", interpolation="nearest")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Norm", "Pred Atk"])
    ax.set_yticklabels(["True Norm", "True Atk"])
    for i in range(2):
        for j in range(2):
            val = mat[i, j]
            color = "white" if val > 20000 else "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontweight="bold", fontsize=10)

plt.tight_layout()
plt.savefig(f"{fig_dir}/generated/fig4_4_confusion_matrices.png", dpi=300)
plt.savefig(f"{fig_dir}/fig4_4_confusion_matrices.png", dpi=300)
plt.close()

# -------------------------------------------------------------
# FIGURE 4.5: Training Loss Convergence
# -------------------------------------------------------------
print("Generating Fig 4.5: Training Loss Convergence...")
with open("results/metrics/baseline_stae_swat.json") as f:
    stae_loss = json.load(f)["training_convergence"]["epoch_losses"]
with open("results/metrics/baseline_rstnet_no_srm_swat.json") as f:
    no_srm_loss = json.load(f)["training_convergence"]["epoch_losses"]
with open("results/metrics/proposed_rstnet_swat.json") as f:
    rstnet_loss = json.load(f)["training_convergence"]["epoch_losses"]

epochs = np.arange(1, 16)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epochs, stae_loss, label="Baseline B (ST-AE)", color="#d95f02", marker="s", linewidth=1.8)
ax.plot(epochs, no_srm_loss, label="Baseline C (RSTNet No SRM)", color="#e7298a", marker="^", linewidth=1.8)
ax.plot(epochs, rstnet_loss, label="Proposed R-STNet (With SRM)", color="#1b9e77", marker="D", linewidth=1.8)
ax.set_xlabel("Training Epoch", fontweight="bold")
ax.set_ylabel("Training MSE Loss", fontweight="bold")
ax.set_title("Training Loss Convergence Across 15 Epochs on SWaT Normal Data", fontweight="bold")
ax.set_xticks(epochs)
ax.legend(framealpha=0.9)
plt.tight_layout()
plt.savefig(f"{fig_dir}/generated/fig4_5_training_loss_curves.png", dpi=300)
plt.savefig(f"{fig_dir}/fig4_5_training_loss_curves.png", dpi=300)
plt.close()

# -------------------------------------------------------------
# FIGURE 4.12: SWaT Channel Shift Diagnostic (AIT201)
# -------------------------------------------------------------
print("Generating Fig 4.12: Channel Shift Diagnostic (AIT201)...")
train_data = np.load("data/processed/SWaT/train.npy")
test_data = np.load("data/processed/SWaT/test.npy")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=False)

# Subplot 1: Train vs Test AIT201
ax1.plot(train_data[:3000, 5], label="Normal Training (AIT201 is Constant / Unrecorded, σ ≈ 0)", color="#2b5c8f", linewidth=1.2)
ax1.set_ylabel("Standardized Value", fontweight="bold")
ax1.set_xlabel("Time Step (First 3,000 Normal Training Windows)", fontweight="bold")
ax1.set_title("Operational Reality of SWaT Channel AIT201: Training Period vs Test Period", fontweight="bold")
ax1.legend(loc="upper right")

ax2.plot(test_data[:3000, 5], label="Test Period (AIT201 Activates Dynamically, Values Explode to z > 1,600)", color="#d95f02", linewidth=1.2)
ax2.set_ylabel("Standardized Value", fontweight="bold")
ax2.set_xlabel("Time Step (First 3,000 Test Windows)", fontweight="bold")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig(f"{fig_dir}/generated/fig4_12_channel_shift_diagnostic.png", dpi=300)
plt.savefig(f"{fig_dir}/fig4_12_channel_shift_diagnostic.png", dpi=300)
plt.close()

print("[Supplementary Figures] ALL FIGURES GENERATED SUCCESSFULLY!")
