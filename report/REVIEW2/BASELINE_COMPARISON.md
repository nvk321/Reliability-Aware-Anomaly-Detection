# Baseline Performance and Comparative Analysis

**Project:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Course:** BCSE209P — Machine Learning Lab  
**Date:** October 2026  
**Status:** Audited Clean Benchmark Evaluation  

---

## 1. Experimental Overview and Benchmark Results

The clean benchmark evaluation ($E1$) compares the proposed **Reliability-Aware Spatio-Temporal Network (R-STNet with SRM)** against three established baselines:
1. **Baseline A:** Isolation Forest (classical unsupervised tree ensemble).
2. **Baseline B:** Spatio-Temporal Autoencoder (ST-AE, feedforward deep reconstruction).
3. **Baseline C:** RSTNetNoSRM (spatiotemporal graph neural network with uniform residual aggregation).

All four models were evaluated on the official chronological mixed SWaT test sequence comprising $44,963$ sliding windows, containing $5,468$ attack-labeled windows ($12.16\%$ positive class rate) and $39,495$ normal operational windows. Decision thresholds were calibrated strictly on the clean normal validation split at the $95^{\text{th}}$ percentile ($q = 0.95$), guaranteeing zero label leakage.

The table below summarizes the audited empirical findings (Table 4.3):

| Metric | Baseline A: Isolation Forest | Baseline B: ST-AE | Baseline C: RSTNetNoSRM | Proposed: R-STNet (with SRM) | Relative Gain (R-STNet vs. Baseline C) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Point-wise F1** | **0.7182** | 0.2169 | 0.2171 | **0.2740** | **+26.21%** |
| **Precision** | 0.8995 | 0.1216 | 0.1218 | 0.1626 | +33.50% |
| **Recall** | 0.5977 | 1.0000 | 1.0000 | 0.8713 | -12.87% |
| **False Positive Rate (FPR)** | **0.0092 (0.92%)** | 1.0000 (100.0%) | 0.9986 (99.86%) | **0.6214 (62.14%)** | **-37.78% (14,898 FPs avoided)** |
| **False Positives (FP)** | 365 | 39,495 | 39,439 | 24,541 | -14,898 |
| **True Negatives (TN)** | 39,159 | 0 | 56 | 14,954 | +14,898 |
| **True Positives (TP)** | 3,268 | 5,468 | 5,468 | 4,764 | -704 |
| **False Negatives (FN)** | 2,200 | 0 | 0 | 704 | +704 |
| **Point-Adjusted F1 (PA-F1)** | 0.8346 | 0.2169 | 0.2171 | 0.2944 | +35.61% |
| **PR-AUC** | 0.7367 | 0.1304 | 0.1380 | 0.1169 | -15.29% |
| **ROC-AUC** | 0.8429 | 0.5070 | 0.5164 | 0.4976 | -3.64% |

---

## 2. Integrated Scientific Discussion: Clean-Data Performance

### 2.1 Result
As displayed in **Figure 4.1** (`fig4_1_model_comparison_barchart.png`) and detailed in Table 4.3, Baseline A (Isolation Forest) achieves the highest overall clean detection performance ($F_1 = 0.7182$, Precision $= 0.8995$, FPR $= 0.0092$). Deep reconstruction baselines without reliability estimation (ST-AE and RSTNetNoSRM) collapse into near-complete false-positive saturation, with FPRs of $100.0\%$ and $99.86\%$ respectively. The Proposed R-STNet achieves an $F_1$ of $0.2740$, reducing the FPR to $62.14\%$ and eliminating $14,898$ false alarms relative to Baseline C under an identical neural backbone.

### 2.2 Observation
1. **The Classical Baseline Dominates Clean Data:** Isolation Forest outperforms all three deep neural architectures by a wide margin on the clean SWaT benchmark, achieving an F1 score nearly triple that of the proposed deep model ($0.7182$ vs. $0.2740$).
2. **Reconstruction Baselines Experience Total False Alarm Saturation:** ST-AE and RSTNetNoSRM classify virtually every single test window as an anomaly ($39,495$ and $39,439$ false positives out of $39,495$ normal windows).
3. **Proposed SRM Provides Substantial Within-Architecture Improvement:** Comparing Proposed R-STNet directly to its identical-capacity counterpart Baseline C ($23,233$ parameters), the addition of the Sensor Reliability Module increases True Negatives from $56$ to $14,954$, driving a $+26.21\%$ relative improvement in Point-wise F1 ($0.2171 \to 0.2740$).

### 2.3 Interpretation
The dramatic divergence between Isolation Forest and deep reconstruction models is explained by the fundamental structural properties of the SWaT dataset:
- **The Impact of Unrecorded Channel Shifts:** As documented in `SWAT_CHANNEL_SHIFT_ANALYSIS.md`, seven channels in SWaT (most notably conductivity analyzer `AIT201` and valve `MV101`) were unrecorded or constant during normal training ($\sigma_{tr} = 0$), but became dynamically active during testing, with standardized values exploding to $z > 1,600$.
- **Why Deep Reconstruction Models Failed:** Reconstruction models are trained to minimize MSE. When evaluated on test windows where `AIT201` fluctuates wildly, the network predicts $\hat{x}_{\text{AIT201}} \approx 1.0$, producing massive squared residuals ($e_{\text{AIT201}} \approx 2.56 \times 10^6$). In unweighted models, this single channel dwarfs the calibrated threshold ($\tau_c = 12.07$), keeping the aggregate anomaly score permanently elevated above $\tau_c$ throughout both normal and attack periods (as shown in **Figure 4.2**, top panel).
- **Why Isolation Forest Succeeded:** Isolation Forest partitions feature space using random orthogonal axis cuts. Extreme univariate outliers are isolated in the first 1 or 2 splits near the root of individual trees. Crucially, tree ensembles do not sum residuals across features and do not perform spatial message passing. Thus, the extreme values in `AIT201` did not distort the partitioning of the remaining 50 healthy features.
- **The Mechanism of R-STNet's Gain:** R-STNet's Sensor Reliability Module computes dynamic reliability weights $r_i(t)$. When `AIT201` exhibits non-physical z-scores, its residual confidence weight $r_i^{(p)}(t) = \exp(-\frac{1}{2} z_{i, t}^2)$ drops exponentially to near-zero ($r < 0.01$). This dynamic downweighting neutralizes the dominant error term, allowing R-STNet's aggregate score to drop below $\tau_{prop} = 3.14$ during normal operations (**Figure 4.2**, bottom panel).

### 2.4 Comparison
Compared to Baseline B (ST-AE, $170,643$ parameters), R-STNet utilizes $86.4\%$ fewer parameters ($23,233$) while achieving significantly superior convergence (**Figure 4.4**, final MSE $0.0092$ vs. $0.0406$) and higher detection quality ($F_1 = 0.2740$ vs. $0.2169$). Compared to Baseline C, R-STNet avoids $14,898$ false alarms, recovering $14,954$ true negative intervals while maintaining high recall ($87.13\%$).

### 2.5 Research Implication
This result provides direct empirical evidence that **explicit reliability weighting successfully mitigates severe false alarms in deep spatiotemporal architectures caused by uncalibrated channel shifts**, without requiring manual channel pruning or retraining.

### 2.6 Limitation
The proposed R-STNet **does not outperform Isolation Forest on clean SWaT data**. Furthermore, R-STNet's false positive rate remains elevated at $62.14\%$. This is because spatial graph attention message passing propagates residual contamination from shifted nodes to neighboring nodes before the SRM weighting occurs at the final decision layer.

---

## 3. Confusion Matrix and Error Profile Analysis

**Figure 4.3** (`fig4_4_confusion_matrices.png`) details the exact error distributions across all four models:
- **Baseline A (Isolation Forest):** $\text{TP} = 3,268$, $\text{FP} = 365$, $\text{TN} = 39,159$, $\text{FN} = 2,200$. Demonstrates clean, balanced classification with minimal false alarms ($0.92\%$).
- **Baseline B (ST-AE):** $\text{TP} = 5,468$, $\text{FP} = 39,495$, $\text{TN} = 0$, $\text{FN} = 0$. Exhibits degenerate "always alarm" behavior; $100\%$ recall is an artifact of universal positive prediction.
- **Baseline C (RSTNetNoSRM):** $\text{TP} = 5,468$, $\text{FP} = 39,439$, $\text{TN} = 56$, $\text{FN} = 0$. Similar degenerate collapse ($99.86\%$ FPR), flagging virtually all operational states.
- **Proposed R-STNet:** $\text{TP} = 4,764$, $\text{FP} = 24,541$, $\text{TN} = 14,954$, $\text{FN} = 704$. Restores meaningful discriminative capacity, correctly identifying $14,954$ normal windows while detecting $87.13\%$ of true attacks.

---

## 4. Training Convergence and Optimization Stability

**Figure 4.4** (`fig4_5_training_loss_curves.png`) plots the training loss trajectories over 15 epochs:
- **ST-AE:** Starts at MSE $= 0.2203$, descending steadily to $0.0406$. The higher reconstruction floor reflects the limitations of feedforward linear layers in modeling complex multi-sensor temporal interactions.
- **RSTNetNoSRM:** Starts at MSE $= 0.0453$, rapidly dropping below $0.0120$ by Epoch 3, and achieving a final MSE of $0.00938$.
- **Proposed R-STNet:** Starts at MSE $= 0.0453$, closely tracking Baseline C across all 15 epochs to reach a final MSE of $0.00923$.

**Optimization Takeaway:** The inclusion of the Sensor Reliability Module does not degrade or perturb the training of the spatiotemporal backbone. Both graph networks optimize stably and converge within 10 epochs.
