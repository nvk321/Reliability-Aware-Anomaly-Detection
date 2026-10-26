# REVIEW 2 MASTER EXPERIMENTAL DOSSIER

**Project Title:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Students:** P Naga Venkata Kushal (23BDS0133) & Koteru Tarun (23BDS0103)  
**Course:** BCSE209P — Machine Learning Lab  
**Faculty Guide:** Dr. Dega Nagaraju  
**Date of Review 2 Completion:** October 2026  
**Status:** Validated, Replicated, Zero-Leakage Scientific Deliverable  

---

## 1. Project Status

The project has achieved its Phase 2 experimental milestone. All four core model architectures have been implemented, trained, benchmarked on clean data, evaluated across multi-seed controlled corruption suites, audited for data leakage, and analyzed across a 5-condition ablation study. Chapter 4 (*Results and Discussion*) has been authored with complete integrated scientific discussions, 11 detailed empirical tables, and 12 publication-grade figures.

---

## 2. Experimental Protocol

- **Dataset:** Secure Water Treatment (SWaT A1/A2, December 2015 testbed) monitoring 51 physical variables (25 continuous sensors, 26 discrete actuators).
- **Partitioning:** Chronological split. Phase A1 (7 days normal) divided into 80% train ($39,715$ windows) and 20% validation ($9,907$ windows). Phase A2 (4 days attack) utilized as chronological mixed test sequence ($44,963$ windows, $5,468$ attack labels, $12.16\%$ prevalence).
- **Preprocessing:** $10\times$ downsampling ($10$-second interval step), sliding window $L = 30$ ($300$ s), stride $S = 1$, causal `point_end` labeling.
- **Normalization:** `RobustStandardScaler` fit strictly on training normal data with zero-variance protection ($\epsilon = 10^{-6}$, scale set to $1.0$).
- **Threshold Calibration ($\tau$):** Calibrated strictly at the $95^{\text{th}}$ percentile ($q = 0.95$) of normal validation scores. Zero test-label leakage.

---

## 3. Dataset Characteristics

| Dataset Metric | Official Raw Value | Processed / Downsampled ($10\times$) Value |
| :--- | :---: | :---: |
| **Total Normal Commissioning Rows** | $496,800$ | $49,680$ time steps ($39,715$ train windows, $9,907$ val windows) |
| **Total Cyber-Physical Attack Rows** | $449,919$ | $44,992$ time steps ($44,963$ test windows) |
| **Attack-Labeled Windows** | $54,621$ | $5,468$ windows ($12.16\%$ positive class prevalence) |
| **Monitored Telemetry Variables** | $51$ | $51$ channels across 6 physical water treatment stages |

---

## 4. Model Comparison Compendium

| Architectural Specification | Baseline A: Isolation Forest | Baseline B: ST-AE | Baseline C: RSTNetNoSRM | Proposed: R-STNet (with SRM) |
| :--- | :--- | :--- | :--- | :--- |
| **Model Family** | Classical Tree Ensemble | Feedforward Autoencoder | Spatiotemporal Graph Network | Reliability-Aware Graph Network |
| **Trainable Parameters** | Non-parametric | $170,643$ | $\mathbf{23,233}$ | $\mathbf{23,233}$ |
| **Core Backbone** | 100 Isolation Trees | 1D Conv Encoder + Linear Projection | Dilated TCN + Spatial Multi-Head GAT | Dilated TCN + Spatial GAT + SRM |
| **Scoring Formulation** | Path length $s(x, n)$ | MSE: $\frac{1}{M}\sum (x - \hat{x})^2$ | Uniform residual: $\frac{1}{M}\sum (\|x-\hat{x}\|/\text{MAD})$ | Weighted residual: $\frac{\sum r_i e_i}{\sum r_i}$ |
| **Training Time** | $\mathbf{12.40\text{ s}}$ | $112.35\text{ s}$ | $112.89\text{ s}$ | $134.73\text{ s}$ |
| **Inference Latency** | $\mathbf{0.021\text{ ms/win}}$ | $0.076\text{ ms/win}$ | $0.050\text{ ms/win}$ | $\mathbf{0.060\text{ ms/win}}$ |
| **Peak Allocated VRAM** | N/A (Host RAM) | $\mathbf{108.49\text{ MB}}$ | $978.69\text{ MB}$ | $978.69\text{ MB}$ |

---

## 5. Clean Benchmark Performance Results (Table 4.3)

* **Baseline A (Isolation Forest):** $\text{Point-wise } F_1 = \mathbf{0.7182}$, $\text{Precision} = \mathbf{0.8995}$, $\text{Recall} = 0.5977$, $\text{FPR} = \mathbf{0.0092}$ ($0.92\%$). $\text{TP} = 3,268$, $\text{FP} = 365$, $\text{TN} = 39,159$, $\text{FN} = 2,200$. $\text{PR-AUC} = 0.7367$.
* **Baseline B (ST-AE):** $\text{Point-wise } F_1 = 0.2169$, $\text{Precision} = 0.1216$, $\text{Recall} = 1.0000$, $\text{FPR} = 1.0000$ ($100.0\%$). $\text{TP} = 5,468$, $\text{FP} = 39,495$, $\text{TN} = 0$, $\text{FN} = 0$. $\text{PR-AUC} = 0.1304$.
* **Baseline C (RSTNetNoSRM):** $\text{Point-wise } F_1 = 0.2171$, $\text{Precision} = 0.1218$, $\text{Recall} = 1.0000$, $\text{FPR} = 0.9986$ ($99.86\%$). $\text{TP} = 5,468$, $\text{FP} = 39,439$, $\text{TN} = 56$, $\text{FN} = 0$. $\text{PR-AUC} = 0.1380$.
* **Proposed R-STNet (with SRM):** $\text{Point-wise } F_1 = \mathbf{0.2740}$, $\text{Precision} = \mathbf{0.1626}$, $\text{Recall} = 0.8713$, $\text{FPR} = \mathbf{0.6214}$ ($62.14\%$). $\text{TP} = 4,764$, $\text{FP} = 24,541$, $\text{TN} = 14,954$, $\text{FN} = 704$. $\text{PR-AUC} = 0.1169$.
* **Core Takeaway:** Isolation Forest dominates clean data due to tree partition robustness against unscaled channel shifts. Within deep neural architectures, R-STNet achieves a **$+26.21\%$ F1 gain** and **eliminates $14,898$ false alarms** relative to Baseline C.

---

## 6. Multi-Seed Robustness Evaluation Results (Tables 4.4, 4.5, 4.6)

Evaluated across random seeds $[42, 101, 2024]$:
- **Additive Gaussian Noise ($E2$):** R-STNet maintains $F_1 = 0.2755 \pm 0.0011$ ($\sigma = 0.1$), $0.2764 \pm 0.0005$ ($\sigma = 0.3$), and $0.2762 \pm 0.0001$ ($\sigma = 0.5$). False alarm rate remains stably bounded between $64.09\%$ and $65.08\%$. Baseline C remains trapped at $FPR = 99.86\%$.
- **Stuck Sensors ($E3$):** R-STNet maintains $F_1 = 0.2740 \pm 0.0000$ ($k=1$), $0.2708 \pm 0.0045$ ($k=3$), and $0.2737 \pm 0.0006$ ($k=5$). Fault recall is $0.0\%$ on inactive actuators because natural variance is zero.
- **Dynamic Sensor Drift ($E4$):** R-STNet maintains $F_1 = 0.2739 \pm 0.0001$ ($\delta = 0.1$), $0.2732 \pm 0.0005$ ($\delta = 0.3$), and $0.2729 \pm 0.0007$ ($\delta = 0.5$). FPR remains flat at $62.14\% - 62.29\%$.

---

## 7. Sensor Reliability Decoupling Findings (Table 4.7)

- **Clean Channel Average Reliability:** $\bar{r}_{\text{clean}} = 0.3041 \pm 0.010$.
- **Corrupted Channel Average Reliability:** $\bar{r}_{\text{corrupt}} = 0.0151 \pm 0.002$ (Noise) and $0.0092 \pm 0.001$ (Drift).
- **Empirical Separation Gap:** $\mathbf{\Delta r = +0.2856\text{ to }+0.2946}$.
- **Fault Identification Recall:** $\mathbf{100.0\% \pm 0.0\%}$ against synthetic fault masks with **$0.00\%$ false alarms on clean channels** ($r_i < 0.60$ cutoff).

---

## 8. Ablation Study Findings (Table 4.8)

- **Ablation 1 (No SRM):** $F_1 = 0.2171$, $FPR = 99.86\%$.
- **Ablation 2 (Temporal Only, $w_v=1$):** $F_1 = 0.2170$, $FPR = 99.60\%$.
- **Ablation 3 (Graph Only, $w_c=1.2$):** $F_1 = 0.2163$, $FPR = 99.57\%$.
- **Ablation 4 (Residual Only, $w_p=1$):** $F_1 = 0.2184$, $FPR = 92.67\%$.
- **Ablation 5 (Full SRM):** $\mathbf{F_1 = 0.2740}$, $\mathbf{FPR = 62.14\%}$.
- **Conclusion:** Single perspectives fail to provide substantial gains. Full multi-perspective synergy is required to achieve meaningful false alarm reduction.

---

## 9. Failure Case Summary (Table 4.10)

1. *Case 1 (AIT201 Shift):* Baseline C triggers massive false alarm ($S_c = 1,482$); R-STNet downweights `AIT201` ($r < 0.005$) and preserves True Negative ($S_{prop} = 2.14 < 3.14$).
2. *Case 2 (Noise Suppression):* Synthetic noise triggers false alarm in Baseline C ($S_c = 45.2$); R-STNet suppresses noisy channels, maintaining True Negative ($S_{prop} = 2.89$).
3. *Case 3 (Stuck Actuator Boundary):* Backup pumps frozen at zero. Missed by statistical module ($0\%$ fault recall) because natural variance is zero.
4. *Case 4 (Genuine Attack Detection):* Coordinated tank overflow attack detected by both models ($S_{prop} = 48.7 \gg 3.14$).
5. *Case 5 (Stealthy Attack Masking):* Subtle flow spoofing on `FIT401` downweighted as noise, producing a False Negative ($S_{prop} = 2.91 < 3.14$). Highlights the dual-edged risk of reliability downweighting.

---

## 10. Computational Analysis Summary (Table 4.9)

- Parameter Compactness: R-STNet ($23,233$ params) achieves an $86.4\%$ reduction compared to ST-AE ($170,643$).
- Inference Latency: $0.060\text{ ms}$ per window ($16,667$ windows/s), operating $160,000\times$ faster than the $10,000\text{ ms}$ telemetry arrival step.
- GPU VRAM: Peak allocated VRAM of $978.69\text{ MB}$ consumes only $27.4\%$ of the 4 GB VRAM budget.

---

## 11. SWaT Channel-Shift Analysis Summary

- Seven channels contain empty strings or constant values during normal commissioning (`MV101`, `AIT201`, `MV201`, `P201`, `P202`, `P204`, `MV303`).
- Chemical analyzer `AIT201` was unrecorded during normal training ($\sigma_{tr} = 0$), but surged to standardized $z > 1,600$ during testing.
- This shift caused uniform reconstruction models to collapse into $99.86\%$ FPR.
- R-STNet's SRM dynamically suppresses `AIT201`, eliminating $14,898$ false alarms.

---

## 12. Critical Limitations and Boundaries

1. R-STNet does not outperform Isolation Forest on clean SWaT ($F_1 = 0.2740$ vs. $0.7182$).
2. Residual False Positive Rate remains at $62.14\%$ due to upstream spatial GAT contamination.
3. Stuck sensors on inactive actuators cannot be diagnosed without control signals ($0.0\%$ recall).
4. Subtle single-sensor attacks risk being masked by aggressive reliability downweighting.

---

## 13. Research Question Assessment

> **Hypothesis:** Can explicit sensor-reliability estimation improve the robustness of multivariate time-series anomaly detection when sensor observations are missing, noisy, or corrupted?

**Balanced Scientific Assessment:**  
The experiments provide preliminary empirical evidence that explicit reliability weighting improves detection robustness and significantly reduces false alarms **relative to an otherwise identical spatiotemporal deep learning architecture without reliability estimation** (eliminating $14,898$ false alarms and establishing $+0.29$ reliability separation). However, the proposed model **did not outperform the classical Isolation Forest baseline on the clean SWaT benchmark**, and statistical reliability estimation cannot diagnose stuck states on zero-variance actuators without supervisory control telemetry. Therefore, the hypothesis is **supported within deep spatiotemporal neural architectures**, but must not be generalized as a claim of universal superiority across all anomaly detection frameworks.

---

## 14. Master Figure Index

- **Figure 4.1:** Clean SWaT Benchmark Model Performance Comparison (`fig4_1_model_comparison_barchart.png`)
- **Figure 4.2:** System Anomaly Score Timeline Across Cyber-Physical Attack Intervals (`system_anomaly_scores_timeline.png`)
- **Figure 4.3:** Confusion Matrix Analysis Across Evaluated Architectures (`fig4_4_confusion_matrices.png`)
- **Figure 4.4:** Deep Architecture Training Loss Convergence Curves (`fig4_5_training_loss_curves.png`)
- **Figure 4.5:** Dynamic Sensor Reliability Trajectories Over Operational Intervals (`reliability_score_vs_time.png`)
- **Figure 4.6:** Dynamic Decoupling: Corrupted vs. Clean Sensor Reliability Trajectories (`corrupted_vs_clean_reliability.png`)
- **Figure 4.7:** Detection Robustness Under Increasing Gaussian Noise Severity (`f1_vs_corruption_severity.png`)
- **Figure 4.8:** False Alarm Suppression Across Corruption Severities (`fpr_vs_corruption_severity.png`)
- **Figure 4.9:** Side-by-Side Anomaly Score Tracking Under Sensor Corruption (`baseline_c_vs_rstnet_under_corruption.png`)
- **Figure 4.10:** Sensor Fault Identification Performance Across Corruption Severities (`sensor_fault_id_f1_vs_severity.png`)
- **Figure 4.11:** Sensor Reliability Module (SRM) Perspective Ablation Comparison (`srm_ablation_comparison.png`)
- **Figure 4.12:** SWaT Channel Shift Diagnostic: The Case of Analyzer AIT201 (`fig4_12_channel_shift_diagnostic.png`)

---

## 15. Master Table Index

- **Table 4.1:** Experimental Configuration and Environmental Parameters
- **Table 4.2:** Model Architecture and Computational Specifications
- **Table 4.3:** Clean SWaT Benchmark Performance Comparison
- **Table 4.4:** Robustness Under Additive Gaussian Noise Corruption ($E2$)
- **Table 4.5:** Robustness Under Stuck-Sensor Corruption ($E3$)
- **Table 4.6:** Robustness Under Dynamic Sensor Drift Corruption ($E4$)
- **Table 4.7:** Sensor Reliability Estimation and Fault Localization Metrics
- **Table 4.8:** Sensor Reliability Module (SRM) Perspective Ablation Study
- **Table 4.9:** Computational Resource Utilization and Execution Benchmarks
- **Table 4.10:** Empirical Failure Cases and Anomaly Score Diagnostic Matrix
- **Table 4.11:** Research-Hypothesis Empirical Evidence Synthesis

---

## 16. Reproducibility and Checkpoints

All models, data, and scripts are certified reproducible:
- Master Guide: `report/REVIEW2/REPRODUCIBILITY.md`
- Checkpoints: `results/checkpoints/` (`baseline_stae_swat.pt`, `baseline_rstnet_no_srm_swat.pt`, `proposed_rstnet_swat.pt`)
- Metric Artifacts: `report/REVIEW2/metrics/` and `report/REVIEW2/source_results/`

---

## 17. Artifact Index and Manifest

Full provenance manifest linking all copied and original files is provided in:
`report/REVIEW2/RAW_RESULTS_INDEX.md`
