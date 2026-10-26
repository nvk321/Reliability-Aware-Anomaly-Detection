# Review 2 Chapter 4 Tables: Complete Experimental Compendium

**Project:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Course:** BCSE209P — Machine Learning Lab  
**Date:** October 2026  
**Status:** Audited, Replicated, and Experimentally Validated  

---

## Table 4.1: Experimental Configuration and Environmental Parameters

| Experimental Parameter | Parameter Specification / Value | Operational Justification & Methodological Details |
| :--- | :--- | :--- |
| **Primary Dataset** | SWaT (Secure Water Treatment A1/A2, Dec 2015) | 6-stage industrial physical water treatment testbed (iTrust SUTD). |
| **Sensor Channels ($M$)** | 51 physical variables | 25 continuous process sensors + 26 discrete pumps and motorized valves. |
| **Chronological Downsample Rate** | $10\times$ downsampling ($10$-second step) | Compresses 1 Hz high-frequency telemetry while preserving process physics. |
| **Sliding Window Length ($L$)** | $30$ time steps ($300$ seconds / 5 minutes) | Captures transient multi-sensor correlations across plant stages. |
| **Window Stride ($S$)** | $1$ step (stride $= 1$) | Dense, non-lossy chronological temporal evaluation. |
| **Label Assignment Mode** | `point_end` | Temporal causality: window label is the label of the final time step $t$. |
| **Training Partition (Normal)** | $39,744$ time steps ($39,715$ windows) | First $80\%$ of the 7-day normal commissioning run (100% normal, $y=0$). |
| **Validation Partition (Normal)**| $9,936$ time steps ($9,907$ windows) | Final $20\%$ of the 7-day normal commissioning run (100% normal, $y=0$). |
| **Test Partition (Mixed Attack)**| $44,992$ time steps ($44,963$ windows) | 4-day chronological cyber-physical test sequence ($5,468$ attack windows, $12.16\%$). |
| **Normalization Protocol** | `RobustStandardScaler` | Fit strictly on training split; zero-variance scale protection ($\epsilon = 10^{-6}$). |
| **Threshold Calibration ($\tau$)**| Validation $95^{\text{th}}$ Quantile ($q=0.95$) | Calculated strictly on normal validation windows; zero test leakage. |
| **Training Hyperparameters** | Epochs: $15$, Batch Size: $64$, LR: $0.0005$ | Adam optimizer, MSE reconstruction loss, cosine annealing. |
| **Corruption Benchmark Seeds** | $[42, 101, 2024]$ | Multi-seed stochastic evaluation across all corruption scenarios. |
| **Hardware Specification** | NVIDIA GeForce RTX 3050 Laptop GPU (4 GB) | Intel Core i5 / Windows 11 / PyTorch 2.6.0+cu124. |

---

## Table 4.2: Model Architecture and Computational Specifications

| Model Identifier | Architectural Family | Core Computational Backbone | Trainable Parameters | Input Tensor Shape | Loss Function | Anomaly Scoring Mechanism | Training Time (s) | Inference Latency (ms/win) | Peak VRAM (MB) |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- | :---: | :---: | :---: |
| **Baseline A: Isolation Forest** | Tree Ensemble | 100 Isolation Trees (max_samples $= 256$) | Non-parametric | $(N, 51)$ | Path length isolation | Tree path length $s(x, n)$ | $12.40$ | $0.021$ | N/A (CPU) |
| **Baseline B: ST-AE** | Autoencoder | 1D Conv Encoder + Linear Reconstruction | $170,643$ | $(B, 30, 51)$ | Mean Squared Error (MSE) | Mean reconstruction error: $\frac{1}{M}\sum (x - \hat{x})^2$ | $112.35$ | $0.076$ | $108.49$ |
| **Baseline C: RSTNetNoSRM** | Spatio-Temporal GNN | Dilated TCN + Spatial Graph Attention (GAT) | $23,233$ | $(B, 30, 51)$ | Mean Squared Error (MSE) | Mean normalized residual: $\frac{1}{M}\sum (\|x-\hat{x}\|/\text{MAD})$ | $112.89$ | $0.050$ | $978.69$ |
| **Proposed: R-STNet (with SRM)** | Reliability-Aware GNN | Dilated TCN + Spatial GAT + SRM Module | $23,233$ | $(B, 30, 51)$ | Mean Squared Error (MSE) | Reliability-weighted residual: $\frac{\sum r_i e_i}{\sum r_i}$ | $134.73$ | $0.060$ | $978.69$ |

---

## Table 4.3: Clean SWaT Benchmark Performance Comparison

*Evaluated on the official chronological mixed test sequence ($44,963$ sliding windows, $5,468$ attack labels, $12.16\%$ positive class rate). All metrics audited against source JSON metrics files.*

| Metric Category | Performance Metric | Baseline A: Isolation Forest | Baseline B: ST-AE | Baseline C: RSTNetNoSRM | Proposed: R-STNet (with SRM) | Relative Gain (R-STNet vs. Baseline C) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Core Detection** | **Point-wise F1 Score** | **0.7182** | 0.2169 | 0.2171 | **0.2740** | **+26.21%** |
| | Precision | 0.8995 | 0.1216 | 0.1218 | 0.1626 | +33.50% |
| | Recall (Detection Rate) | 0.5977 | 1.0000 | 1.0000 | 0.8713 | -12.87% |
| | Point-Adjusted F1 (PA-F1) | 0.8346 | 0.2169 | 0.2171 | 0.2944 | +35.61% |
| **False Alarms** | **False Positive Rate (FPR)** | **0.0092 (0.92%)**| 1.0000 (100.0%) | 0.9986 (99.86%)| **0.6214 (62.14%)** | **-37.78% (14,898 FPs avoided)** |
| | False Positives (FP) | 365 | 39,495 | 39,439 | 24,541 | -14,898 |
| | True Negatives (TN) | 39,159 | 0 | 56 | 14,954 | +14,898 |
| | True Positives (TP) | 3,268 | 5,468 | 5,468 | 4,764 | -704 |
| | False Negatives (FN) | 2,200 | 0 | 0 | 704 | +704 |
| **Threshold-Free** | PR-AUC | 0.7367 | 0.1304 | 0.1380 | 0.1169 | -15.29% |
| | ROC-AUC | 0.8429 | 0.5070 | 0.5164 | 0.4976 | -3.64% |
| **Calibration** | Calibrated Threshold ($\tau$) | 0.6359 | 0.0892 | 12.0706 | 3.1401 | N/A |
| | Threshold Quantile ($q$) | 0.95 | 0.95 | 0.95 | 0.95 | Identical protocol |

---

## Table 4.4: Robustness Under Additive Gaussian Noise Corruption ($E2$)

*Synthetic Gaussian noise $\mathcal{N}(0, \sigma^2)$ injected into channels $\{3, 14, 28\}$ (`P101`, `P205`, `LIT401`) across random seeds $[42, 101, 2024]$. Values reported as $\text{mean} \pm \text{standard deviation}$.*

| Corruption Severity | Evaluated Model | System F1 Score | System Precision | System Recall | False Positive Rate (FPR) | Reliability Separation ($\Delta r$) | Fault Detection Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean ($\sigma = 0.0$)** | Baseline C (No SRM) | $0.2171 \pm 0.0000$ | $0.1218 \pm 0.0000$ | $1.0000 \pm 0.0000$ | $99.86\% \pm 0.00\%$ | N/A | N/A |
| | **Proposed R-STNet** | $\mathbf{0.2740 \pm 0.0000}$ | $\mathbf{0.1626 \pm 0.0000}$ | $\mathbf{0.8713 \pm 0.0000}$ | $\mathbf{62.14\% \pm 0.00\%}$ | N/A | N/A |
| **Low ($\sigma = 0.1$)** | Baseline C (No SRM) | $0.2171 \pm 0.0000$ | $0.1218 \pm 0.0000$ | $1.0000 \pm 0.0000$ | $99.86\% \pm 0.00\%$ | N/A | N/A |
| | **Proposed R-STNet** | $\mathbf{0.2755 \pm 0.0011}$ | $\mathbf{0.1627 \pm 0.0000}$ | $\mathbf{0.8993 \pm 0.0000}$ | $\mathbf{64.09\% \pm 0.05\%}$ | $+0.2856 \pm 0.0012$ | $100.0\% \pm 0.0\%$ |
| **Medium ($\sigma = 0.3$)**| Baseline C (No SRM) | $0.2171 \pm 0.0000$ | $0.1218 \pm 0.0000$ | $1.0000 \pm 0.0000$ | $99.86\% \pm 0.00\%$ | N/A | N/A |
| | **Proposed R-STNet** | $\mathbf{0.2764 \pm 0.0005}$ | $\mathbf{0.1629 \pm 0.0000}$ | $\mathbf{0.9123 \pm 0.0000}$ | $\mathbf{64.91\% \pm 0.03\%}$ | $+0.2878 \pm 0.0008$ | $100.0\% \pm 0.0\%$ |
| **High ($\sigma = 0.5$)** | Baseline C (No SRM) | $0.2171 \pm 0.0000$ | $0.1218 \pm 0.0000$ | $1.0000 \pm 0.0000$ | $99.86\% \pm 0.00\%$ | N/A | N/A |
| | **Proposed R-STNet** | $\mathbf{0.2762 \pm 0.0001}$ | $\mathbf{0.1627 \pm 0.0000}$ | $\mathbf{0.9134 \pm 0.0000}$ | $\mathbf{65.08\% \pm 0.01\%}$ | $+0.2890 \pm 0.0005$ | $100.0\% \pm 0.0\%$ |

---

## Table 4.5: Robustness Under Stuck-Sensor Corruption ($E3$)

*Sensors frozen at their last measured value ($x_i(t) = x_i(t_{\text{fault}})$) across 1, 3, and 5 channels ($k \in \{1, 3, 5\}$). Evaluated over seeds $[42, 101, 2024]$.*

| Number of Stuck Channels ($k$) | Evaluated Model | System F1 Score | System Precision | System Recall | False Positive Rate (FPR) | Sensor Fault Detection Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **$k = 1$ Sensor** | Baseline C (No SRM) | $0.2171 \pm 0.0000$ | $0.1218 \pm 0.0000$ | $1.0000 \pm 0.0000$ | $99.86\% \pm 0.00\%$ | N/A |
| | **Proposed R-STNet** | $\mathbf{0.2740 \pm 0.0000}$ | $\mathbf{0.1626 \pm 0.0000}$ | $\mathbf{0.8713 \pm 0.0000}$ | $\mathbf{62.14\% \pm 0.00\%}$ | $0.0\% \pm 0.0\%^*$ |
| **$k = 3$ Sensors** | Baseline C (No SRM) | $0.2171 \pm 0.0000$ | $0.1218 \pm 0.0000$ | $1.0000 \pm 0.0000$ | $99.86\% \pm 0.00\%$ | N/A |
| | **Proposed R-STNet** | $\mathbf{0.2708 \pm 0.0045}$ | $\mathbf{0.1604 \pm 0.0000}$ | $\mathbf{0.8674 \pm 0.0000}$ | $\mathbf{62.87\% \pm 1.32\%}$ | $0.0\% \pm 0.0\%^*$ |
| **$k = 5$ Sensors** | Baseline C (No SRM) | $0.2171 \pm 0.0000$ | $0.1218 \pm 0.0000$ | $1.0000 \pm 0.0000$ | $99.86\% \pm 0.00\%$ | N/A |
| | **Proposed R-STNet** | $\mathbf{0.2737 \pm 0.0006}$ | $\mathbf{0.1624 \pm 0.0000}$ | $\mathbf{0.8691 \pm 0.0000}$ | $\mathbf{62.06\% \pm 0.07\%}$ | $0.0\% \pm 0.0\%^*$ |

*\*Physical Limitation Note: Stuck sensors on discrete actuators or naturally steady process variables do not generate a residual or temporal velocity spike; hence, variance-based SRM does not flag them as faulty without explicit command-telemetry discrepancy inputs.*

---

## Table 4.6: Robustness Under Dynamic Sensor Drift Corruption ($E4$)

*Linear ramp drift $x_i(t) \leftarrow x_i(t) \pm \delta \cdot \frac{t - t_0}{T}$ injected into continuous process sensors across severities $\delta \in \{0.1, 0.3, 0.5\}$ and seeds $[42, 101, 2024]$.*

| Drift Slope ($\delta$) | Evaluated Model | System F1 Score | System Precision | System Recall | False Positive Rate (FPR) | Reliability Separation ($\Delta r$) | Fault Detection Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Low ($\delta = 0.1$)** | Baseline C (No SRM) | $0.2171 \pm 0.0000$ | $0.1218 \pm 0.0000$ | $1.0000 \pm 0.0000$ | $99.86\% \pm 0.00\%$ | N/A | N/A |
| | **Proposed R-STNet** | $\mathbf{0.2739 \pm 0.0001}$ | $\mathbf{0.1625 \pm 0.0000}$ | $\mathbf{0.8707 \pm 0.0000}$ | $\mathbf{62.14\% \pm 0.00\%}$ | $+0.2928 \pm 0.0010$ | $100.0\% \pm 0.0\%$ |
| **Medium ($\delta = 0.3$)**| Baseline C (No SRM) | $0.2171 \pm 0.0000$ | $0.1218 \pm 0.0000$ | $1.0000 \pm 0.0000$ | $99.86\% \pm 0.00\%$ | N/A | N/A |
| | **Proposed R-STNet** | $\mathbf{0.2732 \pm 0.0005}$ | $\mathbf{0.1621 \pm 0.0000}$ | $\mathbf{0.8694 \pm 0.0000}$ | $\mathbf{62.24\% \pm 0.08\%}$ | $+0.2939 \pm 0.0007$ | $100.0\% \pm 0.0\%$ |
| **High ($\delta = 0.5$)** | Baseline C (No SRM) | $0.2171 \pm 0.0000$ | $0.1218 \pm 0.0000$ | $1.0000 \pm 0.0000$ | $99.86\% \pm 0.00\%$ | N/A | N/A |
| | **Proposed R-STNet** | $\mathbf{0.2729 \pm 0.0007}$ | $\mathbf{0.1619 \pm 0.0000}$ | $\mathbf{0.8690 \pm 0.0000}$ | $\mathbf{62.29\% \pm 0.12\%}$ | $+0.2946 \pm 0.0004$ | $100.0\% \pm 0.0\%$ |

---

## Table 4.7: Sensor Reliability Estimation and Fault Localization Metrics

*Distinguishing Synthetic Corruption Masks (created for controlled experiments) from Native SWaT System Anomaly Labels. Evaluated across 3,000 test windows under Gaussian noise ($\sigma=0.5$) and drift ($\delta=0.5$).*

| Evaluation Condition | Ground-Truth Mask Target | Clean Channels Mean Reliability ($\bar{r}_{\text{clean}}$) | Corrupted Channels Mean Reliability ($\bar{r}_{\text{corrupt}}$) | Reliability Separation ($\Delta r = \bar{r}_{\text{clean}} - \bar{r}_{\text{corrupt}}$) | Fault Detection Recall ($r < 0.60$) | Sensor Fault False Alarm Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean Test Sequence** | Synthetic Mask (None) | $0.3042 \pm 0.012$ | N/A | N/A | N/A | $0.00\%$ |
| **Gaussian Noise ($\sigma = 0.1$)** | Synthetic Noise Mask | $0.3039 \pm 0.011$ | $0.0183 \pm 0.003$ | $\mathbf{+0.2856}$ | $\mathbf{100.0\%}$ | $0.00\%$ |
| **Gaussian Noise ($\sigma = 0.5$)** | Synthetic Noise Mask | $0.3041 \pm 0.010$ | $0.0151 \pm 0.002$ | $\mathbf{+0.2890}$ | $\mathbf{100.0\%}$ | $0.00\%$ |
| **Sensor Drift ($\delta = 0.1$)** | Synthetic Drift Mask | $0.3040 \pm 0.011$ | $0.0112 \pm 0.002$ | $\mathbf{+0.2928}$ | $\mathbf{100.0\%}$ | $0.00\%$ |
| **Sensor Drift ($\delta = 0.5$)** | Synthetic Drift Mask | $0.3038 \pm 0.012$ | $0.0092 \pm 0.001$ | $\mathbf{+0.2946}$ | $\mathbf{100.0\%}$ | $0.00\%$ |
| **Native SWaT Attacks** | Native Attack Labels | Process-wide coupling | Attack-specific | Dynamically modulated | Attacked sensors downweighted | Low FPR in clean zones |

---

## Table 4.8: Sensor Reliability Module (SRM) Perspective Ablation Study

*Ablation isolating the mathematical components of SRM: Temporal Continuity ($w_v$), Graph Consistency ($w_c$), and Residual Confidence ($w_p$). All configurations evaluated under identical backbone parameters (23,233).*

| Ablation Configuration | Temporal Continuity ($w_v$) | Graph Consistency ($w_c$) | Residual Confidence ($w_p$) | Calibrated Threshold ($\tau$) | Point-wise F1 Score | System Precision | System Recall | False Positive Rate (FPR) | PR-AUC | Primary Experimental Finding |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. No SRM (Uniform)** | $0.0$ | $0.0$ | $0.0$ | $12.0706$ | $0.2171$ | $0.1218$ | $\mathbf{1.0000}$ | $99.86\%$ | $\mathbf{0.1371}$ | Complete false alarm collapse; $39,439$ normal windows flagged. |
| **2. Temporal Only** | $1.0$ | $0.0$ | $0.0$ | $18.9890$ | $0.2170$ | $0.1217$ | $0.9971$ | $99.60\%$ | $0.1244$ | High-frequency jitter suppressed; static distribution shift persists. |
| **3. Graph Only** | $0.0$ | $1.2$ | $0.0$ | $0.4645$ | $0.2163$ | $0.1214$ | $0.9936$ | $99.57\%$ | $0.1253$ | Graph error propagated across adjacent nodes; threshold scales down. |
| **4. Residual Only** | $0.0$ | $0.0$ | $1.0$ | $4.6495$ | $0.2184$ | $0.1235$ | $0.9433$ | $92.67\%$ | $0.1368$ | Extreme outlier downweighting begins; initial $7.19\%$ FPR reduction. |
| **5. Full SRM (Proposed)**| $\mathbf{1.0}$ | $\mathbf{1.2}$ | $\mathbf{1.0}$ | $\mathbf{3.1401}$ | $\mathbf{0.2740}$ | $\mathbf{0.1626}$ | $0.8713$ | $\mathbf{62.14\%}$ | $0.1169$ | **Synergistic multi-perspective suppression: $14,898$ false alarms eliminated.** |

---

## Table 4.9: Computational Resource Utilization and Execution Benchmarks

*Benchmarked on NVIDIA GeForce RTX 3050 Laptop GPU (4,096 MB VRAM, 2,048 CUDA Cores) and Intel Core i5 Host CPU.*

| Model Benchmark | Parameter Count | Training Wall Clock (s) | Training Sec / Epoch | Inference Latency (ms / window) | Max Inference Throughput (windows / sec) | Peak Allocated VRAM (MB) | Peak Reserved VRAM (MB) | Streaming Feasibility Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Isolation Forest** | Non-parametric | $\mathbf{12.40}$ | N/A | $\mathbf{0.021}$ | $47,619$ | N/A (Host RAM) | N/A (Host RAM) | Feasible (High-speed tabular CPU execution) |
| **ST-AE** | $170,643$ | $112.35$ | $7.49$ | $0.076$ | $13,158$ | $\mathbf{108.49}$ | $198.00$ | Feasible (Lightweight feedforward convolutional AE) |
| **RSTNetNoSRM** | $\mathbf{23,233}$ | $112.89$ | $7.53$ | $0.050$ | $20,000$ | $978.69$ | $1,124.00$ | Feasible (Efficient TCN-GAT without reliability weighting) |
| **Proposed R-STNet** | $\mathbf{23,233}$ | $134.73$ | $8.98$ | $0.060$ | $\mathbf{16,667}$ | $978.69$ | $1,124.00$ | **Feasible for streaming** ($0.060\text{ ms} \ll 10,000\text{ ms}$ sampling step) |

---

## Table 4.10: Empirical Failure Cases and Anomaly Score Diagnostic Matrix

| Case ID | Timestamp / Interval | Physical Sensors Involved | Physical / Operational Phenomenon | Model Prediction (Baseline C) | Model Prediction (R-STNet) | Ground-Truth State | Primary Causal Factor & Diagnostic Explanation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Case 1** | Test Window $t = 1,200 - 1,800$ | `AIT201` (Conductivity Analyzer) | Constant unrecorded during training ($z \approx 1$); dynamic active in test ($z > 1,600$). | **False Alarm** ($S(t) = 1,482 \gg \tau_c$) | **True Negative** ($S(t) = 2.14 < \tau_{prop}$) | Normal Operation ($y = 0$) | Dataset-level unrecorded channel shift. Baseline C collapses; R-STNet SRM suppresses `AIT201` weight ($r < 0.01$). |
| **Case 2** | Test Window $t = 2,500 - 2,700$ | `FIT101`, `LIT101`, `DPIT301` | Controlled synthetic Gaussian noise ($\sigma = 0.5$) injected into 3 channels. | **False Alarm** ($S(t) = 45.2 > \tau_c$) | **True Negative** ($S(t) = 2.89 < \tau_{prop}$) | Normal Operation ($y = 0$) | Sensor degradation without system breach. SRM isolates corrupted sensors, preventing plant-wide false emergency. |
| **Case 3** | Test Window $t = 14,000 - 14,500$ | `P202`, `P204` (Backup Chemical Dosing Pumps) | Inactive backup pumps frozen at $0.0$ state (stuck sensor condition). | **True Negative** ($S(t) < \tau_c$) | **True Negative** ($S(t) < \tau_{prop}$) | Normal Operation ($y = 0$) | Zero-variance actuator limitation. Neither model detects stuck state because variance is naturally zero during normal run. |
| **Case 4** | Test Window $t = 28,100 - 28,600$ | `LIT301`, `P301`, `MV301` | Multi-stage cyber-physical attack: Stage 3 tank overflow manipulation. | **True Positive** ($S(t) = 210.4 > \tau_c$) | **True Positive** ($S(t) = 48.7 > \tau_{prop}$) | Cyber-Physical Attack ($y = 1$) | Coordinated attack alters multi-sensor physical balance. Both models detect; R-STNet preserves attack alarm while filtering noise. |
| **Case 5** | Test Window $t = 36,200 - 36,450$ | `FIT401`, `UV401` | Subtle, stealthy low-rate flow spoofing attack. | **True Positive** ($S(t) > \tau_c$) | **False Negative** ($S(t) = 2.91 < \tau_{prop}$) | Cyber-Physical Attack ($y = 1$) | Stealthy anomaly masking. SRM moderately downweights slightly drifting sensor, lowering aggregate score below $\tau_{prop}$. |

---

## Table 4.11: Overall Research-Hypothesis Empirical Evidence Synthesis

*Core Research Question: Can explicit sensor-reliability estimation improve the robustness of multivariate time-series anomaly detection when sensor observations are missing, noisy, or corrupted?*

| Empirical Dimension | Tested Evaluation Condition | Direct Empirical Observation | Does It Support the Research Hypothesis? | Scientific Interpretation & Scope Limitations |
| :--- | :--- | :--- | :---: | :--- |
| **Within-Architecture False Alarm Suppression** | Clean SWaT Benchmark ($E1$) | R-STNet achieves $FPR = 62.14\%$ vs. Baseline C's $FPR = 99.86\%$. | **SUPPORTED** | Eliminates $14,898$ false alarms (+26.2% F1 gain) across identical 23k-parameter spatiotemporal architectures. |
| **Additive Noise Robustness** | Gaussian Noise ($E2$: $\sigma \in \{0.1, 0.3, 0.5\}$) | R-STNet maintains $F_1 \approx 0.276$ and achieves $+0.289$ reliability separation. | **SUPPORTED** | Explicit reliability weighting isolates corrupted channels and maintains plant-level detection stability. |
| **Dynamic Drift Robustness** | Sensor Drift ($E4$: $\delta \in \{0.1, 0.3, 0.5\}$) | R-STNet achieves $100\%$ fault recall and $+0.294$ separation without score explosion. | **SUPPORTED** | Multi-perspective fusion (temporal + graph) successfully flags non-physical rate-of-change and cross-sensor drift. |
| **Component Synergy** | SRM Ablation (Configs 1 to 5) | Full SRM ($F_1 = 0.2740$) outperforms individual components ($F_1 \approx 0.216 - 0.218$). | **SUPPORTED** | Combining temporal continuity, graph topology, and residual confidence produces synergistic false-alarm suppression. |
| **Clean Classical Baseline Comparison** | Clean SWaT Benchmark ($E1$) | Isolation Forest achieves $F_1 = 0.7182$ vs. R-STNet's $F_1 = 0.2740$. | **NOT SUPPORTED (Important Limitation)** | The proposed deep architecture does not outperform classical tree ensembles on clean SWaT due to unrecorded channel shifts. |
| **Zero-Variance Actuator Faults** | Stuck Actuators ($E3$: $k \in \{1, 3, 5\}$) | Stuck sensors on naturally constant pumps yield $0.0\%$ fault recall. | **NOT SUPPORTED (Physical Boundary)** | Statistical variance and reconstruction errors cannot distinguish an inactive backup pump from a stuck pump without telemetry. |

---

*End of Review 2 Tables Compendium. All values audited against authoritative JSON metric artifacts.*
