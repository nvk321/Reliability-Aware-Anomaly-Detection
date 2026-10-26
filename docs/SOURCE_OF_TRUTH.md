# Source of Truth: Authoritative Scientific & Technical Specifications

**Project Title:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Students:** P Naga Venkata Kushal (23BDS0133) & Koteru Tarun (23BDS0103)  
**Faculty Guide:** Dr. Dega Nagaraju  
**Course:** BCSE209P — Machine Learning Lab  
**Status:** Frozen Experimental Ground Truth  

---

## 1. Project Identity & Research Core

- **Final Official Title:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data
- **Proposed Model Framework:** R-STNet (Reliability-Aware Spatio-Temporal Network)
- **Primary Auxiliary Mechanism:** Sensor Reliability Module (SRM)
- **Primary Research Question:**
  > Can explicit sensor-reliability estimation improve the robustness of multivariate time-series anomaly detection when sensor observations are missing, noisy, or corrupted?
- **Core Hypothesis:**
  > Explicit multi-perspective sensor-reliability estimation should improve detection robustness and reduce false-alarm saturation under controlled sensor degradation within a spatio-temporal deep-learning anomaly-detection architecture.
- **Defensible Final Research Contribution:**
  > Within the tested spatio-temporal deep-learning architecture, explicit sensor-reliability estimation provides a statistically verified mechanism to mitigate catastrophic false-alarm saturation and improves robustness against controlled sensor degradation relative to uniform residual aggregation, while classical tree-based Isolation Forest remains superior on the clean SWaT benchmark.

---

## 2. Primary Dataset Specification

- **Name:** Secure Water Treatment (SWaT A1/A2, December 2015 Testbed)
- **Institution:** iTrust Centre for Research in Cyber Security, Singapore University of Technology and Design (SUTD)
- **Physical System:** 6-stage municipal-scale water purification plant (P1: Raw water intake, P2: Chemical dosing, P3: Ultrafiltration, P4: Dechlorination, P5: Reverse osmosis, P6: Distribution).
- **Telemetry Channels:** $M = 51$ physical variables ($25$ continuous process sensors, $26$ discrete actuators).
- **Raw Data Composition:**
  - *Phase A1 (Normal Operation):* 7 days of continuous normal operation ($496,800$ raw rows @ 1 Hz).
  - *Phase A2 (Cyber-Physical Attacks):* 4 days of testing with 36 physical attacks ($449,919$ raw rows @ 1 Hz).

---

## 3. Data Preprocessing & Leak-Free Protocol

- **Downsampling:** $10\times$ downsampling ($10$-second interval step), compressing 1 Hz telemetry while preserving process physical dynamics.
- **Data Partitions:**
  - *Training Split (Pure Normal):* First $80\%$ of Phase A1 ($39,744$ time steps $\to 39,715$ sliding windows, $100\%$ normal, $y=0$).
  - *Validation Split (Pure Normal):* Final $20\%$ of Phase A1 ($9,936$ time steps $\to 9,907$ sliding windows, $100\%$ normal, $y=0$). Reserved strictly for threshold tuning.
  - *Test Split (Chronological Mixed Test):* Full Phase A2 ($44,992$ time steps $\to 44,963$ sliding windows, $5,468$ attack labels, $12.16\%$ prevalence).
- **Scaling:** `RobustStandardScaler` fit **strictly on the 80% normal training split** with zero-variance protection ($\epsilon = 10^{-6}$, scale set to $1.0$). Zero test or validation leakage.
- **Sliding Window:** Length $L = 30$ time steps ($300$ s / 5 minutes), Stride $S = 1$, causal `point_end` labeling ($y(\mathbf{X}_t) = y_t$).
- **Threshold Calibration ($\tau$):** Calibrated strictly at the $95^{\text{th}}$ percentile ($q = 0.95$) of normal validation scores. Zero test-label consultation.

---

## 4. Evaluated Model Architectures

1. **Baseline A: Isolation Forest**
   - Classical non-parametric tree ensemble (100 isolation trees, $n=256$ sub-samples).
   - Input: Static flattened vector $\mathbf{x}_t \in \mathbb{R}^{51}$.
   - Score: Tree path length $s(x, n)$.
2. **Baseline B: Spatio-Temporal Autoencoder (ST-AE)**
   - Feedforward convolutional autoencoder ($170,643$ parameters).
   - Input: Tensor $\mathbf{X} \in \mathbb{R}^{B \times 30 \times 51}$.
   - Score: Mean squared error $\frac{1}{M} \sum (x - \hat{x})^2$.
3. **Baseline C: RSTNetNoSRM**
   - Dilated TCN + Spatial Graph Attention (GAT) without reliability weighting ($23,233$ parameters).
   - Input: Tensor $\mathbf{X} \in \mathbb{R}^{B \times 30 \times 51}$.
   - Score: Uniform normalized residual $\frac{1}{M} \sum (|x - \hat{x}| / \text{MAD})$.
4. **Proposed R-STNet (with SRM)**
   - Dilated TCN + Spatial GAT + Sensor Reliability Module ($23,233$ parameters).
   - Score: Reliability-weighted residual $S_{\text{sys}}(t) = \frac{\sum r_i(t) e_i(t)}{\sum r_i(t)}$.

---

## 5. Frozen Validated Benchmark Results

### 5.1 Clean SWaT Benchmark ($E1$)

| Metric | Isolation Forest | ST-AE | RSTNetNoSRM | Proposed R-STNet | R-STNet vs. Baseline C |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Point-wise F1** | **0.7182** | 0.2169 | 0.2171 | **0.2740** | **+26.21% relative gain** |
| **Precision** | 0.8995 | 0.1216 | 0.1218 | 0.1626 | +33.50% gain |
| **Recall** | 0.5977 | 1.0000 | 1.0000 | 0.8713 | Genuine sensitivity |
| **False Positive Rate (FPR)**| **0.0092 (0.92%)**| 1.0000 (100.0%)| 0.9986 (99.86%)| **0.6214 (62.14%)**| **-37.78% (14,898 FPs avoided)**|
| **False Positives (FP)** | 365 | 39,495 | 39,439 | 24,541 | -14,898 normal windows |
| **True Negatives (TN)** | 39,159 | 0 | 56 | 14,954 | +14,898 normal windows |
| **True Positives (TP)** | 3,268 | 5,468 | 5,468 | 4,764 | -704 attack windows |
| **False Negatives (FN)** | 2,200 | 0 | 0 | 704 | +704 attack windows |
| **Point-Adjusted F1 (PA-F1)** | 0.8346 | 0.2169 | 0.2171 | 0.2944 | +35.61% gain |
| **PR-AUC** | 0.7367 | 0.1304 | 0.1380 | 0.1169 | Trade-off acknowledged |
| **ROC-AUC** | 0.8429 | 0.5070 | 0.5164 | 0.4976 | Near 0.50 floor |
| **Calibrated Threshold ($\tau$)**| 0.6359 | 0.0892 | 12.0706 | 3.1401 | Validation quantile $q=0.95$ |

### 5.2 Controlled Robustness Benchmark ($E2 - E4$)
- **Additive Gaussian Noise ($E2$):** R-STNet maintains $F_1 = 0.2755 \pm 0.0011$ ($\sigma = 0.1$), $0.2764 \pm 0.0005$ ($\sigma = 0.3$), and $0.2762 \pm 0.0001$ ($\sigma = 0.5$) with FPR bounded at $64.09\% - 65.08\%$. Baseline C remains collapsed at $FPR = 99.86\%$.
- **Stuck Sensors ($E3$):** R-STNet maintains $F_1 = 0.2740 \pm 0.0000$ ($k=1$), $0.2708 \pm 0.0045$ ($k=3$), and $0.2737 \pm 0.0006$ ($k=5$). Fault recall is $0.0\%$ on inactive actuators because natural variance is zero.
- **Dynamic Sensor Drift ($E4$):** R-STNet maintains $F_1 = 0.2739 \pm 0.0001$ ($\delta = 0.1$), $0.2732 \pm 0.0005$ ($\delta = 0.3$), and $0.2729 \pm 0.0007$ ($\delta = 0.5$) with FPR bounded at $62.14\% - 62.29\%$.

### 5.3 Sensor Reliability Decoupling Metrics
- Clean channels mean reliability: $\bar{r}_{\text{clean}} = 0.3041 \pm 0.010$.
- Corrupted channels mean reliability: $\bar{r}_{\text{corrupt}} = 0.0151 \pm 0.002$ (Noise) and $0.0092 \pm 0.001$ (Drift).
- Reliability separation gap: **$\Delta r = +0.2856\text{ to }+0.2946$**.
- Fault detection recall against synthetic masks: **$100.0\% \pm 0.0\%$** ($r_i < 0.60$ cutoff).

### 5.4 5-Condition Ablation Study
- No SRM: $F_1 = 0.2171$, $FPR = 99.86\%$.
- Temporal Only ($w_v=1$): $F_1 = 0.2170$, $FPR = 99.60\%$.
- Graph Only ($w_c=1.2$): $F_1 = 0.2163$, $FPR = 99.57\%$.
- Residual Only ($w_p=1$): $F_1 = 0.2184$, $FPR = 92.67\%$.
- Full SRM ($w_v=1, w_c=1.2, w_p=1$): **$F_1 = 0.2740$**, **$FPR = 62.14\%$**.

### 5.5 Computational Measurements
- Parameters: $23,233$ (86.4% smaller than ST-AE's $170,643$).
- Training Time: $134.73$ s ($8.98$ s/epoch).
- Inference Latency: $0.060\text{ ms}$ per window ($16,667$ windows/s), operating $160,000\times$ faster than the $10$-second SWaT physical sampling step.
- Peak VRAM: $978.69\text{ MB}$ ($27.4\%$ of a 4 GB RTX 3050 mobile GPU).

---

## 6. Critical Scientific Boundaries & Prohibited Claims

1. ❌ **Do NOT claim SOTA or superiority over Isolation Forest.** Isolation Forest is far superior on clean SWaT ($0.7182$ vs. $0.2740$).
2. ❌ **Do NOT claim complete false alarm elimination.** R-STNet eliminates $14,898$ false alarms, but $24,541$ remain ($FPR = 62.14\%$).
3. ❌ **Do NOT claim universal sensor fault detection.** Inactive zero-variance actuators cannot be diagnosed without supervisory PLC command telemetry.
4. ❌ **Do NOT label SWaT shifted channels as "corrupted sensors."** They are unrecorded commissioning channels and discrete actuator transitions in the public dataset.
5. ❌ **Do NOT claim live industrial deployment.** Only in-memory GPU latency was benchmarked.
