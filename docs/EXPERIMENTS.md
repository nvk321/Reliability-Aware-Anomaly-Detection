# Experimental Protocol & Evaluation Framework

**Project:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Students:** P Naga Venkata Kushal (23BDS0133) & Koteru Tarun (23BDS0103)  
**Faculty Guide:** Dr. Dega Nagaraju  
**Status:** Validated Experimental Framework  

---

## 1. Experimental Overview

The empirical evaluation of R-STNet is structured around four distinct experimental investigations:
1. **Experiment E1: Clean Industrial Benchmark:** Baseline comparison across all four models on the unperturbed SWaT test sequence.
2. **Experiment E2: Additive Gaussian Noise Robustness:** Robustness evaluation under increasing high-frequency sensor noise across multiple random seeds.
3. **Experiment E3: Stuck-Sensor Robustness:** Robustness evaluation when physical sensors freeze at constant values across multiple channels.
4. **Experiment E4: Dynamic Sensor Drift Robustness:** Robustness evaluation under slow linear calibration ramp drifts.
5. **SRM Ablation Suite:** Systematic 5-condition ablation isolating each reliability perspective.

---

## 2. Benchmark Models

| Model Identifier | Architectural Family | Parameter Count | Scoring Formulation | Training Time (s) | Inference Latency (ms/win) |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **Baseline A: Isolation Forest** | Tree Ensemble | Non-parametric | Tree path length $s(x, n)$ | **12.40** | **0.021** |
| **Baseline B: ST-AE** | Convolutional Autoencoder | $170,643$ | Mean squared error $\frac{1}{M}\sum (x - \hat{x})^2$ | $112.35$ | $0.076$ |
| **Baseline C: RSTNetNoSRM** | Spatiotemporal Graph Net | $\mathbf{23,233}$ | Uniform residual $\frac{1}{M}\sum (\|x-\hat{x}\|/\text{MAD})$ | $112.89$ | $0.050$ |
| **Proposed: R-STNet (with SRM)**| Reliability Graph Net | $\mathbf{23,233}$ | Reliability-weighted residual $\frac{\sum r_i e_i}{\sum r_i}$ | $134.73$ | **0.060** |

---

## 3. Data Partitions & Zero-Leakage Protocol

- **Dataset:** SWaT A1/A2 December 2015 testbed ($M = 51$ sensors, $10\times$ downsampling).
- **Training Split (Normal):** $39,715$ sliding windows (First 80% of Phase A1, $y=0$).
- **Validation Split (Normal):** $9,907$ sliding windows (Final 20% of Phase A1, $y=0$). Reserved for threshold calibration ($\tau = \text{Quantile}(S_{\text{val}}, 0.95)$).
- **Mixed Test Split (Attacks):** $44,963$ sliding windows (100% of Phase A2, containing $5,468$ attack labels, $12.16\%$ prevalence).

---

## 4. Controlled Sensor Corruption Suite ($E2 - E4$)

All corruption experiments were evaluated across **three random seeds** ($[42, 101, 2024]$) to compute statistical means and variances:

### 4.1 Experiment E2: Additive Gaussian Noise
- **Injection Formula:** $x_i(t) \leftarrow x_i(t) + \epsilon(t)$, where $\epsilon(t) \sim \mathcal{N}(0, \sigma^2)$.
- **Target Channels:** $\{3, 14, 28\}$ (`P101`, `P205`, `LIT401`).
- **Corruption Rate:** $30\%$ of temporal windows.
- **Severities:** $\sigma \in \{0.1\text{ (Low)}, 0.3\text{ (Medium)}, 0.5\text{ (High)}\}$.

### 4.2 Experiment E3: Stuck Sensors
- **Injection Formula:** $x_i(t) \leftarrow x_i(t_{\text{fault}})$ for all $t \ge t_{\text{fault}}$.
- **Severities:** $k \in \{1, 3, 5\}$ simultaneously frozen sensor channels.
- **Physical Boundary Discovered:** Stuck actuators on naturally inactive backup pumps (`P202`, `P204`) produce $0.0\%$ fault recall because natural operational variance is zero.

### 4.3 Experiment E4: Dynamic Sensor Drift
- **Injection Formula:** $x_i(t) \leftarrow x_i(t) \pm \delta \cdot \frac{t - t_0}{T}$ across duration $T$.
- **Target Channels:** Continuous process flow and level sensors.
- **Severities:** Ramp slopes $\delta \in \{0.1\text{ (Low)}, 0.3\text{ (Medium)}, 0.5\text{ (High)}\}$.

---

## 5. Summary of Verified Experimental Results

### 5.1 Clean Benchmark Results ($E1$)
- **Isolation Forest:** $F_1 = \mathbf{0.7182}$, Precision $= \mathbf{0.8995}$, Recall $= 0.5977$, $\text{FPR} = \mathbf{0.0092}$ ($0.92\%$).
- **ST-AE:** $F_1 = 0.2169$, Precision $= 0.1216$, Recall $= 1.0000$, $\text{FPR} = 1.0000$ ($100.0\%$).
- **RSTNetNoSRM:** $F_1 = 0.2171$, Precision $= 0.1218$, Recall $= 1.0000$, $\text{FPR} = 0.9986$ ($99.86\%$).
- **Proposed R-STNet:** $F_1 = \mathbf{0.2740}$, Precision $= \mathbf{0.1626}$, Recall $= 0.8713$, $\text{FPR} = \mathbf{0.6214}$ ($62.14\%$).
- **Impact of SRM:** Eliminates **$14,898$ false alarms** (-37.8% relative FPR reduction) and yields a **$+26.21\%$ relative improvement in F1** over Baseline C under identical parameters.

### 5.2 Sensor Reliability Separation
- Clean channels mean reliability: $\bar{r}_{\text{clean}} = 0.3041 \pm 0.010$.
- Corrupted channels mean reliability: $\bar{r}_{\text{corrupt}} = 0.0151 \pm 0.002$ (Noise $\sigma=0.5$) and $0.0092 \pm 0.001$ (Drift $\delta=0.5$).
- Empirical separation gap: **$\Delta r = +0.2856\text{ to }+0.2946$**.
- Sensor fault detection recall against synthetic masks: **$100.0\% \pm 0.0\%$** ($r_i < 0.60$ cutoff).

### 5.3 SRM Ablation Findings
- No SRM: $F_1 = 0.2171$, $FPR = 99.86\%$.
- Temporal Only: $F_1 = 0.2170$, $FPR = 99.60\%$.
- Graph Only: $F_1 = 0.2163$, $FPR = 99.57\%$.
- Residual Only: $F_1 = 0.2184$, $FPR = 92.67\%$.
- Full SRM: **$F_1 = 0.2740$**, **$FPR = 62.14\%$**.
