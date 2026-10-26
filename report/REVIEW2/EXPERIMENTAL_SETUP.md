# Experimental Setup and Methodology

**Project:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Course:** BCSE209P — Machine Learning Lab  
**Date:** October 2026  
**Status:** Validated Experimental Framework  

---

## 1. Primary Dataset: Secure Water Treatment (SWaT)

The experimental investigation is conducted on the Secure Water Treatment (SWaT) industrial testbed dataset developed by the iTrust Centre for Research in Cyber Security at Singapore University of Technology and Design (SUTD). SWaT represents a physical, scaled-down water treatment facility producing five gallons per minute of treated water across six distinct operational stages:
1. **Stage 1 (P1):** Raw water intake, chemical dosing, and level control (`LIT101`, `FIT101`, `MV101`, `P101`, `P102`).
2. **Stage 2 (P2):** Chemical pre-treatment, coagulation, and electrochemical water quality monitoring (`AIT201`–`AIT203`, `FIT201`, `MV201`, `P201`–`P206`).
3. **Stage 3 (P3):** Ultrafiltration (UF) feed system, backwash cycle, and membrane differential pressure monitoring (`DPIT301`, `FIT301`, `LIT301`, `MV301`–`MV304`, `P301`, `P302`).
4. **Stage 4 (P4):** Dechlorination, chemical oxidation reduction, and ultraviolet (UV) disinfection (`AIT401`, `AIT402`, `FIT401`, `LIT401`, `P401`–`P404`, `UV401`).
5. **Stage 5 (P5):** Reverse Osmosis (RO) high-pressure membrane desalination and membrane flux tracking (`AIT501`–`AIT504`, `FIT501`–`FIT504`, `P501`, `P502`, `PIT501`–`PIT503`).
6. **Stage 6 (P6):** Treated water storage, distribution pumping, and cleaning-in-place (`FIT601`, `P601`–`P603`).

### Sensor and Actuator Breakdown
The testbed provides $M = 51$ telemetry variables comprising:
- **25 Continuous Physical Sensors:** Float and differential level transmitters, magnetic flow meters, pressure transmitters, and electrochemical water analyzers.
- **26 Discrete Actuators:** Motorized valves, chemical dosing pumps, high-pressure pumps, and ultraviolet units.

---

## 2. Chronological Data Splitting and Leak-Free Protocol

The raw dataset spans two distinct operational operational phases recorded in December 2015:
- **Phase A1 (Normal Operation):** Seven consecutive days of continuous, attack-free normal operation ($496,800$ raw records at 1 Hz).
- **Phase A2 (Cyber-Physical Attacks):** Four consecutive days during which 36 distinct physical and cyber-physical attacks were executed against the plant ($449,919$ raw records at 1 Hz).

### Data Partitions
To ensure complete absence of data leakage and adhere to temporal causality, data was partitioned chronologically:
1. **Training Split (Pure Normal):** The first $80\%$ of Phase A1 ($397,440$ raw seconds $\to 39,744$ time steps at $10\times$ downsampling $\to 39,715$ sliding windows).
2. **Validation Split (Pure Normal):** The final $20\%$ of Phase A1 ($99,360$ raw seconds $\to 9,936$ time steps at $10\times$ downsampling $\to 9,907$ sliding windows). This split is used exclusively for hyperparameter validation and decision threshold calibration.
3. **Test Split (Chronological Mixed Test):** The entire Phase A2 period ($449,919$ raw records $\to 44,992$ time steps at $10\times$ downsampling $\to 44,963$ sliding windows). The test split contains $5,468$ attack-labeled windows, establishing an attack prevalence rate of $12.16\%$.

```
[================ Phase A1: 7 Days Pure Normal (496,800s) ================] [== Phase A2: 4 Days Attack (449,919s) ==]
[----------- Train: 80% (39,715 windows) -----------][-- Val: 20% (9,907) --] [-------- Test: 100% (44,963 windows) --------]
                  ^                                             ^                                   ^
            Fit Scaler                                  Calibrate Threshold                   Evaluate Models
```

---

## 3. Preprocessing, Scaling, and Windowing

### 3.1 Downsampling
Raw 1 Hz industrial data contains substantial high-frequency sensor noise and measurement redundancy. Telemetry is downsampled by a factor of $10\times$ ($10$-second interval step), compressing the sequence while preserving all physical system dynamics and transient attack profiles.

### 3.2 Robust Scaling with Zero-Variance Protection
Data is normalized using `RobustStandardScaler`, fit **strictly on the 80% normal training split**:
$$z_{i, t} = \frac{x_{i, t} - \mu_i}{\sigma_i}$$
where $\mu_i$ is the training sample mean and $\sigma_i$ is the training sample standard deviation.

**Zero-Variance Protection:** Because certain backup actuators (e.g., `P202`, `MV101`) or unrecorded instruments remain constant during normal operation, $\sigma_i$ can equal zero. To prevent division-by-zero or numerical NaN generation:
$$\sigma_i^* = \begin{cases} \sigma_i & \text{if } \sigma_i \ge \epsilon \\ 1.0 & \text{if } \sigma_i < \epsilon \end{cases}$$
where $\epsilon = 10^{-6}$. For constant features, $x_{i, t} - \mu_i = 0$, mapping them cleanly to $0.0$.

### 3.3 Temporal Sliding Window Generation
Sliding windows of length $L = 30$ time steps ($300$ seconds / 5 minutes) with a stride of $S = 1$ time step are extracted:
$$\mathbf{X}_t = [\mathbf{x}_{t-L+1}, \mathbf{x}_{t-L+2}, \dots, \mathbf{x}_t] \in \mathbb{R}^{L \times M}$$
The window label is assigned using `point_end` causality:
$$y(\mathbf{X}_t) = y_t \in \{0, 1\}$$
This ensures that an anomaly is only flagged at time $t$ if the current system state at $t$ is anomalous, preserving online causality.

---

## 4. Evaluated Machine Learning Models

### 4.1 Baseline A: Isolation Forest
- **Family:** Classical Tree Ensemble (Non-parametric).
- **Configuration:** 100 Isolation Trees, sub-sample size $n = 256$, contamination parameter set to expected validation outlier rate.
- **Input:** Static flattened vector $\mathbf{x}_t \in \mathbb{R}^{51}$.
- **Anomaly Score:** Inverse average tree path length:
  $$s(\mathbf{x}, n) = 2^{-\frac{\mathbb{E}(h(\mathbf{x}))}{c(n)}}$$

### 4.2 Baseline B: Spatio-Temporal Autoencoder (ST-AE)
- **Family:** Deep Feedforward Convolutional Autoencoder ($170,643$ parameters).
- **Architecture:** 1D Temporal Convolutional encoder layers followed by dense linear reconstruction layers.
- **Input:** Window tensor $\mathbf{X} \in \mathbb{R}^{B \times 30 \times 51}$.
- **Anomaly Score:** Mean squared reconstruction error:
  $$S_{\text{ST-AE}}(t) = \frac{1}{M} \sum_{i=1}^M (x_{i, t} - \hat{x}_{i, t})^2$$

### 4.3 Baseline C: RSTNet without SRM (RSTNetNoSRM)
- **Family:** Spatiotemporal Graph Neural Network ($23,233$ parameters).
- **Architecture:** 
  1. **Temporal Backbone:** Dilated Temporal Convolutional Network (TCN) with causal dilations $[1, 2, 4]$.
  2. **Spatial Backbone:** Multi-head Graph Attention Network (GAT) modeling inter-sensor dependencies across a learned physical adjacency matrix.
  3. **Reconstruction Head:** Linear projection reconstructing the current sensor vector $\hat{\mathbf{x}}_t$.
- **Anomaly Score:** Unweighted normalized residual error:
  $$S_{\text{Baseline C}}(t) = \frac{1}{M} \sum_{i=1}^M \frac{|x_{i, t} - \hat{x}_{i, t}|}{\text{MAD}_i + \epsilon}$$

### 4.4 Proposed Model: R-STNet (with Sensor Reliability Module)
- **Family:** Reliability-Aware Spatiotemporal Graph Neural Network ($23,233$ parameters).
- **Architecture:** Identical Dilated TCN + Spatial GAT neural backbone as Baseline C, augmented with the **Sensor Reliability Module (SRM)**.
- **Reliability Formulation:** For each sensor $i$ at time $t$, SRM computes a reliability coefficient $r_i(t) \in [0, 1]$ fusing three perspectives:
  $$r_i(t) = \frac{w_v \cdot r_i^{(v)}(t) + w_c \cdot r_i^{(c)}(t) + w_p \cdot r_i^{(p)}(t)}{w_v + w_c + w_p}$$
  where:
  - $r_i^{(v)}(t) = \exp(-\frac{1}{2}(\frac{\Delta x_{i, t}}{\text{MAD}(\Delta x_i)})^2)$ evaluates **temporal continuity** (rate-of-change physics).
  - $r_i^{(c)}(t) = \exp(-\frac{1}{2}(\frac{x_{i, t} - \bar{x}_{\mathcal{N}(i), t}}{\sigma_{\text{neigh}}})^2)$ evaluates **graph consistency** against physical neighbors $\mathcal{N}(i)$.
  - $r_i^{(p)}(t) = \exp(-\frac{1}{2} z_{i, t}^2)$ evaluates **residual confidence** against normal predictive variance.
- **Anomaly Score:** Reliability-weighted residual aggregation:
  $$S_{\text{sys}}(t) = \frac{\sum_{i=1}^M r_i(t) \cdot e_i(t)}{\sum_{i=1}^M r_i(t)}$$

---

## 5. Threshold Calibration Protocol

To prevent test-set label leakage, decision thresholds $\tau$ are calibrated **strictly on the normal validation split**:
1. Normal validation windows are fed through the trained model to compute validation anomaly scores $\{S_{\text{val}}(1), S_{\text{val}}(2), \dots, S_{\text{val}}(N_{\text{val}})\}$.
2. The decision threshold $\tau$ is set to the $95^{\text{th}}$ empirical percentile ($q = 0.95$):
   $$\tau = \text{Quantile}(S_{\text{val}}, 0.95)$$
3. During test-time evaluation, an alarm is triggered if and only if $S_{\text{test}}(t) \ge \tau$.
4. **Leak-Free Guarantee:** Test-set attack labels are never consulted during threshold calibration.

---

## 6. Quantitative Evaluation Metrics

All models are evaluated using standard time-series anomaly detection metrics:
- **Precision:** $\frac{\text{TP}}{\text{TP} + \text{FP}}$ (Accuracy of alarm events).
- **Recall (Detection Rate):** $\frac{\text{TP}}{\text{TP} + \text{FN}}$ (Fraction of attacks detected).
- **Point-wise F1 Score:** $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ (Harmonic mean under strict temporal alignment).
- **False Positive Rate (FPR):** $\frac{\text{FP}}{\text{FP} + \text{TN}}$ (Fraction of normal intervals falsely flagged).
- **Point-Adjusted F1 (PA-F1):** F1 score under the standard anomaly benchmark protocol where if any single point in a contiguous attack segment is detected, the entire segment is counted as detected.
- **PR-AUC & ROC-AUC:** Threshold-independent areas under the Precision-Recall and ROC curves.

---

## 7. Computational Environment

All experiments were executed in a fully reproducible environment:
- **Host Processor:** Intel Core i5 Host CPU (12 cores).
- **Accelerator:** NVIDIA GeForce RTX 3050 Laptop GPU (4,096 MB VRAM, 2,048 CUDA Cores).
- **Operating System:** Microsoft Windows 11 Enterprise.
- **Software Stack:** Python 3.11 / PyTorch 2.6.0+cu124 / CUDA 12.4.
- **Random Seeds:** 42 (primary training), 101, 2024 (multi-seed corruption benchmark).
