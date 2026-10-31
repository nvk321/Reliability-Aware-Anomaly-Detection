# Viva Voce Code Walkthrough Guide: Reliability-Aware Anomaly Detection (R-STNet)

**Course:** BCSE209P — Machine Learning Lab  
**Students:** P Naga Venkata Kushal (23BDS0133) & Koteru Tarun (23BDS0103)  
**Faculty Guide:** Dr. Dega Nagaraju  
**Project:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Repository:** [https://github.com/nvk321/Reliability-Aware-Anomaly-Detection](https://github.com/nvk321/Reliability-Aware-Anomaly-Detection)

---

## Overview

This guide is designed as an exact, step-by-step script for defending the codebase during an academic or technical viva voce examination. It references exact filenames, class names, functions, mathematical equations, and lines of code.

---

## 1. Project Motivation & Problem Context (Start at `README.md`)

- **File to Open:** [`README.md`](file:///d:/Projects/Reliability-Aware-Anomaly-Detection/README.md)
- **What to Say:**
  > "Good morning, Sir/Madam. Our project investigates **Reliability-Aware Anomaly Detection for Multivariate Sensor Data** in Cyber-Physical Systems, evaluated on the Secure Water Treatment (SWaT) testbed.  
  > In conventional industrial anomaly detection, deep learning models aggregate prediction errors across all sensor channels uniformly into a single global score:
  > $$S_{\text{sys}}(t) = \frac{1}{M} \sum_{i=1}^M z_i(t)$$
  > When a physical sensor experiences an ordinary hardware fault—such as high Gaussian jitter, calibration drift, or freezing—its individual error explodes. Because errors are summed uniformly, this benign sensor fault triggers a critical system alarm, halting the entire water treatment facility.  
  > Our core contribution is **R-STNet**, which introduces an online **Sensor Reliability Module (SRM)** to continuously estimate per-sensor trustworthiness $r_i(t) \in [0, 1]$ and downweight faulty sensors, decoupling sensor instrument degradation from true cyber-physical attacks."

---

## 2. Data Ingestion & Preprocessing (Open `src/data/swat_loader.py` & `src/preprocessing/`)

- **Files to Open:**
  - [`src/data/swat_loader.py`](file:///d:/Projects/Reliability-Aware-Anomaly-Detection/src/data/swat_loader.py) &rarr; Class `SWaTLoader`
  - [`src/preprocessing/scaler.py`](file:///d:/Projects/Reliability-Aware-Anomaly-Detection/src/preprocessing/scaler.py) &rarr; Class `RobustStandardScaler`
  - [`src/preprocessing/windowing.py`](file:///d:/Projects/Reliability-Aware-Anomaly-Detection/src/preprocessing/windowing.py) &rarr; Class `SlidingWindowGenerator`
- **What to Explain:**
  1. **SWaT Characteristics:** 51 variables across 6 physical stages (P1 to P6). 7 days of normal operational data (496,800 rows) and 4 days of attack evaluation data (449,919 rows with 36 physical attacks).
  2. **10x Downsampling (`swat_loader.py`):**
     > "We apply 10x temporal downsampling (10s intervals). In a physical water plant, chemical reactions and hydraulic filling take minutes, so 10s sampling respects the physical dynamics while reducing 946k raw points to 94k points, making deep learning computationally edge-viable."
  3. **Leak-Free Zero-Variance Scaling (`scaler.py`):**
     > "We fit our `RobustStandardScaler` **strictly on the normal training split**. In SWaT, backup actuators like pump `P202` remain off throughout normal training, having zero variance. Standard scalers crash due to division by zero. Our scaler clamps $\sigma_i$ to $1.0$ if $\sigma_i < 10^{-8}$, guaranteeing numerical safety."
  4. **Sliding Windows (`windowing.py`):**
     > "We generate 3D sliding windows of shape $(N, L=30, M=51)$ with stride 1. We enforce the `point_end` label mode, meaning the window label corresponds to ground truth at the latest step $t$, preventing future label leakage."

---

## 3. The Classical Baseline: Isolation Forest (Open `src/models/baselines/isolation_forest.py`)

- **File to Open:** [`src/models/baselines/isolation_forest.py`](file:///d:/Projects/Reliability-Aware-Anomaly-Detection/src/models/baselines/isolation_forest.py)
- **Script:** [`scripts/train_baseline.py`](file:///d:/Projects/Reliability-Aware-Anomaly-Detection/scripts/train_baseline.py)
- **What to Explain:**
  > "Why did we implement Isolation Forest?  
  > It establishes the clean tabular benchmark. In [`scripts/train_baseline.py`](file:///d:/Projects/Reliability-Aware-Anomaly-Detection/scripts/train_baseline.py#L45-L72), we fit an ensemble of 100 isolation trees on the normal training split and calibrate the decision threshold $\tau$ strictly on the normal validation split at the 95th quantile.  
  > On clean SWaT, Isolation Forest achieves **$F_1 = 0.7182$** and a false positive rate of only **$0.92\%$**."
- **Critical Viva Defense Point:**
  > "Why does Isolation Forest perform so well on clean SWaT?  
  > Isolation Forest uses axis-aligned orthogonal partitions. In SWaT, certain chemical sensors like `AIT201` exhibit unrecorded baseline drift across the 11-day test period. Because Isolation Forest evaluates features along independent splits, drift in one sensor does not distort other sensor dimensions."

---

## 4. Deep Baseline: RSTNetNoSRM (Open `src/models/baselines/rstnet_no_srm.py`)

- **File to Open:** [`src/models/baselines/rstnet_no_srm.py`](file:///d:/Projects/Reliability-Aware-Anomaly-Detection/src/models/baselines/rstnet_no_srm.py)
- **What to Explain:**
  > "To isolate the exact scientific contribution of sensor reliability, we cannot compare R-STNet against an arbitrary third-party model. We must compare it against an **identical architectural twin**.  
  > `RSTNetNoSRM` contains the exact same dilated 1D temporal convolutions and top-$k$ Graph Attention (23,233 parameters), but computes the anomaly score as an **unweighted mean of residuals**:
  > $$S(t) = \frac{1}{M} \sum_{i=1}^M z_i(t)$$
  > On clean SWaT, Baseline C achieves $F_1 = 0.2171$ with an FPR of $99.86\%$. Because it couples all 51 sensors in a joint representation, natural drift in chemical sensors elevates residuals plant-wide, causing threshold saturation."

---

## 5. The Proposed Model: R-STNet (Open `src/models/rstnet.py`)

- **File to Open:** [`src/models/rstnet.py`](file:///d:/Projects/Reliability-Aware-Anomaly-Detection/src/models/rstnet.py)
- **Functions to Highlight:**
  - `__init__()` (lines 30–75): Show `conv1`, `conv2` (dilation=2), `graph_attention`, `out_head`, and `self.srm`.
  - `forward(x)` (lines 76–103): Show how temporal convolutions embed each sensor independently before passing to spatial graph attention.
  - `evaluate_step(window_np)` (lines 134–184): Show the single-window online diagnostic routine.
- **What to Explain:**
  > "In `rstnet.py`, line 88 reshapes the batch to $(B \cdot M, 1, L)$. Each sensor channel passes through 1D dilated convolutions with a residual skip connection to capture temporal history.  
  > Next, `graph_attention` (lines 61–63) learns a top-15 directed adjacency matrix between sensor embeddings.  
  > Finally, `out_head` predicts the one-step-ahead value $\hat{\mathbf{x}}(t)$ for each sensor."

---

## 6. Sensor Reliability Module: SRM (Open `src/reliability/sensor_reliability.py`)

- **File to Open:** [`src/reliability/sensor_reliability.py`](file:///d:/Projects/Reliability-Aware-Anomaly-Detection/src/reliability/sensor_reliability.py)
- **Functions to Highlight:**
  - `compute_temporal_continuity()` (lines 110–136): Rate-of-change + stuck variance penalty.
  - `compute_neighborhood_consistency()` (lines 137–151): Consensus with topologically linked sensors.
  - `compute_residual_confidence()` (lines 152–171): Tail probability under normal residual distribution.
  - `estimate_reliability()` (lines 172–215): Calibrated logit fusion into $r_i(t) \in [0, 1]$.
  - `compute_weighted_system_anomaly()` (lines 217–233):
    $$S_{\text{sys}}(t) = \frac{\sum_i r_i(t) z_i(t)}{\sum_i r_i(t) + \epsilon}$$
- **What to Explain:**
  > "The SRM does not use arbitrary thresholds. During validation (`fit_calibration()`), it records normal rate-of-change limits, empirical neighbor correlation weights, and residual Median Absolute Deviation (MAD).  
  > At inference time, if a sensor experiences noise or drift, its rate-of-change or neighbor deviation explodes, driving $r_i(t) \to 0$. In line 228, its weight in $S_{\text{sys}}(t)$ vanishes, preventing a false system alarm."

---

## 7. Operational Decision Logic (Open `src/anomaly_detection/discriminator.py`)

- **File to Open:** [`src/anomaly_detection/discriminator.py`](file:///d:/Projects/Reliability-Aware-Anomaly-Detection/src/anomaly_detection/discriminator.py)
- **What to Explain:**
  > "We introduce a 4-way operational state classification engine:
  > - **Case A (Normal):** $S_{\text{sys}} < \tau$ and all $r_i \ge 0.60$. Plant is healthy; sensors are reliable.
  > - **Case B (Sensor Fault):** $S_{\text{sys}} < \tau$ and minimum $r_i < 0.60$. A sensor has degraded, but the plant is safe! Flagged for instrument recalibration without an emergency shutdown.
  > - **Case C (Cyber-Attack):** $S_{\text{sys}} \ge \tau$ and mean $r_i \ge 0.60$. Trustworthy sensors confirm a genuine cyber-physical attack. Critical alarm triggered!
  > - **Case D (Dual Incident):** $S_{\text{sys}} \ge \tau$ and mean $r_i < 0.60$. Simultaneous process emergency and instrument impairment."

---

## 8. Controlled Sensor Corruption (Open `src/corruption/corruptor.py` & `notebooks/05...`)

- **Files to Open:**
  - [`src/corruption/corruptor.py`](file:///d:/Projects/Reliability-Aware-Anomaly-Detection/src/corruption/corruptor.py) &rarr; Class `SensorCorruptor`
  - [`notebooks/05_robustness_and_ablation.ipynb`](file:///d:/Projects/Reliability-Aware-Anomaly-Detection/notebooks/05_robustness_and_ablation.ipynb)
- **What to Explain:**
  > "To validate our hypothesis, we built a deterministic corruptor supporting 9 operational fault modes: Gaussian noise, linear drift, stuck sensors, and multi-sensor combinations.  
  > Crucially, `apply_corruption()` creates a binary `sensor_fault_mask[t, i]`. SWaT labels only indicate cyber-attacks; our synthetic corruption mask provides the **ground truth for sensor-level fault localization**."

---

## 9. Audited Results & Honest Interpretation (Open `results/metrics/`)

- **Files to Open:**
  - [`results/metrics/proposed_rstnet_swat.json`](file:///d:/Projects/Reliability-Aware-Anomaly-Detection/results/metrics/proposed_rstnet_swat.json)
  - [`results/metrics/baseline_rstnet_no_srm_swat.json`](file:///d:/Projects/Reliability-Aware-Anomaly-Detection/results/metrics/baseline_rstnet_no_srm_swat.json)
  - [`results/corruption/corruption_benchmark_summary.json`](file:///d:/Projects/Reliability-Aware-Anomaly-Detection/results/corruption/corruption_benchmark_summary.json)
- **Key Numbers to Present:**
  | Metric | Isolation Forest | RSTNetNoSRM (Base C) | R-STNet (Proposed) |
  | :--- | :--- | :--- | :--- |
  | **Point-wise F1** | **0.7182** | 0.2171 | **0.2740** (+26.2% over Base C) |
  | **Precision** | 0.8995 | 0.1218 | **0.1626** |
  | **Recall** | 0.5977 | 1.0000 | **0.8713** |
  | **False Alarm Rate (FPR)** | **0.92%** | 99.86% | **62.14%** (37.8% relative reduction) |
  | **False Alarms Eliminated**| N/A | Reference | **14,898 false alarms removed** |
  | **Noise/Drift Reliability Sep** | N/A | 0.0000 | **$\Delta r = +0.289$** |
- **What to Conclude Honestly:**
  > "We do not claim that R-STNet is universally superior to Isolation Forest on clean data. Rather, our contribution proves that within deep spatiotemporal neural architectures, explicit reliability estimation eliminates 14,898 false alarms, establishes $+0.29$ reliability separation, and achieves 100% sensor fault recall under noise and drift."

---

## 10. Interactive Streamlit Demonstration (Run `streamlit run demo/app.py`)

- **Action:** Open terminal and run `streamlit run demo/app.py`.
- **Flow to Demonstrate:**
  1. **Tab 1 (Overview):** Reiterate the core research question and model parameters (23,233).
  2. **Tab 4 (Reliability):** Show the bar chart of 51 sensors; show how LIT101 has high trust while degraded sensors are flagged in amber/red.
  3. **Tab 5 (Live Detection):** Select "Attack Scenario 1 (LIT101 Overflow)" from the sidebar. Show the blue anomaly score line crossing the red threshold $\tau = 3.1401$ inside the shaded red attack region. Point to the **STATE C: VERIFIED CYBER-PHYSICAL ATTACK** card.
  4. **Tab 7 (Corruption Sandbox):** Select `LIT101`, choose "Additive Gaussian Noise", and click **Run Live Degradation Simulation**.  
     - Point to Baseline C: Score explodes &rarr; **🚨 FALSE ALARM TRIGGERED!**  
     - Point to R-STNet: Credibility drops &rarr; **🛡️ FALSE ALARM ELIMINATED!**
  5. **Tab 9 (Limitations):** Defend the four academic boundaries (clean benchmark, residual FPR, stuck actuators, offline prototype).
