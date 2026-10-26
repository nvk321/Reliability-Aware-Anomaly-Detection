# Experiment Plan: Robustness and Reliability Validation

**Project:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Primary Benchmark:** SWaT (51 channels)  
**Secondary Benchmark:** WADI (123 channels)  

---

## 1. Research Questions & Working Hypotheses

### Primary Research Question (RQ1)
> *Can explicit sensor-reliability estimation reduce false positive alarms and improve detection robustness in multivariate sensor systems when sensor observations suffer from missingness, noise, drift, spikes, or stuck values?*

- **Hypothesis 1 (Robustness):**  
  Under increasing sensor corruption (e.g., Gaussian noise, calibration drift, or stuck sensors), conventional anomaly detectors (Isolation Forest, Autoencoders) will experience a precipitous rise in False Positive Rate (FPR) and a decline in F1-score, whereas **R-STNet** will maintain high F1 and stable FPR by dynamically discounting corrupted channels.

### Secondary Research Question (RQ2)
> *Can the proposed 4-way discrimination engine accurately separate isolated sensor hardware faults (Case B) from genuine cyber-physical process attacks (Case C)?*

- **Hypothesis 2 (Discrimination Accuracy):**  
  When an isolated sensor is artificially drifted or frozen while the physical system remains normal, R-STNet will correctly flag `SENSOR FAULT` ($\ge 90\%$ accuracy) without triggering a `SYSTEM ANOMALY` emergency alert.

### Tertiary Research Question (RQ3)
> *Which constituent signal of the Sensor Reliability Module (temporal volatility, cross-sensor graph consistency, or residual Z-score) contributes most to robustness?*

- **Hypothesis 3 (Ablation):**  
  Cross-sensor graph consistency provides the primary discriminative power for physical process consistency, while temporal volatility handles sudden spikes and stuck sensor conditions.

---

## 2. Baseline Models

To ensure rigorous and fair comparative evaluation, we evaluate against three distinct classes of baselines:

1. **Baseline 1: Classical Anomaly Detector (Isolation Forest / Robust PCA)**
   - Operates on instantaneous features and differences.
   - Benchmark role: Demonstrates standard industrial off-the-shelf performance.
2. **Baseline 2: Deep Reconstruction Model (Spatio-Temporal Autoencoder / ST-AE)**
   - Uses the identical spatio-temporal encoder/decoder backbone as R-STNet, but **without** the Sensor Reliability Module (unweighted residual sum).
   - Benchmark role: Direct ablation isolating the exact benefit of the reliability mechanism.
3. **Baseline 3: External SOTA Research Baseline (CAROTS)**
   - Official implementation from `reference/CAROTS/` (Kim et al., ICML 2025).
   - Benchmark role: Establishes external reference performance for causality-aware contrastive representation learning.
4. **Proposed Model: R-STNet**
   - Full Dual-Perspective Reliability-Aware Spatio-Temporal Network with 4-way discrimination engine.

---

## 3. Controlled Sensor Corruption Suite (10 Experiments)

All experiments run across identical chronological splits with fixed random seeds (`seed=42`). Synthetic corruption is applied **only to the evaluation sets** (test and validation), preserving the original raw data.

| Exp # | Experiment Name | Corruption Type | Severity / Parameter | Affected Channels | Target Metric Objective |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **Exp 1** | **Clean Benchmark** | None (Original test data) | 0% corruption | None | Establish pristine baseline F1 & PR-AUC. |
| **Exp 2** | **Missing Data (5%)** | Random block missing (zero / forward fill) | 5% random intervals | Random 10% of sensors | Evaluate resilience to packet loss. |
| **Exp 3** | **Missing Data (10%)** | Random block missing | 10% random intervals | Random 10% of sensors | Measure degradation slope. |
| **Exp 4** | **Missing Data (20%)** | Random block missing | 20% random intervals | Random 20% of sensors | Stress test missing data handling. |
| **Exp 5** | **Missing Data (30%)** | Random block missing | 30% random intervals | Random 30% of sensors | Extreme degradation limit test. |
| **Exp 6** | **Low Sensor Noise** | Additive Gaussian Noise $\mathcal{N}(0, \sigma^2)$ | $\sigma = 0.1 \cdot \text{std}(x_i)$ | 3 critical sensors (FIT101, LIT101, AIT202) | Test noise filtering / reliability scoring. |
| **Exp 7** | **High Sensor Noise** | Additive Gaussian Noise $\mathcal{N}(0, \sigma^2)$ | $\sigma = 0.5 \cdot \text{std}(x_i)$ | 3 critical sensors | Test false positive suppression under noise. |
| **Exp 8** | **Single Sensor Failure** | Sensor Drift & Stuck Sensor | Drift: $\Delta x = \alpha t$; Stuck: $x(t) = \text{const}$ | Single sensor: LIT101 (Tank 1 Level) | **Critical test:** Must flag SENSOR FAULT without false SYSTEM ANOMALY. |
| **Exp 9** | **Multi-Sensor Failure** | Mixed Drift + Stuck | 2 sensors: LIT101 stuck + FIT201 drifting | 2 sensors in adjacent stages | Measure multi-sensor fault localization. |
| **Exp 10**| **Mixed Complex Corruption**| Noise + Spikes + Missing | 10% noise + 2% spikes + 5% missing | 5 sensors simultaneously | Holistic real-world harsh environment test. |

---

## 4. Evaluation Metrics

Because industrial anomaly datasets are heavily imbalanced (attacks typically represent 5% to 12% of total operational time), **accuracy is strictly avoided** as a primary metric.

### 4.1. System-Level Anomaly Detection Metrics
- **Precision (P):** $\frac{\text{TP}}{\text{TP} + \text{FP}}$ (Penalizes false alarms).
- **Recall (R):** $\frac{\text{TP}}{\text{TP} + \text{FN}}$ (Penalizes missed attacks).
- **F1-Score ($F_1$):** $2 \cdot \frac{P \cdot R}{P + R}$ (Harmonic mean).
- **PR-AUC (Precision-Recall Area Under Curve):** Threshold-independent evaluation suitable for severe class imbalance.
- **False Positive Rate (FPR):** $\frac{\text{FP}}{\text{FP} + \text{TN}}$ (Monitored specifically during sensor corruption experiments).

### 4.2. Sensor Reliability & Fault Discrimination Metrics
- **Sensor Fault Identification Accuracy:**  
  Percentage of time steps where the injected faulty sensor is correctly identified as $\arg\max_i (1 - r_i(t))$.
- **False Alarm Suppression Rate (FASR):**  
  $$\text{FASR} = 1 - \frac{\text{FPR}_{\text{corrupted}} - \text{FPR}_{\text{clean}}}{\text{FPR}_{\text{baseline\_corrupted}} - \text{FPR}_{\text{baseline\_clean}}}$$
- **Reliability Calibration Error:**  
  Brier score or Expected Calibration Error (ECE) comparing reliability scores to actual ground-truth corruption injection masks.

---

## 5. Ablation Studies Matrix

To scientifically validate each architectural component, we evaluate four model variants on SWaT under Clean (Exp 1), High Noise (Exp 7), and Single Sensor Failure (Exp 8):

1. **Variant A (No SRM):** Standard spatio-temporal backbone with uniform residual sum (Baseline 2).
2. **Variant B (Temporal SRM Only):** R-STNet using only temporal rate-of-change and stuck detection ($v_i$).
3. **Variant C (Graph SRM Only):** R-STNet using only cross-sensor graph consistency ($c_i$).
4. **Variant D (Full R-STNet):** Complete triangulated SRM (Temporal + Graph + Residual Z-Score) with 4-way decision logic.

---

## 6. Machine-Readable Experiment Output Schema

All experiment runs save structured JSON records to `results/metrics/`:
```json
{
  "timestamp": "2026-10-21T14:30:00Z",
  "dataset": "SWaT",
  "model": "R-STNet",
  "experiment_id": "exp_08_single_sensor_failure",
  "corruption": {
    "type": "stuck_sensor",
    "affected_channels": ["LIT101"],
    "severity": 1.0,
    "seed": 42
  },
  "metrics": {
    "precision": 0.884,
    "recall": 0.842,
    "f1": 0.862,
    "pr_auc": 0.891,
    "fpr": 0.018,
    "sensor_fault_accuracy": 0.945,
    "operational_state_distribution": {
      "case_a_normal": 0.82,
      "case_b_sensor_fault": 0.08,
      "case_c_system_anomaly": 0.09,
      "case_d_system_plus_sensor": 0.01
    }
  },
  "hardware": {
    "device": "NVIDIA GeForce RTX 3050 Laptop GPU",
    "peak_vram_mb": 940,
    "train_time_sec": 412
  }
}
```
Manually typed experimental numbers are strictly prohibited.
