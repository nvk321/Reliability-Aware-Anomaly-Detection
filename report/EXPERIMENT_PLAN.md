# Proposed Experimental Plan & Evaluation Protocol (Phase 2 Roadmap)

**Project:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Review Target:** Review 1 (Phase 1)  
**Execution Phase:** Phase 2 (Post-Review 1)  
**Notice:** This document defines the formal experimental design, working hypotheses, corruption scenarios, ablation matrix, and evaluation metrics that will be executed in Phase 2. In strict accordance with Review 1 requirements, **no completed experimental results, performance claims, or final conclusions are presented here.**

---

## 1. Research Hypotheses to Be Tested

The Phase 2 experimental campaign is organized to systematically validate three core scientific hypotheses:

### Hypothesis 1: Robustness Under Sensor Corruption
> **Hypothesis 1:** *Under increasing channel-level sensor corruptions (Gaussian noise, calibration drift, frozen stuck values, and impulse spikes), conventional unweighted multivariate anomaly detectors (Isolation Forest, SpatioTemporal AE, and RSTNet without SRM) will experience a rapid explosion in False Positive Rate (FPR) and a severe degradation in F1-score during normal plant operation. In contrast, R-STNet will maintain a stable FPR and high F1-score by dynamically discounting corrupted channels from the system anomaly score.*

### Hypothesis 2: Four-Way Operational Discrimination
> **Hypothesis 2:** *When an isolated sensor experiences hardware degradation while the monitored cyber-physical plant remains in normal equilibrium, the 4-way decision engine will correctly classify the state as an isolated `SENSOR FAULT` (Case B) with $\ge 90\%$ accuracy, successfully suppressing false plant emergency shutdowns, while preserving immediate detection of genuine cyber-physical process attacks (`SYSTEM ANOMALY`, Case C).*

### Hypothesis 3: Component Ablation & Triangulation Necessity
> **Hypothesis 3:** *Cross-sensor graph neighborhood consistency provides the primary physical foundation for verifying process consensus, while temporal continuity is essential for detecting high-frequency spikes and zero-variance stuck sensors. Consequently, the complete triangulated SRM (Variant D) will achieve superior false-alarm suppression and fault localization compared to temporal-only (Variant B) or graph-only (Variant C) ablations.*

---

## 2. Candidate Baseline Models and Comparative Roles

To ensure rigorous, fair, and scientifically defensible benchmarking, we evaluate five distinct model variants:

| Model ID | Model Name | Category | Architectural Configuration | Primary Evaluation Purpose |
| :---: | :--- | :--- | :--- | :--- |
| **M1** | **Isolation Forest** | Classical Machine Learning | 100 random decision trees, sub-sampling size 256, evaluated on concatenated window features. | Establishes standard industrial off-the-shelf baseline performance. |
| **M2** | **SpatioTemporal Autoencoder (ST-AE)** | Deep Reconstruction Model | 1D dilated convolutional encoder with symmetric decoder; unweighted reconstruction error sum. | Standard unweighted deep learning reconstruction baseline. |
| **M3** | **RSTNet WITHOUT SRM** | Deep Graph Neural Network | Identical temporal convolution and graph attention backbone as R-STNet, but computes **unweighted** residual mean. | **Primary Scientific Ablation:** Isolates the exact empirical gain provided by the Sensor Reliability Module. |
| **M4** | **CAROTS (Kim et al., ICML 2025)** | External Research Baseline | Causal structural discovery + contrastive learning between causally consistent views (`reference/CAROTS/`). | Establishes external state-of-the-art reference performance under published ICML 2025 conventions. |
| **M5** | **Proposed R-STNet WITH SRM** | Proposed Reliability-Aware Framework | Full Spatio-Temporal Graph Attention + Triangulated SRM + 4-Way Operational Decision Engine. | Evaluates the proposed framework across all robustness, fault localization, and attack detection criteria. |

---

## 3. The 10-Experiment Controlled Corruption Suite

Synthetic corruption is injected deterministically into the evaluation test split using `src/corruption/corruptor.py` across three independent random seeds (`seed = 42, 101, 2024`). The raw training and validation sets remain pristine.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               PLANNED EXPERIMENTS MATRIX                               │
├─────────┬──────────────────────┬──────────────────────────────────┬────────────────────┤
│ Exp #   │ Experiment Scenario  │ Target Operational Failure Mode  │ Target Channels    │
├─────────┼──────────────────────┼──────────────────────────────────┼────────────────────┤
│ Exp 1   │ Clean Benchmark      │ Original uncorrupted test data   │ None               │
│ Exp 2   │ Low Gaussian Noise   │ Sensor transmitter noise (0.1)   │ FIT101, LIT101     │
│ Exp 3   │ Medium Gaussian Noise│ Signal conditioning noise (0.3)  │ FIT101, LIT101     │
│ Exp 4   │ High Gaussian Noise  │ Severe electrical noise (0.5)    │ FIT101, LIT101     │
│ Exp 5   │ Single Stuck Sensor  │ Float jammed / AD lockup         │ LIT101             │
│ Exp 6   │ Multi Stuck Sensors  │ Subsystem communication freeze   │ LIT101, AIT202     │
│ Exp 7   │ Incipient Drift      │ Slow membrane fouling drift (0.1)│ FIT101             │
│ Exp 8   │ Severe Drift         │ Rapid calibration shift (0.5)    │ FIT101             │
│ Exp 9   │ Impulse Spikes       │ Power transient ripples (3x std) │ Random 15% channels│
│ Exp 10  │ Mixed Harsh Scenario │ Noise + Drift + Stuck + Spikes   │ 5 channels in P1-P3│
└─────────┴──────────────────────┴──────────────────────────────────┴────────────────────┘
```

### Mathematical Definitions of Injected Corruptions:
1. **Additive Gaussian Noise:**  
   $$x_i^{\text{corrupt}}(t) = x_i(t) + \epsilon(t), \quad \epsilon(t) \sim \mathcal{N}\left(0, (\gamma \cdot \sigma_i)^2\right)$$
   where $\gamma \in \{0.1, 0.3, 0.5\}$ controls severity and $\sigma_i = \text{std}_{\text{train}}(x_i)$.
2. **Frozen / Stuck Sensor:**  
   $$x_i^{\text{corrupt}}(t) = x_i(t_{\text{start}}), \quad \forall t \in [t_{\text{start}}, t_{\text{start}} + \Delta t]$$
   The channel reading freezes at a constant value, causing instantaneous variance over lookback window $K$ to drop to zero ($\text{Var}(x_{i, t-K:t}) < \epsilon_{\text{stuck}}$).
3. **Linear Calibration Drift:**  
   $$x_i^{\text{corrupt}}(t) = x_i(t) + \alpha \cdot \frac{t - t_{\text{start}}}{\Delta t} \cdot \sigma_i, \quad \forall t \in [t_{\text{start}}, t_{\text{start}} + \Delta t]$$
   where $\alpha \in \{0.1, 0.3, 0.5\}$ specifies the total drift magnitude relative to normal channel variance.
4. **Impulse Spikes:**  
   $$x_i^{\text{corrupt}}(t) = x_i(t) \pm 3.0 \cdot \sigma_i$$
   applied randomly at a rate of $2\%$ of time steps.

---

## 4. Dual Ground-Truth Structure

The experimental suite generates **two independent ground-truth vectors**:

1. **System Anomaly Ground Truth ($y_{\text{sys}}(t) \in \{0, 1\}$):**
   - Derived directly from the official iTrust SWaT attack labels.
   - $y_{\text{sys}}(t) = 1$ during the 36 physical cyber-attacks; $0$ during normal operation.
   - Evaluates whether the plant-level monitoring system detects genuine process violations.
2. **Sensor Fault Ground Truth Mask ($M_{\text{fault}}(t, i) \in \{0, 1\}$):**
   - Generated automatically by `SensorCorruptor`.
   - $M_{\text{fault}}(t, i) = 1$ if sensor channel $i$ is actively corrupted at time $t$; $0$ otherwise.
   - Evaluates whether the Sensor Reliability Module accurately identifies which specific sensor is degraded.

---

## 5. Quantitative Evaluation Metrics

Industrial time-series anomaly detection datasets exhibit severe class imbalance (SWaT attack periods constitute $12.14\%$ of test time). Therefore, standard accuracy is strictly prohibited as an evaluation metric.

### 5.1. System-Level Anomaly Detection Metrics
- **Point-wise Precision ($P$):**  
  $$P = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
- **Point-wise Recall ($R$):**  
  $$R = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
- **Point-wise F1-Score ($F_1$):**  
  $$F_1 = 2 \cdot \frac{P \cdot R}{P + R}$$
- **Precision-Recall Area Under Curve (PR-AUC):**  
  $$\text{PR-AUC} = \int_0^1 P(R) \, dR$$
  Threshold-independent metric specifically suited for imbalanced time series.
- **False Positive Rate on Clean Operating Periods ($\text{FPR}_{\text{clean}}$):**  
  $$\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$
  Evaluated specifically during uncorrupted normal operational windows to verify that sensor noise does not induce false alarms.

### 5.2. Sensor Reliability and Fault Localization Metrics
- **False Alarm Suppression Rate (FASR):**  
  $$\text{FASR} = 1 - \frac{\text{FPR}_{\text{model}}^{\text{corrupt}} - \text{FPR}_{\text{model}}^{\text{clean}}}{\text{FPR}_{\text{baseline}}^{\text{corrupt}} - \text{FPR}_{\text{baseline}}^{\text{clean}}}$$
  Quantifies the percentage of sensor-induced false plant alarms successfully eliminated by the reliability mechanism.
- **Reliability Separation Margin ($\Delta r$):**  
  $$\Delta r = \bar{r}_{\text{clean}} - \bar{r}_{\text{corrupted}}$$
  Measures the contrastive separation between healthy sensors ($r \to 1.0$) and degraded sensors ($r \to 0.0$).
- **Sensor Fault Localization Precision, Recall, and F1:**  
  Evaluated point-wise by comparing the most degraded sensor index $\arg\max_i [(1 - r_i) z_i]$ against the synthetic corruption mask $M_{\text{fault}}(t, i)$.

### 5.3. Point-Adjustment Protocol Policy
In accordance with recent critical findings by Kim et al. (AAAI 2022) [40] and Wu & Keogh (IEEE TKDE 2021) [41], the Point-Adjustment (PA) protocol—which labels an entire contiguous anomaly segment as true positives upon detecting a single anomalous point—artificially inflates F1-scores.
- **Our Policy:** Raw point-wise metrics serve as the primary ground truth. Point-adjusted metrics are computed strictly as a labeled secondary reference (`PA-F1`) to enable comparison with historical literature.

---

## 6. Leak-Free Threshold Calibration Protocol

A common methodological flaw in literature is tuning the anomaly threshold $\tau$ directly on the test set to maximize test F1-score (an optimistic leakage flaw). We enforce a strict validation calibration protocol:

1. Models are trained exclusively on the 80% normal training split.
2. Anomaly scores $S_{\text{sys}}(t)$ and reliability vectors $\mathbf{r}(t)$ are computed on the clean 20% normal validation split.
3. The system anomaly decision threshold $\tau_{\text{sys}}$ is calibrated at a pre-specified quantile (e.g., 95th or 99th percentile) of validation normal scores:
   $$\tau_{\text{sys}} = \text{Quantile}(S_{\text{val}}, 0.95)$$
4. The reliability threshold $\theta_{\text{rel}}$ is set to the lower tail of normal validation reliability:
   $$\theta_{\text{rel}} = \max\left(0.40, \text{Quantile}(\min_i r_{i, \text{val}}, 0.01)\right)$$
5. **Thresholds $\tau_{\text{sys}}$ and $\theta_{\text{rel}}$ are completely frozen.**
6. The test split is evaluated blindly using the frozen thresholds.

---

## 7. Ablation Studies Matrix

To scientifically quantify the contribution of each component within the Sensor Reliability Module:

| Variant Code | Model Configuration | Temporal Continuity $v_i$ | Graph Consensus $c_i$ | Tail Confidence $p_i$ | Scientific Question Answered |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **Variant A** | RSTNet WITHOUT SRM (Uniform Mean) | **No** | **No** | **No** | What is the baseline vulnerability of unweighted spatio-temporal graph models? |
| **Variant B** | R-STNet with Temporal SRM Only | **Yes** | **No** | **No** | How much robustness is provided by rate-of-change and stuck-variance checks alone? |
| **Variant C** | R-STNet with Graph SRM Only | **No** | **Yes** | **No** | Can graph neighbor physical consensus detect calibration drift without temporal checks? |
| **Variant D** | Full Proposed R-STNet (Triangulated) | **Yes** | **Yes** | **Yes** | Does multi-signal evidential triangulation maximize false alarm suppression and fault localization? |

---

## 8. Computational Feasibility & Hardware Constraints

- **Execution Environment:** Windows 11 (64-bit), Python 3.11.0 virtual environment (`.venv/`), PyTorch 2.1+ with CUDA 12.1.
- **Hardware Acceleration:** NVIDIA GeForce RTX 3050 Laptop GPU (4096 MiB VRAM).
- **VRAM Budget:** Sequence window $L = 60$, batch size $B = 64$, temporal hidden dimension $64$, node dimension $32$. Peak training VRAM is bounded at $< 1.2\text{ GB}$, fitting comfortably within the 4.0 GB physical limit.
- **Runtime Reproducibility:** Fixed seeds (`42, 101, 2024`), deterministic cuDNN backend, and structured JSON results logging to `results/metrics/`.
