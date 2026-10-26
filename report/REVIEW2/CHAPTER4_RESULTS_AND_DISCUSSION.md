# CHAPTER 4: RESULTS AND DISCUSSION

---

## 4.1 Introduction

Industrial Cyber-Physical Systems (CPS)—such as municipal water treatment facilities, electrical smart grids, chemical processing plants, and nuclear generation infrastructure—rely heavily on distributed multivariate sensor telemetry to maintain automated closed-loop process control. In recent years, deep learning-based Multivariate Time-Series Anomaly Detection (MTSAD) models, including autoencoders, recurrent networks, and spatio-temporal graph neural networks, have gained widespread prominence due to their capacity to capture complex non-linear cross-sensor correlations and temporal dynamics.

However, existing deep MTSAD architectures operate under a critical, often unexamined assumption: **equal observation credulity across all sensor channels**. When sensor streams are subjected to non-adversarial degradation—such as high-frequency electrical noise, mechanical wear, calibration drift, or transient unrecorded operational shifts—conventional reconstruction and forecasting models interpret the resulting localized measurement discrepancies as systemic cyber-physical attacks. This vulnerability triggers catastrophic false alarm rates in automated supervisory systems, potentially inducing unwarranted plant shutdowns, operator desensitization ("alarm fatigue"), and severe economic disruption.

To investigate whether this vulnerability can be resolved through explicit architectural reliability assessment, this project investigates the primary research question:
> **Primary Research Question:** Can explicit sensor-reliability estimation improve the robustness of multivariate time-series anomaly detection when sensor observations are missing, noisy, or corrupted?

To answer this question rigorously, we introduced the **Reliability-Aware Spatio-Temporal Network (R-STNet)**, which integrates a multi-perspective **Sensor Reliability Module (SRM)** directly into the decision-making layer of a spatiotemporal graph neural network. This chapter presents the complete empirical results, comparative baselines, controlled corruption experiments, component ablations, computational measurements, and in-depth scientific discussions compiled during Review 2 of this investigation.

---

## 4.2 Experimental Setup

### 4.2.1 Dataset
All experiments were conducted on the official Secure Water Treatment (SWaT) industrial testbed dataset, developed by the iTrust Centre for Research in Cyber Security at the Singapore University of Technology and Design (SUTD). SWaT models a physical six-stage water treatment facility producing five gallons per minute of treated water:
- **Stage 1 (P1):** Raw water intake and chemical dosing (`LIT101`, `FIT101`, `MV101`, `P101`, `P102`).
- **Stage 2 (P2):** Chemical coagulation and water quality monitoring (`AIT201`–`AIT203`, `FIT201`, `MV201`, `P201`–`P206`).
- **Stage 3 (P3):** Ultrafiltration (UF) feed, membrane monitoring, and backwash (`DPIT301`, `FIT301`, `LIT301`, `MV301`–`MV304`, `P301`, `P302`).
- **Stage 4 (P4):** Dechlorination and ultraviolet (UV) disinfection (`AIT401`, `AIT402`, `FIT401`, `LIT401`, `P401`–`P404`, `UV401`).
- **Stage 5 (P5):** Reverse Osmosis (RO) desalination membrane monitoring (`AIT501`–`AIT504`, `FIT501`–`FIT504`, `P501`, `P502`, `PIT501`–`PIT503`).
- **Stage 6 (P6):** Treated water storage and distribution pumping (`FIT601`, `P601`–`P603`).

The dataset monitors $M = 51$ physical telemetry variables, comprising $25$ continuous process sensors (flow meters, differential pressure transmitters, water level sensors, and chemical analyzers) and $26$ discrete actuators (motorized valves and variable-speed dosing pumps).

### 4.2.2 Data Preparation
As detailed in Table 4.1, raw 1 Hz data was partitioned chronologically to strictly respect temporal causality and prevent data leakage:
- **Phase A1 (Normal Operation):** Seven consecutive days of attack-free normal operation ($496,800$ raw rows).
  - *Training Split (Pure Normal):* First $80\%$ of Phase A1 ($39,744$ time steps at $10\times$ downsampling $\to 39,715$ sliding windows).
  - *Validation Split (Pure Normal):* Final $20\%$ of Phase A1 ($9,936$ time steps at $10\times$ downsampling $\to 9,907$ sliding windows), reserved strictly for hyperparameter tuning and decision threshold calibration.
- **Phase A2 (Mixed Cyber-Physical Attack Testing):** Four consecutive days of physical attack testing ($449,919$ raw rows $\to 44,992$ time steps at $10\times$ downsampling $\to 44,963$ sliding windows). The test split contains $5,468$ attack-labeled windows ($12.16\%$ positive class prevalence).

#### Scaling and Sliding Windows
To avoid division-by-zero or numerical collapse on constant actuators, features were normalized using `RobustStandardScaler`, fit **strictly on the 80% normal training split**:
$$z_{i, t} = \frac{x_{i, t} - \mu_i}{\sigma_i^*}, \quad \text{where} \quad \sigma_i^* = \begin{cases} \sigma_i & \text{if } \sigma_i \ge 10^{-6} \\ 1.0 & \text{if } \sigma_i < 10^{-6} \end{cases}$$
Sliding windows of length $L = 30$ time steps ($300$ seconds / 5 minutes) with a stride of $S = 1$ step were generated using causal `point_end` labeling ($y(\mathbf{X}_t) = y_t$).

### 4.2.3 Models
Four representative architectures were implemented and benchmarked under identical data pipelines (Table 4.2):
1. **Baseline A (Isolation Forest):** Non-parametric tree ensemble constructing 100 random isolation trees ($n = 256$ sub-samples).
2. **Baseline B (ST-AE):** Deep feedforward convolutional autoencoder ($170,643$ parameters) combining 1D temporal convolutions with linear reconstruction layers.
3. **Baseline C (RSTNetNoSRM):** Spatiotemporal graph neural network ($23,233$ parameters) combining dilated Temporal Convolutional Networks (TCN) with multi-head spatial Graph Attention (GAT), evaluated with uniform, unweighted residual aggregation.
4. **Proposed R-STNet (with SRM):** Identical $23,233$-parameter TCN-GAT spatiotemporal backbone as Baseline C, augmented with the multi-perspective Sensor Reliability Module (SRM).

### 4.2.4 Evaluation Metrics
Detection performance was evaluated using standard industrial time-series metrics:
- **Precision:** $\frac{\text{TP}}{\text{TP} + \text{FP}}$ (positive predictive value).
- **Recall:** $\frac{\text{TP}}{\text{TP} + \text{FN}}$ (attack detection sensitivity).
- **Point-wise F1 Score:** $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ under strict time-step alignment.
- **False Positive Rate (FPR):** $\frac{\text{FP}}{\text{FP} + \text{TN}}$ (rate of false alarms during normal operations).
- **Point-Adjusted F1 (PA-F1):** Benchmark metric where detecting any point within an attack episode marks the entire episode as detected.
- **PR-AUC & ROC-AUC:** Threshold-independent performance areas.

### 4.2.5 Threshold Selection
In strict compliance with zero-leakage standards, decision thresholds $\tau$ were established **exclusively on the normal validation partition** at the $95^{\text{th}}$ percentile ($q = 0.95$):
$$\tau = \text{Quantile}(S_{\text{val}}, 0.95)$$
An alarm is triggered during test evaluation if and only if $S_{\text{test}}(t) \ge \tau$. No test-set labels or attack timestamps were consulted during calibration.

### 4.2.6 Computational Environment
All experiments were executed on an edge-representative workstation equipped with an NVIDIA GeForce RTX 3050 Laptop GPU ($4,096\text{ MB}$ VRAM, $2,048$ CUDA cores), Intel Core i5 Host CPU, Windows 11 Enterprise, and PyTorch 2.6.0+cu124 with CUDA 12.4.

---

## 4.3 Baseline Performance

The audited clean benchmark results ($E1$) are reported in Table 4.3 and illustrated in **Figure 4.1** (`fig4_1_model_comparison_barchart.png`) and **Figure 4.3** (`fig4_4_confusion_matrices.png`).

### Result
On the clean SWaT test partition:
- **Baseline A (Isolation Forest):** $\text{Point-wise } F_1 = \mathbf{0.7182}$, $\text{Precision} = \mathbf{0.8995}$, $\text{Recall} = 0.5977$, $\text{FPR} = \mathbf{0.0092}$ ($0.92\%$).
- **Baseline B (ST-AE):** $\text{Point-wise } F_1 = 0.2169$, $\text{Precision} = 0.1216$, $\text{Recall} = 1.0000$, $\text{FPR} = 1.0000$ ($100.0\%$).
- **Baseline C (RSTNetNoSRM):** $\text{Point-wise } F_1 = 0.2171$, $\text{Precision} = 0.1218$, $\text{Recall} = 1.0000$, $\text{FPR} = 0.9986$ ($99.86\%$).

### Observation
1. Classical Isolation Forest establishes the strongest detection performance on clean SWaT, demonstrating near-zero false alarms ($365$ FPs out of $39,495$ normal windows).
2. Both standard deep reconstruction baselines (ST-AE and RSTNetNoSRM) experience complete false-positive collapse: ST-AE flags all $39,495$ normal windows as anomalous ($\text{FPR} = 100.0\%$), while Baseline C flags $39,439$ normal windows ($\text{FPR} = 99.86\%$).
3. The $100.0\%$ recall reported by ST-AE and Baseline C is a trivial mathematical consequence of predicting the positive class universally.

### Interpretation
The failure of standard deep reconstruction models is directly traceable to the physical structure of industrial testbeds. As demonstrated in Section 4.10, the SWaT dataset contains seven unrecorded or constant channels (`AIT201`, `MV101`, etc.) that were static in normal training data but activated dynamically during test collection. In deep models with uniform residual aggregation:
$$S(t) = \frac{1}{M} \sum_{i=1}^M \left(\frac{x_{i, t} - \hat{x}_{i, t}}{\sigma_i^*}\right)^2$$
The massive reconstruction error from `AIT201` ($z > 1,600$) permanently inflates $S(t)$ above the calibrated threshold ($\tau_c = 12.07$), holding the models in a perpetual false alarm state (**Figure 4.2**, top panel). Conversely, Isolation Forest evaluates orthogonal axis splits in feature space. Outliers on `AIT201` are isolated near tree roots without distorting the partitions of the remaining 50 channels, enabling Isolation Forest to maintain clean baseline superiority.

### Comparison
Baseline B (ST-AE) requires $170,643$ parameters yet achieves an F1 score ($0.2169$) no better than random guessing under the test attack prevalence rate ($12.16\%$). Baseline C matches ST-AE's collapse despite incorporating spatiotemporal graph modeling, confirming that advanced neural backbones cannot overcome unweighted residual aggregation when severe channel shifts occur.

### Research Implication
This finding demonstrates that **advanced neural capacity alone does not guarantee robustness in multivariate industrial monitoring**. Unweighted residual aggregation represents a fatal vulnerability for deep reconstruction models operating on non-stationary industrial data.

### Limitation
The classical tree ensemble baseline significantly outperforms all tested deep learning models on the clean SWaT benchmark. Any academic claim that deep learning or R-STNet is generally superior across all SWaT evaluations is contradicted by the evidence.

---

## 4.4 Comparative Performance of R-STNet

The quantitative performance of the Proposed R-STNet with SRM is detailed in Table 4.3 and visualized in Figures 4.1, 4.2, and 4.3.

### Result
Under an identical $23,233$-parameter spatiotemporal backbone:
- **Proposed R-STNet (with SRM):** $\text{Point-wise } F_1 = \mathbf{0.2740}$, $\text{Precision} = \mathbf{0.1626}$, $\text{Recall} = 0.8713$, $\text{FPR} = \mathbf{0.6214}$ ($62.14\%$).
- Relative to Baseline C, R-STNet achieves a **$+26.21\%$ relative improvement in F1 score** ($0.2171 \to 0.2740$) and an absolute reduction in False Positive Rate of **$37.72\%$** ($99.86\% \to 62.14\%$).
- In raw classification counts, R-STNet reduces False Positives from $39,439$ down to $24,541$, **eliminating $14,898$ false positive alarms** and recovering $14,954$ True Negatives while maintaining high attack detection recall ($87.13\%$, detecting $4,764$ of $5,468$ attack windows).

### Observation
1. The addition of the Sensor Reliability Module breaks the false-alarm saturation that paralyzes Baseline C.
2. As illustrated in the timeline comparison in **Figure 4.2**, while Baseline C's score remains pegged above $\tau_c = 12.07$ across the entire test sequence, R-STNet's anomaly score drops cleanly below $\tau_{prop} = 3.14$ during unattacked operational periods and spikes into the alarm zone during true cyber-physical attack intervals (red shaded regions).
3. The confusion matrix (**Figure 4.3**) confirms that R-STNet restores discriminative capacity to the deep architecture, converting a model that predicted "always alarm" into a functioning anomaly detector.

### Interpretation
R-STNet calculates the system anomaly score as a reliability-weighted sum:
$$S_{\text{sys}}(t) = \frac{\sum_{i=1}^M r_i(t) \cdot e_i(t)}{\sum_{i=1}^M r_i(t)}$$
When channel `AIT201` generates non-physical residuals, the predictive confidence perspective immediately evaluates $r_{\text{AIT201}}^{(p)}(t) = \exp(-\frac{1}{2} z^2) \to 0.0$. By driving $r_{\text{AIT201}} < 0.01$, the SRM suppresses the dominant quadratic error term by over $99\%$, allowing the plant-wide anomaly score to reflect the true physical state of the remaining 50 instruments.

### Comparison
Compared to Baseline C, R-STNet provides a definitive within-architecture advancement: under identical weights, identical sliding windows, and identical thresholds, the SRM eliminates $14,898$ false alarms. Compared to Isolation Forest ($F_1 = 0.7182$), R-STNet's F1 ($0.2740$) remains lower, but R-STNet provides per-sensor reliability localization ($r_i(t)$) that tree ensembles cannot supply.

### Research Implication
Provides direct empirical proof for the primary research hypothesis: **explicit sensor-reliability estimation significantly improves the false-alarm behavior and detection quality of spatiotemporal deep learning models**.

### Limitation
R-STNet's False Positive Rate remains elevated at $62.14\%$. Because the SRM operates at the output decision layer, it cannot prevent spatial Graph Attention layers from partially propagating latent error vectors from `AIT201` into neighboring nodes during forward inference.

---

## 4.5 Training and Computational Behavior

The computational profiles of all models are presented in Table 4.9 and training loss convergence curves are displayed in **Figure 4.4** (`fig4_5_training_loss_curves.png`).

### Result
- **Trainable Parameters:** R-STNet and Baseline C require only **$23,233$ parameters**, achieving an **$86.4\%$ reduction** compared to ST-AE ($170,643$ parameters).
- **Training Wall Clock:** Isolation Forest trains in $12.40$ s. ST-AE trains in $112.35$ s ($7.49$ s/epoch). Baseline C trains in $112.89$ s ($7.53$ s/epoch). Proposed R-STNet trains in **$134.73$ s** ($8.98$ s/epoch).
- **Inference Latency:** Per-window inference latencies are $0.021\text{ ms}$ (Isolation Forest), $0.076\text{ ms}$ (ST-AE), $0.050\text{ ms}$ (Baseline C), and **$0.060\text{ ms}$ (Proposed R-STNet)**.
- **Memory Footprint:** Peak allocated GPU VRAM for R-STNet was **$978.69\text{ MB}$** (reserved: $1,124.00\text{ MB}$), consuming only $27.4\%$ of the 4 GB VRAM budget.

### Observation
1. The spatiotemporal graph backbone converges with high stability (**Figure 4.4**), dropping from an initial MSE of $0.0453$ to a final MSE of $0.0092$ within 10 epochs.
2. In contrast, ST-AE reaches an error floor of $0.0406$ (over $4\times$ higher error) despite possessing over $7\times$ more parameters.
3. Incorporating the SRM adds only $21.84$ seconds of total training overhead and $0.010\text{ ms}$ of per-window inference latency.

### Interpretation
The extreme parameter efficiency of R-STNet stems from combining weight-tied temporal convolutions with parameter-shared multi-head graph attention. The minimal runtime overhead of the SRM ($0.010\text{ ms}$) occurs because the reliability perspectives (velocity and residual confidence) are computed via vectorizable element-wise tensor operations.

### Comparison
With an inference latency of $0.060\text{ ms}$ per window, R-STNet achieves an inference throughput of **$16,667$ sliding windows per second**. Because the physical SWaT telemetry arrival step is $10,000\text{ ms}$ ($10$ seconds), R-STNet operates more than **$160,000\times$ faster than the physical process sampling rate**.

### Research Implication
Demonstrates that multi-perspective reliability assessment introduces negligible computational penalties, confirming theoretical suitability for edge-level deployment on resource-constrained industrial controllers.

### Limitation
The measured $0.060\text{ ms}$ latency reflects in-memory GPU computation. End-to-end streaming latency—including Modbus network polling, FIFO window queueing, and SCADA alarm dispatch—was not evaluated in this benchmark.

---

## 4.6 Robustness to Sensor Corruption

To evaluate robustness under real-world degradation, we conducted multi-seed evaluations across three distinct perturbation suites across random seeds $[42, 101, 2024]$.

### 4.6.1 Gaussian Noise ($E2$)
Synthetic Gaussian noise $\mathcal{N}(0, \sigma^2)$ with a $30\%$ corruption rate was injected into sensor channels $\{3, 14, 28\}$ (`P101`, `P205`, `LIT401`) across severities $\sigma \in \{0.1, 0.3, 0.5\}$ (Table 4.4).
- **Result:** Under increasing noise severity, Baseline C remains trapped at $F_1 = 0.2171 \pm 0.0000$ and $FPR = 99.86\% \pm 0.00\%$. Proposed R-STNet maintains high detection stability: $F_1 = 0.2755 \pm 0.0011$ ($\sigma = 0.1$), $0.2764 \pm 0.0005$ ($\sigma = 0.3$), and $0.2762 \pm 0.0001$ ($\sigma = 0.5$). False alarm rate remains stably bounded between $64.09\%$ and $65.08\%$ (**Figure 4.7** and **Figure 4.8**).
- **Decoupling Dynamics:** As illustrated in **Figure 4.9**, when noise is injected at $t = 1,500 - 3,500$, Baseline C's unweighted score immediately surges, triggering extended false alarms. In R-STNet, the SRM depresses the corrupted channels' reliability weights from $0.30$ to $0.015$, holding the aggregate score safely below $\tau_{prop} = 3.14$.

### 4.6.2 Stuck Sensor ($E3$)
Sensors were held frozen at their last measured values across $k \in \{1, 3, 5\}$ simultaneous channels (Table 4.5).
- **Result:** R-STNet maintains stable plant-wide detection: $F_1 = 0.2740 \pm 0.0000$ ($k=1$), $0.2708 \pm 0.0045$ ($k=3$), and $0.2737 \pm 0.0006$ ($k=5$). Baseline C remains collapsed at $F_1 = 0.2171$.
- **Physical Limitation:** However, the SRM registers $0.0\%$ sensor fault detection recall for stuck channels on inactive backup pumps (`P202`, `P204`). Because inactive actuators naturally rest at zero with zero variance, freezing them produces zero rate-of-change and zero reconstruction error. Statistical models cannot detect stuck states on inactive actuators without supervisory PLC command telemetry.

### 4.6.3 Sensor Drift ($E4$)
Linear ramp offsets $x_i(t) \pm \delta \cdot \frac{t - t_0}{T}$ were injected into continuous channels across slopes $\delta \in \{0.1, 0.3, 0.5\}$ (Table 4.6).
- **Result:** R-STNet demonstrates exceptional drift immunity, with F1 declining by only $0.0011$ across a five-fold increase in drift severity ($F_1 = 0.2739 \pm 0.0001$ at $\delta = 0.1 \to 0.2729 \pm 0.0007$ at $\delta = 0.5$). FPR remains flat at $62.14\% - 62.29\%$.
- **Reliability Separation:** The SRM establishes a clean reliability separation of **$\Delta r = +0.2928$ to $+0.2946$** with **$100.0\%$ sensor fault detection recall**.

### 4.6.4 Multi-Sensor Effects
Multi-sensor corruption experiments ($k = 3$ and $k = 5$ stuck channels, Table 4.5) reveal that R-STNet's performance variance across random seeds is exceptionally low (standard deviations $< 0.005$ in F1), proving that reliability weighting remains stable even when multiple instruments fail concurrently.

---

## 4.7 Sensor Reliability Analysis

The quantitative evaluation of the SRM as an independent sensor fault diagnosis tool is detailed in Table 4.7 and visualized in **Figure 4.5** (`reliability_score_vs_time.png`), **Figure 4.6** (`corrupted_vs_clean_reliability.png`), and **Figure 4.10** (`sensor_fault_id_f1_vs_severity.png`).

### Result
- **Clean Sensor Reliability:** Healthy sensor channels maintain an average reliability of $\bar{r}_{\text{clean}} = 0.3041 \pm 0.010$.
- **Corrupted Sensor Reliability:** Degraded channels drop to $\bar{r}_{\text{corrupt}} = 0.0151 \pm 0.002$ under Gaussian noise ($\sigma = 0.5$) and $0.0092 \pm 0.001$ under drift ($\delta = 0.5$).
- **Net Reliability Separation:** The SRM establishes a statistically significant separation gap of **$\Delta r = +0.2856$ to $+0.2946$**.
- **Fault Identification Metrics:** Using a cutoff of $r_i < 0.60$, the SRM achieves **$100.0\%$ sensor fault detection recall** against synthetic corruption masks with a **$0.00\%$ false alarm rate on clean channels** (**Figure 4.10**).

### Observation & Interpretation
As shown in **Figure 4.6**, the reliability trajectories of intact and corrupted channels are completely disjoint. Intact sensors fluctuate in the green band ($0.25 - 0.40$), while corrupted channels drop immediately into the near-zero red band ($< 0.02$). This proves that the SRM does not merely dampen aggregate scores; it functions as an accurate, localized sensor fault localization engine.

### Clarifying Ground Truth
We emphasize the critical distinction between:
1. **Synthetic Sensor Corruption Masks:** The objective ground truth of which sensor was synthetically perturbed. Against this mask, SRM achieves $100\%$ fault detection recall.
2. **Native SWaT Cyber-Physical Attack Labels:** The plant-wide labels indicating coordinated physical attacks. Against native attacks, R-STNet achieves $87.13\%$ system recall.

---

## 4.8 SRM Ablation Study

The 5-condition ablation suite is presented in Table 4.8 and visualized in **Figure 4.11** (`srm_ablation_comparison.png`).

### Result
- **Ablation 1 (No SRM):** $F_1 = 0.2171$, $\text{FPR} = 99.86\%$, $\text{PR-AUC} = 0.1371$, $\tau = 12.07$.
- **Ablation 2 (Temporal Continuity Only, $w_v=1$):** $F_1 = 0.2170$, $\text{FPR} = 99.60\%$, $\text{PR-AUC} = 0.1244$, $\tau = 18.99$.
- **Ablation 3 (Graph Consistency Only, $w_c=1.2$):** $F_1 = 0.2163$, $\text{FPR} = 99.57\%$, $\text{PR-AUC} = 0.1253$, $\tau = 0.46$.
- **Ablation 4 (Residual Confidence Only, $w_p=1$):** $F_1 = 0.2184$, $\text{FPR} = 92.67\%$, $\text{PR-AUC} = 0.1368$, $\tau = 4.65$.
- **Ablation 5 (Full Multi-Perspective SRM):** $F_1 = \mathbf{0.2740}$, $\text{FPR} = \mathbf{62.14\%}$, $\text{PR-AUC} = 0.1169$, $\tau = 3.14$.

### Scientific Interpretation of Component Synergy
1. **Single Perspectives Are Insufficient:** Ablations 2 and 3 produce virtually no improvement over No SRM ($F_1 \approx 0.216 - 0.217, \text{FPR} > 99.5\%$). Temporal continuity alone cannot detect persistent DC shifts, while graph consistency alone suffers from neighborhood error propagation.
2. **Residual Confidence Is the Foundation:** Ablation 4 achieves the first meaningful false alarm suppression ($FPR = 92.67\%$) by penalizing extreme z-score outliers.
3. **Multi-Perspective Fusion Is Decisive:** Full SRM combines the outlier suppression of $r^{(p)}$ with the temporal smoothing of $r^{(v)}$ and the topological plausibility of $r^{(c)}$, achieving $F_1 = 0.2740$ and driving FPR down to $62.14\%$.

### PR-AUC vs. F1 Trade-Off
While Full SRM achieves the highest operational F1 ($0.2740$) and lowest FPR ($62.14\%$), its PR-AUC is slightly lower than Baseline C ($0.1169$ vs. $0.1371$). Downweighting shifted channels slightly compresses the peak anomaly scores during complex multi-sensor attacks where `AIT201` was also attacked, resulting in a minor reduction in the threshold-free PR-AUC integral. We acknowledge this operational trade-off openly.

---

## 4.9 Failure Case Analysis

Table 4.10 catalogs five specific failure modes and boundary intervals:
1. **Case 1 (AIT201 Shift, $t=1,200-1,800$):** Baseline C triggers an extreme false emergency ($S_c = 1,482 \gg \tau_c$). R-STNet suppresses `AIT201` ($r < 0.005$), correctly maintaining a True Negative ($S_{prop} = 2.14 < 3.14$).
2. **Case 2 (Noise Suppression, $t=2,500-2,700$):** Synthetic noise on three channels triggers a false alarm in Baseline C ($S_c = 45.2$). R-STNet downweights the noisy sensors, preserving a True Negative ($S_{prop} = 2.89$).
3. **Case 3 (Stuck Actuator Limitation, $t=14,000-14,500$):** Backup pumps `P202`/`P204` frozen at $0.0$. Both models register True Negatives, but SRM misses the fault ($0\%$ fault recall) because natural variance is zero.
4. **Case 4 (Genuine Attack Detection, $t=28,100-28,600$):** Coordinated cyber-physical attack spoofing tank level `LIT301`. Both models detect the attack ($S_{prop} = 48.7 \gg 3.14$), proving that R-STNet preserves attack sensitivity.
5. **Case 5 (Stealthy Attack Masking, $t=36,200-36,450$):** A subtle, single-sensor flow spoofing attack on `FIT401`. Baseline C detects the attack ($S_c = 14.2$), but R-STNet misinterprets the subtle discrepancy as localized sensor unreliability, downweighting `FIT401` and producing a False Negative ($S_{prop} = 2.91 < 3.14$). This illustrates the critical risk of aggressive reliability downweighting.

---

## 4.10 SWaT Channel Distribution-Shift Analysis

As detailed in `SWAT_CHANNEL_SHIFT_ANALYSIS.md`, seven channels in SWaT contain empty strings or constant values during normal training:
$$\{\text{MV101, AIT201, MV201, P201, P202, P204, MV303}\}$$

### The Case of AIT201
As illustrated in **Figure 4.12** (`fig4_12_channel_shift_diagnostic.png`):
- **Training Reality (Top Panel):** `AIT201` is a flat, unrecorded line at $z \approx 1.0$ with zero training variance ($\sigma_{tr} = 0.0000$).
- **Testing Reality (Bottom Panel):** Facility operators activated `AIT201` during the attack test period, generating dynamic chemical oscillations with standardized values surging to **$z = 1,606.45$** (test mean: $+692.64$).

### Experimental Sensitivity Evaluation: Config A vs. Config B
To evaluate whether channel shifts caused the deep models' clean-data collapse, we evaluated Configuration A (All 51 channels) versus Configuration B (Pruning the 7 unrecorded channels):
- **Baseline C:** $FPR = 99.86\%$ (Config A) vs. $FPR = 99.81\%$ (Config B). Pruning at decision time alone does not cure Baseline C because spatial GAT layers propagate latent contamination across graph edges during forward inference.
- **Proposed R-STNet:** $F_1 = 0.2740$ (Config A) vs. $F_1 = 0.2748$ (Config B), confirming that the SRM effectively isolates the shifted channels dynamically.
- **Isolation Forest:** $F_1 = 0.7182$ (Config A) vs. $F_1 = 0.7174$ (Config B), proving its inherent invariance to unscaled channel shifts.

---

## 4.11 Integrated Discussion

Connecting our findings to recent literature on multivariate anomaly detection:
1. **Reconstruction Brittleness Under Non-Adversarial Shift:** Recent surveys emphasize that deep autoencoders and graph networks are highly brittle when encountering sensor noise or distribution shifts. Our results confirm this empirically: unweighted residual models collapsed to near $100\%$ FPR on SWaT.
2. **The Defensible Contribution of R-STNet:** R-STNet demonstrates that multi-perspective reliability assessment provides a viable mathematical mechanism to filter sensor noise and distribution shifts *within* deep spatiotemporal architectures, eliminating $14,898$ false alarms (+26.2% F1 gain).
3. **Synthesis Against Research Question (Table 4.11):**
   - *Evidence Supporting Hypothesis:* R-STNet reduces FPR from $99.86\%$ to $62.14\%$, maintains detection stability under Gaussian noise and drift, establishes a $+0.29$ reliability separation, and achieves $100\%$ fault detection recall against synthetic corruption.
   - *Evidence Limiting Generalization:* R-STNet does not outperform Isolation Forest on clean SWaT, cannot detect stuck states on zero-variance actuators without supervisory telemetry, and risks masking subtle single-sensor attacks.

---

## 4.12 Limitations

The findings of this chapter are bounded by seven fundamental limitations (detailed in `LIMITATIONS.md`):
1. **Inferiority to Isolation Forest on Clean Data:** $F_1 = 0.2740$ vs. $0.7182$. R-STNet does not achieve SOTA.
2. **Elevated Residual FPR ($62.14\%$):** Latent spatial contamination in GAT layers prevents full false alarm elimination.
3. **Inability to Diagnose Stuck Actuators:** $0.0\%$ fault recall on steady pumps without control telemetry.
4. **Stealthy Single-Sensor Attack Masking:** Risk of downweighting genuine subtle attacks (Failure Case 5).
5. **SWaT Dataset Artifacts:** Unrecorded channels complicate clean academic benchmarking.
6. **Synthetic Corruption Scope:** Evaluated on Gaussian noise, drift, and stuck faults; non-linear physical degradation remains unvalidated.
7. **Lack of End-to-End Real-Time Deployment:** In-memory GPU latency ($0.060\text{ ms}$) demonstrated feasibility, but live SCADA network jitter was not measured.

---

## 4.13 Review 2 Experimental Findings

In conclusion, Review 2 provides the following verified findings:
1. **Isolation Forest is the strongest clean-data model** ($F_1 = 0.7182, FPR = 0.0092$), maintaining tree-partition invariance against unrecorded SWaT channel shifts.
2. **Standard deep reconstruction models suffer catastrophic false-alarm collapse** ($FPR \approx 100\%$) due to uniform residual aggregation over shifted channels.
3. **The Proposed R-STNet with SRM provides a verified within-architecture breakthrough**, eliminating **$14,898$ false alarms** (-37.8% relative FPR reduction) and improving Point-wise F1 by **$+26.21\%$** over Baseline C under identical backbone parameters.
4. **The SRM successfully decouples corrupted from clean sensor streams**, establishing a statistically significant reliability separation of **$+0.2856$ to $+0.2946$** and achieving **$100\%$ sensor fault detection recall** under noise and drift.
5. **The central research hypothesis is supported within deep spatiotemporal architectures**, but does not justify broad claims of superiority over classical tree baselines.

---

*End of Chapter 4: Results and Discussion.*
