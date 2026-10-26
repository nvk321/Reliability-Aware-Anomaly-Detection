# RELIABILITY-AWARE ANOMALY DETECTION FOR MULTIVARIATE SENSOR DATA

**Course:** BCSE209P — Machine Learning Lab  
**Project Milestone:** Final Project Report & Thesis Package  
**Academic Year:** 2026  

### Students:
- **P Naga Venkata Kushal** — Register No: 23BDS0133
- **Koteru Tarun** — Register No: 23BDS0103

### Faculty Guide:
- **Dr. Dega Nagaraju**, Associate Professor

### Academic Institution:
School of Computer Science and Engineering  
Vellore Institute of Technology, India  

---

## ABSTRACT

Multivariate Time-Series Anomaly Detection (MTSAD) is vital for maintaining the physical safety, operational continuity, and cybersecurity of industrial Cyber-Physical Systems (CPS). Deep learning architectures—such as convolutional autoencoders and spatio-temporal graph attention networks—have become the dominant paradigm for industrial anomaly detection. However, existing deep models operate under a foundational vulnerability: **equal observation credulity across all sensor channels**. In operational industrial environments, physical sensors routinely experience non-adversarial degradation (high-frequency electrical noise, mechanical calibration drift, frozen sensor transducers, or transient uncalibrated operational shifts). When this occurs, conventional models with uniform residual aggregation interpret localized sensor faults as plant-wide cyber-physical attacks, inducing severe false alarm saturation, operator alarm fatigue, and unnecessary plant shutdowns.

To solve this problem, this thesis investigates whether explicit multi-perspective sensor-reliability estimation can improve the detection robustness of spatio-temporal deep neural architectures. We develop the **Reliability-Aware Spatio-Temporal Network (R-STNet)**, which integrates an auxiliary **Sensor Reliability Module (SRM)** directly into a Causal Dilated Temporal Convolutional Network (TCN) and Multi-Head Spatial Graph Attention (GAT) backbone. The SRM continuously computes a scalar reliability weight $r_i(t) \in [0, 1]$ for each sensor by fusing temporal rate-of-change continuity ($r^{(v)}$), graph neighborhood topological consistency ($r^{(c)}$), and predictive residual confidence ($r^{(p)}$), dynamically downweighting degraded channels at the decision layer.

The framework was benchmarked on the official Secure Water Treatment (SWaT) testbed dataset across 51 physical variables (25 continuous sensors and 26 discrete actuators) under a strict zero-leakage chronological protocol. On the clean benchmark ($E1$), classical tree-based Isolation Forest achieved the highest clean performance ($F_1 = 0.7182, FPR = 0.0092$) due to orthogonal tree cuts isolating unrecorded dataset channel shifts (`AIT201`) without joint space distortion. Within deep neural architectures, Proposed R-STNet achieved a **$+26.21\%$ relative improvement in F1 score** ($0.2171 \to 0.2740$) and an absolute reduction in False Positive Rate from **$99.86\%$ to $62.14\%$**, **eliminating $14,898$ false alarms** relative to Baseline C (RSTNetNoSRM) under identical backbone parameters ($23,233$). Under multi-seed controlled corruption suites ($[42, 101, 2024]$) across Gaussian noise ($\sigma \in [0.1, 0.5]$) and linear drift ($\delta \in [0.1, 0.5]$), the SRM established an empirical reliability separation gap of **$\Delta r = +0.2856\text{ to }+0.2946$** and achieved **$100.0\%$ sensor fault detection recall** against synthetic corruption masks. R-STNet executed inference in $0.060\text{ ms}$ per window ($16,667$ windows/s), operating $160,000\times$ faster than the physical process sampling rate. These findings support the research hypothesis within deep spatio-temporal neural architectures while transparently acknowledging that classical tree baselines remain superior on clean industrial data.

---

## TABLE OF CONTENTS

- **Chapter 1: Introduction**
  - 1.1 Context and Motivation
  - 1.2 Problem Statement
  - 1.3 Research Question and Hypothesis
  - 1.4 Scope and Objectives
  - 1.5 Thesis Organization
- **Chapter 2: Literature Review**
  - 2.1 Classical Anomaly Detection in Industrial Systems
  - 2.2 Deep Learning for Multivariate Time Series
  - 2.3 Graph Neural Networks & Spatial Dependency Modeling
  - 2.4 Sensor Fault Diagnosis vs. System Anomaly Detection
  - 2.5 Research Gaps and Theoretical Positioning
- **Chapter 3: Methodology**
  - 3.1 Mathematical Problem Formulation
  - 3.2 Spatio-Temporal Neural Backbone (TCN-GAT)
  - 3.3 Multi-Perspective Sensor Reliability Module (SRM)
  - 3.4 Reliability-Weighted System Anomaly Scoring
  - 3.5 Leak-Free Scaling and Validation Quantile Thresholding
- **Chapter 4: Results and Discussion**
  - 4.1 Experimental Setup and Dataset Specifications
  - 4.2 Baseline Performance Comparison
  - 4.3 Comparative Performance of R-STNet
  - 4.4 Training Convergence and Computational Benchmarks
  - 4.5 Robustness to Sensor Corruption (Noise, Stuck, Drift)
  - 4.6 Sensor Reliability Decoupling Analysis
  - 4.7 SRM Perspective Component Ablation Study
  - 4.8 Qualitative Failure Case Analysis
  - 4.9 SWaT Channel Distribution-Shift Diagnosis
  - 4.10 Integrated Discussion and Literature Context
  - 4.11 Scientific Limitations
- **Chapter 5: Conclusion**
  - 5.1 Work Carried Out in the Thesis
  - 5.2 Conclusions from Results and Discussion
  - 5.3 Managerial and Operational Implications
  - 5.4 Future Scope of the Work
- **References** (40 Verified IEEE Citations, 90.0% ScienceDirect)
- **Appendix** (Appendices A to J: Core Executable Code Repository)

---

# CHAPTER 1: INTRODUCTION

### 1.1 Context and Motivation
Modern critical infrastructure—such as municipal water purification facilities, oil and chemical refineries, power generation grids, and automated discrete manufacturing plants—operates as complex Cyber-Physical Systems (CPS). These plants integrate physical processes (chemical dosing, hydraulic pressure, thermal heat exchange) with distributed Supervisory Control and Data Acquisition (SCADA) networks, Programmable Logic Controllers (PLCs), and dense multivariate sensor arrays. Telemetry streams continuously convey real-time measurements of tank levels, flow rates, pipeline pressures, and chemical concentrations.

In recent years, the convergence of Operational Technology (OT) with Information Technology (IT) networks has exposed industrial systems to sophisticated cyber-physical attacks (e.g., Stuxnet, Industroyer, and the Florida water treatment poisoning incident). Consequently, automated anomaly detection has become a foundational layer of industrial defense. Traditional threshold alarms and statistical Process Control (SPC) techniques fail to capture the multi-dimensional, non-linear physical interactions that govern large-scale industrial plants. This has led to the widespread adoption of deep learning architectures, specifically unsupervised reconstruction models, to detect anomalies by identifying deviations from learned normal physical correlations.

### 1.2 Problem Statement
Despite substantial architectural advancements, current deep learning models for multivariate anomaly detection share an intrinsic, dangerous vulnerability: **they treat all incoming sensor observations with equal credulity**. In actual industrial deployments, physical sensors routinely degrade due to harsh operating environments, mechanical wear, calibration drift, fouling, electrical noise, or transient communication packet loss. 

When a physical sensor degrades, conventional reconstruction and forecasting architectures uniformly sum squared reconstruction residuals across all channels:
$$S(t) = \frac{1}{M} \sum_{i=1}^M \left(\frac{x_i(t) - \hat{x}_i(t)}{\sigma_i}\right)^2$$
A single noisy, drifting, or unrecorded sensor channel can generate massive residuals that completely dominate the aggregate anomaly score. As a result, the detection system triggers plant-wide emergency alarms during completely safe, nominal operational regimes. In real-world control rooms, this causes catastrophic operator "alarm fatigue," leading operators to silence, ignore, or disable automated safety alerts, ultimately leaving the plant defenseless when a genuine cyber-physical attack occurs.

### 1.3 Research Question and Hypothesis
To address this fundamental limitation, this thesis poses and experimentally answers the following core research question:
> **Primary Research Question:** Can explicit sensor-reliability estimation improve the robustness of multivariate time-series anomaly detection when sensor observations are missing, noisy, or corrupted?

We formulate the corresponding scientific hypothesis:
> **Core Hypothesis:** Explicit multi-perspective sensor-reliability estimation will significantly reduce false-alarm saturation and improve robustness to controlled sensor degradation within a spatio-temporal deep-learning anomaly-detection architecture, relative to uniform residual aggregation.

### 1.4 Scope and Objectives
The specific objectives of this research are:
1. **Develop R-STNet:** Engineer a compact, lightweight spatio-temporal deep learning architecture combining Causal Dilated Temporal Convolutions (TCN) and Spatial Multi-Head Graph Attention (GAT) with a dynamic Sensor Reliability Module (SRM).
2. **Formulate Multi-Perspective Reliability (SRM):** Design a mathematical framework fusing temporal continuity, topological graph consistency, and predictive residual confidence to decouple sensor unreliability from plant-wide anomalies.
3. **Establish Leak-Free Industrial Benchmarks:** Evaluate R-STNet on the 51-sensor Secure Water Treatment (SWaT) testbed under strict chronological partitioning and validation quantile thresholding ($\tau$).
4. **Evaluate Multi-Seed Robustness:** Test model stability under additive Gaussian noise, stuck sensor states, and linear drift across multiple random seeds.
5. **Conduct Component Ablations and Failure Diagnostics:** Systematically ablate SRM components and analyze qualitative failure intervals to identify physical operational boundaries.

---

# CHAPTER 2: LITERATURE REVIEW

### 2.1 Classical Anomaly Detection in Industrial Systems
Early industrial process monitoring relied heavily on statistical dimensionality reduction and classical machine learning. Dynamic Principal Component Analysis (DPCA) and Partial Least Squares (PLS) established linear multivariate relationships between sensor channels, flagging anomalies via Hotelling's $T^2$ and Squared Prediction Error (SPE) statistics [5], [7]. While computationally efficient and interpretable, linear methods struggle with non-linear multi-phase fluid dynamics and chemical reaction kinetics [6], [25].

To address non-linearities, non-parametric tree ensembles—most notably Isolation Forest [3]—were introduced. Isolation Forest constructs random orthogonal axis cuts to isolate anomalous instances near tree roots. As established in this thesis, Isolation Forest remains remarkably robust against uncalibrated single-channel shifts because tree splits evaluate feature subsets independently rather than aggregating residuals across all dimensions [36]. However, tabular tree ensembles do not explicitly model temporal sequential dynamics, cannot provide physical topological graph explanations, and lack mechanisms to output continuous sensor reliability weights [16].

### 2.2 Deep Learning for Multivariate Time Series
To capture temporal dynamics, deep autoencoders and recurrent neural networks (RNNs/LSTMs) were widely adopted for reconstruction-based anomaly detection [10], [14], [15]. These models train exclusively on normal operational telemetry, minimizing reconstruction error under the assumption that unseen cyber-physical attacks will violate historical patterns and produce high reconstruction residuals. 

Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs) introduced stochastic latent representations to capture operational variance [9], [10]. Transformer-based architectures, such as Anomaly Transformer [12] and TranAD [11], incorporated self-attention mechanisms to model long-range temporal correlations. However, standard deep sequence models treat multivariate observations as flat feature vectors, ignoring the underlying physical connectivity and physical conservation laws that link adjacent industrial instruments.

### 2.3 Graph Neural Networks & Spatial Dependency Modeling
Recognizing the physical topology of industrial plants, recent research has explored Graph Neural Networks (GNNs) for Multivariate Time-Series Anomaly Detection [4], [19], [20]. Graph Deviation Networks (GDN) [4] learn structural relationships between sensors using graph attention, detecting anomalies when learned inter-sensor relationships are broken. Chen et al. [14] and Han & Lee [15] developed spatio-temporal graph convolutional networks to simultaneously capture temporal dynamics and spatial multi-sensor correlations.

However, existing graph architectures suffer from a critical vulnerability: **spatial error propagation**. When a single sensor degrades, message-passing across graph edges propagates corrupted latent embeddings into healthy neighboring nodes, distorting representations across the entire subsystem [8], [24], [27].

### 2.4 Sensor Fault Diagnosis vs. System Anomaly Detection
A major division in literature exists between **System Anomaly Detection** (identifying coordinated cyber-physical attacks against the plant) and **Sensor Fault Diagnosis (FDD)** (detecting localized sensor hardware failure) [8], [9], [11], [13]. Model-based FDD techniques utilize analytical redundancy and state observers (Kalman filters, Luenberger observers) to generate directional fault residuals [6], [10], [25]. While effective in small-scale linear systems, analytical observers require precise physical plant differential equations that are unavailable in complex 51-sensor water treatment facilities [30].

Recent research has attempted data-driven sensor validation using confidence estimation [13], [17], [24], [32]. Zhao et al. [38] explored dynamic residual weighting for chemical fault detection, while Zhang et al. [40] investigated dual-perspective spatio-temporal networks. However, integrating dynamic multi-perspective reliability assessment directly into the decision-making layer of deep graph networks without sacrificing real-time streaming throughput remains an open research challenge.

### 2.5 Summary of Research Gap
Existing deep learning MTSAD frameworks:
1. Uniformly aggregate residuals, causing catastrophic false-alarm saturation when non-adversarial sensor shifts occur.
2. Lack integrated mechanisms to output per-sensor reliability health scores alongside system-level alarms.
3. Propagate single-sensor corruption across spatial graph attention edges.
R-STNet is designed specifically to address these gaps through explicit, multi-perspective reliability estimation.

---

# CHAPTER 3: METHODOLOGY

*(For complete mathematical derivations and architectural diagrams, see [docs/METHODOLOGY.md](file:///d:/Projects/Reliability-Aware-Anomaly-Detection/docs/METHODOLOGY.md) and Appendix C–D).*

### 3.1 Architecture Overview
R-STNet combines an efficient $23,233$-parameter spatiotemporal neural backbone with an auxiliary Sensor Reliability Module (SRM):
- **Temporal Modeling:** Causal Dilated TCN with dilations $d \in \{1, 2, 4\}$ and kernel size $K = 3$.
- **Spatial Modeling:** Multi-head Graph Attention Network (GAT) with a learned adjacency topology $\mathbf{A} \in \mathbb{R}^{M \times M}$.
- **Reconstruction:** Reconstructs the physical sensor state at the final step of sliding window $\mathbf{X}_t \in \mathbb{R}^{30 \times 51}$.

### 3.2 Sensor Reliability Module (SRM) Formulation
For each sensor $i$ at time step $t$, the SRM computes a continuous scalar reliability score $r_i(t) \in [0, 1]$:
$$r_i(t) = \frac{w_v \cdot r_i^{(v)}(t) + w_c \cdot r_i^{(c)}(t) + w_p \cdot r_i^{(p)}(t)}{w_v + w_c + w_p}$$
with calibrated perspective weights $w_v = 1.0$, $w_c = 1.2$, and $w_p = 1.0$:
1. **Temporal Continuity:** $r_i^{(v)}(t) = \exp \left( - \frac{1}{2} \left(\frac{|x_{i, t} - x_{i, t-1}|}{\text{MAD}(\Delta x_i) + \epsilon}\right)^2 \right)$
2. **Graph Consistency:** $r_i^{(c)}(t) = \exp \left( - \frac{1}{2} \left(\frac{|x_{i, t} - \sum_{j \in \mathcal{N}(i)} \alpha_{ij} x_{j, t}|}{\sigma_{\text{neigh}, i} + \epsilon}\right)^2 \right)$
3. **Residual Confidence:** $r_i^{(p)}(t) = \exp \left( - \frac{1}{2} \left(\frac{|x_{i, t} - \hat{x}_{i, t}|}{\text{MAD}(e_i) + \epsilon}\right)^2 \right)$

### 3.3 Reliability-Weighted System Anomaly Scoring
The system anomaly score aggregates normalized residuals using dynamic reliability weights:
$$S_{\text{sys}}(t) = \frac{\sum_{i=1}^M r_i(t) \cdot \left( \frac{|x_{i, t} - \hat{x}_{i, t}|}{\text{MAD}(e_i) + \epsilon} \right)}{\sum_{i=1}^M r_i(t)}$$
When a sensor degrades, $r_i(t) \to 0$, suppressing its influence by over $95\%$ and preventing false system alarms.

---

# CHAPTER 4: RESULTS AND DISCUSSION

*(Integrated directly from the fully validated Review 2 package [report/REVIEW2/CHAPTER4_RESULTS_AND_DISCUSSION.md](file:///d:/Projects/Reliability-Aware-Anomaly-Detection/report/REVIEW2/CHAPTER4_RESULTS_AND_DISCUSSION.md)).*

### 4.1 Clean Benchmark Results (Table 4.3)
Evaluated on the official chronological mixed SWaT test sequence ($44,963$ windows, $5,468$ attack labels, $12.16\%$ positive prevalence):
- **Isolation Forest (Baseline A):** $F_1 = \mathbf{0.7182}$, Precision $= \mathbf{0.8995}$, Recall $= 0.5977$, $\text{FPR} = \mathbf{0.0092}$ ($0.92\%$).
- **ST-AE (Baseline B):** $F_1 = 0.2169$, Precision $= 0.1216$, Recall $= 1.0000$, $\text{FPR} = 1.0000$ ($100.0\%$). Flagged all $39,495$ normal windows as anomalous.
- **RSTNetNoSRM (Baseline C):** $F_1 = 0.2171$, Precision $= 0.1218$, Recall $= 1.0000$, $\text{FPR} = 0.9986$ ($99.86\%$). Flagged $39,439$ normal windows.
- **Proposed R-STNet (with SRM):** $F_1 = \mathbf{0.2740}$, Precision $= \mathbf{0.1626}$, Recall $= 0.8713$, $\text{FPR} = \mathbf{0.6214}$ ($62.14\%$).
- **SRM Gain:** Achieves a **$+26.21\%$ relative F1 gain** and eliminates **$14,898$ false alarms** (-37.78% relative FPR reduction), recovering $14,954$ True Negatives while preserving $87.13\%$ genuine attack recall.

### 4.2 Robustness Under Controlled Sensor Corruption (Tables 4.4, 4.5, 4.6)
Across multi-seed evaluations ($[42, 101, 2024]$):
- **Gaussian Noise ($\sigma \in [0.1, 0.5]$):** R-STNet maintains $F_1 = 0.2755 \pm 0.0011$ ($\sigma = 0.1$) to $0.2762 \pm 0.0001$ ($\sigma = 0.5$) with FPR bounded at $64.09\% - 65.08\%$. Baseline C remains trapped at $FPR = 99.86\%$.
- **Sensor Drift ($\delta \in [0.1, 0.5]$):** R-STNet maintains $F_1 = 0.2739 \pm 0.0001$ ($\delta = 0.1$) to $0.2729 \pm 0.0007$ ($\delta = 0.5$) with FPR bounded at $62.14\% - 62.29\%$.
- **Sensor Reliability Decoupling (Table 4.7):** Clean channels maintain $\bar{r} \approx 0.304$, while corrupted channels drop to $\bar{r} \approx 0.015$, establishing a separation gap of **$\Delta r = +0.2856\text{ to }+0.2946$** and **$100.0\%$ sensor fault detection recall** ($r_i < 0.60$ cutoff).
- **Stuck Actuator Boundary:** On inactive backup pumps (`P202`, `P204`), SRM registered $0.0\%$ fault recall because natural variance was zero during training and testing. Statistical models require supervisory PLC command telemetry to detect stuck states on inactive actuators.

### 4.3 SRM Perspective Ablation (Table 4.8)
- No SRM: $F_1 = 0.2171$, $FPR = 99.86\%$.
- Temporal Only: $F_1 = 0.2170$, $FPR = 99.60\%$.
- Graph Only: $F_1 = 0.2163$, $FPR = 99.57\%$.
- Residual Only: $F_1 = 0.2184$, $FPR = 92.67\%$.
- Full SRM: **$F_1 = 0.2740$**, **$FPR = 62.14\%$**.
Proves that single perspectives fail; full multi-perspective fusion is essential to achieve meaningful false-alarm suppression.

### 4.4 SWaT Channel Shift Diagnostic (Figure 4.12)
Seven channels (`MV101`, `AIT201`, `MV201`, `P201`, `P202`, `P204`, `MV303`) were unrecorded or constant in normal training data ($\sigma_{tr} = 0$). In test data, analyzer `AIT201` surged to standardized $z > 1,600$, forcing unweighted reconstruction models into $99.86\%$ FPR. R-STNet downweights `AIT201` to $r < 0.005$, preventing false plant-wide alarms.

### 4.5 Computational Measurements (Table 4.9)
- Trainable Parameters: $23,233$ (86.4% smaller than ST-AE's $170,643$).
- Inference Latency: $0.060\text{ ms}$ per window ($16,667$ windows/s), operating $160,000\times$ faster than the $10$-second SWaT sampling interval.
- Peak VRAM: $978.69\text{ MB}$ ($27.4\%$ of a 4 GB mobile GPU).

---

# CHAPTER 5: CONCLUSION

*(See complete text in [report/FINAL/CHAPTER5_CONCLUSION.md](file:///d:/Projects/Reliability-Aware-Anomaly-Detection/report/FINAL/CHAPTER5_CONCLUSION.md)).*

The thesis conclusively establishes that:
1. **Isolation Forest is superior on clean SWaT data** ($F_1 = 0.7182$ vs. $0.2740$).
2. **R-STNet significantly outperforms Baseline C within deep architectures**, achieving a $+26.21\%$ relative F1 gain and eliminating $14,898$ false alarms.
3. **The SRM successfully decouples sensor faults from plant-wide attacks**, achieving $+0.29$ reliability separation and $100\%$ fault recall against synthetic corruption.
4. **The research hypothesis is supported within deep spatiotemporal models**, establishing a robust foundation for reliability-aware industrial CPS monitoring.

---

# REFERENCES

*(Complete verified IEEE references detailed in [report/FINAL/REFERENCES.md](file:///d:/Projects/Reliability-Aware-Anomaly-Detection/report/FINAL/REFERENCES.md) and [report/FINAL/references.bib](file:///d:/Projects/Reliability-Aware-Anomaly-Detection/report/FINAL/references.bib)).*

---

# APPENDIX

*(Complete executable Python source code implementations detailed in [report/FINAL/APPENDIX.md](file:///d:/Projects/Reliability-Aware-Anomaly-Detection/report/FINAL/APPENDIX.md)).*
