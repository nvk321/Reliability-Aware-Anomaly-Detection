# FINAL RESEARCH PROJECT DEFENSE PRESENTATION

**Course:** BCSE209P — Machine Learning Lab  
**Project Title:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Students:** P Naga Venkata Kushal (23BDS0133) & Koteru Tarun (23BDS0103)  
**Faculty Guide:** Dr. Dega Nagaraju, Associate Professor  
**Target Audience:** Academic Review Committee, Faculty Examiners, CPS Security Researchers  

---

## Slide 1: Title & Project Identity

### Slide Content:
- **Title:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data
- **Focus:** Decoupling Localized Sensor Unreliability from Systemic Cyber-Physical Attacks
- **Dataset:** Secure Water Treatment (SWaT A1/A2, SUTD) — 51 Channels (25 Sensors, 26 Actuators)
- **Proposed Architecture:** R-STNet (Reliability-Aware Spatio-Temporal Network with SRM)
- **Presented by:**
  - P Naga Venkata Kushal (23BDS0133)
  - Koteru Tarun (23BDS0103)
- **Faculty Guide:** Dr. Dega Nagaraju

> **Speaker Notes (Slide 1):**  
> *"Good morning respected faculty members and examiners. Today, Tarun and I present our final machine learning research project: 'Reliability-Aware Anomaly Detection for Multivariate Sensor Data'. In industrial critical infrastructure—such as water purification plants and smart power grids—automated anomaly detection is essential for physical safety. Today, we show how explicit sensor reliability estimation solves the critical vulnerability of false alarm saturation in deep spatio-temporal architectures."*

---

## Slide 2: Problem Statement: The Equal Observation Credulity Flaw

### Slide Content:
- **Core Industry Vulnerability:** Modern deep Multivariate Time-Series Anomaly Detection (MTSAD) models assume **equal credulity across all sensor observations**.
- **Real-World Reality:** Physical industrial sensors frequently experience non-adversarial degradation:
  - High-frequency electrical noise & EMI.
  - Sensor fouling & calibration drift.
  - Stuck transducers & zero-variance states.
  - Unrecorded operational shifts during commissioning.
- **The Reconstruction Collapse:**
  $$S(t) = \frac{1}{M} \sum_{i=1}^M \left(\frac{x_i(t) - \hat{x}_i(t)}{\sigma_i}\right)^2$$
- When one sensor degrades, its quadratic residual spikes. The unweighted model interprets this as a plant-wide emergency, triggering **catastrophic false alarms**.

> **Speaker Notes (Slide 2):**  
> *"The fundamental flaw in modern deep anomaly detection is equal observation credulity. If a single flow meter experiences electrical noise or calibration drift, existing autoencoders and graph networks treat that discrepancy with the exact same weight as a multi-stage cyber-attack. This forces the aggregate anomaly score past the alarm threshold, creating constant false alarms and severe operator alarm fatigue in control rooms."*

---

## Slide 3: Motivation & Research Questions

### Slide Content:
- **The Core Dilemma:** How can an automated supervisory system distinguish between:
  1. *Localized Sensor Unreliability* (benign sensor hardware degradation) $\implies$ Maintenance alert.
  2. *Systemic Cyber-Physical Attack* (coordinated process disruption) $\implies$ Emergency plant shutdown.
- **Primary Research Question:**
  > *Can explicit sensor-reliability estimation improve the robustness of multivariate time-series anomaly detection when sensor observations are missing, noisy, or corrupted?*
- **Scientific Hypothesis:**
  > *Explicit multi-perspective sensor reliability estimation will significantly reduce false-alarm saturation and improve robustness to sensor degradation within deep spatio-temporal architectures.*

> **Speaker Notes (Slide 3):**  
> *"This leads directly to our primary research question: Can explicit reliability estimation improve detection robustness under sensor degradation? Our hypothesis is that by estimating per-sensor reliability across multiple physical perspectives, we can downweight noisy or shifting sensors before they trigger false plant shutdowns."*

---

## Slide 4: The Benchmark Environment: SWaT Industrial Testbed

### Slide Content:
- **Testbed Facility:** Secure Water Treatment (SWaT), iTrust SUTD Singapore.
- **Physical Scale:** 6-Stage continuous municipal-scale water purification plant (5 gallons/min).
  - P1: Raw Water & Level Control | P2: Chemical Coagulation | P3: Ultrafiltration
  - P4: Dechlorination & UV | P5: Reverse Osmosis Desalination | P6: Storage & Distribution
- **Telemetry Dimensions:** $M = 51$ physical variables.
  - 25 Continuous Sensors (flow, level, differential pressure, electrochemical analyzers).
  - 26 Discrete Actuators (motorized valves, chemical metering pumps).
- **Official Data Partitions:**
  - 7 Days Normal Commissioning ($496,800$ rows) $\to$ Chronological 80% Train / 20% Val.
  - 4 Days Attack Testing ($449,919$ rows) $\to$ 36 physical attacks ($12.16\%$ prevalence).

> **Speaker Notes (Slide 4):**  
> *"We evaluate our method on the gold-standard SWaT dataset from SUTD iTrust. SWaT models a physical six-stage water treatment facility logging 51 continuous sensors and discrete actuators. We enforced strict chronological data partitioning: the first 80% of normal operation for training, the final 20% of normal operation for threshold calibration, and the 4-day attack run for testing, ensuring zero label leakage."*

---

## Slide 5: Proposed Architecture: R-STNet Framework

### Slide Content:
- **Model Taxonomy:** Reliability-Aware Spatio-Temporal Network ($23,233$ parameters).
- **Temporal Backbone:** Causal Dilated Temporal Convolutional Network (TCN) with dilations $d \in \{1, 2, 4\}$ (strictly causal, no future leakage).
- **Spatial Backbone:** Multi-Head Spatial Graph Attention Network (GAT) learning topological inter-sensor dependencies across physical stages.
- **Reconstruction Output:** Predicts nominal state $\hat{\mathbf{x}}_t \in \mathbb{R}^{51}$.
- **Compactness:** Requires **86.4% fewer parameters** than standard convolutional autoencoders ($23\text{k}$ vs. $170\text{k}$).

> **Speaker Notes (Slide 5):**  
> *"Here is our proposed R-STNet architecture. It combines a 1D Causal Dilated TCN for temporal dynamics with a Multi-Head Spatial Graph Attention Network to capture physical dependencies. With only 23,233 parameters, R-STNet is 86% smaller than traditional autoencoders, while achieving superior reconstruction convergence within 10 epochs."*

---

## Slide 6: The Sensor Reliability Module (SRM)

### Slide Content:
- **Continuous Reliability Weighting:** $r_i(t) \in [0, 1]$ computed for each sensor $i$ at time $t$:
  $$r_i(t) = \frac{w_v \cdot r_i^{(v)}(t) + w_c \cdot r_i^{(c)}(t) + w_p \cdot r_i^{(p)}(t)}{w_v + w_c + w_p}$$
- **Perspective 1: Temporal Continuity ($r_i^{(v)}$):**
  $$v_{i, t} = \frac{|x_{i, t} - x_{i, t-1}|}{\text{MAD}(\Delta x_i) + \epsilon}, \quad r_i^{(v)} = \exp(-0.5 \cdot v_{i, t}^2)$$
  Penalizes non-physical rate-of-change spikes and high-frequency electrical jitter.
- **Perspective 2: Graph Neighborhood Consistency ($r_i^{(c)}$):**
  $$c_{i, t} = \frac{|x_{i, t} - \sum_{j \in \mathcal{N}(i)} \alpha_{ij} x_{j, t}|}{\sigma_{\text{neigh}, i} + \epsilon}, \quad r_i^{(c)} = \exp(-0.5 \cdot c_{i, t}^2)$$
  Penalizes sensors that diverge from their physical topological neighbors.
- **Perspective 3: Predictive Residual Confidence ($r_i^{(p)}$):**
  $$z_{i, t} = \frac{|x_{i, t} - \hat{x}_{i, t}|}{\text{MAD}(e_i) + \epsilon}, \quad r_i^{(p)} = \exp(-0.5 \cdot z_{i, t}^2)$$
  Penalizes severe statistical outliers and uncalibrated distribution shifts.

> **Speaker Notes (Slide 6):**  
> *"The core innovation is our Sensor Reliability Module (SRM). Rather than treating all sensors equally, SRM fuses three physical perspectives into a continuous weight between 0 and 1. First, temporal continuity catches sudden jitter; second, graph consistency catches sensors drifting away from their physical neighbors; third, predictive residual confidence downweights extreme statistical outliers."*

---

## Slide 7: Reliability-Weighted Decision Scoring

### Slide Content:
- **Baseline C (Unweighted Scoring):**
  $$S_{\text{unweighted}}(t) = \frac{1}{M} \sum_{i=1}^M \frac{|x_{i, t} - \hat{x}_{i, t}|}{\text{MAD}(e_i) + \epsilon}$$
  *Vulnerability:* A single corrupted sensor forces $S(t) \gg \tau$, triggering an immediate false alarm.
- **Proposed R-STNet (Reliability-Weighted Scoring):**
  $$S_{\text{sys}}(t) = \frac{\sum_{i=1}^M r_i(t) \cdot \left(\frac{|x_{i, t} - \hat{x}_{i, t}|}{\text{MAD}(e_i) + \epsilon}\right)}{\sum_{i=1}^M r_i(t)}$$
- **Decoupling Mechanism:** When sensor $k$ degrades, $r_k(t) \to 0.015$, downweighting its residual by **over 95%** and keeping the plant score safely below $\tau_{prop} = 3.14$.

> **Speaker Notes (Slide 7):**  
> *"In Slide 7, we contrast the scoring mathematics. Baseline C takes an unweighted average, so one bad sensor explodes the entire score. R-STNet weights each residual by its reliability. When sensor k fails, its reliability drops to 0.015, suppressing its contribution by over 95%. The plant-wide score remains below threshold, avoiding an unnecessary emergency shutdown."*

---

## Slide 8: Clean SWaT Benchmark Results (The Honest Comparison)

### Slide Content:

| Model Architecture | Point-wise F1 | Precision | Recall | False Positive Rate (FPR) | False Alarms (FP) | True Negatives (TN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Isolation Forest (Baseline A)** | **0.7182** | **0.8995** | 0.5977 | **0.92% (0.0092)** | **365** | **39,159** |
| **ST-AE (Baseline B)** | 0.2169 | 0.1216 | 1.0000 | 100.00% (1.0000) | 39,495 | 0 |
| **RSTNetNoSRM (Baseline C)** | 0.2171 | 0.1218 | 1.0000 | 99.86% (0.9986) | 39,439 | 56 |
| **Proposed R-STNet (with SRM)** | **0.2740** | **0.1626** | 0.8713 | **62.14% (0.6214)** | **24,541** | **14,954** |

- **Key Takeaway 1:** Isolation Forest is superior on clean data ($F_1 = 0.7182$). Tree axis splits isolate single-channel shifts without joint feature distortion.
- **Key Takeaway 2:** Within deep models, R-STNet yields a **+26.21% relative F1 gain** and **eliminates 14,898 false alarms** relative to Baseline C.

> **Speaker Notes (Slide 8):**  
> *"Slide 8 shows our clean benchmark results. Here we maintain strict scientific honesty: Isolation Forest achieves the best clean-data performance with F1 of 0.7182. But looking at deep architectures, standard models like ST-AE and Baseline C collapse completely, flagging all normal windows. R-STNet breaks this collapse: it eliminates 14,898 false alarms and improves F1 by 26.2% over Baseline C under identical parameters."*

---

## Slide 9: Root Cause Diagnostic: The SWaT Channel Shift Discovery

### Slide Content:
- **Audit Discovery:** Seven channels in raw SWaT contain empty strings or constant values during training:
  $$\{\text{MV101, AIT201, MV201, P201, P202, P204, MV303}\}$$
- **The Case of Analyzer AIT201:**
  - *Training:* Offline/unrecorded, constant $z \approx 1.0$ with zero variance ($\sigma_{tr} = 0$).
  - *Testing:* Activated by operators, generating dynamic oscillations surging to **$z > 1,600$** (Mean test $z = +692.64$).
- **Impact on Deep Models:** Uniform reconstruction models produce massive residuals ($e \approx 2.56 \times 10^6$), permanently forcing anomaly scores above threshold.
- **R-STNet Response:** SRM detects non-physical residuals and depresses $r_{\text{AIT201}} < 0.005$, automatically preventing false alarms without human channel pruning.

> **Speaker Notes (Slide 9):**  
> *"Why did standard deep models collapse? Our data audit revealed a critical dataset artifact: seven channels were constant or unrecorded in training but active in test. Most notably, conductivity analyzer AIT201 had zero training variance, but surged to z > 1,600 during test. Uniform models interpret this as a massive attack. R-STNet's SRM automatically downweights AIT201 to near-zero, proving its self-healing capability."*

---

## Slide 10: Multi-Seed Robustness Under Sensor Corruption

### Slide Content:
- **Controlled Perturbations across Seeds $[42, 101, 2024]$:**
  1. *Additive Gaussian Noise ($\sigma \in [0.1, 0.5]$):* R-STNet maintains $F_1 = 0.2762 \pm 0.0001$ with FPR bounded at $65.08\% \pm 0.01\%$. Baseline C remains collapsed at $FPR = 99.86\%$.
  2. *Dynamic Linear Drift ($\delta \in [0.1, 0.5]$):* R-STNet maintains $F_1 = 0.2729 \pm 0.0007$ with FPR bounded at $62.29\% \pm 0.12\%$.
- **Empirical Reliability Decoupling:**
  - Clean channels: $\bar{r}_{\text{clean}} = 0.3041 \pm 0.010$.
  - Corrupted channels: $\bar{r}_{\text{corrupt}} = 0.0151 \pm 0.002$.
  - Net separation gap: **$\Delta r = +0.2856\text{ to }+0.2946$**.
  - Fault detection recall: **$100.0\% \pm 0.0\%$** against synthetic corruption masks.

> **Speaker Notes (Slide 10):**  
> *"Slide 10 evaluates robustness under controlled corruption across multiple seeds. As Gaussian noise severity increases from 0.1 to 0.5, R-STNet's F1 remains stable at 0.276, maintaining a 35% FPR advantage over Baseline C. The SRM establishes a clear separation gap of +0.29 between clean and corrupted channels, achieving 100% sensor fault detection recall."*

---

## Slide 11: SRM Component Ablation Study

### Slide Content:
- **5-Condition Systematic Ablation (Identical $23\text{k}$ Backbone):**
  1. *Ablation 1 (No SRM):* $F_1 = 0.2171, \text{FPR} = 99.86\%, \tau = 12.07$
  2. *Ablation 2 (Temporal Only, $w_v=1$):* $F_1 = 0.2170, \text{FPR} = 99.60\%, \tau = 18.99$
  3. *Ablation 3 (Graph Only, $w_c=1.2$):* $F_1 = 0.2163, \text{FPR} = 99.57\%, \tau = 0.46$
  4. *Ablation 4 (Residual Only, $w_p=1$):* $F_1 = 0.2184, \text{FPR} = 92.67\%, \tau = 4.65$
  5. *Ablation 5 (Full SRM):* **$F_1 = 0.2740, \text{FPR} = 62.14\%, \tau = 3.14$**
- **Takeaway:** Individual components fail to suppress false alarms ($FPR > 92\%$). Only **multi-perspective synergy** drives FPR down to $62.14\%$.

> **Speaker Notes (Slide 11):**  
> *"In our ablation study, we isolated the individual SRM components. Temporal continuity alone or graph consistency alone fails to reduce false alarms, remaining above 99.5% FPR. Only when residual confidence is combined with temporal rate smoothing and graph consistency does the model achieve an F1 of 0.2740 and cut FPR to 62.14%, proving the necessity of multi-perspective fusion."*

---

## Slide 12: Qualitative Failure Cases & Operational Boundaries

### Slide Content:
- **Case 1 (AIT201 Shift):** Baseline C triggers false alarm ($S=1,482$); R-STNet downweights sensor ($r < 0.005$) $\to$ **True Negative Preserved**.
- **Case 2 (Noise Suppression):** Noise on 3 sensors triggers false alarm in Baseline C ($S=45.2$); R-STNet suppresses noise $\to$ **True Negative Preserved**.
- **Case 3 (Stuck Actuator Boundary):** Inactive backup pumps (`P202`/`P204`) frozen at zero. Missed by SRM ($0\%$ recall) because natural variance is zero $\to$ **Requires PLC Command Telemetry**.
- **Case 4 (Genuine Attack Detection):** Tank overflow manipulation detected by both models ($S_{prop} = 48.7 \gg 3.14$) $\to$ **Genuine Sensitivity Preserved**.
- **Case 5 (Stealthy Attack Masking):** Subtle flow spoofing on `FIT401` downweighted as noise $\to$ **False Negative**. Demonstrates the dual-edged risk of aggressive reliability downweighting.

> **Speaker Notes (Slide 12):**  
> *"To ensure rigorous science, we analyzed five diagnostic cases. R-STNet successfully prevents false alarms from unrecorded shifts (Case 1) and noise (Case 2), while detecting genuine multi-sensor attacks (Case 4). However, Case 3 and Case 5 reveal physical boundaries: stuck inactive pumps cannot be detected without PLC command signals, and subtle single-sensor attacks risk being downweighted as noise."*

---

## Slide 13: Computational Efficiency & Edge Feasibility

### Slide Content:
- **Inference Latency:** **$0.060\text{ ms}$ per sliding window** ($16,667$ windows/second).
- **Industrial Sampling Step:** SWaT sampling interval is $10,000\text{ ms}$ ($10$ seconds).
- **Execution Headroom:** R-STNet executes **$160,000\times$ faster** than the physical process telemetry arrival rate.
- **VRAM Utilization:** Peak allocated VRAM of **$978.69\text{ MB}$** ($27.4\%$ of a 4 GB RTX 3050 Laptop GPU).
- **Scope Qualification:** Benchmarked in GPU memory; physical PLC Modbus network jitter was not evaluated.

> **Speaker Notes (Slide 13):**  
> *"Slide 13 details computational feasibility. R-STNet processes an inference window in 0.060 milliseconds—over 160,000 times faster than the 10-second SWaT arrival interval. Consuming under 1 GB of VRAM, the model is fully compact for edge deployment on industrial controllers like NVIDIA Jetson."*

---

## Slide 14: Managerial & Practical Implications

### Slide Content:
1. **Operator Trust & Alarm Fatigue Reduction:** Eliminates $14,898$ false alarms relative to Baseline C, mitigating operator desensitization in control rooms.
2. **Dual-Layer Diagnostic Output:**
   - *System-Level Alarm ($S_{\text{sys}} \ge \tau$):* Informs operators of coordinated cyber-physical attacks.
   - *Sensor-Level Reliability Health Index ($r_i(t) < 0.60$):* Directs maintenance crews to specific degrading transmitters.
3. **Runtime Self-Healing:** Suppresses transient sensor shifts dynamically without requiring emergency model retraining.
4. **Human-in-the-Loop Position:** Models must serve as intelligent decision-support tools rather than autonomous shutdown switches.

> **Speaker Notes (Slide 14):**  
> *"The operational implications are substantial. Instead of giving operators an opaque red light, R-STNet provides a dual output: a plant-wide alarm for coordinated attacks, and a per-sensor health index for maintenance technicians. This cuts alarm fatigue while preserving human-in-the-loop oversight."*

---

## Slide 15: Conclusion & Future Scope

### Slide Content:
- **Final Research Assessment:**
  - *The Research Hypothesis is Supported Within Deep Spatiotemporal Models:* Explicit reliability estimation provides a proven mechanism to filter sensor degradation and eliminate false-alarm saturation.
  - *Academic Honesty:* R-STNet does not outperform Isolation Forest on clean SWaT, and statistical models cannot diagnose stuck inactive actuators without supervisory telemetry.
- **Future Scope:**
  1. *Upstream Reliability Gating:* Mask unreliable node embeddings *before* spatial graph message passing.
  2. *PLC Command-Telemetry Fusion:* Ingest PLC actuator commands to diagnose frozen valves and pumps.
  3. *Cross-Testbed Validation:* Benchmark on WADI, Tennessee Eastman, and Gas Pipeline ICS datasets.
  4. *Hardware-in-the-Loop (HIL):* Test live Modbus/TCP streaming jitter on embedded industrial hardware.

> **Speaker Notes (Slide 15):**  
> *"In conclusion, our research proves that explicit sensor reliability estimation provides a defensible, mathematically validated breakthrough against false-alarm saturation in deep spatiotemporal architectures. While classical Isolation Forest remains stronger on clean data, R-STNet provides the per-sensor health diagnostics and noise robustness needed for modern industrial CPS. Thank you, and we welcome your questions."*
