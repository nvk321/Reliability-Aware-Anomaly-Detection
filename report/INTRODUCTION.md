# Chapter 1: Introduction

**Course:** Machine Learning Lab (BCSE209P)  
**Slot:** L47 + L48 | Fall Semester 2026-27  
**Project Title:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Authors:** P Naga Venkata Kushal (23BDS0133), Koteru Tarun (23BDS0103)  
**Guide:** Dr. Dega Nagaraju  
**Institution:** School of Computer Science and Engineering (SCOPE), VIT Vellore  

---

## 1.1 Background

Modern industrial automation, infrastructure management, and manufacturing systems have undergone a profound digital transformation driven by the principles of Industry 4.0 and Cyber-Physical Systems (CPS). Across mission-critical sectors—including municipal water treatment facilities, thermal and nuclear power plants, chemical manufacturing refineries, smart electric grids, and autonomous aerospace platforms—continuous computational control loops are coupled with physical processes [6], [7]. 

Ensuring the uninterrupted safety, operational stability, and physical integrity of these plants requires continuous real-time supervisory monitoring. In traditional supervisory architectures, human operators relied heavily on manual inspections and static alarm thresholds configured within Supervisory Control and Data Acquisition (SCADA) systems. However, as the physical scale, operational complexity, and interconnectedness of modern engineering infrastructure have expanded, manual and static threshold-based monitoring strategies have proven fundamentally inadequate [3], [16]. 

Complex physical processes are subject to multivariate interactions, non-stationary environmental shifts, and sophisticated cyber-physical intrusions. Consequently, data-driven machine learning techniques—specifically Multivariate Time-Series Anomaly Detection (MTSAD)—have emerged as an indispensable foundation for modern industrial safety, automated diagnostics, and predictive maintenance [16], [17].

---

## 1.2 Multivariate Sensor Systems

A multivariate sensor system comprises a spatially distributed and functionally heterogeneous array of sensing instruments deployed across a physical installation [38], [39]. These instruments record continuous physical, chemical, and thermodynamic variables alongside discrete actuator commands at regular sampling intervals. 

Mathematically, at any discrete time step $t \in \{1, 2, \dots, T\}$, the collective observation from a network of $M$ monitoring channels is represented as an $M$-dimensional vector:

$$\mathbf{X}(t) = \left[ x_1(t), x_2(t), \dots, x_M(t) \right]^T \in \mathbb{R}^M$$

where each channel $x_i(t)$ represents the measurement of physical variable $i$ (e.g., fluid level, volumetric flow, differential pressure, or chemical concentration).

In cyber-physical sensor systems, these variables do not evolve in isolation. Instead, they are tightly coupled through two distinct structural dimensions:
1. **Temporal Dependencies:** Individual sensor channels exhibit autoregressive memory, rate-of-change constraints, inertia lags, and settling times governed by the underlying physical laws of motion, thermodynamics, and mass transfer [16], [25].
2. **Spatial and Cross-Sensor Dependencies:** Distinct sensor channels are bound by physical conservation principles (e.g., conservation of mass, conservation of energy, hydraulic continuity, and chemical stoichiometry) [19], [33]. For instance, opening an upstream control valve predictably triggers an increase in downstream flow rate and a subsequent change in storage tank level.

Consequently, an effective monitoring system must simultaneously model intra-sensor temporal evolutions and inter-sensor spatial correlations [20], [21].

---

## 1.3 Industrial Sensor Monitoring

Industrial sensor monitoring operates under stringent operational requirements: high availability, minimal computational latency, and ultra-low false alarm rates [6], [43]. In critical infrastructure such as water treatment plants (e.g., the Secure Water Treatment testbed [38]), continuous telemetry streams from flow meters, level transmitters, motorized pumps, and chemical analyzers to Programmable Logic Controllers (PLCs) executing closed-loop proportional-integral-derivative (PID) control.

In these environments, anomalous behaviors can emerge from diverse operational vectors:
- **Physical Equipment Failures:** Mechanical pump cavitation, valve seizing, pipe ruptures, or membrane fouling.
- **Cyber-Physical Attacks:** Malicious adversaries manipulating sensor measurements (sensor spoofing) or overriding actuator states (valve hijacking) via compromised network communications [43], [44].
- **Process Disturbances:** Severe fluctuations in inflow composition, power supply transients, or ambient environmental surges.

Automated anomaly detection frameworks must continuously ingest multivariate streams $\mathbf{X}(t)$, learn the normal operational envelope, and provide early warning flags prior to catastrophic system failure.

---

## 1.4 Anomaly Detection

In machine learning and statistics, an anomaly (or outlier) is defined as an observation or sequence of observations that deviates so substantially from the established distribution of normal data as to arouse suspicions that it was generated by a different underlying mechanism [3], [8].

Classical literature categorizes anomalies into three distinct taxonomies [3]:
1. **Point Anomalies:** An individual data instance that deviates significantly from the rest of the dataset (e.g., an extreme, physically impossible pressure reading).
2. **Contextual (Conditional) Anomalies:** A data instance that appears normal in a global statistical sense, but deviates anomalously within its specific context (e.g., high heating power consumption during peak summer, or low tank level when inflow valves are fully open).
3. **Collective Anomalies:** A collection of related data instances that appear normal individually, but whose collective sequential pattern or joint combination violates expected system behavior over time [16].

In cyber-physical sensor monitoring, the vast majority of critical operational failures and cyber-attacks manifest as subtle **contextual and collective anomalies**, requiring models capable of reasoning over temporal windows rather than isolated static points [10], [19].

---

## 1.5 Multivariate Time-Series Anomaly Detection

Multivariate Time-Series Anomaly Detection (MTSAD) formalizes the process of assigning an anomaly score $S(t) \in \mathbb{R}$ to a sequence window of multivariate observations:

$$\mathbf{W}_t = \left[ \mathbf{X}(t - L + 1), \mathbf{X}(t - L + 2), \dots, \mathbf{X}(t) \right] \in \mathbb{R}^{L \times M}$$

where $L$ denotes the temporal lookback horizon.

MTSAD frameworks typically adopt an **unsupervised or semi-supervised one-class learning paradigm** [10], [17]. Because catastrophic industrial accidents and cyber-attacks are extremely rare, historical datasets contain overwhelming volumes of normal operational data, while labeled anomalies are scarce, highly diverse, and unpredictable. Consequently, models are trained exclusively on healthy baseline operation to capture normal data distributions. At inference time, any substantial deviation between the observed window and the model's learned representation is interpreted as an anomaly [9], [19].

---

## 1.6 Sensor Faults versus System-Level Anomalies

A critical yet frequently overlooked reality in industrial monitoring is the fundamental difference between a **sensor fault** and a **system-level anomaly** [29], [33]:

### Definition 1: System-Level Anomaly (Process / Cyber-Physical Attack)
A system-level anomaly occurs when the physical process itself deviates from safe operational boundaries (e.g., an attacker forces a valve open, causing a storage tank to overflow, or a physical pipe fractures). Under a genuine system anomaly:
- The underlying physical state is abnormal.
- Physical sensors are functioning correctly and reporting true physical measurements.
- Neighboring, physically coupled sensors register correlated deviations consistent with conservation laws.
- **Required Action:** Immediate emergency plant alarm, operator intervention, and automated fail-safe trip [38], [43].

### Definition 2: Sensor Fault (Instrument / Observation Corruption)
A sensor fault occurs when the physical plant is operating in complete, safe equilibrium, but an individual sensing instrument produces an abnormal or corrupted measurement due to hardware degradation [28], [30]:
- **High Gaussian Noise:** Electromagnetic interference or aging signal amplifiers.
- **Sensor Drift:** Gradual calibration shifts resulting from chemical fouling, membrane coating, or mechanical fatigue [30].
- **Frozen / Stuck Sensors:** Complete loss of telemetry dynamics due to mechanical float jamming or analog-to-digital converter lockup.
- **Impulse Spikes:** Power supply switching ripples or mechanical shock.
- **The underlying physical plant remains completely safe and normal.**
- Neighboring sensors show normal, healthy operation.
- **Required Action:** Suppress plant shutdown; alert instrumentation and control (I&C) technicians to clean, recalibrate, or replace the faulty transmitter.

---

## 1.7 Problem Statement

We formulate the core research problem mathematically:

Let an industrial cyber-physical process be monitored by $M$ continuous and discrete channels streaming observations $\mathbf{X}(t) = [x_1(t), x_2(t), \dots, x_M(t)]^T \in \mathbb{R}^M$ at time step $t$. Let $\hat{\mathbf{X}}(t) \in \mathbb{R}^M$ denote the expected normal observation vector generated by a predictive or reconstructive spatio-temporal model, yielding instantaneous residual deviations:

$$e_i(t) = \left| x_i(t) - \hat{x}_i(t) \right|, \quad \forall i \in \{1, 2, \dots, M\}$$

Standard multivariate anomaly detection models aggregate these residuals uniformly into a global scalar anomaly score:

$$S_{\text{global}}(t) = \frac{1}{M} \sum_{i=1}^M \phi(e_i(t))$$

where $\phi(\cdot)$ is a non-negative scaling function.

When an isolated sensor $k$ experiences an observation corruption (such that $e_k(t) \gg 0$) while the underlying physical process remains completely nominal ($x_j(t) \approx \hat{x}_j(t)$ for all $j \neq k$), the unweighted summation forces:

$$S_{\text{global}}(t) \ge \tau_{\text{sys}}$$

triggering a **false plant-wide emergency alarm**.

**The Research Problem:**  
The objective of this research is to formulate, implement, and rigorously evaluate an explicit reliability-aware mechanism that:
1. Dynamically estimates a continuous, channel-specific reliability vector $\mathbf{r}(t) = [r_1(t), \dots, r_M(t)]^T \in [0, 1]^M$ online, reflecting the trustworthiness of each individual sensor observation.
2. Incorporates $\mathbf{r}(t)$ into system-level anomaly scoring such that degraded sensors are down-weighted, preventing false system shutdowns:
   $$S_{\text{sys}}(t) = \frac{\sum_{i=1}^M r_i(t) \cdot z_i(t)}{\sum_{i=1}^M r_i(t) + \epsilon}$$
3. Simultaneously isolates the corrupted channel for maintenance while preserving rapid detection of genuine multi-sensor cyber-physical attacks.

---

## 1.8 Motivation

The primary motivation of this study stems from the severe operational and economic consequences of false positive alarms in modern automated industries:
1. **Economic Cost of Unwarranted Downtime:** In continuous processing plants (such as chemical refineries or municipal water treatment works), an emergency shutdown triggers extensive restart protocols, equipment thermal stress, chemical wastage, and millions of dollars in downtime costs [6], [7].
2. **Alarm Fatigue:** When monitoring systems regularly trigger alarms caused by noisy, drifting, or glitched sensors, human operators develop "alarm fatigue," leading them to silence, ignore, or disable automated safety interlocks—a primary contributor to historical industrial disasters.
3. **The Architectural Gap in Deep Learning:** While modern deep learning architectures (e.g., Graph Neural Networks, Transformers, Causal Contrastive Learning) have drastically improved representation capacity on pristine benchmark splits, they remain vulnerable to individual channel corruptions due to their unweighted residual aggregation paradigms [19], [21], [27].

---

## 1.9 Significance of the Study

This investigation bridges the divide between two historically segregated engineering disciplines:
- **Computer Science & Machine Learning:** Focuses on high-capacity spatio-temporal representation learning, graph attention, and unsupervised representation learning for anomaly detection [10], [19], [20].
- **Control Engineering & Process Systems:** Focuses on analytical redundancy, sensor validation, fault isolation, and hardware reliability estimation [28], [33], [36].

By integrating physical consensus principles (graph neighbor agreement and temporal continuity) directly into deep neural anomaly detection, this study establishes a unified, scientifically grounded framework capable of operating robustly in harsh, real-world industrial environments.

---

## 1.10 Existing Approaches

Existing literature for multivariate sensor anomaly detection spans four broad paradigms:
1. **Classical Statistical & Machine Learning Baselines:** Principal Component Analysis (PCA) [6], Dynamic PCA [5], One-Class Support Vector Machines (OC-SVM) [2], and Isolation Forest [1].
2. **Deep Reconstruction & Prediction Models:** Long Short-Term Memory Autoencoders (LSTM-AE) [14], Unsupervised Anomaly Detection (USAD) [9], Stochastic Recurrent Networks (OmniAnomaly) [10], and Multi-Scale Convolutional Recurrent Autoencoders (MSCRED) [13].
3. **Graph Neural Networks (GNNs):** Graph Deviation Network (GDN) [19], Multivariate Time-Series Anomaly Detection via Graph Attention (MTAD-GAT) [20], and Spatio-Temporal Graph Convolutional Networks [23].
4. **Causal & Contrastive Learning:** Causality-Aware Contrastive Learning (CAROTS) [21], which enforces representation invariance under domain shifts.

---

## 1.11 Limitations of Existing Approaches

Despite significant algorithmic advances, current approaches exhibit fundamental limitations:
1. **Uniform Residual Aggregation:** Every established deep detector (USAD, OmniAnomaly, GDN, MTAD-GAT, TranAD) aggregates channel errors uniformly. A single corrupted sensor is mathematically capable of triggering a system alarm [19], [20].
2. **Absence of Channel Trust Metrics:** Existing models output an anomaly score for the *timestamp*, but do not produce a continuous, interpretable reliability score $r_i(t) \in [0, 1]$ for each individual sensor channel $i$.
3. **Latent Representation Contamination:** In autoencoder architectures (e.g., USAD, OmniAnomaly), feeding corrupted inputs through dense encoder layers contaminates the latent bottleneck, corrupting the reconstruction of clean channels [10].
4. **Binary Decision Rigidity:** Conventional systems produce a single binary decision ($\text{Normal}$ vs. $\text{Anomaly}$), failing to differentiate whether an alarm represents instrument maintenance or a plant emergency.
5. **Evaluation Disconnect in Benchmarks:** Benchmarks like SWaT and WADI provide attack labels, but lack ground truth for isolated sensor hardware faults, leaving sensor fault isolation largely untested in mainstream deep learning literature [38], [40].

---

## 1.12 Proposed Approach: R-STNet

To overcome these limitations, we propose **R-STNet (Dual-Perspective Reliability-Aware Spatio-Temporal Network)**:
- **Temporal Backbone:** Dilated 1D Convolutions with residual skip connections capture autoregressive dynamics and physical settling times per channel.
- **Graph Attention Backbone:** Learnable node embeddings dynamically discover directed physical dependencies among sensors, executing message passing over topological neighbors.
- **Sensor Reliability Module (SRM):** Triangulates three independent physical signals:
  1. *Temporal Continuity ($v_i(t)$):* Rate-of-change boundaries and variance-based stuck sensor detection.
  2. *Neighborhood Consistency ($c_i(t)$):* Deviation from topological neighbor predictions.
  3. *Residual Tail Confidence ($p_i(t)$):* Median Absolute Deviation (MAD)-standardized tail probability.
  Fuses these signals via calibrated logit fusion into continuous reliability scores $r_i(t) \in [0, 1]$.
- **Reliability-Weighted Aggregator:** Weights individual residuals by $r_i(t)$, actively discounting corrupted channels from the system anomaly score $S_{\text{sys}}(t)$.
- **Four-Way Operational State Classifier:** Distinguishes Case A (Normal), Case B (Sensor Fault), Case C (System Anomaly), and Case D (System Anomaly with Degradation).

---

## 1.13 Research Questions

### Central Research Question
> **RQ1:** *Can explicit channel-level sensor reliability estimation improve the robustness of multivariate time-series anomaly detection when sensor observations suffer from missingness, high noise, calibration drift, or stuck values?*

### Supporting Research Questions
- **RQ2:** *Can a dual-perspective decision engine accurately separate isolated sensor hardware faults from genuine cyber-physical process attacks without requiring manual first-principles differential equations?*
- **RQ3:** *What is the relative empirical contribution of temporal continuity, graph neighbor consensus, and residual tail confidence toward overall sensor reliability estimation?*
- **RQ4:** *Does calibrating anomaly and reliability thresholds strictly on clean normal validation data eliminate the optimistic evaluation biases prevalent in published time-series anomaly detection literature?*

---

## 1.14 Objectives

To systematically answer these research questions, the project establishes eight measurable research objectives:
1. **Study and Formalize:** Conduct an in-depth literature audit of existing multivariate time-series anomaly detection models, graph neural networks, and sensor fault diagnosis techniques.
2. **Diagnose Vulnerabilities:** Mathematically analyze and empirically demonstrate the vulnerability of unweighted anomaly detectors to isolated sensor-level corruptions.
3. **Develop Architecture:** Formulate the proposed R-STNet architecture integrating temporal dilated convolutions and graph attention message passing.
4. **Formulate SRM:** Mathematically design the Sensor Reliability Module to compute continuous channel trustworthiness $r_i(t) \in [0, 1]$ via multi-signal triangulation.
5. **Formulate Weighted Scoring:** Derive the reliability-weighted system anomaly aggregation mechanism and the complementary sensor fault localization score.
6. **Design 4-Way Engine:** Implement the multi-state operational discrimination logic to decouple sensor maintenance alerts from plant emergency shutdowns.
7. **Build Controlled Corruption Suite:** Implement a deterministic, reproducible synthetic corruption framework supporting 9 operational failure modes with exact ground-truth masks.
8. **Establish Rigorous Benchmarking:** Formulate a leakage-free evaluation protocol comparing R-STNet against classical, deep reconstruction, and unweighted graph baselines across raw point-wise metrics.

---

## 1.15 Scope

### In-Scope
- Multivariate sensor and actuator telemetry from continuous industrial cyber-physical systems.
- Primary benchmark evaluation on the Secure Water Treatment (SWaT) testbed (51 channels).
- Secondary benchmark preparation for the Water Distribution (WADI) testbed (123 channels).
- Investigation of five primary operational failure modes: Gaussian noise, sensor drift, stuck/frozen values, impulse spikes, and missing data.
- Execution within a constrained computational budget (local NVIDIA RTX 3050 Laptop GPU, 4GB VRAM).

### Out-of-Scope (for Phase 1 / Review 1)
- Large-scale experimental execution and final empirical performance claims (reserved for Phase 2).
- Non-cyber-physical IT server metrics (e.g., PSM, SMD) that lack physical conservation laws.
- Online edge hardware deployment into physical PLC microcontrollers.

---

## 1.16 Expected Contributions

The expected contributions of this research project upon completion include:
1. **A Novel Reliability-Aware Architecture (R-STNet):** A unified deep learning framework that integrates spatio-temporal graph representation learning with an explicit Sensor Reliability Module.
2. **Triangulated Sensor Reliability Formulation:** A continuous evidential trust formulation combining temporal dynamics, graph topological consensus, and standardized residual statistics.
3. **Reliability-Weighted Anomaly Aggregation:** An aggregation mechanism that protects plant-level anomaly detection from false alarms induced by degraded sensors.
4. **A Four-Way Operational Decision Engine:** A structured diagnostic engine providing actionable industrial classification across normal, sensor fault, and attack states.
5. **Controlled Corruption Evaluation Protocol:** A reproducible methodology and dataset suite that bridges the gap between cyber-physical attack labels and sensor fault ground truth.

---

## 1.17 Organization of the Report

The remainder of this Review 1 report is structured as follows:
- **Chapter 2 (Literature Review):** Provides an extensive, thematically organized critical survey of classical anomaly detection, deep multivariate time-series models, graph and causal architectures, sensor fault diagnosis, cyber-physical benchmarks, and evaluation pitfalls, culminating in the formal synthesis of the research gap.
- **Chapter 3 (Methodology):** Formulates the complete proposed R-STNet framework, mathematical derivations of the Sensor Reliability Module, reliability-weighted scoring, the 4-way decision engine, benchmark dataset preparation, baseline configurations, the 10-experiment controlled corruption suite, and the Phase 2 evaluation protocol.
