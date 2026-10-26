# RELIABILITY-AWARE ANOMALY DETECTION FOR MULTIVARIATE SENSOR DATA

**Course:** Machine Learning Lab (BCSE209P)  
**Slot:** L47 + L48  
**Academic Term:** Fall Semester 2026-27  

### Course Project Report — Review 1 (Phase 1)

**Submitted by:**  
**P NAGA VENKATA KUSHAL** (Registration No: **23BDS0133**)  
**KOTERU TARUN** (Registration No: **23BDS0103**)  

**Under the Guidance of:**  
**Dr. DEGA NAGARAJU**  
Associate Professor / Assistant Professor  
School of Computer Science and Engineering (SCOPE)  

**School of Computer Science and Engineering (SCOPE)**  
**Vellore Institute of Technology (VIT), Vellore**  
**October 2026**  

---

## BONAFIDE CERTIFICATE

This is to certify that the project report titled **"RELIABILITY-AWARE ANOMALY DETECTION FOR MULTIVARIATE SENSOR DATA"** is a bonafide record of work carried out by **P NAGA VENKATA KUSHAL (23BDS0133)** and **KOTERU TARUN (23BDS0103)** in partial fulfillment of the requirements for the award of the degree of **Bachelor of Technology in Computer Science and Engineering** with specialization in Data Science during the Fall Semester 2026-27 in the course **Machine Learning Lab (BCSE209P)**, under the faculty guidance of **Dr. DEGA NAGARAJU**, School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology (VIT), Vellore.

The work presented in this report embodies the original research, problem formulation, literature survey, and methodology designed for the **Review 1 (Phase 1)** milestone. The results and conclusions of the proposed experimental framework will be carried out during Phase 2.

<br><br>

----------------------------------------- &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -----------------------------------------  
**Guide Signature** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Head of Department / Dean**  
**Dr. DEGA NAGARAJU** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **SCOPE, VIT Vellore**  
SCOPE, VIT Vellore  

Date: October 2026  
Place: VIT Vellore  

---

## ABSTRACT

In industrial cyber-physical systems (CPS)—including water treatment facilities, power grids, and chemical manufacturing plants—multivariate sensor arrays continuously monitor physical processes. Existing state-of-the-art multivariate time-series anomaly detection (MTSAD) frameworks typically aggregate individual channel reconstruction or forecasting errors uniformly into a single global scalar anomaly score. However, this conventional formulation operates under an unverified assumption: that all sensor observations are equally trustworthy. Consequently, when an isolated sensor experiences an operational corruption—such as high Gaussian noise, calibration drift, frozen stuck values, or impulse spikes—the unweighted aggregation inflates the global anomaly score, incorrectly triggering a critical plant-wide emergency alarm. In mission-critical infrastructure, these false alarms result in costly unwarranted shutdowns, equipment wear, and operator alarm fatigue.

This project investigates whether explicit channel-level sensor reliability estimation can improve the robustness of multivariate anomaly detection by decoupling sensor-level observation corruption from genuine physical process anomalies. We formulate **R-STNet (Dual-Perspective Reliability-Aware Spatio-Temporal Network)**, an integrated deep learning architecture that combines 1D dilated temporal convolutions with inter-sensor graph attention message passing to capture physical conservation laws. The architecture introduces a novel **Sensor Reliability Module (SRM)** that continuously estimates a trustworthiness score $r_i(t) \in [0, 1]$ for each sensor channel by triangulating temporal continuity, cross-sensor graph neighborhood consistency, and standardized residual tail confidence. The resulting reliability scores are incorporated into a reliability-weighted anomaly aggregation mechanism that actively down-weights corrupted channels during system scoring, coupled with a 4-way operational decision engine (Normal, Sensor Fault, System Anomaly, and System Anomaly with Degradation).

The proposed framework is formulated for evaluation on the benchmark **Secure Water Treatment (SWaT)** dataset (51 channels) using a leakage-free chronological splitting protocol and a 10-experiment controlled corruption suite. This Review 1 report formalizes the theoretical foundations, extensive literature survey (45 verified references), mathematical framework, and planned evaluation methodology for Phase 1.

**Keywords:** Multivariate Time-Series Anomaly Detection, Sensor Reliability Module, Cyber-Physical Systems, Spatio-Temporal Graph Attention, Sensor Fault Isolation, False Alarm Suppression, Secure Water Treatment (SWaT).

---

## TABLE OF CONTENTS

- **Certificate**
- **Abstract**
- **Table of Contents**
- **List of Figures**
- **List of Tables**
- **Chapter 1: Introduction**
  - 1.1 Background
  - 1.2 Multivariate Sensor Systems
  - 1.3 Industrial Sensor Monitoring
  - 1.4 Anomaly Detection
  - 1.5 Multivariate Time-Series Anomaly Detection
  - 1.6 Sensor Faults versus System-Level Anomalies
  - 1.7 Problem Statement
  - 1.8 Motivation
  - 1.9 Significance of the Study
  - 1.10 Existing Approaches
  - 1.11 Limitations of Existing Approaches
  - 1.12 Proposed Approach (R-STNet)
  - 1.13 Research Questions
  - 1.14 Objectives
  - 1.15 Scope
  - 1.16 Expected Contributions
  - 1.17 Organization of the Report
- **Chapter 2: Literature Review**
  - 2.1 Introduction
  - 2.2 Classical Anomaly Detection
  - 2.3 Machine Learning-Based Anomaly Detection
  - 2.4 Deep Learning-Based Multivariate Time-Series Anomaly Detection
  - 2.5 Reconstruction-Based Methods
  - 2.6 Prediction-Based Methods
  - 2.7 Graph and Spatial-Temporal Methods
  - 2.8 Causal and Contrastive Learning Methods
  - 2.9 Sensor Fault Detection and Diagnosis (FDI)
  - 2.10 Sensor Reliability and Data Quality
  - 2.11 Robustness to Noise and Sensor Corruption
  - 2.12 Industrial / Cyber-Physical System Anomaly Detection
  - 2.13 Benchmark Datasets
  - 2.14 Evaluation Challenges in Time-Series Anomaly Detection
  - 2.15 Comparative Analysis
  - 2.16 Research Gap
  - 2.17 Summary
- **Chapter 3: Proposed Methodology & Planned Evaluation Framework**
  - 3.1 Methodology Overview
  - 3.2 System Architecture
  - 3.3 Dataset (Secure Water Treatment - SWaT)
  - 3.4 Dataset Preparation
  - 3.5 Preprocessing
  - 3.6 Temporal Windowing
  - 3.7 Baseline Models
  - 3.8 Proposed R-STNet Architecture Formulation
  - 3.9 Sensor Reliability Module (SRM)
  - 3.10 Reliability-Weighted System Anomaly Score
  - 3.11 Sensor Corruption Model
  - 3.12 Four-Way Operational State Interpretation
  - 3.13 Experimental Design
  - 3.14 Evaluation Metrics
  - 3.15 Threshold Selection Protocol
  - 3.16 Experimental Comparison Plan
  - 3.17 Ablation Study Plan
  - 3.18 Computational Environment & Reproducibility
- **References (IEEE Style)**

---

## LIST OF FIGURES

- **Figure 1:** Conceptual Dilemma Between Observation Corruption and Process Anomalies (`report/FIGURES/01_problem_concept.mmd`)
- **Figure 2:** End-to-End Scientific Research Framework (`report/FIGURES/02_research_framework.mmd`)
- **Figure 3:** Overall Research Methodology and Data Processing Pipeline (`report/FIGURES/03_overall_methodology.mmd`)
- **Figure 4:** Detailed Architecture of the Proposed R-STNet Framework (`report/FIGURES/04_rstnet_architecture.mmd`)
- **Figure 5:** Triangulated Sensor Reliability Module (SRM) Formulation (`report/FIGURES/05_sensor_reliability_module.mmd`)
- **Figure 6:** Four-Way Operational State Discrimination Logic (`report/FIGURES/06_four_state_framework.mmd`)
- **Figure 7:** Controlled Sensor Corruption Matrix and Dual Evaluation Protocol (`report/FIGURES/07_experimental_design.mmd`)

---

## LIST OF TABLES

- **Table 1:** Problem Characteristics in Industrial Multivariate Sensor Monitoring
- **Table 2:** Literature Comparison Across Anomaly Detection Paradigms
- **Table 3:** Synthesis of the Research Gap
- **Table 4:** Benchmark Dataset Characteristics (SWaT vs. WADI)
- **Table 5:** Proposed R-STNet Architecture Components
- **Table 6:** Comparative Overview of Evaluation Baselines
- **Table 7:** Controlled Sensor Corruption Suite (Planned Evaluation Scenarios)
- **Table 8:** Comprehensive Evaluation Metrics Framework
- **Table 9:** Planned Experimental Suite Matrix (Phase 2 Execution)

---

# CHAPTER 1: INTRODUCTION

### Section Purpose & Structure
This introductory chapter establishes the operational context of industrial cyber-physical sensor systems, differentiates between observation corruptions and true process anomalies, formulates the formal mathematical problem statement, outlines the research questions and measurable objectives, defines the scope, and articulates the expected contributions of the project.

---

### 1.1 Background
Modern industrial automation, infrastructure management, and manufacturing systems have undergone a profound digital transformation driven by Industry 4.0 and Cyber-Physical Systems (CPS) [6], [7]. In mission-critical sectors—such as municipal water treatment facilities, thermal and nuclear power plants, chemical manufacturing refineries, and smart electric grids—continuous computational control loops are coupled with physical processes. Ensuring the uninterrupted safety, operational stability, and physical integrity of these plants requires continuous real-time supervisory monitoring. 

In traditional supervisory architectures, human operators relied on manual inspections and static alarm thresholds configured within Supervisory Control and Data Acquisition (SCADA) systems. However, as the physical scale, operational complexity, and interconnectedness of modern engineering infrastructure have expanded, static threshold-based monitoring has proven fundamentally inadequate [3], [16]. Complex physical processes are subject to multivariate interactions, non-stationary environmental shifts, and sophisticated cyber-physical intrusions. Consequently, data-driven machine learning techniques—specifically Multivariate Time-Series Anomaly Detection (MTSAD)—have emerged as an indispensable foundation for modern industrial safety, automated diagnostics, and predictive maintenance [16], [17].

---

### 1.2 Multivariate Sensor Systems
A multivariate sensor system comprises a spatially distributed and functionally heterogeneous array of sensing instruments deployed across a physical installation [38], [39]. These instruments record continuous physical, chemical, and thermodynamic variables alongside discrete actuator commands at regular sampling intervals. 

Mathematically, at any discrete time step $t \in \{1, 2, \dots, T\}$, the collective observation from a network of $M$ monitoring channels is represented as an $M$-dimensional vector:

$$\mathbf{X}(t) = \left[ x_1(t), x_2(t), \dots, x_M(t) \right]^T \in \mathbb{R}^M$$

where each channel $x_i(t)$ represents the measurement of physical variable $i$ (e.g., fluid level, volumetric flow, differential pressure, or chemical concentration).

In cyber-physical sensor systems, these variables do not evolve in isolation. Instead, they are tightly coupled through two distinct structural dimensions:
1. **Temporal Dependencies:** Individual sensor channels exhibit autoregressive memory, rate-of-change constraints, inertia lags, and settling times governed by physical laws of motion, thermodynamics, and mass transfer [16], [25].
2. **Spatial and Cross-Sensor Dependencies:** Distinct sensor channels are bound by physical conservation principles (e.g., conservation of mass, conservation of energy, hydraulic continuity, and chemical stoichiometry) [19], [33]. For instance, opening an upstream control valve predictably triggers an increase in downstream flow rate and a subsequent change in storage tank level.

Consequently, an effective monitoring system must simultaneously model intra-sensor temporal evolutions and inter-sensor spatial correlations [20], [21].

---

### 1.3 Industrial Sensor Monitoring
Industrial sensor monitoring operates under stringent operational requirements: high availability, minimal computational latency, and ultra-low false alarm rates [6], [43]. In critical infrastructure such as water treatment plants (e.g., the Secure Water Treatment testbed [38]), continuous telemetry streams from flow meters, level transmitters, motorized pumps, and chemical analyzers to Programmable Logic Controllers (PLCs) executing closed-loop proportional-integral-derivative (PID) control.

In these environments, anomalous behaviors can emerge from diverse operational vectors:
- **Physical Equipment Failures:** Mechanical pump cavitation, valve seizing, pipe ruptures, or membrane fouling.
- **Cyber-Physical Attacks:** Malicious adversaries manipulating sensor measurements (sensor spoofing) or overriding actuator states (valve hijacking) via compromised network communications [43], [44].
- **Process Disturbances:** Severe fluctuations in inflow composition, power supply transients, or ambient environmental surges.

Automated anomaly detection frameworks must continuously ingest multivariate streams $\mathbf{X}(t)$, learn the normal operational envelope, and provide early warning flags prior to catastrophic system failure.

---

### 1.4 Anomaly Detection
In machine learning and statistics, an anomaly (or outlier) is defined as an observation or sequence of observations that deviates so substantially from the established distribution of normal data as to arouse suspicions that it was generated by a different underlying mechanism [3], [8].

Classical literature categorizes anomalies into three distinct taxonomies [3]:
1. **Point Anomalies:** An individual data instance that deviates significantly from the rest of the dataset (e.g., an extreme, physically impossible pressure reading).
2. **Contextual (Conditional) Anomalies:** A data instance that appears normal in a global statistical sense, but deviates anomalously within its specific context (e.g., high heating power consumption during peak summer, or low tank level when inflow valves are fully open).
3. **Collective Anomalies:** A collection of related data instances that appear normal individually, but whose collective sequential pattern or joint combination violates expected system behavior over time [16].

In cyber-physical sensor monitoring, the vast majority of critical operational failures and cyber-attacks manifest as subtle **contextual and collective anomalies**, requiring models capable of reasoning over temporal windows rather than isolated static points [10], [19].

---

### 1.5 Multivariate Time-Series Anomaly Detection
Multivariate Time-Series Anomaly Detection (MTSAD) formalizes the process of assigning an anomaly score $S(t) \in \mathbb{R}$ to a sequence window of multivariate observations:

$$\mathbf{W}_t = \left[ \mathbf{X}(t - L + 1), \mathbf{X}(t - L + 2), \dots, \mathbf{X}(t) \right] \in \mathbb{R}^{L \times M}$$

where $L$ denotes the temporal lookback horizon.

MTSAD frameworks typically adopt an **unsupervised or semi-supervised one-class learning paradigm** [10], [17]. Because catastrophic industrial accidents and cyber-attacks are extremely rare, historical datasets contain overwhelming volumes of normal operational data, while labeled anomalies are scarce, highly diverse, and unpredictable. Consequently, models are trained exclusively on healthy baseline operation to capture normal data distributions. At inference time, any substantial deviation between the observed window and the model's learned representation is interpreted as an anomaly [9], [19].

---

### 1.6 Sensor Faults versus System-Level Anomalies
A critical reality in industrial monitoring is the fundamental difference between a **sensor fault** and a **system-level anomaly** [29], [33]:

#### System-Level Anomaly (Process / Cyber-Physical Attack)
A system-level anomaly occurs when the physical process itself deviates from safe operational boundaries (e.g., an attacker forces a valve open, causing a storage tank to overflow, or a physical pipe fractures). Under a genuine system anomaly:
- The underlying physical state is abnormal.
- Physical sensors are functioning correctly and reporting true physical measurements.
- Neighboring, physically coupled sensors register correlated deviations consistent with conservation laws.
- **Required Action:** Immediate emergency plant alarm, operator intervention, and automated fail-safe trip [38], [43].

#### Sensor Fault (Instrument / Observation Corruption)
A sensor fault occurs when the physical plant is operating in complete, safe equilibrium, but an individual sensing instrument produces an abnormal or corrupted measurement due to hardware degradation [28], [30]:
- **High Gaussian Noise:** Electromagnetic interference or aging signal amplifiers.
- **Sensor Drift:** Gradual calibration shifts resulting from chemical fouling, membrane coating, or mechanical fatigue [30].
- **Frozen / Stuck Sensors:** Complete loss of telemetry dynamics due to mechanical float jamming or analog-to-digital converter lockup.
- **Impulse Spikes:** Power supply switching ripples or mechanical shock.
- **The underlying physical plant remains completely safe and normal.**
- Neighboring sensors show normal, healthy operation.
- **Required Action:** Suppress plant shutdown; alert instrumentation and control (I&C) technicians to clean, recalibrate, or replace the faulty transmitter.

---

### 1.7 Problem Statement
We formulate the core research problem mathematically:

Let an industrial cyber-physical process be monitored by $M$ continuous and discrete channels streaming observations $\mathbf{X}(t) = [x_1(t), x_2(t), \dots, x_M(t)]^T \in \mathbb{R}^M$ at time step $t$. Let $\hat{\mathbf{X}}(t) \in \mathbb{R}^M$ denote the expected normal observation vector generated by a predictive or reconstructive spatio-temporal model, yielding instantaneous residual deviations:

$$e_i(t) = \left| x_i(t) - \hat{x}_i(t) \right|, \quad \forall i \in \{1, 2, \dots, M\}$$

Standard multivariate anomaly detection models aggregate these residuals uniformly into a global scalar anomaly score:

$$S_{\text{global}}(t) = \frac{1}{M} \sum_{i=1}^M \phi(e_i(t))$$

where $\phi(\cdot)$ is a non-negative scaling function.

When an isolated sensor $k$ experiences an observation corruption (such that $e_k(t) \gg 0$) while the underlying physical process remains completely nominal ($x_j(t) \approx \hat{x}_j(t)$ for all $j \neq k$), the unweighted summation forces:

$$S_{\text{global}}(t) \ge \tau_{\text{sys}}$$

triggering a **false plant-wide emergency alarm**.

**The Formal Research Problem:**  
The objective of this research is to formulate, implement, and rigorously evaluate an explicit reliability-aware mechanism that:
1. Dynamically estimates a continuous, channel-specific reliability vector $\mathbf{r}(t) = [r_1(t), \dots, r_M(t)]^T \in [0, 1]^M$ online, reflecting the trustworthiness of each individual sensor observation.
2. Incorporates $\mathbf{r}(t)$ into system-level anomaly scoring such that degraded sensors are down-weighted, preventing false system shutdowns:
   $$S_{\text{sys}}(t) = \frac{\sum_{i=1}^M r_i(t) \cdot z_i(t)}{\sum_{i=1}^M r_i(t) + \epsilon}$$
3. Simultaneously isolates the corrupted channel for maintenance while preserving rapid detection of genuine multi-sensor cyber-physical attacks.

---

### 1.8 Motivation
The primary motivation of this study stems from the severe operational and economic consequences of false positive alarms in modern automated industries:
1. **Economic Cost of Unwarranted Downtime:** In continuous processing plants (such as chemical refineries or municipal water treatment works), an emergency shutdown triggers extensive restart protocols, equipment thermal stress, chemical wastage, and millions of dollars in downtime costs [6], [7].
2. **Alarm Fatigue:** When monitoring systems regularly trigger alarms caused by noisy, drifting, or glitched sensors, human operators develop "alarm fatigue," leading them to silence, ignore, or disable automated safety interlocks—a primary contributor to historical industrial disasters.
3. **The Architectural Gap in Deep Learning:** While modern deep learning architectures (e.g., Graph Neural Networks, Transformers, Causal Contrastive Learning) have drastically improved representation capacity on pristine benchmark splits, they remain vulnerable to individual channel corruptions due to their unweighted residual aggregation paradigms [19], [21], [27].

---

### 1.9 Significance of the Study
This investigation bridges the divide between two historically segregated engineering disciplines:
- **Computer Science & Machine Learning:** Focuses on high-capacity spatio-temporal representation learning, graph attention, and unsupervised representation learning for anomaly detection [10], [19], [20].
- **Control Engineering & Process Systems:** Focuses on analytical redundancy, sensor validation, fault isolation, and hardware reliability estimation [28], [33], [36].

By integrating physical consensus principles (graph neighbor agreement and temporal continuity) directly into deep neural anomaly detection, this study establishes a unified, scientifically grounded framework capable of operating robustly in harsh, real-world industrial environments.

---

### 1.10 Existing Approaches
Existing literature for multivariate sensor anomaly detection spans four broad paradigms:
1. **Classical Statistical & Machine Learning Baselines:** Principal Component Analysis (PCA) [6], Dynamic PCA [5], One-Class Support Vector Machines (OC-SVM) [2], and Isolation Forest [1].
2. **Deep Reconstruction & Prediction Models:** Long Short-Term Memory Autoencoders (LSTM-AE) [14], Unsupervised Anomaly Detection (USAD) [9], Stochastic Recurrent Networks (OmniAnomaly) [10], and Multi-Scale Convolutional Recurrent Autoencoders (MSCRED) [13].
3. **Graph Neural Networks (GNNs):** Graph Deviation Network (GDN) [19], Multivariate Time-Series Anomaly Detection via Graph Attention (MTAD-GAT) [20], and Spatio-Temporal Graph Convolutional Networks [23].
4. **Causal & Contrastive Learning:** Causality-Aware Contrastive Learning (CAROTS) [21], which enforces representation invariance under domain shifts.

---

### 1.11 Limitations of Existing Approaches
Despite significant algorithmic advances, current approaches exhibit fundamental limitations:
1. **Uniform Residual Aggregation:** Every established deep detector (USAD, OmniAnomaly, GDN, MTAD-GAT, TranAD) aggregates channel errors uniformly. A single corrupted sensor is mathematically capable of triggering a system alarm [19], [20].
2. **Absence of Channel Trust Metrics:** Existing models output an anomaly score for the *timestamp*, but do not produce a continuous, interpretable reliability score $r_i(t) \in [0, 1]$ for each individual sensor channel $i$.
3. **Latent Representation Contamination:** In autoencoder architectures (e.g., USAD, OmniAnomaly), feeding corrupted inputs through dense encoder layers contaminates the latent bottleneck, corrupting the reconstruction of clean channels [10].
4. **Binary Decision Rigidity:** Conventional systems produce a single binary decision ($\text{Normal}$ vs. $\text{Anomaly}$), failing to differentiate whether an alarm represents instrument maintenance or a plant emergency.
5. **Evaluation Disconnect in Benchmarks:** Benchmarks like SWaT and WADI provide attack labels, but lack ground truth for isolated sensor hardware faults, leaving sensor fault isolation largely untested in mainstream deep learning literature [38], [40].

---

### 1.12 Proposed Approach: R-STNet
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

### 1.13 Research Questions
#### Central Research Question
> **RQ1:** *Can explicit channel-level sensor reliability estimation improve the robustness of multivariate time-series anomaly detection when sensor observations suffer from missingness, high noise, calibration drift, or stuck values?*

#### Supporting Research Questions
- **RQ2:** *Can a dual-perspective decision engine accurately separate isolated sensor hardware faults from genuine cyber-physical process attacks without requiring manual first-principles differential equations?*
- **RQ3:** *What is the relative empirical contribution of temporal continuity, graph neighbor consensus, and residual tail confidence toward overall sensor reliability estimation?*
- **RQ4:** *Does calibrating anomaly and reliability thresholds strictly on clean normal validation data eliminate the optimistic evaluation biases prevalent in published time-series anomaly detection literature?*

---

### 1.14 Objectives
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

### 1.15 Scope
#### In-Scope
- Multivariate sensor and actuator telemetry from continuous industrial cyber-physical systems.
- Primary benchmark evaluation on the Secure Water Treatment (SWaT) testbed (51 channels).
- Secondary benchmark preparation for the Water Distribution (WADI) testbed (123 channels).
- Investigation of five primary operational failure modes: Gaussian noise, sensor drift, stuck/frozen values, impulse spikes, and missing data.
- Execution within a constrained computational budget (local NVIDIA RTX 3050 Laptop GPU, 4GB VRAM).

#### Out-of-Scope (for Phase 1 / Review 1)
- Large-scale experimental execution and final empirical performance claims (reserved for Phase 2).
- Non-cyber-physical IT server metrics (e.g., PSM, SMD) that lack physical conservation laws.
- Online edge hardware deployment into physical PLC microcontrollers.

---

### 1.16 Expected Contributions
The expected contributions of this research project upon completion include:
1. **A Novel Reliability-Aware Architecture (R-STNet):** A unified deep learning framework that integrates spatio-temporal graph representation learning with an explicit Sensor Reliability Module.
2. **Triangulated Sensor Reliability Formulation:** A continuous evidential trust formulation combining temporal dynamics, graph topological consensus, and standardized residual statistics.
3. **Reliability-Weighted Anomaly Aggregation:** An aggregation mechanism that protects plant-level anomaly detection from false alarms induced by degraded sensors.
4. **A Four-Way Operational Decision Engine:** A structured diagnostic engine providing actionable industrial classification across normal, sensor fault, and attack states.
5. **Controlled Corruption Evaluation Protocol:** A reproducible methodology and dataset suite that bridges the gap between cyber-physical attack labels and sensor fault ground truth.

---

### 1.17 Organization of the Report
The remainder of this Review 1 report is structured as follows:
- **Chapter 2 (Literature Review):** Provides an extensive, thematically organized critical survey of classical anomaly detection, deep multivariate time-series models, graph and causal architectures, sensor fault diagnosis, cyber-physical benchmarks, and evaluation pitfalls, culminating in the formal synthesis of the research gap.
- **Chapter 3 (Methodology):** Formulates the complete proposed R-STNet framework, mathematical derivations of the Sensor Reliability Module, reliability-weighted scoring, the 4-way decision engine, benchmark dataset preparation, baseline configurations, the 10-experiment controlled corruption suite, and the Phase 2 evaluation protocol.

---

# CHAPTER 2: LITERATURE REVIEW

### Section Purpose & Structure
This chapter presents a comprehensive, thematically organized critical review of 45 verified academic papers (spanning 1998–2025, with emphasis on 2018–2025 and premier Elsevier/ScienceDirect journals). It synthesizes existing paradigms, identifies strengths and limitations, and formally articulates the research gap addressed by this study.

---

### 2.1 Introduction
Multivariate time-series anomaly detection (MTSAD) in complex engineering and cyber-physical systems (CPS) has attracted substantial research over the past three decades [3], [16]. The challenge of reliably detecting subtle, safety-critical deviations while remaining robust to inevitable measurement imperfections lies at the intersection of applied machine learning, statistical process monitoring, graph representation learning, and control engineering [6], [19], [33].

To provide a rigorous, thematic synthesis of the state-of-the-art—rather than a disconnected chronological summary—this literature review is organized around **five major thematic pillars**:
1. **Classical and Statistical Anomaly Detection (Sections 2.2–2.3):** Distance, density, linear projection, and tree-based partition methods.
2. **Machine Learning and Deep Learning for Multivariate Sequences (Sections 2.4–2.6):** Reconstruction, autoregressive forecasting, and stochastic latent architectures.
3. **Spatial, Graph, and Causal Multivariate Modeling (Sections 2.7–2.8):** Graph neural networks, graph attention, topological message passing, and causal structural discovery.
4. **Sensor Faults, Sensor Reliability, and Robustness (Sections 2.9–2.11):** Fault detection and diagnosis (FDD), analytical redundancy, sensor data validation, and noise-tolerant representations.
5. **Industrial CPS Benchmarks and Evaluation Protocols (Sections 2.12–2.14):** Testbed architectures (SWaT, WADI), cyber-physical attack profiles, and critical evaluation methodology pitfalls.

---

### 2.2 Classical Anomaly Detection
Classical anomaly detection foundations were established on cross-sectional tabular data, where observations are treated as static, independent, and identically distributed (i.i.d.) vectors [3], [8]. These techniques generally classify into statistical, distance-based, and density-based formulations:
- **Statistical Methods:** Assume normal data follows a parametric probability distribution (e.g., Gaussian, Student's $t$, or Gaussian Mixture Models). Points falling into low-probability density tails ($p(\mathbf{x}) < \alpha$) are flagged as outliers [8].
- **Distance-Based Methods:** Operate under the assumption that normal data instances reside in dense clusters, whereas anomalies are located far from their $k$-nearest neighbors. Global distance metrics (e.g., Euclidean or Mahalanobis distance) quantify isolation.
- **Density-Based Methods:** Acknowledge that real-world datasets exhibit clusters of varying densities. The seminal **Local Outlier Factor (LOF)** algorithm developed by Breunig et al. (2000) [4] compares the local density of an object to the local densities of its $k$-nearest neighbors. Points with substantially lower local reachability densities are identified as local outliers.

**Strengths and Limitations for Multivariate Sensor Streams:**  
Non-parametric methods like LOF [4] do not require rigid distributional assumptions and provide intuitive geometric interpretations for tabular data. However, classical distance and density methods suffer catastrophically from the **curse of dimensionality** when applied to multivariate sensor streams ($M \ge 50$). In high-dimensional spaces, distance metrics concentrate, rendering pairwise Euclidean distances equidistant and uninformative [3]. Crucially, static classical methods discard sequential temporal ordering, rendering them incapable of detecting contextual and collective time-series anomalies where individual readings fall within nominal ranges but violate temporal dynamics [16].

---

### 2.3 Machine Learning-Based Anomaly Detection
To scale beyond distance calculations, machine learning introduced tree-partitioning and margin-based boundaries:
- **Isolation Forest:** Developed by Liu, Ting, and Zhou (2008) [1], Isolation Forest isolates anomalies rather than profiling normal points. It constructs an ensemble of randomized binary trees (iTrees). Because anomalous points are few and possess attribute values divergent from the norm, they are isolated closer to the root of the trees, resulting in noticeably shorter average path lengths $h(\mathbf{x})$. The anomaly score is defined as:
  $$s(\mathbf{x}, n) = 2^{-\frac{\mathbb{E}(h(\mathbf{x}))}{c(n)}}$$
  where $c(n)$ is the average path length of unsuccessful searches in a Binary Search Tree of $n$ instances.
- **One-Class Support Vector Machines (OC-SVM):** Formulated by Schölkopf et al. (1999) [2], OC-SVM maps training vectors into a high-dimensional reproducing kernel Hilbert space (RKHS) via non-linear kernels (e.g., Radial Basis Function) and finds a maximum-margin hyperplane that separates the normal training data from the coordinate origin.

In industrial and chemical process engineering, statistical projection methods have long dominated automated monitoring [6], [7]:
- **Principal Component Analysis (PCA):** Projects correlated sensor variables onto an orthogonal low-dimensional subspace capturing dominant variance. Residual deviations in the orthogonal complement subspace are monitored via the Squared Prediction Error (SPE or $Q$-statistic), while in-subspace shifts are monitored via Hotelling's $T^2$ [6], [36].
- **Dynamic PCA (DPCA):** To capture serial correlation, Wang & He (2020) [5] (*Computers & Chemical Engineering*) stack time-lagged observation vectors into augmented Hankel matrices before computing principal components, enabling detection of dynamic process shifts in complex loops like the Tennessee Eastman Process (TEP).

**Limitations in Cyber-Physical Sensor Systems:**  
While Isolation Forest [1] and DPCA [5] remain widely used industrial baselines, they cannot capture complex non-linear spatio-temporal interactions across dozens of continuous stages [16]. A single noisy or drifted sensor causes severe **smearing effects** across projection matrices [6]: a fault on sensor $k$ contaminates the SPE contributions across multiple clean channels, making fault isolation difficult. Tree-based and linear models cannot natively learn continuous sensor reliability scores online.

---

### 2.4 Deep Learning-Based Multivariate Time-Series Anomaly Detection
The advent of deep neural networks enabled end-to-end representation learning directly from continuous time-series without requiring manual feature engineering [16], [17]. Deep MTSAD architectures process multivariate sliding windows $\mathbf{W}_t \in \mathbb{R}^{L \times M}$ and are broadly classified into **reconstruction-based**, **prediction-based**, and **hybrid frameworks** [17].

---

### 2.5 Reconstruction-Based Methods
Reconstruction models operate on the assumption that an encoder network $f_\theta(\cdot)$ can compress normal multivariate windows into a low-dimensional bottleneck representation $\mathbf{z}$, and a decoder network $g_\phi(\cdot)$ can reconstruct the original input $\tilde{\mathbf{X}} = g_\phi(f_\theta(\mathbf{X}))$ [14], [16]. Because the network is trained exclusively on normal patterns, it fails to reconstruct unseen anomalous patterns, yielding elevated reconstruction errors:

$$S_{\text{recon}}(t) = \|\mathbf{X}(t) - \tilde{\mathbf{X}}(t)\|_2^2$$

#### 1. USAD (UnSupervised Anomaly Detection)
- **Reference:** Audibert, Michiardi, Guyard, Marti, and Zuluaga (ACM KDD 2020) [9].
- **Architecture:** Employs an autoencoder architecture with an adversarial training policy. A single encoder $E$ feeds two independent decoders, $D_1$ and $D_2$. During phase one, $E$ and $D_1$ are trained to reconstruct normal windows; during phase two, $D_2$ is trained adversarially to discriminate between the real input and the reconstruction from $D_1$.
- **Strengths:** Fast training; adversarial objective amplifies anomalous reconstruction errors.
- **Limitations:** Flattens temporal windows into 1D vectors, discarding sequential structure. Crucially, **adversarial error amplification exacerbates sensor noise**: when an isolated sensor experiences high Gaussian noise or a spike, the adversarial decoders aggressively amplify the error, guaranteeing a massive false plant emergency shutdown.

#### 2. OmniAnomaly (Stochastic Recurrent Networks)
- **Reference:** Su, Zhao, Niu, Liu, Sun, and Pei (ACM KDD 2019) [10].
- **Architecture:** Combines Gated Recurrent Units (GRU) with a stochastic variational latent space augmented by planar normalizing flows. Captures temporal dependencies via GRU cells and models complex multimodal normal data distributions through non-Gaussian stochastic latent variables $\mathbf{z}_t \sim q_\phi(\mathbf{z}_t | \mathbf{x}_{\le t})$.
- **Strengths:** Robust probabilistic modeling of normal operational noise; effective on server telemetry benchmarks (SMD, SMAP).
- **Limitations:** Ignores inter-sensor physical topologies. Suffers from **latent space contamination**: because all $M$ sensor channels are fed into dense recurrent encoder layers, an extreme reading on a single corrupted sensor contaminates the latent state $\mathbf{z}_t$, degrading the reconstruction of all remaining clean channels.

#### 3. MSCRED (Multi-Scale Convolutional Recurrent Encoder-Decoder)
- **Reference:** Zhang, Song, Chen, Shen, and Mount (AAAI 2019) [13].
- **Architecture:** Constructs multi-scale cross-correlation signature matrices over various lookback horizons to capture spatial correlations, encoding them via 2D convolutional autoencoders coupled with ConvLSTM networks.
- **Strengths:** Retains spatial cross-correlation matrices across multiple temporal scales.
- **Limitations:** Computing continuous signature matrices across 50+ channels incurs prohibitive computational overhead; sums signature reconstruction errors uniformly without channel reliability weighting.

---

### 2.6 Prediction-Based Methods
Prediction-based (forecasting) methods train autoregressive deep models to forecast the expected one-step-ahead sensor values $\hat{\mathbf{X}}(t) = f_\theta(\mathbf{W}_{t-1})$. The anomaly score is computed from the prediction residual:

$$S_{\text{pred}}(t) = \|\mathbf{X}(t) - \hat{\mathbf{X}}(t)\|_2^2$$

#### 1. TranAD (Deep Transformer Networks for Time-Series Anomaly Detection)
- **Reference:** Tuli, Casale, and Jennings (VLDB 2022) [11].
- **Architecture:** Leverages multi-head self-attention mechanisms to model long-range temporal dependencies, using an adversarial two-phase training procedure with focus scores to stabilize attention maps.
- **Strengths:** Captures subtle contextual temporal shifts; achieves strong benchmark F1-scores.
- **Limitations:** Quadratic computational and memory complexity $\mathcal{O}(L^2)$ creates severe VRAM bottlenecks on local hardware. Treats all feature residual deviations equally during anomaly scoring, leaving system alerts vulnerable to corrupted channels.

#### 2. Anomaly Transformer (Association Discrepancy)
- **Reference:** Xu, Wu, Wang, and Long (ICLR 2022) [12].
- **Architecture:** Introduces the concept of **association discrepancy**. Combines classic self-attention (series association) with a Gaussian-kernel prior association. In normal periods, series and prior associations match; during anomalies, anomalous points struggle to build associations with distant normal points, creating a large discrepancy.
- **Strengths:** Principled mathematical formulation capturing temporal association breakdowns.
- **Limitations:** Evaluates association discrepancy globally across all channels; lacks channel-specific sensor reliability estimation.

---

### 2.7 Graph and Spatial-Temporal Methods
In industrial cyber-physical systems, sensors are deployed across physical piping networks, chemical reactors, and electrical buses [38], [39]. Treating sensors as independent channels (univariate modeling) or passing them through unconstrained dense latent layers discards the **physical topological network** that governs the plant.

Graph Neural Networks (GNNs) represent sensor channels as nodes $\mathcal{V} = \{v_1, \dots, v_M\}$ and physical or statistical relationships as edges $\mathcal{E}$. This allows graph convolutional message passing to enforce physical conservation principles [19], [25].

#### 1. Graph Deviation Network (GDN)
- **Reference:** Deng and Hooi (AAAI / IJCAI 2021) [19].
- **Architecture:** Learns a directed graph structure dynamically using node embeddings $\mathbf{v}_i \in \mathbb{R}^d$. Evaluates pairwise similarity $e_{ji} = \frac{\mathbf{v}_i^T \mathbf{v}_j}{\|\mathbf{v}_i\| \|\mathbf{v}_j\|}$, applies Top-$K$ sparsification to retain the strongest directed edges, and computes edge weights via Softmax. A graph attention network aggregates messages from topological neighbors, forecasting the next expected reading for each sensor.
- **Anomaly Score:** Standardizes prediction residuals: $a_i(t) = \frac{|x_i(t) - \hat{x}_i(t)|}{\text{Median}_i + \text{MAD}_i}$, and takes the maximum: $S(t) = \max_i a_i(t)$.
- **Relevance to Our Work:** GDN provides the foundational inspiration for our learned graph dependency module.
- **Critical Limitation Addressed by R-STNet:** By taking the maximum standardized residual $\max_i a_i(t)$ or summing errors uniformly, **GDN is completely unprotected against single-sensor corruption**. If sensor $k$ develops calibration drift, spikes, or high noise, $a_k(t)$ spikes, instantly triggering a false plant alarm. GDN lacks any Sensor Reliability Module to evaluate whether sensor $k$ is trustworthy.

#### 2. MTAD-GAT (Multivariate Time-Series Anomaly Detection via GAT)
- **Reference:** Zhao, Wang, Duan, Huang, Cao, Tong, and Xu (IEEE ICDM 2020) [20].
- **Architecture:** Employs two parallel Graph Attention Networks: a **Feature-Oriented GAT** (capturing spatial dependencies among sensors) and a **Time-Oriented GAT** (capturing temporal correlations across sequence steps within the window). Combines joint forecasting and reconstruction heads.
- **Limitations:** Treats all feature errors equally during anomaly aggregation; lacks an explicit sensor trustworthiness metric.

#### 3. Recent Elsevier Advancements in Spatio-Temporal GNNs
- **Chen et al. (2022) [22] (*Neurocomputing*):** Deep variational graph convolutional recurrent networks for MTSAD.
- **Han & Lee (2023) [23] (*Expert Systems with Applications*):** Multi-scale spatio-temporal graph convolutional networks capturing multi-frequency oscillations.
- **Li, Zhang, and Zhao (2023) [24] (*Computers in Industry*):** Spatio-temporal graph neural networks applied specifically to industrial sensor fault diagnosis.

---

### 2.8 Causal and Contrastive Learning Methods
- **Reference:** Kim, Mok, Lee, Lew, Kim, and Yoon (ICML 2025 / arXiv:2506.03964) [21] (**CAROTS**).
- **Theoretical Mechanism:** Real-world multivariate sensor data is subject to domain shifts and external operational variations. CAROTS integrates causal structural equation modeling (DAG discovery) with contrastive representation learning. It constructs augmented temporal views that preserve underlying causal physical topologies while perturbing non-causal spurious correlations.
- **Role in Our Project:** CAROTS is maintained as a **read-only external state-of-the-art research reference and benchmark** in `reference/CAROTS-master/` (in accordance with ADR-006 [Technical References]).
- **Conceptual Differentiation:**  
  CAROTS operates at the **latent contrastive representation invariance level**. It does **not** estimate an explicit per-sensor reliability vector $\mathbf{r}(t) \in [0, 1]^M$, does not localize which specific sensor transmitter is degraded, and does not provide an operational 4-way decision framework. Our proposed R-STNet focuses specifically on explicit, channel-level reliability estimation and false-alarm suppression.

---

### 2.9 Sensor Fault Detection and Diagnosis (FDI)
While computer science literature has traditionally focused on anomaly detection, the control systems and reliability engineering literature has spent decades studying **Fault Detection and Isolation (FDI)** [33], [34], [35]:
- **Analytical Redundancy:** Formulated by Gertler (1998) [33] and Isermann (2006) [34], analytical redundancy generates parity equations using physical conservation laws. If three sensors monitor water level, inflow, and outflow of a tank, the mass balance $\frac{dV}{dt} = Q_{\text{in}} - Q_{\text{out}}$ provides an independent, redundant estimate of the level sensor.
- **Process History-Based Methods:** Comprehensive surveys by Venkatasubramanian et al. (2003) [36] (*Computers & Chemical Engineering*) and Qin (2012) [6] (*Annual Reviews in Control*) review qualitative, quantitative, and statistical methods for isolating sensor faults in chemical processing plants.
- **Nonlinear Observers:** Sharifi & Langari (2014) [29] (*Control Engineering Practice*) demonstrated non-linear observer-based sensor fault diagnosis specifically on water treatment testbeds.

**The Scalability Bottleneck of Classical FDI:**  
Classical model-based FDI requires **exact mathematical differential equations** of the entire physical plant (differential equations governing every valve, pump curve, friction factor, and chemical reaction rate) [33], [34]. For modern large-scale facilities (e.g., SWaT with 51 channels, WADI with 123 channels), hand-crafting exact physical ODEs is economically and mathematically intractable.

**Our Proposed Solution:** R-STNet **learns analytical redundancy directly from normal operational data** using data-driven spatio-temporal representations and graph attention, eliminating the need for manual physics modeling.

---

### 2.10 Sensor Reliability and Data Quality
Recent literature in premier Elsevier journals has recognized the necessity of quantifying sensor data reliability and validation:
- **Zhu & Fei (2022) [32] (*ISA Transactions*):** Proposed multi-sensor data validation and reconstruction in industrial CPS using deep autoencoders with confidence estimation. While establishing the validity of confidence metrics, their approach lacks explicit graph neighbor consensus and 4-way operational state logic.
- **Huang, Yan, and Liu (2021) [30] (*Measurement*):** Combined spatial-temporal correlation with deep learning for sensor fault diagnosis and data reconstruction, demonstrating that physical correlation matrices can reconstruct missing or corrupted sensor readings.
- **Liu & Chen (2022) [31] (*Journal of Process Control*):** Developed dynamic graph attention networks for robust process monitoring and sensor fault isolation, confirming that attention weights reflect physical topological coupling.
- **Saeed, Jan, Lee, and Koo (2021) [28] (*Reliability Engineering & System Safety*):** Investigated extremely randomized trees for classifying wireless sensor network faults into bias, drift, performance degradation, and noise.

---

### 2.11 Robustness to Noise and Sensor Corruption
- **Reference:** Audibert, Michiardi, Guyard, and Zuluaga (IEEE TKDE 2022) [27] (**RANSynCoders**).
- **Mechanism:** Inspired by the RANSAC algorithm in computer vision, RANSynCoders trains an ensemble of autoencoders on random subsets of sensor channels. At inference time, it evaluates consensus across the subset reconstructions: if an anomaly is present across all subsets, it represents a genuine system anomaly; if it affects only subsets containing a specific sensor, it is treated as sensor noise.
- **Strengths:** Highly robust against noisy and corrupted channels.
- **Limitations:** Training and executing dozens of random subset autoencoders incurs severe computational overhead. It relies on random brute-force subset selection rather than learning the physical graph topology, and does not output an interpretable, continuous reliability score $r_i(t) \in [0, 1]$.

---

### 2.12 Industrial / Cyber-Physical System Anomaly Detection
Industrial Control Systems (ICS) and SCADA networks present unique operational challenges [43], [44]:
- **High Consequence of False Alarms:** Emergency plant shutdowns risk physical equipment damage, water contamination, and massive restart expenses [38].
- **Stealthy Cyber-Physical Attacks:** Modern cyber-attacks (e.g., Stuxnet-style physical attacks) manipulate physical processes within nominal static threshold bounds, making them invisible to univariate threshold detectors [43].
- **Lightweight Requirements:** Research by Kravchik & Shabtai (2022) [43] and Al-Abassi et al. (2020) [44] emphasizes the necessity of lightweight convolutional and recurrent architectures capable of executing on local industrial edge hardware.

---

### 2.13 Benchmark Datasets
Mainstream multivariate anomaly detection literature evaluates models across several standard public benchmarks:
- **SWaT (Secure Water Treatment) [38], [45]:** 51 channels across 6 continuous physical stages; 11 days duration; 36 cyber-physical attacks; strict physical conservation laws present. (Selected as Primary Benchmark).
- **WADI (Water Distribution) [39]:** 123 channels across municipal distribution loops; 16 days duration; 15 complex attacks. (Selected as Secondary External Benchmark).
- **SMD, PSM, SMAP/MSL [10]:** Internet server metrics and spacecraft telemetry lacking cyber-physical conservation laws. (Documented in `docs/DATA_SETUP.md`; deferred).

---

### 2.14 Evaluation Challenges in Time-Series Anomaly Detection
Recent landmark papers in premier machine learning venues have uncovered severe methodological and evaluation flaws in published MTSAD literature:

#### 1. The Point-Adjustment (PA) Distortion
In the Point-Adjustment (PA) evaluation heuristic popularized by OmniAnomaly [10] and USAD [9], if an anomaly detection model flags even a single time step within a contiguous ground-truth anomaly segment, the entire segment is retroactively marked as **True Positives (TP)**.  
Kim, Choi, Choi, Lee, and Yoon (AAAI 2022) [40] (*"Towards a Rigorous Evaluation of Time-Series Anomaly Detection"*) and Wu & Keogh (IEEE TKDE 2021) [41] (*"Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress"*) mathematically proved that **PA dramatically inflates F1-scores**. Kim et al. [40] showed that a completely random noise generator or an inverted detector can achieve near-perfect F1-scores ($\ge 0.90$) under the PA protocol!  
**Our Policy:** R-STNet enforces **raw point-wise Precision, Recall, F1, and PR-AUC** as primary ground-truth metrics. Point-adjusted metrics are computed strictly as a labeled secondary reference (`PA-F1`).

#### 2. Threshold Snooping / Test Set Tuning
Wu & Keogh [41] and Garg et al. (NeurIPS 2021) [42] highlighted that many published papers select decision thresholds $\tau$ by exhaustively searching the test split to maximize test F1-score—a direct violation of scientific machine learning.  
**Our Policy:** Decision thresholds ($\tau_{\text{sys}}, \theta_{\text{rel}}$) are calibrated strictly on the clean normal validation split and **frozen** prior to blind test evaluation.

---

### 2.15 Comparative Analysis
As synthesized in the Literature Matrix (`report/LITERATURE_MATRIX.md`), existing paradigms each solve isolated aspects of the problem:
- GDN [19] and MTAD-GAT [20] learn graph topologies, but sum residuals uniformly without reliability weighting.
- CAROTS [21] achieves causal contrastive invariance, but does not output channel-level reliability scores or separate sensor faults from attacks.
- RANSynCoders [27] provides noise robustness through random subsets, but incurs heavy computational costs and ignores physical graph topology.
- Classical FDI [33], [34] isolates sensor faults, but requires unattainable manual mathematical differential equations.

---

### 2.16 Research Gap
Synthesizing all five literature themes reveals the precise research gap:

> **The Formal Research Gap:**  
> Existing literature provides robust foundations for multivariate anomaly detection, graph neural networks, and sensor fault analysis; however, there is an absence of an **integrated, lightweight framework that explicitly estimates continuous, channel-level sensor reliability online, dynamically down-weights untrusted sensor observations during system anomaly scoring, and operationalizes a multi-state decision engine that decouples isolated sensor corruptions from plant-wide process anomalies under a controlled, reproducible evaluation protocol.**

---

### 2.17 Summary
This literature review confirms that:
1. Standard multivariate anomaly detectors conflate observation corruption with process anomalies due to uniform unweighted error aggregation.
2. An isolated degraded sensor triggers costly, unwarranted plant emergency shutdowns.
3. Estimating an explicit, continuous trustworthiness score $r_i(t) \in [0, 1]$ via physical graph consensus, temporal continuity, and residual statistics provides a mathematically sound, physically grounded solution.
4. The proposed **R-STNet** architecture directly operationalizes these insights into an integrated, defensible research framework.

---

# CHAPTER 3: PROPOSED METHODOLOGY & PLANNED EVALUATION FRAMEWORK

### Section Purpose & Structure
This chapter presents the complete proposed research methodology and planned evaluation framework for Review 1 (Phase 1). It details the mathematical formulation of R-STNet, the Sensor Reliability Module, the reliability-weighted anomaly aggregation mechanism, the 4-way decision engine, benchmark dataset preparation, baseline model architectures, the 10-experiment controlled corruption suite, and the Phase 2 evaluation protocol.

---

### 3.1 Methodology Overview
The proposed research methodology establishes an end-to-end, leak-free framework designed to ingest high-dimensional multivariate sensor time series, learn spatio-temporal representations and physical topological relationships, estimate continuous channel-level sensor reliability online, and decouple isolated instrument degradations from genuine plant-wide cyber-physical attacks.

The operational workflow spans six integrated stages, as depicted in Figure 3 (`report/FIGURES/03_overall_methodology.mmd`):
1. **Raw Telemetry Ingestion & Chronological Splitting:** Reconstructs the official SWaT benchmark into a 7-day continuous normal sequence (496,800 samples) and a 4-day continuous attack sequence (449,919 samples). Implements a project-specific chronological partition (80% train / 20% validation on normal data).
2. **Leak-Free Preprocessing & Windowing:** Scales features strictly using training distribution parameters with zero-variance protection, constructing sliding sequence tensors $\mathbf{W}_t \in \mathbb{R}^{L \times M}$.
3. **Spatio-Temporal Graph Representation:** 1D dilated temporal convolutions capture local autoregressive dynamics per sensor, while learnable graph attention captures directed physical interdependencies across channels.
4. **Online Sensor Reliability Estimation (SRM):** Evaluates temporal continuity, graph neighborhood consistency, and standardized residual tail confidence, fusing them via calibrated evidential logit fusion into continuous reliability scores $r_i(t) \in [0, 1]$.
5. **Reliability-Weighted Anomaly Scoring:** Aggregates residuals weighted by reliability, actively discounting degraded sensors to prevent false alarms:
   $$S_{\text{sys}}(t) = \frac{\sum_{i=1}^M r_i(t) \cdot z_i(t)}{\sum_{i=1}^M r_i(t) + \epsilon}$$
6. **Multi-State Operational Diagnosis:** Compares $S_{\text{sys}}(t)$ and $\mathbf{r}(t)$ against validation-calibrated thresholds to classify plant status into four mutually exclusive operational regimes (Normal, Sensor Fault, System Anomaly, and System Anomaly with Sensor Degradation).

---

### 3.2 System Architecture: R-STNet
The primary proposed architecture is **R-STNet (Dual-Perspective Reliability-Aware Spatio-Temporal Network)**. R-STNet unifies deep spatio-temporal representation learning with a dedicated online Sensor Reliability Module (SRM). The system architecture integrates:
- A Temporal Representation Backbone based on dilated 1D convolutions with residual skip connections.
- An Inter-Sensor Graph Attention Network utilizing learnable node embeddings and Top-$K$ directed message passing.
- A Joint Forecasting Head projecting combined temporal and spatial latent vectors.
- An Online Sensor Reliability Module (SRM) outputting continuous trust scores $\mathbf{r}(t) \in [0, 1]^M$.
- A Reliability-Weighted System Anomaly Scorer $S_{\text{sys}}(t)$ and Sensor Fault Scorer $S_{\text{fault}}(t)$.
- A Four-Way Operational State Discrimination Engine.

---

### 3.3 Dataset: Secure Water Treatment (SWaT)
The primary benchmark utilized in this study is the **Secure Water Treatment (SWaT)** dataset collected by the iTrust Centre for Research in Cyber Security at the Singapore University of Technology and Design (SUTD) [38], [45]. SWaT represents an operational physical pilot plant that purifies non-potable water through six sequential stages:
- **P1:** Raw water intake, storage, and chemical dosing.
- **P2:** Coagulation, flocculation, and chemical pre-treatment (pH/ORP monitoring).
- **P3:** Ultrafiltration (UF) membrane separation.
- **P4:** Dechlorination via chemical dosing and ultraviolet (UV) irradiation.
- **P5:** Reverse Osmosis (RO) desalinization under high hydraulic pressure.
- **P6:** Clean water permeate storage and automated backwash cycling.

The testbed records $M = 51$ continuous and discrete channels at 1 Hz (1 sample per second):
- **25 Continuous Physical Sensors:** Float and differential level transmitters (`LIT101`, `LIT301`, `LIT401`), electromagnetic flow meters (`FIT101`–`FIT601`), pressure transmitters (`PIT501`–`PIT503`, `DPIT301`), and electrochemical analyzers (`AIT201`–`AIT504`).
- **26 Discrete Actuators & Valves:** Motorized modulating valves (`MV101`–`MV504`), centrifugal transfer pumps (`P101`–`P603`), and UV disinfection lamps (`UV401`).

**Operational Sequences and Attack Labels:**
- **Official Normal Run (`normal.csv`):** Spans 7 continuous days (496,800 seconds, rows 395,298 to 892,097 of the raw archive) under 100% healthy nominal operation ($y = 0$).
- **Official Attack Test Run (`attack.csv`):** Spans 4 continuous days (449,919 seconds) containing 36 distinct physical and cyber-attack scenarios designed by ICS security engineers (12.14% attack density, 54,621 attack seconds).
- **Reconstructed Test Run:** Recombines normal intervals (395,298 rows) and attack intervals (54,621 rows) chronologically by timestamp (`src/data/swat_loader.py`).

**Critical Ground-Truth Distinction:**  
Official SWaT attack labels ($y_{\text{sys}} \in \{0, 1\}$) represent **cyber-physical attack events affecting the process**. They do **not** provide ground truth that an individual sensor instrument is physically damaged, uncalibrated, or degraded. Sensor-fault ground truth is provided strictly via our controlled corruption framework masks ($M_{\text{fault}}(t, i)$).

---

### 3.4 Dataset Preparation
Dataset ingestion and preparation are implemented in `src/data/swat_loader.py`:
1. **Timestamp Parsing & Column Normalization:** Strip leading/trailing whitespaces from column headers (e.g., `' FIT101'` $\to$ `'FIT101'`).
2. **Chronological Splitting (Project-Specific Partition):**
   - **Training Set (80% Normal):** First 397,440 raw seconds (39,744 samples at 10x downsample). Contains pure normal operation ($y = 0$). All neural parameters and scalers are trained exclusively on this split.
   - **Validation Set (20% Normal):** Next 99,360 raw seconds (9,936 samples at 10x downsample). Contains pure normal operation ($y = 0$). Reserved strictly for threshold and calibration parameter tuning.
   - **Test Set (Attack Sequence):** Continuous 449,919 raw seconds (44,992 samples at 10x downsample) containing 12.14% attack density. Evaluated blindly.
3. **Downsampling:** Downsampling by factor **10x** yields clean 10-second intervals, preserving hydraulic time constants while accelerating training throughput by an order of magnitude.
4. **Data Integrity Checks:** Enforce assertions verifying zero NaNs, zero Infs, correct dimensionality $(N, L, M)$, and binary label validity.

---

### 3.5 Preprocessing
#### 3.5.1 Leak-Free Scaling
To prevent information leakage from the evaluation split into model training:
1. Feature scaling parameters ($\mu_{\text{train}, i}$ and $\sigma_{\text{train}, i}$) are computed **strictly on the 80% normal training split**.
2. The validation and test sets are transformed using the fitted training parameters:
   $$\tilde{x}_i(t) = \frac{x_i(t) - \mu_{\text{train}, i}}{\sigma_{\text{train}, i} + \epsilon}$$
3. **Zero-Variance Protection:** Certain discrete backup actuators (e.g., `P102`, `P202`) remain constantly inactive throughout normal training ($\sigma_i = 0$). To prevent division-by-zero, our scaler sets $\sigma_i = 1.0$ for any channel where $\sigma_i < 10^{-6}$.

#### 3.5.2 Importance of Chronological Processing
Unlike cross-sectional tabular data, time series possess temporal autocorrelation. Shuffling or random K-fold cross-validation allows the model to access future information when predicting past states (lookahead leakage). Enforcing strict chronological splitting guarantees authentic out-of-sample evaluation [40], [41].

---

### 3.6 Temporal Windowing
Static point-wise observations do not provide temporal context. Given the standardized multivariate sequence, we construct chronological sliding windows:

$$\mathbf{W}_t = \left[ \mathbf{X}(t - L + 1), \mathbf{X}(t - L + 2), \dots, \mathbf{X}(t) \right] \in \mathbb{R}^{L \times M}$$

- **Lookback Length ($L$):** Set to $L = 60$ steps ($\approx 10$ minutes at 10s downsampling, or 5 minutes at 5s downsampling), matching physical settling times.
- **Stride ($S$):** $S_{\text{test}} = 1$ during evaluation to maintain continuous monitoring; $S_{\text{train}} = 1$ (or $S=5$) during training.

---

### 3.7 Baseline Models
To benchmark R-STNet rigorously, four baseline architectures are implemented in `src/models/baselines/`:
1. **Baseline A: Classical Isolation Forest (`isolation_forest.py`):** 100 randomized decision isolation trees [1] operating on flattened sliding window features. Establishes standard off-the-shelf industrial baseline performance.
2. **Baseline B: SpatioTemporal Autoencoder (`reconstruction_ae.py`):** Dilated 1D convolutional temporal encoder coupled to a symmetric convolutional decoder [14]. Unweighted Mean Squared Reconstruction Error: $S_{\text{ST-AE}}(t) = \frac{1}{M} \sum_{i=1}^M (x_i(t) - \tilde{x}_i(t))^2$.
3. **Baseline C: RSTNet WITHOUT SRM (`rstnet_no_srm.py`):** Identical temporal convolution and graph attention backbone as R-STNet, but **without the Sensor Reliability Module**. Unweighted Mean Standardized Residual: $S_{\text{No-SRM}}(t) = \frac{1}{M} \sum_{i=1}^M z_i(t)$. **Primary scientific ablation** isolating the exact empirical gain provided by the reliability mechanism.
4. **Baseline 3: CAROTS (`reference/CAROTS-master/`):** Causal structural discovery with contrastive representation learning [21]. Establishes external state-of-the-art reference performance under published ICML 2025 conventions.

---

### 3.8 Proposed R-STNet Architecture Formulation
The neural backbone of R-STNet is implemented in `src/models/rstnet.py`:

#### 3.8.1 Temporal Representation Block
For an input window $\mathbf{W}_t \in \mathbb{R}^{B \times L \times M}$, we permute and reshape to $(B \cdot M, 1, L)$. Each sensor channel passes through two 1D dilated convolutional layers with kernel size $K=3$ and dilation factor $d=2$:

$$\mathbf{H}_{\text{conv1}} = \text{ReLU}\left( \text{Conv1D}_1(\mathbf{W}_t) \right)$$
$$\mathbf{H}_{\text{conv2}} = \text{ReLU}\left( \text{Conv1D}_2(\mathbf{H}_{\text{conv1}}) \right) + \mathbf{H}_{\text{conv1}}$$

A linear projection maps the final time step into node feature dimension $d_{\text{node}} = 32$:

$$\mathbf{H}_{\text{temp}} = \text{Linear}(\mathbf{H}_{\text{conv2}}[:, :, -1]) \in \mathbb{R}^{B \times M \times d_{\text{node}}}$$

#### 3.8.2 Graph Attention Dependency Module
To model physical correlations (conservation of mass and energy) without manual plant modeling, R-STNet maintains two learnable node embedding dictionaries $\mathbf{E}_1, \mathbf{E}_2 \in \mathbb{R}^{M \times d_{\text{node}}}$. The directed similarity matrix is:

$$\mathbf{S} = \text{ReLU}\left( \mathbf{E}_1 \mathbf{E}_2^T \right), \quad S_{ii} = 0 \quad (\text{Self-loops masked})$$

To retain the strongest physical couplings, we apply Top-$K$ sparsification ($K=15$):

$$\tilde{S}_{ij} = \begin{cases} S_{ij}, & \text{if } S_{ij} \in \text{TopK}(S_{i, :}) \\ 0, & \text{otherwise} \end{cases}$$

Edge weights are normalized via Softmax over incoming edges:

$$A_{ij} = \frac{\exp(\tilde{S}_{ij})}{\sum_{k=1}^M \exp(\tilde{S}_{ik}) + \epsilon}$$

Spatial message passing aggregates information from topological neighbors:

$$\mathbf{H}_{\text{spat}} = \mathbf{A} \mathbf{H}_{\text{temp}} \in \mathbb{R}^{B \times M \times d_{\text{node}}}$$

#### 3.8.3 Joint Forecasting Head
The temporal embedding $\mathbf{H}_{\text{temp}}$ and spatial embedding $\mathbf{H}_{\text{spat}}$ are concatenated:

$$\mathbf{H}_{\text{joint}} = \left[ \mathbf{H}_{\text{temp}} \,\|\, \mathbf{H}_{\text{spat}} \right] \in \mathbb{R}^{B \times M \times 2 d_{\text{node}}}$$

A multi-layer perceptron forecasts one-step-ahead expected values:

$$\hat{\mathbf{X}}(t) = \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \mathbf{H}_{\text{joint}} + \mathbf{b}_1) + \mathbf{b}_2 \in \mathbb{R}^{B \times M}$$

The model is trained end-to-end on normal data using Mean Squared Error loss:

$$\mathcal{L}_{\text{MSE}} = \frac{1}{B \cdot M} \sum_{b=1}^B \sum_{i=1}^M \left( x_{b, i}(t) - \hat{x}_{b, i}(t) \right)^2$$

---

### 3.9 Sensor Reliability Module (SRM)
The Sensor Reliability Module (implemented in `src/reliability/sensor_reliability.py`) evaluates three complementary, physically motivated signals for each channel $i$:

#### 1. Temporal Continuity ($v_i(t) \in [0, 1]$)
Evaluates rate-of-change violations and frozen/stuck sensor conditions:
- **Rate-of-Change Limit:** $\Delta x_i(t) = |x_i(t) - x_i(t-1)|$. An exponential penalty dampens abrupt unphysical step violations against validation quantile bound $\delta_i^{\max}$:
  $$\text{rate\_score}_i(t) = \exp\left( -0.5 \cdot \max\left(0, \frac{\Delta x_i(t)}{\delta_i^{\max}} - 1.0\right)^2 \right)$$
- **Stuck-Sensor Variance Check:** For dynamic continuous channels, a moving variance over lookback window $K=15$ detects frozen telemetry:
  $$\text{stuck\_penalty}_i(t) = \begin{cases} 0.1, & \text{if } \text{Var}(x_{i, t-K:t}) < \epsilon_{\text{stuck}} \text{ and dynamic}_i \\ 1.0, & \text{otherwise} \end{cases}$$
- **Combined Temporal Continuity:**
  $$v_i(t) = \text{clip}\left( \text{rate\_score}_i(t) \cdot \text{stuck\_penalty}_i(t), \, 0.01, \, 0.99 \right)$$

#### 2. Graph Neighborhood Consistency ($c_i(t) \in [0, 1]$)
Evaluates agreement between sensor $i$ and its physical topological neighbors:
$$\hat{x}_i^{\mathcal{N}}(t) = \sum_{j \in \mathcal{N}(i)} W_{ji} \cdot x_j(t)$$
where $W_{ji}$ is the column-normalized graph adjacency. The physical discrepancy is scaled by the validation Median Absolute Deviation ($\text{MAD}_i$):
$$c_i(t) = \text{clip}\left( \exp\left( -\frac{|x_i(t) - \hat{x}_i^{\mathcal{N}}(t)|}{2.5 \cdot \text{MAD}_i} \right), \, 0.01, \, 0.99 \right)$$
If a sensor fails or drifts, its reading deviates from its correlated neighbors, pulling $c_i(t)$ down.

#### 3. Standardized Residual Tail Confidence ($p_i(t) \in [0, 1]$)
Computes the Gaussian tail probability of the MAD-standardized prediction residual $z_i(t)$:
$$z_i(t) = \frac{|x_i(t) - \hat{x}_i(t)| - \text{Median}_i(e)}{\text{MAD}_i(e) + \epsilon}$$
$$p_i(t) = \text{clip}\left( \exp\left( -0.5 \cdot \left(\frac{\max(0, z_i(t))}{2.0}\right)^2 \right), \, 0.01, \, 0.99 \right)$$

#### 4. Calibrated Evidential Logit Fusion
To avoid arbitrary threshold cuts, component scores are fused via a smooth evidential formulation in logit space:

$$\text{logit}(s) = \ln\left( \frac{s}{1 - s} \right)$$
$$\text{logit\_fusion}_i(t) = w_v \cdot \text{logit}(v_i(t)) + w_c \cdot \text{logit}(c_i(t)) + w_p \cdot \text{logit}(p_i(t)) + b_r$$
$$r_i(t) = \frac{1}{1 + \exp\left( -\text{logit\_fusion}_i(t) \right)} \in [0, 1]$$

Weights $w_v = 1.0$, $w_c = 1.2$, $w_p = 1.0$, and $b_r = 0.0$ are calibrated systematically on the validation normal split (`fit_calibration`).

---

### 3.10 Reliability-Weighted System Anomaly Score
Standard detectors evaluate the unweighted sum:
$$S_{\text{unweighted}}(t) = \frac{1}{M} \sum_{i=1}^M z_i(t)$$
When sensor $k$ fails ($z_k \to \infty$), $S_{\text{unweighted}}(t)$ spikes, causing a false plant alarm.

In **R-STNet**, we compute the reliability-weighted aggregation:

$$S_{\text{sys}}(t) = \frac{\sum_{i=1}^M r_i(t) \cdot z_i(t)}{\sum_{i=1}^M r_i(t) + \epsilon}$$

**Mathematical Behavior & Physical Intuition:**
1. **Under an Isolated Sensor Fault:** Sensor $k$ degrades ($r_k \to 0$, $z_k \gg 0$). Its contribution $r_k(t) z_k(t) \to 0$. The remaining $M-1$ reliable sensors ($r_j \approx 1.0, z_j \approx 0$) dominate the numerator and denominator, keeping $S_{\text{sys}}(t) < \tau_{\text{sys}}$ and **suppressing the false system alarm**.
2. **Under a Genuine Cyber-Physical Attack:** Attackers manipulate the physical process (e.g., closing valves). The physical state changes. Multiple correlated sensors show elevated residuals $z$, but because their readings match physical conservation laws, their reliability remains high ($r \ge \theta_{\text{rel}}$). Consequently, $S_{\text{sys}}(t)$ spikes, **triggering a legitimate plant alarm**.

---

### 3.11 Sensor Corruption Model
The controlled corruption framework (`src/corruption/corruptor.py`) generates deterministic, reproducible synthetic corruptions across 9 operational failure modes:
1. **Gaussian Noise:** Additive white noise $\mathcal{N}(0, (\gamma \sigma_i)^2)$ with $\gamma \in \{0.1, 0.3, 0.5\}$.
2. **Stuck / Frozen Sensor:** Observation frozen at constant value $x_i(t_{\text{start}})$.
3. **Linear Calibration Drift:** Ramp offset $\Delta x(t) = \alpha \cdot \frac{t - t_{\text{start}}}{\Delta t} \sigma_i$ with $\alpha \in \{0.1, 0.5\}$.
4. **Impulse Spikes:** Sharp deviations $\pm 3.0 \sigma_i$ at 2% probability.
5. **Missing Telemetry:** Forward-filled or zero-imputed blocks.
6. **Multi-Sensor Failure:** Simultaneous stuck and drift across adjacent stages.
7. **Mixed Harsh Environment:** Combined noise, drift, spikes, and stuck readings on 5 channels.

Generates exact binary ground-truth masks:
$$M_{\text{fault}}(t, i) = \begin{cases} 1, & \text{if sensor } i \text{ is synthetically corrupted at time } t \\ 0, & \text{otherwise} \end{cases}$$

---

### 3.12 Four-Way Operational State Interpretation
Implemented in `src/anomaly_detection/discriminator.py`:
- **Sensor Fault Localization Score:**
  $$S_{\text{fault}}(t) = \max_{i \in \{1 \dots M\}} \left[ (1 - r_i(t)) \cdot z_i(t) \right]$$
  $$\text{FaultySensor}(t) = \arg\max_{i \in \{1 \dots M\}} \left[ (1 - r_i(t)) \cdot z_i(t) \right]$$

#### Four Mutually Exclusive Operational Regimes:
- **Case A (NORMAL):**  
  $S_{\text{sys}}(t) < \tau_{\text{sys}}$ **AND** $\min_i r_i(t) \ge \theta_{\text{rel}}$.  
  *Plant in control; all sensors verified trustworthy.*
- **Case B (ISOLATED SENSOR FAULT):**  
  $S_{\text{sys}}(t) < \tau_{\text{sys}}$ **AND** $\min_i r_i(t) < \theta_{\text{rel}}$.  
  *Plant safe; sensor $\text{FaultySensor}(t)$ degraded. False emergency shutdown suppressed; maintenance alerted.*
- **Case C (CRITICAL SYSTEM ANOMALY):**  
  $S_{\text{sys}}(t) \ge \tau_{\text{sys}}$ **AND** $\frac{1}{M} \sum_i r_i(t) \ge \theta_{\text{rel}}$.  
  *Genuine cyber-physical attack or process rupture confirmed by trustworthy sensors. Emergency intervention triggered.*
- **Case D (SYSTEM ANOMALY WITH SENSOR DEGRADATION):**  
  $S_{\text{sys}}(t) \ge \tau_{\text{sys}}$ **AND** $\frac{1}{M} \sum_i r_i(t) < \theta_{\text{rel}}$.  
  *Process failure accompanied by concurrent instrument impairment. Highest operational risk level.*

---

### 3.13 Experimental Design
The planned experimental campaign spans 10 controlled scenarios across 3 random seeds (`seed = 42, 101, 2024`):
- **E1:** Clean SWaT Benchmark (Pristine baseline).
- **E2–E4:** Low (0.1), Medium (0.3), High (0.5) Gaussian Noise.
- **E5–E6:** Single Stuck Sensor (`LIT101`) and Multi-Stuck Sensors (Stages P1, P2, P3).
- **E7–E8:** Incipient (0.1) and Severe (0.5) Calibration Drift (`FIT101`).
- **E9:** Impulse Spikes (2% frequency, 3x amplitude).
- **E10:** Mixed Harsh Operational Environment.

---

### 3.14 Evaluation Metrics
In accordance with time-series evaluation integrity guidelines [40], [41]:
- **System Anomaly Detection Metrics:** Raw Point-wise Precision ($P$), Recall ($R$), F1-Score ($F_1$), Precision-Recall AUC (PR-AUC), and False Positive Rate on clean periods ($\text{FPR}$).
- **Robustness & Reliability Metrics:**
  - False Alarm Suppression Rate ($\text{FASR}$):
    $$\text{FASR} = 1 - \frac{\text{FPR}_{\text{model}}^{\text{corrupt}} - \text{FPR}_{\text{model}}^{\text{clean}}}{\text{FPR}_{\text{baseline}}^{\text{corrupt}} - \text{FPR}_{\text{baseline}}^{\text{clean}}}$$
  - Reliability Separation Margin ($\Delta r = \bar{r}_{\text{clean}} - \bar{r}_{\text{corrupted}}$).
  - Sensor Fault Localization Precision, Recall, and F1 comparing $\arg\max_i [(1 - r_i) z_i]$ against $M_{\text{fault}}(t, i)$.
- **Point-Adjusted F1 (PA):** Reported strictly as a labeled secondary metric (`PA-F1`) with explicit documentation of its optimistic bias [40].

---

### 3.15 Threshold Selection Protocol
To guarantee zero test-label leakage [41]:
1. Model weights are trained strictly on the 80% normal training split.
2. System scores $S_{\text{sys}}(t)$ are evaluated on the 20% normal validation split.
3. The system anomaly threshold $\tau_{\text{sys}}$ is calibrated at the 95th percentile of normal validation scores:
   $$\tau_{\text{sys}} = \text{Quantile}(S_{\text{val}}, 0.95)$$
4. The reliability threshold is calibrated on validation reliability:
   $$\theta_{\text{rel}} = \max\left(0.40, \text{Quantile}(\min_i r_{i, \text{val}}, 0.01)\right)$$
5. **Thresholds are permanently frozen** before evaluating the test sequence blindly.

---

### 3.16 Experimental Comparison Plan
The core scientific comparison isolates the contribution of the Sensor Reliability Module:
- Comparing **Baseline C (RSTNet without SRM)** vs. **Proposed R-STNet (with SRM)** isolates the exact impact of reliability-weighted scoring under identical neural backbone capacity.
- Comparing against **Baseline A (Isolation Forest)** and **Baseline B (ST-AE)** benchmarks classical and deep reconstruction behavior.
- Comparing against **Baseline 3 (CAROTS)** provides external state-of-the-art reference under ICML 2025 conventions.

---

### 3.17 Ablation Study Plan
Four model variants will be evaluated on SWaT under Clean (E1), High Noise (E4), and Stuck Sensor (E5):
1. **Variant A (No SRM):** Uniform residual mean.
2. **Variant B (Temporal SRM Only):** Evaluates $v_i(t)$ alone.
3. **Variant C (Graph SRM Only):** Evaluates $c_i(t)$ alone.
4. **Variant D (Full Triangulated R-STNet):** Complete $v_i(t) + c_i(t) + p_i(t)$ fusion.

Answers Hypothesis 3 regarding the necessity of multi-signal evidential triangulation.

---

### 3.18 Computational Environment & Reproducibility
- **Operating System:** Windows 11 (64-bit).
- **Compute Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4096 MiB VRAM), Driver 592.82, CUDA 12.1 runtime.
- **Python Runtime:** Python 3.11.0 strictly isolated in virtual environment `.venv/`.
- **Frameworks:** PyTorch 2.1.0+cu121, NumPy 1.24.3, Pandas 2.0.3, Scikit-learn 1.3.0.
- **Memory Footprint:** Sequence window $L=60$, batch size $B=64$. Peak allocated VRAM during forward/backward passes is measured at $< 1.2\text{ GB}$, fitting well within the 4.0 GB physical hardware limit.
- **Reproducibility:** Centrally seeded RNG (`seed = 42, 101, 2024`), deterministic cuDNN backends, and configuration-driven execution via YAML files (`configs/swat_rstnet.yaml`).

---

# REFERENCES (IEEE STYLE)

[1] F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation forest," in *Proc. 2008 Eighth IEEE International Conference on Data Mining (ICDM)*, Pisa, Italy, 2008, pp. 413–422. DOI: 10.1109/ICDM.2008.17.

[2] B. Schölkopf, R. C. Williamson, A. J. Smola, J. Shawe-Taylor, and J. C. Platt, "Support vector method for novelty detection," in *Advances in Neural Information Processing Systems (NeurIPS 12)*, Denver, CO, USA, 1999, pp. 582–588.

[3] V. Chandola, A. Banerjee, and V. Kumar, "Anomaly detection: A survey," *ACM Computing Surveys*, vol. 41, no. 3, Article 15, pp. 1–58, Jul. 2009. DOI: 10.1145/1541880.1541882.

[4] M. M. Breunig, H.-P. Kriegel, R. T. Ng, and J. Sander, "LOF: Identifying density-based local outliers," *ACM SIGMOD Record*, vol. 29, no. 2, pp. 93–104, May 2000. DOI: 10.1145/335191.335388.

[5] X. Wang and Q. He, "Dynamic principal component analysis based fault detection for multivariate processes with industrial applications," *Computers & Chemical Engineering*, vol. 136, p. 106787, May 2020. DOI: 10.1016/j.compchemeng.2020.106787. (Elsevier)

[6] S. J. Qin, "Survey on data-driven industrial process monitoring and diagnosis," *Annual Reviews in Control*, vol. 36, no. 2, pp. 220–234, Dec. 2012. DOI: 10.1016/j.arcontrol.2012.09.004. (Elsevier)

[7] L. H. Chiang, E. L. Russell, and R. D. Braatz, *Fault Detection and Diagnosis in Industrial Systems*. London, U.K.: Springer Science & Business Media, 2000. DOI: 10.1007/978-1-4471-0347-9.

[8] V. Hodge and J. Austin, "A survey of outlier detection methodologies," *Artificial Intelligence Review*, vol. 22, no. 2, pp. 85–126, Oct. 2004. DOI: 10.1023/B:AIRE.0000045502.10941.a9.

[9] J. Audibert, P. Michiardi, F. Guyard, S. Marti, and M. A. Zuluaga, "USAD: UnSupervised Anomaly Detection on multivariate time series," in *Proc. 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '20)*, Virtual Event, CA, USA, 2020, pp. 3395–3404. DOI: 10.1145/3394486.3403392.

[10] Y. Su, Y. Zhao, C. Niu, R. Liu, W. Sun, and D. Pei, "Robust anomaly detection for multivariate time series through stochastic recurrent neural networks," in *Proc. 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '19)*, Anchorage, AK, USA, 2019, pp. 2828–2837. DOI: 10.1145/3292500.3330672.

[11] S. Tuli, G. Casale, and N. R. Jennings, "TranAD: Deep transformer networks for anomaly detection in multivariate time series," *Proc. VLDB Endowment*, vol. 15, no. 6, pp. 1202–1214, Feb. 2022. DOI: 10.14778/3514061.3514067.

[12] J. Xu, H. Wu, J. Wang, and M. Long, "Anomaly Transformer: Time series anomaly detection with association discrepancy," in *Proc. International Conference on Learning Representations (ICLR 2022)*, Virtual, 2022.

[13] C. Zhang, D. Song, Y. Chen, X. Shen, and C. Mount, "A deep neural network for unsupervised anomaly detection and diagnosis in multivariate time series data," in *Proc. AAAI Conference on Artificial Intelligence*, vol. 33, no. 01, Jul. 2019, pp. 1409–1416. DOI: 10.1609/aaai.v33i01.33011409.

[14] P. Malhotra, A. Ramakrishnan, G. Anand, L. Vig, P. Agarwal, and G. Shroff, "LSTM-based encoder-decoder for multi-sensor anomaly detection," in *Proc. ICML 2016 Anomaly Detection Workshop*, New York, NY, USA, 2016. arXiv:1607.00148.

[15] M. Munir, S. A. Siddiqui, A. Dengel, and S. Ahmed, "DeepAnT: A deep learning approach for unsupervised anomaly detection in time series," *IEEE Access*, vol. 7, pp. 1991–2005, Dec. 2018. DOI: 10.1109/ACCESS.2018.2886457.

[16] A. Blázquez-García, A. Conde, U. Mori, and J. A. Lozano, "A review on outlier/anomaly detection in time series data," *ACM Computing Surveys*, vol. 54, no. 3, Article 56, pp. 1–33, Apr. 2021. DOI: 10.1145/3444690.

[17] K. Choi, J. Yi, C. Park, and S. Yoon, "Deep learning for anomaly detection in time-series data: Review, analysis, and guidelines," *IEEE Access*, vol. 9, pp. 120043–120065, Aug. 2021. DOI: 10.1109/ACCESS.2021.3107975.

[18] B. Zong, Q. Song, M. R. Min, W. Cheng, C. Lumezanu, D. Cho, and H. Chen, "Deep autoencoding Gaussian mixture model for unsupervised anomaly detection," in *Proc. International Conference on Learning Representations (ICLR 2018)*, Vancouver, BC, Canada, 2018.

[19] A. Deng and B. Hooi, "Graph neural network-based anomaly detection in multivariate time series," in *Proc. 35th AAAI Conference on Artificial Intelligence / IJCAI 2021*, Virtual, 2021, pp. 2213–2221. DOI: 10.24963/ijcai.2021/305.

[20] H. Zhao, Y. Wang, J. Duan, C. Huang, D. Cao, Y. Tong, and B. Xu, "Multivariate time-series anomaly detection via graph attention network," in *Proc. 2020 IEEE International Conference on Data Mining (ICDM)*, Sorrento, Italy, 2020, pp. 841–850. DOI: 10.1109/ICDM50108.2020.00093.

[21] H. Kim, J. Mok, D. Lee, J. Lew, S. Kim, and S. Yoon, "Causality-aware contrastive learning for robust multivariate time-series anomaly detection," in *Proc. 42nd International Conference on Machine Learning (ICML 2025)*, 2025. PMLR. arXiv:2506.03964.

[22] W. Chen, L. Tian, B. Chen, L. Dai, Z. Duan, and M. Zhou, "Deep variational graph convolutional recurrent network for multivariate time series anomaly detection," *Neurocomputing*, vol. 491, pp. 134–145, Jun. 2022. DOI: 10.1016/j.neucom.2022.03.056. (Elsevier)

[23] S. Han and S. S. Lee, "Multi-scale spatio-temporal graph convolutional networks for multivariate time series anomaly detection," *Expert Systems with Applications*, vol. 224, p. 119933, Aug. 2023. DOI: 10.1016/j.eswa.2023.119933. (Elsevier)

[24] S. Li, Y. Zhang, and C. Zhao, "Spatio-temporal graph neural networks for sensor fault diagnosis in industrial processes," *Computers in Industry*, vol. 148, p. 103896, Jun. 2023. DOI: 10.1016/j.compind.2023.103896. (Elsevier)

[25] Z. Wu, S. Pan, G. Long, J. Jiang, and C. Zhang, "Connecting the dots: Multivariate time series forecasting with graph neural networks," in *Proc. 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '20)*, 2020, pp. 753–763. DOI: 10.1145/3394486.3403118.

[26] E. Dai and S. Chen, "Graph-augmented normalizing flows for anomaly detection of multiple time series," in *Proc. International Conference on Learning Representations (ICLR 2022)*, Virtual, 2022.

[27] J. Audibert, P. Michiardi, F. Guyard, and M. A. Zuluaga, "RANSynCoders: Random sampling autoencoder ensembles for robust multivariate time series anomaly detection," *IEEE Transactions on Knowledge and Data Engineering*, vol. 35, no. 6, pp. 6128–6140, Jun. 2023. DOI: 10.1109/TKDE.2022.3168128.

[28] U. Saeed, S. U. Jan, Y. D. Lee, and I. Koo, "Fault diagnosis based on extremely randomized trees in wireless sensor networks," *Reliability Engineering & System Safety*, vol. 205, p. 107284, Jan. 2021. DOI: 10.1016/j.ress.2020.107284. (Elsevier)

[29] R. Sharifi and R. Langari, "A sensor fault diagnosis framework for nonlinear systems: Application to a water treatment testbed," *Control Engineering Practice*, vol. 30, pp. 93–104, Sep. 2014. DOI: 10.1016/j.conengprac.2014.05.002. (Elsevier)

[30] J. Huang, X. Yan, and C. Liu, "Sensor fault diagnosis and data reconstruction based on spatial-temporal correlation and deep learning," *Measurement*, vol. 184, p. 109923, Nov. 2021. DOI: 10.1016/j.measurement.2021.109923. (Elsevier)

[31] J. Liu and J. Chen, "Robust multivariate process monitoring and sensor fault isolation using dynamic graph attention networks," *Journal of Process Control*, vol. 114, pp. 1–12, Jun. 2022. DOI: 10.1016/j.jprocont.2022.03.008. (Elsevier)

[32] Q. Zhu and Z. Fei, "Multi-sensor data validation and reconstruction in industrial cyber-physical systems using deep autoencoders with confidence estimation," *ISA Transactions*, vol. 128, pp. 312–323, Sep. 2022. DOI: 10.1016/j.isatra.2021.11.018. (Elsevier)

[33] J. Gertler, *Fault Detection and Diagnosis in Engineering Systems*. New York, NY, USA: Marcel Dekker, 1998.

[34] R. Isermann, *Fault-Diagnosis Systems: An Introduction from Fault Detection to Fault Tolerance*. Berlin, Germany: Springer-Verlag, 2006. DOI: 10.1007/3-540-30368-5.

[35] S. X. Ding, *Data-Driven Design of Fault Diagnosis and Fault-Tolerant Control Systems*. London, U.K.: Springer-Verlag, 2014. DOI: 10.1007/978-1-4471-6410-4.

[36] V. Venkatasubramanian, R. Rengaswamy, K. Yin, and S. N. Kavuri, "A review of process fault detection and diagnosis: Part I: Quantitative model-based methods," *Computers & Chemical Engineering*, vol. 27, no. 3, pp. 293–311, Mar. 2003. DOI: 10.1016/S0098-1354(02)00160-6. (Elsevier)

[37] W. Yan, "Toward automatic detection and isolation of sensor faults in cyber-physical energy systems," *IEEE Transactions on Industrial Informatics*, vol. 16, no. 11, pp. 7006–7016, Nov. 2020. DOI: 10.1109/TII.2019.2957134.

[38] A. P. Mathur and N. O. Tippenhauer, "SWaT: A water treatment testbed for research and training on cyber physical systems security," in *Proc. 2016 International Workshop on Cyber-physical Systems for Smart Water Networks (CySWater)*, Vienna, Austria, 2016, pp. 31–36. DOI: 10.1109/CySWater.2016.7469060.

[39] C. M. Ahmed, V. R. Melesse, and A. P. Mathur, "WADI: A water distribution testbed for research in the design of secure cyber physical systems," in *Proc. 3rd International Workshop on Cyber-Physical Systems for Smart Water Networks (CySWater '17)*, Pittsburgh, PA, USA, 2017, pp. 25–28. DOI: 10.1145/3055366.3055375.

[40] S. Kim, K. Choi, H.-S. Choi, B. Lee, and S. Yoon, "Towards a rigorous evaluation of time-series anomaly detection," in *Proc. 36th AAAI Conference on Artificial Intelligence*, vol. 36, no. 7, Jun. 2022, pp. 7194–7201. DOI: 10.1609/aaai.v36i7.20680.

[41] R. Wu and E. Keogh, "Current time series anomaly detection benchmarks are flawed and are creating the illusion of progress," *IEEE Transactions on Knowledge and Data Engineering*, vol. 35, no. 3, pp. 2421–2429, Mar. 2023. DOI: 10.1109/TKDE.2021.3112126.

[42] S. Garg, S. Zhang, Y. Zhang, and A. Smola, "Evaluation metrics for time series anomaly detection," in *Proc. NeurIPS 2021 Workshop on Time Series*, Virtual, 2021. arXiv:2109.05257.

[43] M. Kravchik and A. Shabtai, "Efficient cyber attack detection in industrial control systems using lightweight neural networks," *IEEE Transactions on Dependable and Secure Computing*, vol. 19, no. 4, pp. 2179–2192, Jul.–Aug. 2022. DOI: 10.1109/TDSC.2021.3050101.

[44] A. Al-Abassi, H. Karimipour, A. Dehghantanha, and R. M. Parizi, "An ensemble deep learning-based cyber-attack detection in industrial control systems," *IEEE Access*, vol. 8, pp. 83473–83487, May 2020. DOI: 10.1109/ACCESS.2020.2992249.

[45] J. Goh, S. Adepu, K. N. Junejo, and A. Mathur, "A dataset to support research in the design of secure water treatment systems," in *Critical Information Infrastructures Security (CRITIS 2016)*, Lecture Notes in Computer Science, vol. 10242, G. Havarneanu et al., Eds. Cham, Switzerland: Springer, 2017, pp. 88–99. DOI: 10.1007/978-3-319-71368-7_8.
