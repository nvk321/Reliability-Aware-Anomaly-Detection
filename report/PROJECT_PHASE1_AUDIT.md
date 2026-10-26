# Project Phase 1 Audit: Reliability-Aware Anomaly Detection for Multivariate Sensor Data

**Document Type:** Technical & Academic Audit  
**Review Target:** Review 1 (Phase 1 Evaluation)  
**Academic Institution:** School of Computer Science and Engineering (SCOPE), VIT Vellore  
**Course:** Machine Learning Lab (BCSE209P), Slot L47 + L48  
**Faculty Guide:** Dr. DEGA NAGARAJU  
**Student Investigators:** P NAGA VENKATA KUSHAL (23BDS0133), KOTERU TARUN (23BDS0103)  
**Date of Audit:** October 2026  

---

## Executive Summary of Phase 1 Status

This audit establishes the baseline truth of the repository as of Phase 1 (Review 1 milestone). The project has completed architectural design, baseline mathematical modeling, leak-free data preprocessing pipeline implementation, and initial code scaffolding. However, in accordance with the Phase 1 milestone, **all large-scale experimental runs, comparative evaluations, and final ablation studies remain proposed future work**. No empirical performance claims, final F1-scores, or superiority claims are made at this stage.

---

## A. Current Project Objective

The primary objective of this project is to develop, implement, and rigorously evaluate **R-STNet (Dual-Perspective Reliability-Aware Spatio-Temporal Network)**—a deep learning framework specifically designed for multivariate sensor time-series in industrial cyber-physical systems (CPS). 

Traditional multivariate anomaly detectors treat all sensor streams as uniformly trustworthy, aggregating prediction or reconstruction residuals across all channels indiscriminately. Consequently, an isolated hardware imperfection (e.g., severe Gaussian noise, calibration drift, or a frozen/stuck reading) inflates global anomaly scores, triggering false plant-level shutdowns. 

The project aims to:
1. Estimate continuous, channel-specific sensor reliability scores $r_i(t) \in [0, 1]$ online using temporal continuity, cross-sensor graph neighborhood agreement, and residual confidence.
2. Formulate a reliability-weighted anomaly aggregation mechanism that down-weights untrusted channels to prevent false alarms.
3. Establish a 4-way operational state discrimination engine that classifies each operating instant into:
   - **Case A (Normal Operation):** Safe process, reliable sensors.
   - **Case B (Sensor Fault):** Safe process, isolated degraded sensor(s); alerts maintenance without triggering false system shutdown.
   - **Case C (System Anomaly):** Genuine cyber-physical attack or process failure confirmed by trustworthy sensors.
   - **Case D (System Anomaly with Sensor Degradation):** Process anomaly occurring concurrently with instrument impairment.
4. Benchmark the proposed architecture against classical (Isolation Forest), deep reconstruction (SpatioTemporal Autoencoder), and unweighted graph baselines (RSTNet without SRM) under controlled, reproducible synthetic corruptions.

---

## B. Current Research Questions

### Central Research Question
> **RQ1 (Central):** *Can explicit channel-level sensor reliability estimation improve the robustness of multivariate time-series anomaly detection when sensor observations suffer from missingness, high noise, calibration drift, or stuck values?*

### Supporting Research Questions
- **RQ2 (Sensor vs. Process Discrimination):** *Can an explicit dual-perspective decision engine reliably separate isolated instrument-level corruptions from genuine plant-wide cyber-physical attacks without requiring manual physics modeling?*
- **RQ3 (Component Contribution / Ablation):** *To what degree do temporal signal continuity, graph-based topological consensus, and standardized residual tail confidence individually contribute to total sensor reliability estimation?*
- **RQ4 (Evaluation Protocol Integrity):** *Does threshold calibration strictly performed on normal validation data prevent optimistic evaluation biases that commonly occur in published time-series anomaly detection benchmarks?*

---

## C. Current Proposed Architecture (R-STNet)

The proposed architecture is **R-STNet (Dual-Perspective Reliability-Aware Spatio-Temporal Network)**:

1. **Temporal Feature Extractor:**
   - 1D Dilated Convolutional layers per channel with residual skip connections.
   - Extracts localized autoregressive patterns and physical inertia dynamics over sequence window $\mathbf{W}_t \in \mathbb{R}^{L \times M}$.
2. **Inter-Sensor Graph Attention Backbone:**
   - Learnable node embeddings $\mathbf{E}_1, \mathbf{E}_2 \in \mathbb{R}^{M \times d}$.
   - Directed adjacency matrix $\mathbf{A} = \text{Softmax}(\text{TopK}(\text{ReLU}(\mathbf{E}_1 \mathbf{E}_2^T)))$.
   - Explicit message passing over topological neighbors to model physical coupling (mass/energy balance).
3. **Forecasting Head:**
   - Joint representation layer combining temporal and spatial embeddings to forecast one-step-ahead expected sensor values $\hat{\mathbf{X}}(t) \in \mathbb{R}^M$.
4. **Sensor Reliability Module (SRM):**
   - Triangulates three physically motivated signals:
     1. *Temporal Continuity ($v_i(t)$):* Rate-of-change boundaries and variance-based stuck sensor detection.
     2. *Cross-Sensor Neighborhood Consistency ($c_i(t)$):* Residual deviation between observed sensor value and prediction from correlated graph neighbors.
     3. *Residual Confidence ($p_i(t)$):* Median Absolute Deviation (MAD)-standardized residual Z-score tail probability.
   - Smooth evidential logit fusion produces calibrated reliability vector $\mathbf{r}(t) \in [0, 1]^M$.
5. **Reliability-Weighted System Anomaly Score:**
   $$S_{\text{sys}}(t) = \frac{\sum_{i=1}^M r_i(t) \cdot z_i(t)}{\sum_{i=1}^M r_i(t) + \epsilon}$$
6. **Sensor Fault Localization Score:**
   $$S_{\text{fault}}(t) = \max_{i \in \{1 \dots M\}} \left[ (1 - r_i(t)) \cdot z_i(t) \right]$$
7. **Four-Way Operational State Classifier:**
   - Jointly tests system anomaly score against validation threshold $\tau_{\text{sys}}$ and reliability vector against confidence threshold $\theta_{\text{rel}}$.

---

## D. What Has Already Been Implemented

1. **Data Ingestion and Chronological Reconstruction (`src/data/swat_loader.py`):**
   - Reconstructs the authentic official SWaT A1/A2 December 2015 testbed.
   - Recombines the 7-day continuous normal training run (496,800 rows at 1 Hz).
   - Reconstructs the 4-day continuous physical cyber-attack testing run (449,919 rows, 12.14% attack density).
   - Establishes a project-specific chronological split: 80% train (39,744 samples at 10x downsample) and 20% validation (9,936 samples at 10x downsample) on normal data.
2. **Leak-Free Preprocessing Pipeline (`src/preprocessing/scaler.py`, `windowing.py`):**
   - `RobustStandardScaler` fit strictly on the 80% normal training split to eliminate data leakage.
   - Memory-efficient sliding window generation supporting configurable lookback windows ($L=30$ or $L=60$).
3. **Sensor Reliability Module (`src/reliability/sensor_reliability.py`):**
   - Implements temporal continuity, graph neighbor consistency, and standardized residual tail confidence.
   - Vectorized batch computation and validation calibration routines (`fit_calibration`).
4. **Proposed R-STNet Architecture (`src/models/rstnet.py`):**
   - Full PyTorch implementation of temporal dilated convolutions, graph attention dependency module, forecasting head, and integrated SRM.
5. **Baseline Implementations (`src/models/baselines/`):**
   - Baseline A: Classical Isolation Forest (`isolation_forest.py`).
   - Baseline B: SpatioTemporal Autoencoder (`reconstruction_ae.py`).
   - Baseline C: SpatioTemporal Graph Backbone without SRM (`rstnet_no_srm.py` - direct ablation baseline).
6. **Controlled Corruption Suite (`src/corruption/corruptor.py`):**
   - Deterministic synthetic injection supporting 9 failure modes: missingness, Gaussian noise, impulse spikes, stuck sensor, drift, intermittent failure, single sensor, multi sensor, and mixed corruption.
   - Generates exact binary ground-truth sensor fault masks $M_{\text{fault}}(t, i) \in \{0, 1\}$.
7. **Evaluation and Metrics Engine (`src/evaluation/metrics.py`):**
   - Point-wise Precision, Recall, F1-Score, PR-AUC, ROC-AUC, FPR.
   - Labeled Point-Adjusted F1 (PA) for baseline alignment, accompanied by distortion warnings.
   - Sensor fault localization metrics (Precision, Recall, F1, identification accuracy).
8. **Unit Tests & Scaffolding (`tests/`):**
   - Unit tests covering scaler leak prevention, window indexing, stuck-sensor penalization, and false-alarm discounting.

---

## E. What Is Still Under Development (Planned for Phase 2)

1. **Full-Scale Model Training & Checkpoint Optimization:**
   - Hyperparameter grid search and multi-epoch convergence across all baselines and R-STNet on the full dataset.
2. **Execution of the 10-Experiment Controlled Corruption Matrix:**
   - Systematic execution of Clean (E1), Gaussian Noise (E2), Stuck Sensor (E3), Drift (E4), Missing Data (E5–E7), Multi-Sensor (E8–E9), and Mixed Corruption (E10) across multiple random seeds (42, 101, 2024).
3. **Ablation Studies:**
   - Quantitative evaluation of Variant A (No SRM), Variant B (Temporal SRM only), Variant C (Graph SRM only), and Variant D (Full R-STNet).
4. **External Validation on WADI Testbed:**
   - Secondary evaluation on the 123-channel Water Distribution testbed.
5. **Interactive Diagnostic & Explanation Dashboard:**
   - Real-time sensor trust waterfall visualization and graph edge violation attribution.

---

## F. Dataset Status

- **Primary Dataset: SWaT (Secure Water Treatment):**
  - Origin: iTrust Centre for Research in Cyber Security, Singapore University of Technology and Design (SUTD).
  - Physical Scope: 6 physical water treatment stages (P1–P6), 51 continuous sensors and discrete actuators.
  - Raw Location: `reference/swat_data/` (Normal: 402 MB, Attack: 15 MB, Merged: 426 MB).
  - Processed Location: `data/processed/SWaT/` (`train.npy`, `val.npy`, `test.npy`, `test_labels.npy`).
  - Preprocessing Status: Verified clean, zero NaNs, zero Infs, zero test-label leakage.
  - Important Structural Distinction: Official SWaT attack labels indicate system-level cyber-physical attacks; they do **not** provide ground truth for isolated sensor hardware faults. Sensor-fault ground truth is provided strictly via the controlled corruption generator masks.
- **Secondary Dataset: WADI (Water Distribution):**
  - Origin: iTrust, SUTD. 123 sensor/actuator channels across 3 water distribution loops.
  - Raw Location: `reference/wadi_data/` (Normal 14 days: 493 MB, Attack: 109 MB).
  - Status: Documented and planned for Phase 2 external validation.

---

## G. Reference Research Used

1. **CAROTS (Kim et al., ICML 2025 / arXiv:2506.03964):**
   - *Role:* External state-of-the-art reference baseline in causality-aware contrastive time-series anomaly detection.
   - *Location in workspace:* `reference/CAROTS-master/` (maintained read-only).
   - *Decoupling:* R-STNet is an independent architecture with an explicit sensor-reliability estimation module and 4-way decision logic; it does not clone or rebrand CAROTS.
2. **GDN (Deng & Hooi, IJCAI 2021):**
   - *Role:* Foundational reference for learning directed sensor graphs via node embeddings.
3. **MTAD-GAT (Zhao et al., IEEE ICDM 2020):**
   - *Role:* Dual-graph attention reference (feature and temporal correlations).
4. **USAD (Audibert et al., ACM KDD 2020):**
   - *Role:* Adversarial reconstruction autoencoder reference.
5. **OmniAnomaly (Su et al., ACM KDD 2019):**
   - *Role:* Stochastic RNN reference demonstrating latent space representation of multivariate sequences.
6. **RANSynCoders (Audibert et al., IEEE TKDE 2022):**
   - *Role:* Random sampling ensemble autoencoders for robust anomaly detection.
7. **Classical FDI (Gertler 1998, Isermann 2006, Ding 2014):**
   - *Role:* Theoretical foundation of analytical redundancy and parity space relations for sensor fault isolation.
8. **Evaluation Pitfalls (Kim et al. AAAI 2022, Wu & Keogh IEEE TKDE 2021):**
   - *Role:* Critical guidelines identifying the distortion of the Point-Adjustment (PA) protocol.

---

## H. Current Assumptions

1. **Physical Redundancy:** The monitored cyber-physical system possesses intrinsic physical coupling (mass, volume, and chemical conservation) across sensor channels, enabling topological neighbors to estimate each other's expected values.
2. **Normal Training Baseline:** Normal operational training data is substantially free of severe persistent sensor faults and attacks.
3. **Chronological Validity:** Sensor time-series follow non-stationary temporal dynamics; random k-fold cross-validation is invalid and causes future-data leakage. Chronological partitioning is mandatory.
4. **Validation Calibration:** Validation data reflects healthy operating envelopes and can be used to calibrate residual bounds (MAD) and decision thresholds ($\tau_{\text{sys}}, \theta_{\text{rel}}$) without referencing test labels.

---

## I. Current Limitations

1. **Computational Ceiling:** Executed locally on Windows 11 with an NVIDIA GeForce RTX 3050 Laptop GPU (4096 MiB VRAM). Requires bounded batch size ($B \le 64$) and sequence window ($L \le 60$) to prevent CUDA Out-of-Memory faults.
2. **Discrete Actuator Variance:** Binary actuators (e.g., pumps, motorized valves) exhibit near-zero variance during long steady-state periods, requiring numerical clamping to prevent division-by-zero during standardization.
3. **Synthetic Corruption Reliance for Sensor Fault Ground Truth:** Because real-world benchmark datasets lack separate ground truth for isolated sensor hardware faults, sensor-level evaluation relies on mathematically defined synthetic corruption masks.
4. **Phase 1 Incompleteness:** Full hyperparameter exploration, extensive multi-seed experimental runs, and statistical significance testing have not yet been conducted.

---

## J. What Is Appropriate to Claim at Review 1

- The comprehensive problem formulation addressing the vulnerability of standard multivariate anomaly detectors to sensor-level corruptions.
- The theoretical motivation and mathematical derivation of the Sensor Reliability Module (SRM).
- The mathematical formulation of reliability-weighted system anomaly scoring and the 4-way operational state discrimination engine.
- The design of the leak-free preprocessing and chronological splitting protocol.
- The design and implementation of the 9-mode controlled sensor corruption framework.
- The baseline model architectures (Isolation Forest, SpatioTemporal AE, RSTNet without SRM) and their comparative evaluation plan.
- The planned evaluation framework incorporating raw point-wise metrics, threshold calibration, and sensor-fault localization metrics.

---

## K. What Must NOT Yet Be Claimed

- **DO NOT CLAIM** that R-STNet has achieved superior F1, Precision, or Recall over baselines.
- **DO NOT CLAIM** specific empirical performance numbers (e.g., "R-STNet achieved 0.88 F1").
- **DO NOT CLAIM** that the 10 controlled corruption experiments have been completed.
- **DO NOT CLAIM** that the ablation study has empirically demonstrated the superiority of the triangulated SRM.
- **DO NOT CLAIM** that R-STNet is state-of-the-art across all benchmarks.
- **DO NOT CLAIM** that no previous work has ever studied noisy sensors.
- **DO NOT CLAIM** that the project is complete.
