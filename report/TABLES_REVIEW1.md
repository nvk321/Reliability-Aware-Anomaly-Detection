# Tables for Review 1 Report

**Project:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Review Target:** Review 1 (Phase 1)  
**Notice:** Contains structural, methodological, architectural, and planning tables ONLY. In accordance with Phase 1 scope rules, **no completed experimental results or performance tables are included**.

---

## Table 1: Problem Characteristics in Industrial Multivariate Sensor Monitoring

| Dimension | Typical Industrial Cyber-Physical System | Conventional Anomaly Detection Assumption | Consequence of Misalignment |
| :--- | :--- | :--- | :--- |
| **Observation Reliability** | Physical sensors frequently experience hardware noise, calibration drift, fouling, spikes, or stuck readings. | Sensors are 100% trustworthy; any observation deviation $|x_i - \hat{x}_i|$ represents a physical process failure. | Isolated sensor degradation triggers false emergency plant shutdowns ($S_{\text{global}} > \tau$). |
| **Channel Interdependence** | Sensors are physically coupled through conservation laws (mass, energy, pressure, fluid flow). | Many classical algorithms treat channels independently or project them into unconstrained latent spaces. | Misses coordinated multi-channel attack patterns; unable to use cross-sensor consensus to identify faulty sensors. |
| **Temporal Dynamics** | Physical processes exhibit inertia, valve opening times, chemical reaction delays, and settling times. | Point-based models evaluate static snapshots without sequence memory. | Inability to distinguish transient spikes from sustained physical state changes. |
| **Operational Impact** | Unnecessary plant shutdowns incur extreme financial losses, operational wear, and water wastage. | Detection models optimize global residual sums without considering the cost of false positive alarms. | High False Positive Rate (FPR) destroys operator confidence in automated monitoring systems. |
| **Diagnostic Need** | Plant operators require actionable intelligence: *Is the physical process failing, or is a sensor broken?* | Detectors output a binary scalar flag ($\text{Anomaly} \in \{0, 1\}$) without fault isolation. | Maintenance teams cannot pinpoint whether to inspect a process valve or replace a sensor transmitter. |

---

## Table 2: Literature Comparison Across Anomaly Detection Paradigms

| Paradigm | Exemplar References | Spatial Modeling | Temporal Modeling | Noise Robustness | Explicit Channel Reliability $r_i(t)$ | Decouples Sensor Faults from System Anomalies |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Classical / Statistical** | Isolation Forest [1], OCSVM [2], DPCA [5] | Low (Linear subspace) | None to Low | Low | **No** | No (Uniform residual) |
| **Deep Autoencoders** | USAD [9], MSCRED [13], ST-AE [14] | Moderate (Latent bottleneck) | Moderate (Conv/LSTM) | Low to Moderate | **No** | No (Uniform reconstruction sum) |
| **Stochastic Deep Models**| OmniAnomaly [10], DAGMM [18] | Moderate (Stochastic flows) | High (Recurrent GRU) | Low (Latent contamination) | **No** | No (Latent distribution score) |
| **Transformers** | TranAD [11], Anomaly Transformer [12] | High (Cross-channel attention) | High (Multi-head attention) | Moderate | **No** | No (Global association score) |
| **Graph Neural Networks** | GDN [19], MTAD-GAT [20], ST-GCN [23] | High (Directed graph attention) | High (TCN / GRU) | Low (Unweighted node sum) | **No** | No (Uniform node error sum) |
| **Causal Contrastive** | CAROTS [21] | High (Causal structural discovery)| High (Contrastive views) | High (against shift) | **No** | No (Latent invariance) |
| **Random Ensembles** | RANSynCoders [27] | Moderate (Random subsets) | Moderate (Dense windows) | High | **No** (Implicit voting) | Partial |
| **Model-Based FDI** | Gertler [33], Isermann [34], Ding [35] | High (Analytical redundancy) | High (Differential ODEs) | High | **No** (Parity residuals) | **Yes** (Requires manual physics ODEs) |
| **Proposed R-STNet** | **This Work** | **High (Learned graph attention)** | **High (Dilated 1D Convolutions)** | **High (Evaluated under 9 modes)** | **YES (Native online SRM)** | **YES (4-Way Discrimination Engine)** |

---

## Table 3: Synthesis of the Research Gap

| Capability / Feature | State of the Art in Literature | Remaining Unaddressed Gap | Proposed Resolution in R-STNet |
| :--- | :--- | :--- | :--- |
| **Sensor Trustworthiness** | Outputting a single anomaly score for the entire timestamp or sliding window. | Absence of an explicit, continuous trustworthiness vector $\mathbf{r}(t) \in [0, 1]^M$ for individual sensor channels. | **Sensor Reliability Module (SRM):** Continuously estimates $r_i(t)$ by triangulating temporal continuity, graph consensus, and residual tail confidence. |
| **System Anomaly Scoring** | Uniform, unweighted summation of prediction/reconstruction errors across all channels. | Inability to prevent a corrupted sensor from dominating the global anomaly score. | **Reliability-Weighted Aggregation:** Dynamically down-weights untrusted channels: $S_{\text{sys}}(t) = \frac{\sum r_i z_i}{\sum r_i + \epsilon}$. |
| **Operational Classification**| Binary output: Normal ($0$) vs. Anomaly ($1$). | Cannot distinguish whether an anomaly is an instrument failure or a cyber-physical plant emergency. | **4-Way Discrimination Engine:** Classifies state into Normal, Sensor Fault, System Anomaly, and System Anomaly with Sensor Degradation. |
| **Physical Redundancy** | Requiring complex mathematical differential equations for the entire physical plant. | Infeasible to hand-craft differential equations for large-scale multi-stage facilities (51–123 channels). | **Data-Driven Graph Attention:** Automatically learns physical topological correlations directly from normal operational data. |
| **Sensor Evaluation** | Evaluating models strictly on cyber-physical attack labels without sensor fault ground truth. | Inability to evaluate whether an algorithm can isolate faulty sensors without false system alarms. | **Controlled Corruption Framework:** Synthesizes 9 operational failure modes with exact binary ground-truth masks $M_{\text{fault}}(t, i)$. |

---

## Table 4: Benchmark Dataset Characteristics

| Dataset Metric / Property | Secure Water Treatment (SWaT) [38], [45] | Water Distribution (WADI) [39] |
| :--- | :--- | :--- |
| **Domain** | Multi-stage industrial water treatment plant | Scaled-down municipal water distribution network |
| **Physical Infrastructure** | 6 physical treatment stages (P1 to P6) | 3 distribution loops with water tanks, pumps, valves |
| **Total Monitored Channels ($M$)**| **51 channels** (25 continuous sensors, 26 discrete actuators) | **123 channels** (54 continuous sensors, 69 discrete actuators) |
| **Sampling Frequency** | 1 Hz (1 reading per second) | 1 Hz (1 reading per second) |
| **Normal Operation Duration** | 7 continuous days (504,000 seconds total, 496,800 rows valid) | 14 continuous days (1,209,600 rows valid) |
| **Attack Test Duration** | 4 continuous days (449,919 rows valid) | 2 continuous days (172,801 rows valid) |
| **Total Cyber-Physical Attacks** | 36 distinct physical and cyber-attack scenarios | 15 distinct multi-stage attack scenarios |
| **Test Set Anomaly Density** | **12.14% attack samples** (54,621 attack seconds) | **5.99% attack samples** |
| **Raw Data File Location** | `reference/swat_data/` (`normal.csv`, `attack.csv`) | `reference/wadi_data/` (`WADI_14days_new.csv`, `WADI_attackdataLABLE.csv`) |
| **Project Role** | **Primary Benchmark (Tier 1)** — Immediate implementation | **Secondary Benchmark (Tier 2)** — External validation |
| **Project Chronological Split** | Train: 39,744 windows (80% normal at 10x downsample) <br> Val: 9,936 windows (20% normal at 10x downsample) <br> Test: 44,992 windows (100% mixed attack at 10x downsample) | Documented and scheduled for Phase 2 external generalization study. |

---

## Table 5: Proposed R-STNet Architecture Components

| Module Name | Sub-Components | Input Dimension | Output Dimension | Mathematical / Functional Purpose | Status in Codebase |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Temporal Feature Extractor** | 1D Dilated Convolutions (`Conv1d`, dilation=2, residual skip) + Linear projection | Window $\mathbf{W}_t \in \mathbb{R}^{B \times L \times M}$ | Temporal features $\mathbf{H}_{\text{temp}} \in \mathbb{R}^{B \times M \times d}$ | Extracts local autoregressive dynamics, physical inertia, and temporal lag signatures per channel. | **Implemented** (`src/models/rstnet.py`) |
| **Graph Dependency Module** | Learnable node embeddings $\mathbf{E}_1, \mathbf{E}_2$, Top-K sparsification, Softmax message passing | Temporal features $\mathbf{H}_{\text{temp}} \in \mathbb{R}^{B \times M \times d}$ | Spatial features $\mathbf{H}_{\text{spat}} \in \mathbb{R}^{B \times M \times d}$ + Adjacency $\mathbf{A} \in \mathbb{R}^{M \times M}$ | Learns directed physical correlation topology and aggregates consensus messages from topological neighbors. | **Implemented** (`src/models/baselines/rstnet_no_srm.py`) |
| **Forecasting Output Head** | Multi-Layer Perceptron (`Linear` $\to$ `ReLU` $\to$ `Dropout` $\to$ `Linear`) | Joint features $[\mathbf{H}_{\text{temp}} \,\|\, \mathbf{H}_{\text{spat}}] \in \mathbb{R}^{B \times M \times 2d}$ | Expected forecast $\hat{\mathbf{X}}(t) \in \mathbb{R}^{B \times M}$ | Predicts one-step-ahead expected physical sensor values based on joint spatio-temporal representations. | **Implemented** (`src/models/rstnet.py`) |
| **Sensor Reliability Module (SRM)** | 1. Temporal Continuity ($v_i$)<br>2. Graph Neighbor Consensus ($c_i$)<br>3. Standardized Residual Tail ($p_i$) | Recent window $\mathbf{W}_t$ + Observed $\mathbf{X}(t)$ + Forecast $\hat{\mathbf{X}}(t)$ | Continuous reliability vector $\mathbf{r}(t) \in [0, 1]^M$ | Triangulates three independent physical signals to compute online channel trustworthiness via calibrated logit fusion. | **Implemented** (`src/reliability/sensor_reliability.py`) |
| **Reliability-Weighted Aggregator** | Weighted residual normalization | Residuals $\mathbf{z}(t) \in \mathbb{R}^M$, Reliability $\mathbf{r}(t) \in [0, 1]^M$ | Scalar system score $S_{\text{sys}}(t) \in \mathbb{R}_+$ | Insulates system anomaly score from corrupted sensors: $S_{\text{sys}} = \frac{\sum r_i z_i}{\sum r_i + \epsilon}$. | **Implemented** (`src/models/rstnet.py`) |
| **Sensor Fault Scorer** | Complementary weighting | Residuals $\mathbf{z}(t)$, Reliability $\mathbf{r}(t)$ | Fault score $S_{\text{fault}}(t)$ + Channel index | Pinpoints most degraded sensor: $\arg\max_i [(1 - r_i) z_i]$. | **Implemented** (`src/models/rstnet.py`) |
| **State Discrimination Engine** | Calibrated dual threshold comparison | $S_{\text{sys}}(t)$, $\mathbf{r}(t)$, $\tau_{\text{sys}}$, $\theta_{\text{rel}}$ | 4-way operational state label $\in \{\text{Case A, B, C, D}\}$ | Decouples instrument maintenance alerts from genuine plant emergency alarms. | **Implemented** (`src/anomaly_detection/discriminator.py`) |

---

## Table 6: Comparative Overview of Evaluation Baselines

| Baseline Model | Category | Architectural Description | Anomaly Scoring Mechanism | Role in Experimental Evaluation |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline A: Isolation Forest** | Classical Machine Learning | Ensemble of 100 random decision trees partitioning feature space. | Average path length in isolation trees: $s(x) = 2^{-\frac{E(h(x))}{c(n)}}$. | Benchmarks off-the-shelf industrial tabular detector performance. |
| **Baseline B: SpatioTemporal Autoencoder (ST-AE)** | Deep Learning Reconstruction | Dilated temporal convolutional encoder with symmetric decoder. | Unweighted Mean Squared Reconstruction Error: $S(t) = \frac{1}{M} \sum_{i=1}^M (x_i(t) - \tilde{x}_i(t))^2$. | Represents standard unweighted deep reconstruction baseline. |
| **Baseline C: RSTNet without SRM** | Deep Graph Neural Network | Identical temporal convolution + graph attention backbone as R-STNet, but **without SRM**. | Unweighted Mean Standardized Residual: $S(t) = \frac{1}{M} \sum_{i=1}^M z_i(t)$. | **Primary Ablation Baseline:** Isolates the exact scientific contribution of the Sensor Reliability Module. |
| **Baseline 3: CAROTS (Kim et al., 2025)** | External State-of-the-Art | Causal structural discovery + contrastive learning between causally consistent views. | Contrastive representation distance + prediction residual. | Establishes external reference benchmark under ICML 2025 conventions. |
| **Proposed: R-STNet with SRM** | Proposed Reliability-Aware Framework | Full Spatio-Temporal Graph Attention + Sensor Reliability Module + 4-Way Engine. | Reliability-weighted residual aggregation: $S_{\text{sys}}(t) = \frac{\sum r_i z_i}{\sum r_i + \epsilon}$. | Evaluates proposed hypothesis: suppressing false alarms while preserving true attack detection. |

---

## Table 7: Controlled Sensor Corruption Suite (Planned Evaluation Scenarios)

| Exp # | Scenario Name | Physical Failure Mode | Mathematical Formulation | Target Channels | Severity Parameters | Ground Truth Availability |
| :---: | :--- | :--- | :--- | :--- | :--- | :---: |
| **E1** | **Clean Benchmark** | Normal operation / Attacks only | $x_i(t) = x_i^{\text{orig}}(t)$ | None | 0% corruption | SWaT attack labels |
| **E2** | **Low Gaussian Noise** | Signal conditioning noise | $x_i(t) = x_i(t) + \mathcal{N}(0, \sigma_i^2)$ | 3 critical sensors (`FIT101`, `LIT101`, `AIT202`) | $\sigma_i = 0.1 \cdot \text{std}(x_i)$ | Synthetic mask $M(t, i) = 1$ |
| **E3** | **Medium Gaussian Noise** | Electromagnetic interference | $x_i(t) = x_i(t) + \mathcal{N}(0, \sigma_i^2)$ | 3 critical sensors | $\sigma_i = 0.3 \cdot \text{std}(x_i)$ | Synthetic mask $M(t, i) = 1$ |
| **E4** | **High Gaussian Noise** | Severe transmitter fault | $x_i(t) = x_i(t) + \mathcal{N}(0, \sigma_i^2)$ | 3 critical sensors | $\sigma_i = 0.5 \cdot \text{std}(x_i)$ | Synthetic mask $M(t, i) = 1$ |
| **E5** | **Single Stuck Sensor** | Mechanical freeze / AD lock | $x_i(t) = x_i(t_{\text{start}})$ for $t \ge t_{\text{start}}$ | 1 sensor (`LIT101`, Tank 1 Level) | Duration = 10% of test run | Synthetic mask $M(t, i) = 1$ |
| **E6** | **Multi Stuck Sensors** | Bus freeze / PLC card fault | $x_i(t) = x_i(t_{\text{start}})$ | 3 sensors across stages P1, P2, P3 | Duration = 10% of test run | Synthetic mask $M(t, i) = 1$ |
| **E7** | **Incipient Sensor Drift** | Aging / Diaphragm fatigue | $x_i(t) = x_i(t) + \alpha \cdot (t - t_{\text{start}})$ | 1 sensor (`FIT101`, Raw Water Flow) | Drift slope $\alpha = 0.1 \cdot \frac{\text{std}(x_i)}{N_{\text{drift}}}$ | Synthetic mask $M(t, i) = 1$ |
| **E8** | **Severe Sensor Drift** | Rapid calibration shift | $x_i(t) = x_i(t) + \alpha \cdot (t - t_{\text{start}})$ | 1 sensor (`FIT101`) | Drift slope $\alpha = 0.5 \cdot \frac{\text{std}(x_i)}{N_{\text{drift}}}$ | Synthetic mask $M(t, i) = 1$ |
| **E9** | **Impulse Spikes** | Power supply transient ripple | $x_i(t) = x_i(t) \pm 3.0 \cdot \text{std}(x_i)$ | Random 15% of channels | 2% impulse frequency | Synthetic mask $M(t, i) = 1$ |
| **E10**| **Mixed Harsh Environment**| Combined multi-fault stress | Noise (0.2) + Drift + Stuck + Spikes | 5 simultaneous channels | Multi-severity combined | Synthetic mask $M(t, i) = 1$ |

---

## Table 8: Comprehensive Evaluation Metrics Framework

| Metric Category | Metric Symbol / Name | Mathematical Definition | Evaluation Objective / Scientific Justification |
| :--- | :--- | :--- | :--- |
| **System Detection Quality** | **Precision ($P$)** | $\frac{\text{TP}}{\text{TP} + \text{FP}}$ | Measures the exactness of alarms; penalizes false alarms caused by noisy or degraded sensors. |
| | **Recall ($R$)** | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ | Measures the completeness of detection; ensures genuine cyber-physical attacks are not missed. |
| | **F1-Score ($F_1$)** | $2 \cdot \frac{P \cdot R}{P + R}$ | Primary point-wise harmonic mean balancing false alarms and missed attacks without point adjustment. |
| | **PR-AUC** | $\int P(R) \, dR$ | **Threshold-independent evaluation.** Essential for imbalanced benchmarks (12% attack density). |
| | **False Positive Rate (FPR)**| $\frac{\text{FP}}{\text{FP} + \text{TN}}$ | Monitored explicitly during clean operating periods to measure false alarm induction under corruption. |
| **Reliability & Robustness** | **False Alarm Suppression Rate (FASR)** | $1 - \frac{\text{FPR}_{\text{model}}^{\text{corrupt}} - \text{FPR}_{\text{model}}^{\text{clean}}}{\text{FPR}_{\text{baseline}}^{\text{corrupt}} - \text{FPR}_{\text{baseline}}^{\text{clean}}}$ | Measures the percentage of sensor-induced false alarms successfully suppressed by the SRM. |
| | **Reliability Separation ($\Delta r$)** | $\bar{r}_{\text{clean}} - \bar{r}_{\text{corrupted}}$ | Quantifies the margin between the reliability scores assigned to healthy vs. corrupted sensor channels. |
| | **Sensor Fault Precision ($P_{\text{fault}}$)** | $\frac{\text{TP}_{\text{fault}}}{\text{TP}_{\text{fault}} + \text{FP}_{\text{fault}}}$ | Evaluates accuracy in isolating the specific degraded channel against synthetic ground-truth masks. |
| | **Sensor Fault Recall ($R_{\text{fault}}$)** | $\frac{\text{TP}_{\text{fault}}}{\text{TP}_{\text{fault}} + \text{FN}_{\text{fault}}}$ | Evaluates the proportion of corrupted sensor steps successfully detected by the SRM. |
| | **Sensor Fault F1 ($F_{1, \text{fault}}$)**| $2 \cdot \frac{P_{\text{fault}} \cdot R_{\text{fault}}}{P_{\text{fault}} + R_{\text{fault}}}$ | Harmonic mean of sensor-level fault localization performance. |
| **Benchmark Reference** | **Point-Adjusted F1 (PA)** | Point-adjustment heuristic [10] | Reported **strictly as a labeled secondary metric** to facilitate comparison with historical literature. |

---

## Table 9: Planned Experimental Suite Matrix (Phase 2 Execution)

| Experiment Code | Model Variants Evaluated | Datasets Tested | Target Hypothesis Evaluated | Planned Evaluation Protocol |
| :---: | :--- | :--- | :--- | :--- |
| **EXP-BENCH** | Isolation Forest, ST-AE, RSTNet (No SRM), CAROTS, Proposed R-STNet | SWaT (Clean E1) | Establish baseline clean performance across all models. | Train on 80% normal; calibrate threshold at 95th percentile of normal validation scores; evaluate blindly on mixed test set. |
| **EXP-NOISE** | All models | SWaT (E2, E3, E4) | **Hypothesis 1:** Unweighted models suffer rapid FPR explosion under noise; R-STNet maintains low FPR via SRM. | Evaluate across 3 noise tiers (0.1, 0.3, 0.5) over 3 deterministic seeds (42, 101, 2024). |
| **EXP-STUCK** | All models | SWaT (E5, E6) | **Hypothesis 1:** Frozen sensor causes permanent false alarm in AE; R-STNet flags Case B (Sensor Fault). | Inject stuck sensor into Tank 1 Level (`LIT101`); measure Case B classification vs. false Case C alerts. |
| **EXP-DRIFT** | All models | SWaT (E7, E8) | **Hypothesis 1:** Gradual calibration drift escapes static checks but is caught by graph consistency. | Inject linear ramp drift into Raw Water Flow (`FIT101`); measure detection lag and false alarm rate. |
| **EXP-ABLAT** | 1. Variant A (No SRM)<br>2. Variant B (Temporal SRM)<br>3. Variant C (Graph SRM)<br>4. Variant D (Full R-STNet) | SWaT (E1, E4, E5) | **Hypothesis 3:** Graph consistency provides primary process agreement; temporal volatility handles sudden spikes/stuck. | Ablate individual SRM components; measure impact on $F_1$, FASR, and $\Delta r$. |
| **EXP-EXTVAL** | Best baseline vs. Proposed R-STNet | WADI (123 channels) | Verify generalization of reliability-aware framework to larger industrial cyber-physical networks. | Train on 14-day WADI normal run; evaluate on 2-day attack run under controlled sensor corruption. |
