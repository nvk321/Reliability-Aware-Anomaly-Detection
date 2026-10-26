# Chapter 3: Proposed Methodology & Planned Evaluation Framework

**Course:** Machine Learning Lab (BCSE209P)  
**Slot:** L47 + L48 | Fall Semester 2026-27  
**Project Title:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Authors:** P Naga Venkata Kushal (23BDS0133), Koteru Tarun (23BDS0103)  
**Guide:** Dr. Dega Nagaraju  
**Institution:** School of Computer Science and Engineering (SCOPE), VIT Vellore  
**Milestone:** REVIEW 1 (PHASE 1)  

---

## 3.1 Methodology Overview

The proposed research methodology establishes an end-to-end, leak-free framework designed to ingest high-dimensional multivariate sensor time series, learn spatio-temporal representations and physical topological relationships, estimate continuous channel-level sensor reliability online, and decouple isolated instrument degradations from genuine plant-wide cyber-physical attacks.

The operational workflow spans six integrated stages, as depicted in the conceptual pipeline (refer to Figure 3 in `report/FIGURES/03_overall_methodology.mmd`):
1. **Raw Telemetry Ingestion & Chronological Splitting:** Reconstructs the official SWaT benchmark into a 7-day continuous normal sequence (496,800 samples) and a 4-day continuous attack sequence (449,919 samples). Implements a project-specific chronological partition (80% train / 20% validation on normal data).
2. **Leak-Free Preprocessing & Windowing:** Scales features strictly using training distribution parameters with zero-variance protection, constructing sliding sequence tensors $\mathbf{W}_t \in \mathbb{R}^{L \times M}$.
3. **Spatio-Temporal Graph Representation:** 1D dilated temporal convolutions capture local autoregressive dynamics per sensor, while learnable graph attention captures directed physical interdependencies across channels.
4. **Online Sensor Reliability Estimation (SRM):** Evaluates temporal continuity, graph neighborhood consistency, and standardized residual tail confidence, fusing them via calibrated evidential logit fusion into continuous reliability scores $r_i(t) \in [0, 1]$.
5. **Reliability-Weighted Anomaly Scoring:** Aggregates residuals weighted by reliability, actively discounting degraded sensors to prevent false alarms:
   $$S_{\text{sys}}(t) = \frac{\sum_{i=1}^M r_i(t) \cdot z_i(t)}{\sum_{i=1}^M r_i(t) + \epsilon}$$
6. **Multi-State Operational Diagnosis:** Compares $S_{\text{sys}}(t)$ and $\mathbf{r}(t)$ against validation-calibrated thresholds to classify plant status into four mutually exclusive operational regimes (Normal, Sensor Fault, System Anomaly, and System Anomaly with Sensor Degradation).

---

## 3.2 System Architecture: R-STNet

The primary proposed architecture is **R-STNet (Dual-Perspective Reliability-Aware Spatio-Temporal Network)**. R-STNet unifies deep spatio-temporal representation learning with a dedicated online Sensor Reliability Module (SRM).

```
┌────────────────────────────────────────────────────────────────────────┐
│                      R-STNet NEURAL ARCHITECTURE                       │
├────────────────────────────────────────────────────────────────────────┤
│  Input Window Tensor: W_t in R^(B x L x M)                             │
│       │                                                                │
│       ├──> Temporal Feature Extractor (Dilated Conv1D + Skip)          │
│       │         └──> H_temp in R^(B x M x d)                           │
│       │                   │                                            │
│       ├──> Inter-Sensor Graph Attention (Node Embeddings E_1, E_2)     │
│       │         └──> Learned Adjacency Matrix A in R^(M x M)           │
│       │         └──> Spatial Message Passing H_spat in R^(B x M x d)   │
│       │                   │                                            │
│       └──> Forecasting Head: [H_temp || H_spat] -> X_hat(t) in R^M    │
│                           │                                            │
│  ┌────────────────────────┴────────────────────────────────────────┐   │
│  │               SENSOR RELIABILITY MODULE (SRM)                   │   │
│  │  1. Temporal Volatility v_i(t) (Rate limit + Stuck variance)   │   │
│  │  2. Graph Neighbor Consistency c_i(t) (Physical agreement)      │   │
│  │  3. Residual Confidence p_i(t) (MAD tail probability)           │   │
│  │  4. Calibrated Logit Fusion -> r_i(t) in [0, 1]                │   │
│  └────────────────────────┬────────────────────────────────────────┘   │
│                           │                                            │
│  Dual System Outputs:     ├──> System Anomaly Score S_sys(t)           │
│                           ├──> Sensor Fault Score S_fault(t)           │
│                           └──> 4-Way Operational Decision Engine       │
└────────────────────────────────────────────────────────────────────────┘
```

The mathematical formulation of each constituent component is established in Sections 3.8 through 3.12.

---

## 3.3 Dataset: Secure Water Treatment (SWaT)

### 3.3.1 Physical Industrial Environment
The primary benchmark utilized in this study is the **Secure Water Treatment (SWaT)** dataset collected by the iTrust Centre for Research in Cyber Security at the Singapore University of Technology and Design (SUTD) [38], [45]. SWaT represents an operational physical pilot plant that purifies non-potable water through six sequential stages:
- **P1:** Raw water intake, storage, and chemical dosing.
- **P2:** Coagulation, flocculation, and chemical pre-treatment (pH/ORP monitoring).
- **P3:** Ultrafiltration (UF) membrane separation.
- **P4:** Dechlorination via chemical dosing and ultraviolet (UV) irradiation.
- **P5:** Reverse Osmosis (RO) desalinization under high hydraulic pressure.
- **P6:** Clean water permeate storage and automated backwash cycling.

### 3.3.2 Multivariate Structure
The testbed records $M = 51$ continuous and discrete channels at 1 Hz (1 sample per second):
- **25 Continuous Physical Sensors:** Float and differential level transmitters (`LIT101`, `LIT301`, `LIT401`), electromagnetic flow meters (`FIT101`–`FIT601`), pressure transmitters (`PIT501`–`PIT503`, `DPIT301`), and electrochemical analyzers (`AIT201`–`AIT504`).
- **26 Discrete Actuators & Valves:** Motorized modulating valves (`MV101`–`MV504`), centrifugal transfer pumps (`P101`–`P603`), and UV disinfection lamps (`UV401`).

### 3.3.3 Operational Sequences and Attack Labels
- **Official Normal Run (`normal.csv`):** Spans 7 continuous days (496,800 seconds, rows 395,298 to 892,097 of the raw archive) under 100% healthy nominal operation ($y = 0$).
- **Official Attack Test Run (`attack.csv`):** Spans 4 continuous days (449,919 seconds) containing 36 distinct physical and cyber-attack scenarios designed by ICS security engineers (12.14% attack density, 54,621 attack seconds).
- **Reconstructed Test Run:** Recombines normal intervals (395,298 rows) and attack intervals (54,621 rows) chronologically by timestamp (`src/data/swat_loader.py`).

### 3.3.4 Critical Ground-Truth Distinction
As formally detailed in `report/DATASET_RESEARCH.md`, official SWaT attack labels ($y_{\text{sys}} \in \{0, 1\}$) represent **cyber-physical attack events affecting the process**. They do **not** provide ground truth that an individual sensor instrument is physically damaged, uncalibrated, or degraded. Sensor-fault ground truth is provided strictly via our controlled corruption framework masks ($M_{\text{fault}}(t, i)$).

---

## 3.4 Dataset Preparation

Dataset ingestion and preparation are implemented in `src/data/swat_loader.py`:
1. **Timestamp Parsing & Column Normalization:** Strip leading/trailing whitespaces from column headers (e.g., `' FIT101'` $\to$ `'FIT101'`).
2. **Chronological Splitting (Project-Specific Partition):**
   - **Training Set (80% Normal):** First 397,440 raw seconds (39,744 samples at 10x downsample). Contains pure normal operation ($y = 0$).
   - **Validation Set (20% Normal):** Next 99,360 raw seconds (9,936 samples at 10x downsample). Contains pure normal operation ($y = 0$). Reserved strictly for threshold and calibration parameter tuning.
   - **Test Set (Attack Sequence):** Continuous 449,919 raw seconds (44,992 samples at 10x downsample) containing 12.14% attack density. Evaluated blindly.
3. **Downsampling:** Downsampling by factor **10x** yields clean 10-second intervals, preserving hydraulic time constants while accelerating training throughput by an order of magnitude.
4. **Data Integrity Checks:** Enforce assertions verifying zero NaNs, zero Infs, correct dimensionality $(N, L, M)$, and binary label validity.

---

## 3.5 Preprocessing

### 3.5.1 Leak-Free Scaling
To prevent information leakage from the evaluation split into model training:
1. Feature scaling parameters ($\mu_{\text{train}, i}$ and $\sigma_{\text{train}, i}$) are computed **strictly on the 80% normal training split**.
2. The validation and test sets are transformed using the fitted training parameters:
   $$\tilde{x}_i(t) = \frac{x_i(t) - \mu_{\text{train}, i}}{\sigma_{\text{train}, i} + \epsilon}$$
3. **Zero-Variance Protection:** Certain discrete backup actuators (e.g., `P102`, `P202`) remain constantly inactive throughout normal training ($\sigma_i = 0$). To prevent division-by-zero, our scaler sets $\sigma_i = 1.0$ for any channel where $\sigma_i < 10^{-6}$.

### 3.5.2 Importance of Chronological Processing
Unlike cross-sectional tabular data, time series possess temporal autocorrelation. Shuffling or random K-fold cross-validation allows the model to access future information when predicting past states (lookahead leakage). Enforcing strict chronological splitting guarantees authentic out-of-sample evaluation [40], [41].

---

## 3.6 Temporal Windowing

Static point-wise observations do not provide temporal context. Given the standardized multivariate sequence, we construct chronological sliding windows:

$$\mathbf{W}_t = \left[ \mathbf{X}(t - L + 1), \mathbf{X}(t - L + 2), \dots, \mathbf{X}(t) \right] \in \mathbb{R}^{L \times M}$$

- **Lookback Length ($L$):** Set to $L = 60$ steps ($\approx 10$ minutes at 10s downsampling, or 5 minutes at 5s downsampling), matching physical settling times.
- **Stride ($S$):** $S_{\text{test}} = 1$ during evaluation to maintain continuous monitoring; $S_{\text{train}} = 1$ (or $S=5$) during training.

---

## 3.7 Baseline Models

To benchmark R-STNet rigorously, four baseline architectures are implemented in `src/models/baselines/`:

### 1. Baseline A: Classical Isolation Forest
- **Architecture:** Ensemble of 100 randomized decision isolation trees (`isolation_forest.py`) [1].
- **Input:** Flattened sliding window features.
- **Scoring:** Normalized path length $s(\mathbf{x})$.
- **Benchmark Role:** Establishes standard off-the-shelf industrial baseline performance.

### 2. Baseline B: SpatioTemporal Autoencoder (ST-AE)
- **Architecture:** Dilated 1D convolutional temporal encoder coupled to a symmetric convolutional decoder (`reconstruction_ae.py`) [14].
- **Input:** Window tensor $\mathbf{W}_t \in \mathbb{R}^{L \times M}$.
- **Scoring:** Unweighted Mean Squared Reconstruction Error:
  $$S_{\text{ST-AE}}(t) = \frac{1}{M} \sum_{i=1}^M \left( x_i(t) - \tilde{x}_i(t) \right)^2$$
- **Benchmark Role:** Represents standard unweighted deep learning reconstruction.

### 3. Baseline C: RSTNet WITHOUT SRM (Primary Scientific Ablation)
- **Architecture:** Identical temporal convolution and graph attention backbone as R-STNet, but **without the Sensor Reliability Module** (`rstnet_no_srm.py`).
- **Scoring:** Unweighted Mean Standardized Residual:
  $$S_{\text{No-SRM}}(t) = \frac{1}{M} \sum_{i=1}^M z_i(t)$$
- **Benchmark Role:** Directly isolates the exact scientific contribution of the reliability mechanism under identical representation capacity.

### 4. Baseline 3: CAROTS (Kim et al., ICML 2025)
- **Architecture:** Causal structural discovery with contrastive representation learning (`reference/CAROTS-master/`) [21].
- **Benchmark Role:** Establishes external state-of-the-art reference performance under published ICML 2025 conventions.

---

## 3.8 Proposed R-STNet Architecture Formulation

The neural backbone of R-STNet is implemented in `src/models/rstnet.py`:

### 3.8.1 Temporal Representation Block
For an input window $\mathbf{W}_t \in \mathbb{R}^{B \times L \times M}$, we permute and reshape to $(B \cdot M, 1, L)$. Each sensor channel passes through two 1D dilated convolutional layers with kernel size $K=3$ and dilation factor $d=2$:

$$\mathbf{H}_{\text{conv1}} = \text{ReLU}\left( \text{Conv1D}_1(\mathbf{W}_t) \right)$$
$$\mathbf{H}_{\text{conv2}} = \text{ReLU}\left( \text{Conv1D}_2(\mathbf{H}_{\text{conv1}}) \right) + \mathbf{H}_{\text{conv1}}$$

A linear projection maps the final time step into node feature dimension $d_{\text{node}} = 32$:

$$\mathbf{H}_{\text{temp}} = \text{Linear}(\mathbf{H}_{\text{conv2}}[:, :, -1]) \in \mathbb{R}^{B \times M \times d_{\text{node}}}$$

### 3.8.2 Graph Attention Dependency Module
To model physical correlations (conservation of mass and energy) without manual plant modeling, R-STNet maintains two learnable node embedding dictionaries $\mathbf{E}_1, \mathbf{E}_2 \in \mathbb{R}^{M \times d_{\text{node}}}$. The directed similarity matrix is:

$$\mathbf{S} = \text{ReLU}\left( \mathbf{E}_1 \mathbf{E}_2^T \right), \quad S_{ii} = 0 \quad (\text{Self-loops masked})$$

To retain the strongest physical couplings, we apply Top-$K$ sparsification ($K=15$):

$$\tilde{S}_{ij} = \begin{cases} S_{ij}, & \text{if } S_{ij} \in \text{TopK}(S_{i, :}) \\ 0, & \text{otherwise} \end{cases}$$

Edge weights are normalized via Softmax over incoming edges:

$$A_{ij} = \frac{\exp(\tilde{S}_{ij})}{\sum_{k=1}^M \exp(\tilde{S}_{ik}) + \epsilon}$$

Spatial message passing aggregates information from topological neighbors:

$$\mathbf{H}_{\text{spat}} = \mathbf{A} \mathbf{H}_{\text{temp}} \in \mathbb{R}^{B \times M \times d_{\text{node}}}$$

### 3.8.3 Joint Forecasting Head
The temporal embedding $\mathbf{H}_{\text{temp}}$ and spatial embedding $\mathbf{H}_{\text{spat}}$ are concatenated:

$$\mathbf{H}_{\text{joint}} = \left[ \mathbf{H}_{\text{temp}} \,\|\, \mathbf{H}_{\text{spat}} \right] \in \mathbb{R}^{B \times M \times 2 d_{\text{node}}}$$

A multi-layer perceptron forecasts one-step-ahead expected values:

$$\hat{\mathbf{X}}(t) = \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \mathbf{H}_{\text{joint}} + \mathbf{b}_1) + \mathbf{b}_2 \in \mathbb{R}^{B \times M}$$

The model is trained end-to-end on normal data using Mean Squared Error loss:

$$\mathcal{L}_{\text{MSE}} = \frac{1}{B \cdot M} \sum_{b=1}^B \sum_{i=1}^M \left( x_{b, i}(t) - \hat{x}_{b, i}(t) \right)^2$$

---

## 3.9 Sensor Reliability Module (SRM)

The Sensor Reliability Module (implemented in `src/reliability/sensor_reliability.py`) evaluates three complementary, physically motivated signals for each channel $i$:

### 1. Temporal Continuity ($v_i(t) \in [0, 1]$)
Evaluates rate-of-change violations and frozen/stuck sensor conditions:
- **Rate-of-Change Limit:** $\Delta x_i(t) = |x_i(t) - x_i(t-1)|$. An exponential penalty dampens abrupt unphysical step violations against validation quantile bound $\delta_i^{\max}$:
  $$\text{rate\_score}_i(t) = \exp\left( -0.5 \cdot \max\left(0, \frac{\Delta x_i(t)}{\delta_i^{\max}} - 1.0\right)^2 \right)$$
- **Stuck-Sensor Variance Check:** For dynamic continuous channels, a moving variance over lookback window $K=15$ detects frozen telemetry:
  $$\text{stuck\_penalty}_i(t) = \begin{cases} 0.1, & \text{if } \text{Var}(x_{i, t-K:t}) < \epsilon_{\text{stuck}} \text{ and dynamic}_i \\ 1.0, & \text{otherwise} \end{cases}$$
- **Combined Temporal Continuity:**
  $$v_i(t) = \text{clip}\left( \text{rate\_score}_i(t) \cdot \text{stuck\_penalty}_i(t), \, 0.01, \, 0.99 \right)$$

### 2. Graph Neighborhood Consistency ($c_i(t) \in [0, 1]$)
Evaluates agreement between sensor $i$ and its physical topological neighbors:
$$\hat{x}_i^{\mathcal{N}}(t) = \sum_{j \in \mathcal{N}(i)} W_{ji} \cdot x_j(t)$$
where $W_{ji}$ is the column-normalized graph adjacency. The physical discrepancy is scaled by the validation Median Absolute Deviation ($\text{MAD}_i$):
$$c_i(t) = \text{clip}\left( \exp\left( -\frac{|x_i(t) - \hat{x}_i^{\mathcal{N}}(t)|}{2.5 \cdot \text{MAD}_i} \right), \, 0.01, \, 0.99 \right)$$
If a sensor fails or drifts, its reading deviates from its correlated neighbors, pulling $c_i(t)$ down.

### 3. Standardized Residual Tail Confidence ($p_i(t) \in [0, 1]$)
Computes the Gaussian tail probability of the MAD-standardized prediction residual $z_i(t)$:
$$z_i(t) = \frac{|x_i(t) - \hat{x}_i(t)| - \text{Median}_i(e)}{\text{MAD}_i(e) + \epsilon}$$
$$p_i(t) = \text{clip}\left( \exp\left( -0.5 \cdot \left(\frac{\max(0, z_i(t))}{2.0}\right)^2 \right), \, 0.01, \, 0.99 \right)$$

### 4. Calibrated Evidential Logit Fusion
To avoid arbitrary threshold cuts, component scores are fused via a smooth evidential formulation in logit space:

$$\text{logit}(s) = \ln\left( \frac{s}{1 - s} \right)$$
$$\text{logit\_fusion}_i(t) = w_v \cdot \text{logit}(v_i(t)) + w_c \cdot \text{logit}(c_i(t)) + w_p \cdot \text{logit}(p_i(t)) + b_r$$
$$r_i(t) = \frac{1}{1 + \exp\left( -\text{logit\_fusion}_i(t) \right)} \in [0, 1]$$

Weights $w_v = 1.0$, $w_c = 1.2$, $w_p = 1.0$, and $b_r = 0.0$ are calibrated systematically on the validation normal split (`fit_calibration`).

---

## 3.10 Reliability-Weighted System Anomaly Score

Standard detectors evaluate the unweighted sum:
$$S_{\text{unweighted}}(t) = \frac{1}{M} \sum_{i=1}^M z_i(t)$$
When sensor $k$ fails ($z_k \to \infty$), $S_{\text{unweighted}}(t)$ spikes, causing a false plant alarm.

In **R-STNet**, we compute the reliability-weighted aggregation:

$$S_{\text{sys}}(t) = \frac{\sum_{i=1}^M r_i(t) \cdot z_i(t)}{\sum_{i=1}^M r_i(t) + \epsilon}$$

### Mathematical Behavior & Physical Intuition:
1. **Under an Isolated Sensor Fault:** Sensor $k$ degrades ($r_k \to 0$, $z_k \gg 0$). Its contribution $r_k(t) z_k(t) \to 0$. The remaining $M-1$ reliable sensors ($r_j \approx 1.0, z_j \approx 0$) dominate the numerator and denominator, keeping $S_{\text{sys}}(t) < \tau_{\text{sys}}$ and **suppressing the false system alarm**.
2. **Under a Genuine Cyber-Physical Attack:** Attackers manipulate the physical process (e.g., closing valves). The physical state changes. Multiple correlated sensors show elevated residuals $z$, but because their readings match physical conservation laws, their reliability remains high ($r \ge \theta_{\text{rel}}$). Consequently, $S_{\text{sys}}(t)$ spikes, **triggering a legitimate plant alarm**.

---

## 3.11 Sensor Corruption Model (Planned Experimental Suite)

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

## 3.12 Four-Way Operational State Interpretation

Implemented in `src/anomaly_detection/discriminator.py`:
- **Sensor Fault Localization Score:**
  $$S_{\text{fault}}(t) = \max_{i \in \{1 \dots M\}} \left[ (1 - r_i(t)) \cdot z_i(t) \right]$$
  $$\text{FaultySensor}(t) = \arg\max_{i \in \{1 \dots M\}} \left[ (1 - r_i(t)) \cdot z_i(t) \right]$$

### Four Mutually Exclusive Operational Regimes:
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

## 3.13 Experimental Design (Phase 2 Suite)

The planned experimental campaign spans 10 controlled scenarios across 3 random seeds (`seed = 42, 101, 2024`):
- **E1:** Clean SWaT Benchmark (Pristine baseline).
- **E2–E4:** Low (0.1), Medium (0.3), High (0.5) Gaussian Noise.
- **E5–E6:** Single Stuck Sensor (`LIT101`) and Multi-Stuck Sensors (Stages P1, P2, P3).
- **E7–E8:** Incipient (0.1) and Severe (0.5) Calibration Drift (`FIT101`).
- **E9:** Impulse Spikes (2% frequency, 3x amplitude).
- **E10:** Mixed Harsh Operational Environment.

---

## 3.14 Evaluation Metrics

In accordance with time-series evaluation integrity guidelines [40], [41]:
- **System Anomaly Detection Metrics:** Raw Point-wise Precision ($P$), Recall ($R$), F1-Score ($F_1$), Precision-Recall AUC (PR-AUC), and False Positive Rate on clean periods ($\text{FPR}$).
- **Robustness & Reliability Metrics:**
  - False Alarm Suppression Rate ($\text{FASR}$):
    $$\text{FASR} = 1 - \frac{\text{FPR}_{\text{model}}^{\text{corrupt}} - \text{FPR}_{\text{model}}^{\text{clean}}}{\text{FPR}_{\text{baseline}}^{\text{corrupt}} - \text{FPR}_{\text{baseline}}^{\text{clean}}}$$
  - Reliability Separation Margin ($\Delta r = \bar{r}_{\text{clean}} - \bar{r}_{\text{corrupted}}$).
  - Sensor Fault Localization Precision, Recall, and F1 comparing $\arg\max_i [(1 - r_i) z_i]$ against $M_{\text{fault}}(t, i)$.
- **Point-Adjusted F1 (PA):** Reported strictly as a labeled secondary metric (`PA-F1`) with explicit documentation of its optimistic bias [40].

---

## 3.15 Threshold Selection Protocol

To guarantee zero test-label leakage [41]:
1. Model weights are trained strictly on the 80% normal training split.
2. System scores $S_{\text{sys}}(t)$ are evaluated on the 20% normal validation split.
3. The system anomaly threshold $\tau_{\text{sys}}$ is calibrated at the 95th percentile of normal validation scores:
   $$\tau_{\text{sys}} = \text{Quantile}(S_{\text{val}}, 0.95)$$
4. The reliability threshold is calibrated on validation reliability:
   $$\theta_{\text{rel}} = \max\left(0.40, \text{Quantile}(\min_i r_{i, \text{val}}, 0.01)\right)$$
5. **Thresholds are permanently frozen** before evaluating the test sequence blindly.

---

## 3.16 Experimental Comparison Plan

The core scientific comparison isolates the contribution of the Sensor Reliability Module:
- Comparing **Baseline C (RSTNet without SRM)** vs. **Proposed R-STNet (with SRM)** isolates the exact impact of reliability-weighted scoring under identical neural backbone capacity.
- Comparing against **Baseline A (Isolation Forest)** and **Baseline B (ST-AE)** benchmarks classical and deep reconstruction behavior.
- Comparing against **Baseline 3 (CAROTS)** provides external state-of-the-art reference under ICML 2025 conventions.

---

## 3.17 Ablation Study Plan

Four model variants will be evaluated on SWaT under Clean (E1), High Noise (E4), and Stuck Sensor (E5):
1. **Variant A (No SRM):** Uniform residual mean.
2. **Variant B (Temporal SRM Only):** Evaluates $v_i(t)$ alone.
3. **Variant C (Graph SRM Only):** Evaluates $c_i(t)$ alone.
4. **Variant D (Full Triangulated R-STNet):** Complete $v_i(t) + c_i(t) + p_i(t)$ fusion.

Answers Hypothesis 3 regarding the necessity of multi-signal evidential triangulation.

---

## 3.18 Computational Environment & Reproducibility

- **Operating System:** Windows 11 (64-bit).
- **Compute Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4096 MiB VRAM), Driver 592.82, CUDA 12.1 runtime.
- **Python Runtime:** Python 3.11.0 strictly isolated in virtual environment `.venv/`.
- **Frameworks:** PyTorch 2.1.0+cu121, NumPy 1.24.3, Pandas 2.0.3, Scikit-learn 1.3.0.
- **Memory Footprint:** Sequence window $L=60$, batch size $B=64$. Peak allocated VRAM during forward/backward passes is measured at $< 1.2\text{ GB}$, fitting well within the 4.0 GB physical hardware limit.
- **Reproducibility:** Centrally seeded RNG (`seed = 42, 101, 2024`), deterministic cuDNN backends, and configuration-driven execution via YAML files (`configs/swat_rstnet.yaml`).
