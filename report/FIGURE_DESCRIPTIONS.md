# Figure Descriptions for Review 1 Report

**Project:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Review Target:** Review 1 (Phase 1)  
**Format:** Detailed figure captions, architectural narratives, and academic explanations for Figures 1 through 7.

---

### Figure 1: Conceptual Dilemma Between Observation Corruption and Process Anomalies
- **File:** `report/FIGURES/01_problem_concept.mmd`
- **Short Caption:** The fundamental dilemma of multivariate sensor monitoring: distinguishing observation corruptions from physical process anomalies.
- **Detailed Narrative & Academic Explanation:**  
  Figure 1 illustrates the foundational dilemma encountered in cyber-physical systems (CPS) when an abnormal reading is detected. Physical sensors in facilities like water treatment plants (e.g., flow transmitters, level sensors, differential pressure gauges) feed continuous telemetry into automated monitoring systems. When a reading deviates substantially from nominal expectations, conventional multivariate detectors (e.g., GDN, USAD, OmniAnomaly) sum errors uniformly across all channels, automatically mapping any severe deviation into a plant-wide critical emergency alarm. 
  
  In reality, the deviation can originate from two completely different physical pathways:
  1. **A genuine cyber-physical attack or process failure** (e.g., unauthorized valve closure, pump shutdown, tank overflow), where physical laws ensure that multiple correlated sensors respond in a coordinated fashion.
  2. **An isolated sensor degradation** (e.g., high Gaussian noise from electromagnetic interference, calibration drift from aging, or mechanical freezing into a stuck constant), where the underlying physical process remains in complete control and healthy equilibrium.
  
  Under the conventional paradigm, sensor degradation causes catastrophic false alarms and costly unneeded shutdowns. In contrast, the proposed reliability-aware paradigm (R-STNet) evaluates the trustworthiness of the sensor stream online, down-weights the corrupted channel during system-level scoring, suppresses the false emergency shutdown, and directs maintenance personnel specifically to the degraded instrument.

---

### Figure 2: End-to-End Scientific Research Framework
- **File:** `report/FIGURES/02_research_framework.mmd`
- **Short Caption:** Conceptual research framework mapping literature foundations to the identified research gap, proposed methodology, and planned evaluation.
- **Detailed Narrative & Academic Explanation:**  
  Figure 2 demonstrates the logical research progression governing this study across four structured phases:
  1. **Literature Audit (1998–2025):** Synthesizes classical statistical anomaly detection (Isolation Forest, DPCA), deep temporal autoencoders (USAD, OmniAnomaly, TranAD), spatio-temporal graph neural networks (GDN, MTAD-GAT), causal representation learning (CAROTS), and control-theoretic fault diagnosis (FDI).
  2. **Identified Research Gap:** Formulates the specific unaddressed challenge: current deep multivariate detectors lack an online channel-level reliability mechanism, causing unweighted residual aggregations to trigger false alarms under sensor-specific corruptions, while benchmark datasets lack independent sensor-fault ground truth.
  3. **Proposed Methodological Framework (R-STNet):** Resolves this gap through a unified deep architecture integrating a spatio-temporal graph backbone, an explicit multi-signal Sensor Reliability Module (SRM), reliability-weighted residual aggregation, and a 4-way operational state discrimination engine.
  4. **Planned Experimental Evaluation (Phase 2):** Outlines the rigorous scientific protocol to validate the framework using a 10-experiment controlled corruption suite, multi-seed deterministic runs, fair baseline comparisons, and leakage-free validation threshold calibration.

---

### Figure 3: Overall Research Methodology and Data Processing Pipeline
- **File:** `report/FIGURES/03_overall_methodology.mmd`
- **Short Caption:** End-to-end data processing, model inference, and operational decision workflow.
- **Detailed Narrative & Academic Explanation:**  
  Figure 3 traces the step-by-step data and tensor transformations throughout the proposed system:
  - **Stage 1 (Data Ingestion & Chronological Split):** Raw SWaT telemetry is ingested from `reference/swat_data/`, downsampled chronologically by a factor of 10 to preserve physical settling dynamics, and partitioned into an 80% normal training split, a 20% normal validation split (reserved strictly for threshold and calibration parameter tuning), and a mixed attack test split.
  - **Stage 2 (Preprocessing & Windowing):** A `RobustStandardScaler` is fitted strictly on the training split to eliminate data leakage. Chronological sliding windows of length $L=60$ are constructed.
  - **Stage 3 (Spatio-Temporal Backbone):** 1D dilated convolutions capture temporal lags and physical inertia, while a learned graph attention module captures cross-sensor physical dependencies (mass and volume balances).
  - **Stage 4 (Sensor Reliability Module):** Three independent physical signals (temporal continuity, graph neighborhood consistency, and tail confidence) are evaluated and fused via calibrated logit fusion into a reliability vector $\mathbf{r}(t) \in [0, 1]^M$.
  - **Stage 5 (Decision Engine):** The system computes a reliability-weighted system anomaly score $S_{\text{sys}}(t)$ and a sensor fault localization score $S_{\text{fault}}(t)$, feeding a 4-way classifier that produces operational recommendations.

---

### Figure 4: Detailed Architecture of the Proposed R-STNet Framework
- **File:** `report/FIGURES/04_rstnet_architecture.mmd`
- **Short Caption:** Neural layer architecture of R-STNet detailing temporal convolutions, graph attention, forecasting head, and integrated SRM.
- **Detailed Narrative & Academic Explanation:**  
  Figure 4 provides the deep learning architectural specification implemented in `src/models/rstnet.py`:
  - **Temporal Encoder:** Permutes the input window tensor $\mathbf{W}_t \in \mathbb{R}^{B \times L \times M}$ into $(B \cdot M, 1, L)$, passing each sensor channel through two sequential 1D convolutional layers with kernel size 3, dilation factor 2, and a residual skip connection, followed by linear projection into node embedding dimension $d=32$.
  - **Graph Attention Block:** Maintains two learnable node embedding dictionaries $\mathbf{E}_1, \mathbf{E}_2 \in \mathbb{R}^{M \times 32}$. Evaluates directed similarity $\mathbf{S} = \text{ReLU}(\mathbf{E}_1 \mathbf{E}_2^T)$, zeros out self-loops, sparsifies the graph by retaining the Top-$K$ ($K=15$) strongest physical relationships, and normalizes incoming weights via Softmax to execute graph message passing.
  - **Joint Output Head:** Concatenates temporal and spatial latent matrices into $[\mathbf{H}_{\text{temp}} \,\|\, \mathbf{H}_{\text{spat}}] \in \mathbb{R}^{B \times M \times 64}$, feeding a multi-layer perceptron to forecast one-step-ahead expected readings $\hat{\mathbf{X}}(t)$.
  - **SRM & Output Heads:** Computes normalized prediction residuals $z_i(t)$, evaluates the three reliability components, and produces the final dual scores: $S_{\text{sys}}(t)$ and $S_{\text{fault}}(t)$.

---

### Figure 5: Triangulated Sensor Reliability Module (SRM) Formulation
- **File:** `report/FIGURES/05_sensor_reliability_module.mmd`
- **Short Caption:** Mathematical structure of the Sensor Reliability Module illustrating three-signal triangulation and calibrated evidential logit fusion.
- **Detailed Narrative & Academic Explanation:**  
  Figure 5 illustrates how channel-level sensor reliability $r_i(t) \in [0, 1]$ is estimated online without resorting to arbitrary rules:
  1. **Temporal Continuity ($v_i$):** Evaluates instantaneous rate-of-change violations against validation quantile bounds ($\delta_i^{\max}$) using an exponential penalty for spikes, combined with a moving variance check over lookback window $K$ to detect frozen/stuck telemetry.
  2. **Neighborhood Consistency ($c_i$):** Reconstructs an analytical estimate of sensor $i$ from its physical graph neighbors $\hat{x}_i^{\mathcal{N}}(t) = \sum_j W_{ji} x_j(t)$. Significant deviation indicates that sensor $i$ has broken physical conservation laws with its topological cluster.
  3. **Residual Tail Confidence ($p_i$):** Computes the Gaussian tail probability of the Median Absolute Deviation (MAD)-standardized residual $z_i(t)$ against the clean validation distribution.
  4. **Calibrated Evidential Fusion:** Transforms component scores into logit space, computes a weighted evidential sum, and applies a logistic sigmoid function to yield a smooth, continuous reliability score $r_i(t) \in [0, 1]$.

---

### Figure 6: Four-Way Operational State Discrimination Logic
- **File:** `report/FIGURES/06_four_state_framework.mmd`
- **Short Caption:** Decision tree for the 4-way operational state discrimination engine.
- **Detailed Narrative & Academic Explanation:**  
  Figure 6 details the multi-perspective decision engine implemented in `src/anomaly_detection/discriminator.py`:
  - The engine continuously monitors two calibrated indicators: the reliability-weighted system anomaly score $S_{\text{sys}}(t)$ against threshold $\tau_{\text{sys}}$, and the channel reliability vector $\mathbf{r}(t)$ against threshold $\theta_{\text{rel}}$.
  - **Case A (Normal Operation):** $S_{\text{sys}} < \tau_{\text{sys}}$ and $\min_i r_i \ge \theta_{\text{rel}}$. The physical process is healthy and all instruments are operating within nominal limits.
  - **Case B (Isolated Sensor Fault):** $S_{\text{sys}} < \tau_{\text{sys}}$ and $\min_i r_i < \theta_{\text{rel}}$. The process is healthy, but an isolated sensor has failed, drifted, or stuck. The system suppresses false emergency shutdowns and alerts maintenance with the exact sensor identifier.
  - **Case C (Critical System Anomaly):** $S_{\text{sys}} \ge \tau_{\text{sys}}$ and $\text{mean}_i r_i \ge \theta_{\text{rel}}$. Multiple reliable sensors confirm a physical process violation (e.g., cyber-attack altering valve positions). Triggers an immediate emergency plant intervention.
  - **Case D (System Anomaly with Sensor Degradation):** $S_{\text{sys}} \ge \tau_{\text{sys}}$ and $\text{mean}_i r_i < \theta_{\text{rel}}$. Critical process failure accompanied by concurrent instrument degradation, indicating maximum operational threat.

---

### Figure 7: Controlled Sensor Corruption Matrix and Dual Evaluation Protocol
- **File:** `report/FIGURES/07_experimental_design.mmd`
- **Short Caption:** Overview of the planned 10-experiment controlled corruption suite and dual-perspective evaluation framework.
- **Detailed Narrative & Academic Explanation:**  
  Figure 7 depicts the planned experimental evaluation suite (Phase 2):
  - In real-world cyber-physical datasets like SWaT, attack labels reflect process attacks but provide no ground truth for sensor hardware faults. To resolve this, the controlled corruption suite (`src/corruption/corruptor.py`) injects 9 deterministic failure modes across 4 severity tiers into the evaluation set: Gaussian noise (E2–E4), stuck sensors (E5–E6), calibration drift (E7–E8), impulse spikes (E9), and mixed harsh corruptions (E10).
  - This architecture generates **dual ground truth**: official SWaT attack labels $y_{\text{sys}}(t)$ for system anomalies, and synthetic binary masks $M_{\text{fault}}(t, i)$ for sensor-level fault localization.
  - Evaluates all five candidate models (Isolation Forest, ST-AE, RSTNet without SRM, CAROTS, and proposed R-STNet) across system detection metrics (Precision, Recall, F1, PR-AUC, FPR) and reliability metrics (FASR, Reliability Separation $\Delta r$, and Sensor Fault F1) across 3 deterministic random seeds (42, 101, 2024).
