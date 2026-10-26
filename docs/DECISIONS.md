# Architecture Decision Records (ADR)

**Project:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Repository:** `Reliability-Aware-Anomaly-Detection`  
**Discipline:** ML Research & System Architecture  

---

## Record Index

- [ADR-001: Selection of Primary Research Architecture (R-STNet)](#adr-001-selection-of-primary-research-architecture-r-stnet)
- [ADR-002: Dataset Prioritization and Benchmark Selection](#adr-002-dataset-prioritization-and-benchmark-selection)
- [ADR-003: Triangulated Sensor Reliability Formulation](#adr-003-triangulated-sensor-reliability-formulation)
- [ADR-004: Four-Way State Discrimination Engine](#adr-004-four-way-state-discrimination-engine)
- [ADR-005: Hardware Feasibility and 4GB VRAM Memory Budgeting](#adr-005-hardware-feasibility-and-4gb-vram-memory-budgeting)
- [ADR-006: Academic Attribution and Decoupling of Reference Code](#adr-006-academic-attribution-and-decoupling-of-reference-code)
- [ADR-007: Python 3.11 Runtime and PyTorch CUDA Environment](#adr-007-python-311-runtime-and-pytorch-cuda-environment)

---

### ADR-001: Selection of Primary Research Architecture (R-STNet)

- **Date:** 2026-10-21
- **Status:** **APPROVED & LOCKED**
- **Context:**  
  The initial presentation concept proposed `Preprocessing -> Autoencoder -> Isolation Forest -> SHAP`. However, an autoencoder + Isolation Forest evaluates global residual magnitude without assessing individual sensor trustworthiness. Under single-sensor corruption (e.g. noise, drift, spikes), global residuals spike and trigger severe false system alarms. Furthermore, computing SHAP post-hoc across 500k time points is computationally prohibitive and cannot prevent false alarms during online inference.
- **Options Considered:**
  1. *Option 1:* Strictly implement the presentation proposal (`Autoencoder -> Isolation Forest -> SHAP`).
  2. *Option 2:* Direct adoption of CAROTS (Causal Contrastive Learning).
  3. *Option 3:* Large Transformer architectures (e.g. Anomaly Transformer, PatchTST).
  4. *Option 4 (Chosen):* **R-STNet (Dual-Perspective Reliability-Aware Spatio-Temporal Network)**.
- **Chosen Option:** Option 4 (R-STNet).
- **Reason:**  
  R-STNet decouples sensor trustworthiness from system anomaly detection via an explicit Sensor Reliability Module (SRM). It incorporates temporal dynamics (Dilated Convolutions / Bi-GRU) and inter-sensor physical topology (Graph Attention). The reliability scores dynamically down-weight untrusted sensors, directly validating the core research hypothesis: reducing false positives under sensor degradation.
- **Trade-offs:**  
  Requires training a graph-attention spatio-temporal model and calibrating the reliability fusion parameters on a validation split, rather than using an off-the-shelf tabular Isolation Forest.

---

### ADR-002: Dataset Prioritization and Benchmark Selection

- **Date:** 2026-10-21
- **Status:** **APPROVED & LOCKED**
- **Context:**  
  Multiple time-series anomaly detection benchmarks exist in literature: SWaT, WADI, PSM, SMD, SMAP, and MSL. Downloading and preprocessing all datasets at once introduces clutter, storage overhead, and risks losing focus on the primary cyber-physical industrial sensor problem.
- **Options Considered:**
  1. *Option 1:* Ingest all 6 datasets simultaneously.
  2. *Option 2:* Use synthetic datasets only (e.g., Lorenz96, VAR).
  3. *Option 3 (Chosen):* Two-tiered real-world industrial dataset protocol:
     - **Primary:** **SWaT (Secure Water Treatment)** — 51 continuous sensors/actuators across 6 treatment stages.
     - **Secondary:** **WADI (Water Distribution)** — 123 sensors/actuators across larger distribution network for external validation.
     - **Deferred:** PSM (eBay servers) and SMD (server clusters) kept documented in `docs/DATA_SETUP.md` for potential expansion.
- **Chosen Option:** Option 3.
- **Reason:**  
  SWaT and WADI are real cyber-physical water infrastructure testbeds where physical relationships (flow, tank level, pump status, pressure) strictly govern normal operations. They are the ideal platforms to distinguish sensor instrument faults from true physical process failures.
- **Trade-offs:**  
  SWaT and WADI are large (~400MB+ raw CSVs), requiring structured downsampling (downsample rate 5 or 10) consistent with established benchmark literature.

---

### ADR-003: Triangulated Sensor Reliability Formulation

- **Date:** 2026-10-21
- **Status:** **APPROVED & LOCKED**
- **Context:**  
  How should a sensor's reliability $r_i(t) \in [0, 1]$ be mathematically defined without producing arbitrary or ungrounded percentages?
- **Options Considered:**
  1. *Option 1:* Binary threshold on reconstruction error ($r_i = 1$ if $e_i < \tau$ else $0$).
  2. *Option 2:* Rule-based heuristics only (min/max range and zero variance check).
  3. *Option 3 (Chosen):* **Triangulated Multi-Signal Evidential Formulation**:
     - *Temporal Volatility ($v_i$):* Evaluates physical rate-of-change and dead-band/stuck sensor behavior.
     - *Graph Neighborhood Agreement ($c_i$):* Compares observed value with reconstruction from physically correlated neighbor channels.
     - *Standardized Residual Z-Score ($p_i$):* Gaussian tail probability of the prediction error against clean validation distribution.
     - *Calibrated Logistic Fusion:* Smooth mapping into $[0, 1]$.
- **Chosen Option:** Option 3.
- **Reason:**  
  Provides mathematically defensible, continuous reliability scores. If a single sensor spikes or sticks, neighbors remain consistent, pulling down only that sensor's reliability $r_i(t)$ while preserving overall system monitoring.
- **Trade-offs:**  
  Requires computing neighbor predictions via graph weights, introducing slight computational overhead over a naive rule check.

---

### ADR-004: Four-Way State Discrimination Engine

- **Date:** 2026-10-21
- **Status:** **APPROVED & LOCKED**
- **Context:**  
  The system must not only output an anomaly score, but also make a structured operational diagnosis distinguishing sensor faults from system anomalies.
- **Options Considered:**
  1. *Option 1:* Single binary threshold ($\text{Anomaly} \in \{0, 1\}$).
  2. *Option 2:* Post-hoc heuristic classification.
  3. *Option 3 (Chosen):* **Four-Way Operational State Regime**:
     - **Case A (NORMAL):** Healthy system, reliable sensors.
     - **Case B (SENSOR FAULT):** Healthy system, one or more degraded sensors. Suppresses false system shutdown; alerts maintenance.
     - **Case C (SYSTEM ANOMALY):** Process anomaly confirmed by reliable sensors.
     - **Case D (SYSTEM ANOMALY + SENSOR DEGRADATION):** Critical process fault accompanied by instrument impairment.
- **Chosen Option:** Option 3.
- **Reason:**  
  Directly operationalizes the core research hypothesis and provides actionable industrial intelligence.
- **Trade-offs:**  
  Requires establishing calibrated dual thresholds: system anomaly threshold $\tau_{\text{sys}}$ and reliability threshold $\theta_{\text{rel}}$. Both are determined systematically on validation splits.

---

### ADR-005: Hardware Feasibility and 4GB VRAM Memory Budgeting

- **Date:** 2026-10-21
- **Status:** **APPROVED & LOCKED**
- **Context:**  
  The target execution hardware is an NVIDIA GeForce RTX 3050 Laptop GPU with 4096 MiB VRAM. Models with high quadratic attention memory (e.g. dense multi-head transformers across 1000 steps) will cause Out-of-Memory (OOM) crashes.
- **Options Considered:**
  1. *Option 1:* Mandate external cloud GPUs (A100/V100).
  2. *Option 2:* Run on CPU only with severe downsampling.
  3. *Option 3 (Chosen):* **Budget-Constrained Local Model Architecture**:
     - Window size $L = 60$ (representing 5 minutes of physical process at 5s downsampling).
     - Batch size $B = 64$.
     - Temporal Dilated Convolutions / Bi-GRU backbone ($\approx 1.5\text{M}$ parameters).
     - Peak VRAM footprint $< 1.2\text{ GB}$, fitting comfortably within 4.0 GB VRAM.
- **Chosen Option:** Option 3.
- **Reason:**  
  Guarantees rapid experimentation, local reproducibility, and zero dependency on expensive cloud infrastructure.
- **Trade-offs:**  
  Sequence context window is bounded at $L = 60$ to $100$ steps rather than very long horizons.

---

### ADR-006: Academic Attribution and Decoupling of Reference Code

- **Date:** 2026-10-21
- **Status:** **APPROVED & LOCKED**
- **Context:**  
  The workspace includes `reference/CAROTS-master` (Kim et al., ICML 2025) and other reference implementations. We must define how this code is utilized.
- **Options Considered:**
  1. *Option 1:* Directly modify and build the new project inside `reference/CAROTS`.
  2. *Option 2:* Conceal the existence of reference code.
  3. *Option 3 (Chosen):* **Strict Research Decoupling with Full Academic Attribution**:
     - `reference/` remains read-only, pristine reference code and baseline benchmark.
     - The proposed project is implemented from scratch under `src/`, with its own module design, data loaders, training pipeline, and reliability mechanisms.
     - External work is properly attributed in `docs/references/technical_references.md`.
- **Chosen Option:** Option 3.
- **Reason:**  
  Ensures absolute academic integrity, clarity of original contributions, and prevents dependency spaghetti between reference code and our implementation.
- **Trade-offs:**  
  Requires maintaining independent data loaders and pipeline logic in `src/`.

---

### ADR-007: Python 3.11 Runtime and PyTorch CUDA Environment

- **Date:** 2026-10-21
- **Status:** **APPROVED & LOCKED**
- **Context:**  
  The host environment has Python 3.13 default and Python 3.11 installed. Python 3.13 currently has incomplete CUDA wheel support across PyTorch ecosystem packages. Python 3.11 is the proven, stable baseline for PyTorch + CUDA on Windows.
- **Options Considered:**
  1. *Option 1:* Use system default Python 3.13.
  2. *Option 2 (Chosen):* Create dedicated `.venv` using Python 3.11 (`py -3.11`).
- **Chosen Option:** Option 2.
- **Reason:**  
  Guarantees zero wheel compilation issues for PyTorch, SciPy, Scikit-learn, and CUDA acceleration.
- **Trade-offs:**  
  Requires explicitly activating `.venv` or calling `.venv\Scripts\python.exe`.
