# System Architecture Specification: Reliability-Aware Anomaly Detection (R-STNet)

**Document Type:** Architectural Contract  
**Document Status:** Locked & Approved (Phase 0)  
**Primary Model Name:** **R-STNet (Dual-Perspective Reliability-Aware Spatio-Temporal Network)**  
**Target Environments:** SWaT (Primary: 51 channels), WADI (Secondary: 123 channels)

> [!IMPORTANT]
> **Architectural Positioning:**  
> R-STNet is **our proposed project architecture and framework**, not an established previously published paper. It draws inspiration from established components (dilated convolutions/GRU, graph attention, analytical redundancy, and residual evaluation), but the unified combination—specifically the explicit online Sensor Reliability Module (SRM), the reliability-weighted anomaly aggregation, and the 4-way operational state discrimination engine—constitutes our proposed research contribution.

---

## 1. Executive Summary & Architectural Motivation

Conventional multivariate time-series anomaly detection pipelines (including Autoencoders, VAEs, and Isolation Forests) aggregate individual channel residuals into a single global scalar anomaly score:
$$S_{\text{global}}(t) = \sum_{i=1}^M \mathcal{L}(x_i(t), \hat{x}_i(t))$$

When a single sensor experiences an isolated operational fault—such as high Gaussian noise, sensor drift, a stuck reading, or sudden electrical spikes—its individual residual blows up. This causes $S_{\text{global}}(t)$ to exceed the alarm threshold, incorrectly triggering a **CRITICAL SYSTEM ANOMALY** alert. In industrial cyber-physical systems (such as water purification or power generation), this produces costly false shutdowns.

**R-STNet** solves this problem by decoupling **sensor trustworthiness** from **system-state integrity**. It introduces an explicit **Sensor Reliability Module (SRM)** that estimates a continuous reliability vector $\mathbf{r}(t) \in [0, 1]^M$ for all $M$ sensors. It then dynamically down-weights untrusted sensors during system anomaly aggregation, while simultaneously flagging the degraded sensor for maintenance.
The reliability score formulation is a **research hypothesis** whose weights and thresholds are not arbitrary constants, but are systematically calibrated against normal validation behavior and controlled synthetic perturbations.

---

## 2. End-to-End System Pipeline

```
┌────────────────────────────────────────────────────────────────────────┐
│                      RAW MULTIVARIATE SENSOR DATA                      │
│                  X(t) = [x_1(t), x_2(t), ..., x_M(t)]                  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   PREPROCESSING & WINDOW GENERATOR                     │
│  - Robust Z-score / MinMax scaling (fit strictly on normal training)   │
│  - Chronological sliding window: W_t = [X(t - L + 1), ..., X(t)]       │
│  - Controlled corruption injector (for robustness evaluation only)     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  SPATIO-TEMPORAL REPRESENTATION BACKBONE               │
│                                                                        │
│   ┌────────────────────────────────┐  ┌─────────────────────────────┐  │
│   │    Temporal Feature Encoder    │  │ Graph Dependency Estimator  │  │
│   │ (Dilated Conv / Bidirectional  │  │ (Learned Adjacency Matrix   │  │
│   │    GRU with Residual Skips)    │  │  A_ij + Dynamic Attention)  │  │
│   └───────────────┬────────────────┘  └──────────────┬──────────────┘  │
│                   └───────────────┬──────────────────┘                 │
│                                   ▼                                    │
│                     Joint Latent Tensor H(t)                           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    SENSOR RELIABILITY MODULE (SRM)                     │
│                                                                        │
│  Evaluates 3 complementary signals for each sensor i:                  │
│                                                                        │
│  1. Signal Continuity / Temporal Volatility:                           │
│     - Rate-of-change limit checks, dead-band/stuck sensor detection    │
│  2. Graph Neighborhood Agreement (Cross-Sensor Consistency):           │
│     - Deviation between x_i(t) and predictions from correlated sensors │
│  3. Standardized Residual Deviation:                                   │
│     - Rolling Z-score against historical normal distribution           │
│                                                                        │
│  Calibrated Logistic / Evidential Fusion:                              │
│       r_i(t) in [0, 1]  (1.0 = Highly Reliable, 0.0 = Corrupted)       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               RELIABILITY-AWARE SYSTEM ANOMALY AGGREGATOR              │
│                                                                        │
│   Weighted System Anomaly Score:                                       │
│          S_sys(t) = ( sum_i r_i(t) * e_i(t) ) / ( sum_i r_i(t) + eps ) │
│                                                                        │
│   Sensor Fault Score:                                                  │
│          S_fault(t) = max_i [ (1 - r_i(t)) * (e_i(t) / MAD_i) ]       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               FOUR-WAY DISCRIMINATION DECISION ENGINE                  │
│                                                                        │
│   Case A: S_sys < tau_sys  AND  min_i r_i >= theta_rel  --> NORMAL     │
│   Case B: S_sys < tau_sys  AND  min_i r_i <  theta_rel  --> SENSOR FAULT│
│   Case C: S_sys >= tau_sys AND  mean r_i >= theta_rel  --> SYSTEM ANOMALY
│   Case D: S_sys >= tau_sys AND  mean r_i <  theta_rel  --> SYSTEM ANOMALY
│                                                         + SENSOR WARNING│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      INTRINSIC EXPLANATION LAYER                       │
│  - Per-sensor reliability vector: r_i(t) for all i in [1..M]           │
│  - Top-k suspicious channels rank-ordered                              │
│  - Specific corruption signature identified (Drift, Spike, Stuck, Noise)│
│  - Graph edge attribution: which physical neighbor bonds were broken   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Mathematical Formulation of Key Components

### 3.1. Spatio-Temporal Prediction & Residuals
For a multivariate sequence window $\mathbf{W}_t \in \mathbb{R}^{L \times M}$, the backbone generates a one-step-ahead forecast $\hat{\mathbf{X}}(t)$ and/or reconstruction $\tilde{\mathbf{X}}(t)$.  
The instantaneous prediction error for channel $i$ is:
$$e_i(t) = |x_i(t) - \hat{x}_i(t)|$$

To normalize across heterogeneously scaled sensors (e.g. differential pressure DPIT vs flow FIT vs pH analyzer AIT), we compute the Median Absolute Deviation (MAD) over clean validation data:
$$z_i(t) = \frac{e_i(t) - \text{Median}(e_i)}{\text{MAD}_i + \epsilon}$$

### 3.2. Sensor Reliability Module (SRM) Formulation
Reliability $r_i(t) \in [0, 1]$ is estimated from three physically motivated components:

1. **Temporal Volatility Score ($v_i(t)$):**
   Detects stuck sensors (zero change over time) and abrupt unphysical step changes (derivative explosion):
   $$\Delta x_i(t) = |x_i(t) - x_i(t-1)|$$
   $$v_i(t) = \exp\left( - \frac{\max(0, \Delta x_i(t) - \delta_i^{\max})^2}{2 \sigma_{\Delta, i}^2} \right) \cdot \mathbb{I}(\text{Var}(\mathbf{x}_{i, t-K:t}) > \epsilon_{\text{stuck}})$$

2. **Cross-Sensor Neighborhood Agreement ($c_i(t)$):**
   In cyber-physical systems, sensors are bound by physical laws (conservation of mass, energy, Bernoulli relations). If channel $j$ is connected to $i$ with learned adjacency weight $A_{ji}$, then neighbor channels provide a redundant estimator $\hat{x}_i^{\mathcal{N}}(t) = \sum_{j \in \mathcal{N}(i)} \frac{A_{ji}}{\sum_k A_{ki}} x_j(t)$.
   $$c_i(t) = \exp\left( - \frac{|x_i(t) - \hat{x}_i^{\mathcal{N}}(t)|}{\lambda_c \cdot \text{MAD}_i} \right)$$

3. **Residual Confidence ($p_i(t)$):**
   Continuous probability based on standard residual Z-scores:
   $$p_i(t) = 2 \cdot \left(1 - \Phi(z_i(t))\right) = \text{erfc}\left( \frac{z_i(t)}{\sqrt{2}} \right)$$

4. **Composite Reliability Score ($r_i(t)$):**
   The combined trust score is computed via a smooth evidential product:
   $$r_i(t) = \sigma\left( w_v \cdot \ln\left(\frac{v_i(t)}{1 - v_i(t) + \epsilon}\right) + w_c \cdot \ln\left(\frac{c_i(t)}{1 - c_i(t) + \epsilon}\right) + w_p \cdot \ln\left(\frac{p_i(t)}{1 - p_i(t) + \epsilon}\right) + b_r \right)$$
   where weights $w_v, w_c, w_p, b_r$ are calibrated on validation splits with synthetic perturbation.

### 3.3. Reliability-Weighted System Anomaly Score
Standard detectors sum all errors uniformly:
$$S_{\text{unweighted}}(t) = \frac{1}{M} \sum_{i=1}^M z_i(t)$$
When sensor $k$ fails ($z_k(t) \to \infty$), $S_{\text{unweighted}}(t)$ causes a false system alarm.

In **R-STNet**, we compute the reliability-weighted aggregation:
$$S_{\text{sys}}(t) = \frac{\sum_{i=1}^M r_i(t) \cdot z_i(t)}{\sum_{i=1}^M r_i(t) + \epsilon}$$
When sensor $k$ is degraded ($r_k(t) \to 0$), its contribution to $S_{\text{sys}}(t)$ drops towards zero, insulating the system-level alarm from sensor corruption!

### 3.4. Sensor Fault Discrimination Engine
To explicitly identify when a sensor has failed even while the system is healthy, we define the **Sensor Fault Score**:
$$S_{\text{fault}}(t) = \max_{i \in \{1 \dots M\}} \left( (1 - r_i(t)) \cdot z_i(t) \right)$$
$$\text{FaultySensor}(t) = \arg\max_{i \in \{1 \dots M\}} \left( (1 - r_i(t)) \cdot z_i(t) \right)$$

The operational state is classified into four mutually exclusive operational regimes:
- **CASE A (NORMAL):**  
  $S_{\text{sys}}(t) < \tau_{\text{sys}}$ **AND** $\min_i r_i(t) \ge \theta_{\text{rel}}$  
  *Interpretation:* Physical process is in control; all sensors are functioning within verified bounds.
- **CASE B (SENSOR FAULT):**  
  $S_{\text{sys}}(t) < \tau_{\text{sys}}$ **AND** $\min_i r_i(t) < \theta_{\text{rel}}$  
  *Interpretation:* Physical plant is safe and operating stably; specific sensor(s) have failed or drifted. Flagged for instrument maintenance without false emergency shutdown.
- **CASE C (SYSTEM ANOMALY):**  
  $S_{\text{sys}}(t) \ge \tau_{\text{sys}}$ **AND** $\frac{1}{M} \sum_i r_i(t) \ge \theta_{\text{rel}}$  
  *Interpretation:* Genuine physical anomaly (e.g. cyber-attack altering valve positions, physical leak, tank overflow). Sensors are trustworthy, confirming real process deviation.
- **CASE D (SYSTEM ANOMALY WITH SENSOR DEGRADATION):**  
  $S_{\text{sys}}(t) \ge \tau_{\text{sys}}$ **AND** $\frac{1}{M} \sum_i r_i(t) < \theta_{\text{rel}}$  
  *Interpretation:* Physical system is anomalous while one or more monitoring instruments are concurrently impaired. Highest operational risk alert.

---

## 4. Evaluation of the Initial Presentation Proposal vs. R-STNet

| Criterion | Initial Presentation Concept (`Autoencoder -> Isolation Forest -> SHAP`) | Proposed Architecture (**R-STNet**) | Justification for Enhancement |
| :--- | :--- | :--- | :--- |
| **Sensor Reliability Modeling** | **None.** Operates on unweighted latent vectors or residuals. | **Explicit SRM.** Computes continuous trustworthiness vector $\mathbf{r}_t \in [0, 1]^M$. | Solves the primary project question: conventional models cannot decouple sensor faults from system anomalies. |
| **False Alarm Suppression** | **Poor.** A single corrupted sensor inflates total reconstruction error and triggers Isolation Forest. | **Strong.** Reliability weighting discounts corrupted channels from system anomaly score. | Direct experimental validation under controlled noise, spikes, drift, and stuck sensors. |
| **Temporal Dynamics** | Flattened dense autoencoder ignores sequential order and multi-scale temporal lags. | Dilated Temporal Convolutions / Bi-GRU captures autoregressive lags and physical settling time. | Industrial testbeds (SWaT/WADI) exhibit inertia and transport delays between stages. |
| **Inter-Sensor Physics** | Implicitly mixed in latent bottleneck; cannot audit pair-wise consistency. | Graph attention / learned physical topology explicitly checks physical conservation laws. | If a pump starts, flow downstream must rise. Disagreement indicates sensor failure. |
| **Explainability** | Post-hoc SHAP on tabular representations. Prohibitively slow for continuous time-series (500k points). | **Intrinsic explainability.** Per-sensor reliability $r_i$, residual contribution $z_i$, and edge violation directly computed. | Real-time diagnostic capability without expensive kernel approximations. |
| **Scientific Defensibility** | Standard tutorial combination; vulnerable to academic review critique ("what is novel?"). | Formulates an explicit reliability-aware mechanism with 4-way operational state discrimination. | Scientifically defensible contribution suitable for rigorous peer/academic evaluation. |

---

## 5. Computational Feasibility & Hardware Constraints

- **Host GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (4096 MiB VRAM).
- **VRAM Budget Allocation:**
  - Model parameters: $\sim 1.5\text{M}$ to $3.5\text{M}$ parameters ($\approx 15\text{ MB}$ FP32).
  - Optimizer states (AdamW): $\approx 30\text{ MB}$.
  - Forward/Backward Activations (Batch size 64, Window length 60, Channels 51): $\approx 180\text{ MB}$.
  - Total Peak VRAM during training: $< 1.2\text{ GB}$, leaving ample headroom under the 4.0 GB limit.
- **CPU Fallback:** Fully vectorizable in pure PyTorch with standard matrix operations; complete pipeline is executable on multi-core CPU if CUDA is unavailable.

---

## 6. Implementation Modules & Directory Mapping

- `src/data/`: `swat_loader.py`, `wadi_loader.py` (Dataset ingestion and chronological partitioning).
- `src/preprocessing/`: `scaler.py`, `windowing.py` (Robust scaling and sliding window generation).
- `src/corruption/`: `corruptor.py` (Synthetic missing, noise, spike, drift, stuck, multi-sensor corruption).
- `src/models/`:
  - `spatiotemporal_backbone.py` (TCN / Bi-GRU + Graph Attention).
  - `reliability_module.py` (Temporal continuity, neighborhood agreement, calibrated fusion).
  - `rst_net.py` (End-to-end R-STNet model class).
- `src/anomaly_detection/`: `discriminator.py` (4-way state decision logic, threshold calibration).
- `src/evaluation/`: `metrics.py`, `robustness.py` (Precision, Recall, F1, PR-AUC, false alarm rate under corruption).
- `src/visualization/`: `plots.py`, `explanation.py` (Sensor trust waterfall plots, diagnostic dashboard).
