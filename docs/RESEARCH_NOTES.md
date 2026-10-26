# Research Notes: Focused Literature Audit & Positioning

**Project:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Discipline:** Machine Learning Research for Cyber-Physical Systems (CPS)  
**Status:** Audit Completed & Documented  

---

## 1. Focused Literature Audit: Top 8 Closest Approaches

To establish a defensible research positioning, we audit the closest existing methods across multivariate anomaly detection, graph neural networks, and sensor fault diagnosis.

### Paper 1: Graph Deviation Network (GDN)
- **Reference:** Deng, A., & Hooi, B. (2021). *Graph Neural Network-Based Anomaly Detection in Multivariate Time Series.* Proc. IJCAI 2021, pp. 2213–2221.
- **What it does:**  
  Learns a directed graph structure between sensors using node embeddings, forecasts future values of each sensor via graph attention over topological neighbors, and computes anomaly scores from prediction errors.
- **What our system does differently:**  
  GDN sums all sensor prediction errors uniformly into a single global score. When an isolated sensor experiences high noise, drift, or a stuck value, GDN's global score spikes and incorrectly flags a **CRITICAL PROCESS ANOMALY**. Our system (R-STNet) uses the graph neighborhood agreement not merely to forecast values, but to **estimate a continuous sensor trustworthiness score $r_i(t)$**, and **down-weights untrusted sensors** from the system anomaly score.
- **Remaining Gap Addressed:**  
  GDN lacks any sensor reliability mechanism and cannot separate sensor hardware faults from plant process anomalies.

---

### Paper 2: Causality-Aware Contrastive Learning (CAROTS)
- **Reference:** Kim, H., Mok, J., Lee, D., Lew, J., Kim, S., & Yoon, S. (2025). *Causality-Aware Contrastive Learning for Robust Multivariate Time-Series Anomaly Detection.* ICML 2025 (arXiv:2506.03964).
- **What it does:**  
  Estimates causal DAGs via structural equation discovery and enforces contrastive learning between causally consistent augmented views to achieve robustness against distribution shifts on SWaT and WADI.
- **What our system does differently:**  
  CAROTS seeks invariant representations under domain shifts at the latent representation level. It does **not** estimate an explicit per-sensor reliability vector $\mathbf{r}(t) \in [0, 1]^M$, does not pinpoint which sensor is degraded, and does not provide an operational 4-way decision engine.
- **Remaining Gap Addressed:**  
  Explicit online sensor fault localization and dynamic false-alarm discounting.

---

### Paper 3: MTAD-GAT (Multivariate Time-Series Anomaly Detection via GAT)
- **Reference:** Zhao, H., Wang, Y., Duan, J., Huang, C., Cao, D., Tong, Y., & Xu, B. (2020). *Multivariate Time-series Anomaly Detection via Graph Attention Network.* IEEE ICDM 2020, pp. 841–850.
- **What it does:**  
  Combines feature-oriented graph attention (inter-sensor dependencies) with time-oriented graph attention (temporal dependencies) in a joint forecasting and reconstruction framework.
- **What our system does differently:**  
  Like GDN, MTAD-GAT treats all feature errors equally during anomaly scoring. A corrupted sensor directly inflates the overall score. Our framework introduces the Sensor Reliability Module (SRM) on top of the spatio-temporal representations.
- **Remaining Gap Addressed:**  
  Protection of system-level anomaly scoring against individual corrupted sensors.

---

### Paper 4: RANSynCoders (Random Sampling Autoencoders)
- **Reference:** Audibert, J., Michiardi, P., Guyard, F., & Zuluaga, M. A. (2022). *RANSynCoders: Robust Anomaly Detection for Multivariate Time Series.* IEEE Transactions on Knowledge and Data Engineering.
- **What it does:**  
  Inspired by RANSAC, it trains an ensemble of autoencoders on random subsets of sensors and takes bootstrap consensus to detect anomalies in the presence of noisy/corrupted channels.
- **What our system does differently:**  
  RANSynCoders relies on brute-force random subset training, which is computationally expensive, ignores learned physical graph topology, and does not produce an interpretable graph-attributed reliability score. Our R-STNet uses a single unified model that learns the physical topology directly and continuously calibrates sensor reliability.
- **Remaining Gap Addressed:**  
  Principled graph-based reliability estimation with low computational complexity (single model vs. large ensemble).

---

### Paper 5: Stochastic Latent Anomaly Detection (OmniAnomaly)
- **Reference:** Su, Y., Zhao, Y., Niu, C., Liu, R., Sun, W., & Pei, D. (2019). *Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Networks.* ACM KDD 2019, pp. 2828–2837.
- **What it does:**  
  Models temporal dependencies with GRUs and stochastic latent spaces with planar normalizing flows to capture complex normal data distributions.
- **What our system does differently:**  
  OmniAnomaly does not model inter-sensor spatial relationships explicitly and suffers from **latent contamination**: corrupted input features corrupt the latent state $z_t$, corrupting the reconstruction of all other channels. Our SRM evaluates temporal continuity and spatial neighbor consensus before aggregating system status.
- **Remaining Gap Addressed:**  
  Prevention of latent representation contamination from single-channel sensor faults.

---

### Paper 6: Analytical Redundancy and Classical FDI in Cyber-Physical Systems
- **Reference:** Gertler, J. (1998). *Fault Detection and Diagnosis in Engineering Systems.* Marcel Dekker; Isermann, R. (2006). *Fault-Diagnosis Systems.* Springer.
- **What it does:**  
  Uses physical conservation equations (mass/energy balance) to construct parity relations that isolate sensor faults from process faults.
- **What our system does differently:**  
  Classical FDI requires exact analytical mathematical equations for the entire physical plant (differential equations of every tank, pump, valve), which is impractical for large complex multi-stage testbeds like SWaT (51 channels) and WADI (123 channels). R-STNet **learns** these redundancy relationships directly from normal operational data via data-driven spatio-temporal representations and graph attention.
- **Remaining Gap Addressed:**  
  Data-driven learning of analytical redundancy relationships without requiring manual physics modeling.

---

### Paper 7: USAD (UnSupervised Anomaly Detection)
- **Reference:** Audibert, J., Michiardi, P., Guyard, F., Marti, S., & Zuluaga, M. A. (2020). *USAD: UnSupervised Anomaly Detection on Multivariate Time Series.* ACM KDD 2020.
- **What it does:**  
  Uses an adversarially trained dual-decoder autoencoder to provide rapid reconstruction and amplify anomalous deviations.
- **What our system does differently:**  
  USAD operates on flattened sliding windows without channel-level structure. Adversarial amplification exacerbates the problem: when a sensor has noise or spikes, the adversarial objective amplifies the error, guaranteeing a massive false alarm.
- **Remaining Gap Addressed:**  
  Channel-selective damping rather than indiscriminate adversarial error amplification.

---

## 2. Synthesis of Research Gaps

| Capability | GDN | MTAD-GAT | CAROTS | RANSynCoders | OmniAnomaly | Proposed R-STNet |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Temporal Dependency Modeling | Yes | Yes | Yes | Weak | Yes | **Yes (TCN/GRU)** |
| Inter-Sensor Graph Modeling | Yes | Yes | Yes (Causal) | No | No | **Yes (Graph Attention)** |
| Explicit Online Sensor Reliability $r_i(t) \in [0, 1]$ | **No** | **No** | **No** | **No** | **No** | **Yes (Native SRM)** |
| Reliability-Weighted Anomaly Aggregation | **No** | **No** | **No** | Implicit | **No** | **Yes** |
| Decouples Sensor Faults from System Anomalies | **No** | **No** | **No** | Weak | **No** | **Yes (4-Way Engine)** |
| Robustness to Noise/Drift/Stuck Sensors | Low | Low | Moderate | Moderate | Low | **High (Evaluated)** |
| Intrinsic Explanation (Trust + Graph Violation) | Partial | Partial | Low | Low | Low | **High** |

---

## 3. Safe, Scientifically Defensible Novelty Statement

> **Defensible Research Position:**  
> "This project does not claim that time-series anomaly detection, graph attention, or sensor fault detection are individually novel. Rather, the proposed contribution is the **integrated reliability-aware framework (R-STNet)**: an explicit online Sensor Reliability Module (SRM) that combines temporal signal continuity, cross-sensor graph neighborhood agreement, and standardized residual tail confidence to compute continuous per-sensor trustworthiness scores. By dynamically down-weighting degraded sensors during system-level aggregation, the framework insulates plant-level anomaly detection from sensor corruption while operating a 4-way decision engine that separates isolated instrument degradation from genuine cyber-physical process attacks."

---

## 4. Architecture Positioning Clarification

- **R-STNet is OUR proposed project architecture/framework.**  
- It is NOT a previously published off-the-shelf paper name.  
- Its constituent building blocks (dilated temporal convolution, graph attention, and residual evaluation) are drawn from established literature, but their integration with the Sensor Reliability Module (SRM) and the 4-way decision logic represents our proposed research design.
