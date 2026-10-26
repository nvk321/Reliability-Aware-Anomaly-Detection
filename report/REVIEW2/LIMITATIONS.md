# Experimental Limitations and Methodological Boundaries

**Project:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Course:** BCSE209P — Machine Learning Lab  
**Date:** October 2026  
**Status:** Comprehensive Scientific Risk and Limitation Audit  

---

## 1. Overview and Scientific Integrity

Rigorous scientific evaluation requires an honest, transparent delineation of the conditions under which a proposed method succeeds, where it fails, and what boundaries constrain its generalizability. This document catalogs the **seven fundamental limitations and methodological boundaries** identified during the Phase 2 experimental investigation.

---

## 2. Catalog of Critical Limitations

### Limitation 1: Inferiority to Classical Isolation Forest on Clean Data
* **Empirical Reality:** On the official clean SWaT benchmark ($E1$), Baseline A (Isolation Forest) achieves an $F_1$ score of **$0.7182$**, precision of **$0.8995$**, and False Positive Rate of **$0.0092$** ($0.92\%$). In contrast, Proposed R-STNet achieves an $F_1$ score of **$0.2740$**, precision of **$0.1626$**, and False Positive Rate of **$62.14\%$**.
* **Scientific Boundary:** The proposed deep architecture does **not** outperform the classical tree ensemble baseline on the clean SWaT evaluation. Claims of "state-of-the-art" superiority are objectively unsupported by the data and must never be asserted. R-STNet's contribution is strictly confined to improving robustness and reducing false alarms *within* deep spatiotemporal neural architectures relative to unweighted reconstruction baselines.

---

### Limitation 2: Residual False Alarm Rate ($62.14\%$) Due to Spatial Contamination
* **Empirical Reality:** While R-STNet eliminates $14,898$ false alarms relative to Baseline C, its False Positive Rate remains elevated at $62.14\%$ ($24,541$ normal test windows flagged as anomalous).
* **Architectural Mechanism:** The Sensor Reliability Module (SRM) operates at the *decision aggregation layer*, dynamically weighting the output residuals:
  $$S_{\text{sys}}(t) = \frac{\sum r_i(t) e_i(t)}{\sum r_i(t)}$$
  However, in the upstream neural backbone, the input window $\mathbf{X}_t$ passes through spatial Graph Attention Network (GAT) layers. When an unrecorded or shifted sensor (`AIT201`) enters the network, message-passing across graph edges propagates corrupted latent embeddings into healthy neighboring nodes (`AIT202`, `FIT201`). While the SRM successfully downweights `AIT201`, the neighboring nodes' reconstruction errors remain moderately inflated, preventing the aggregate system score from dropping fully below $\tau$ at all times.

---

### Limitation 3: Physical Inability to Diagnose Stuck States on Zero-Variance Actuators
* **Empirical Reality:** Under Experiment $E3$ (stuck-sensor corruption), the SRM achieved **$0.0\%$ sensor fault detection recall** when backup chemical pumps (`P202`, `P204`) were frozen at $0.0$.
* **Physical Boundary:** Statistical anomaly detection models infer sensor reliability from *temporal velocity* ($\Delta x$), *spatial divergence* ($x - \bar{x}_{\mathcal{N}}$), and *reconstruction error* ($x - \hat{x}$). If a physical device (such as an inactive redundant pump or closed valve) naturally rests at zero during both normal training and normal testing, freezing it at zero produces zero rate-of-change and zero reconstruction error.
* **Operational Implication:** Pure unsupervised sensor time-series models cannot distinguish an inactive device from a stuck device without supervisory control context (PLC command state vs. sensor telemetry feedback).

---

### Limitation 4: The Dual-Edged Sword of Reliability Downweighting (Stealthy Attack Masking)
* **Empirical Reality:** In Failure Case 5 (SWaT Attack #33, Table 4.10), R-STNet produced a False Negative ($S_{prop} = 2.91 < \tau_{prop} = 3.14$), whereas Baseline C detected the attack ($S_c = 14.2 > \tau_c = 12.07$).
* **Methodological Risk:** When an adversary executes a stealthy, low-magnitude cyber-attack against a single physical sensor (e.g., gradually biasing flow meter `FIT401`), the SRM interprets the subtle single-sensor discrepancy as localized sensor unreliability rather than a malicious physical attack. By downweighting the attacked sensor, the SRM inadvertently suppresses the true attack signal, allowing stealthy attacks to slip past the system threshold.

---

### Limitation 5: Dataset-Level Artifacts in the SWaT Benchmark
* **Empirical Reality:** Seven channels in the raw SWaT dataset (`MV101`, `AIT201`, `MV201`, `P201`, `P202`, `P204`, `MV303`) were unrecorded or constant in normal training data, but active in test data.
* **Benchmarking Impact:** These shifts violate the fundamental machine learning assumption of independent and identically distributed (i.i.d.) normal operational states. They penalize deep reconstruction models disproportionately compared to tree ensembles. While this provided a realistic scenario to test R-STNet's reliability dampening, it complicates clean academic benchmarking and prevents pure clean-data comparisons.

---

### Limitation 6: Scope of Synthetic Corruption Suite
* **Methodological Limitation:** Controlled robustness experiments ($E2-E4$) evaluated synthetic Gaussian noise, constant stuck channels, and linear ramp drift.
* **Unexplored Modalities:** Real-world industrial sensor failure can include non-linear sensor degradation, power-supply harmonic distortion, intermittent packet dropouts, and coordinated multi-sensor adversarial spoofing. Generalizing R-STNet's robustness to non-linear physical failure modes remains an unvalidated research question.

---

### Limitation 7: Absence of End-to-End Real-Time Deployment Benchmarking
* **Computational Limitation:** While R-STNet demonstrated an inference latency of $0.060\text{ ms}$ per window in GPU memory, this does not constitute an end-to-end industrial deployment.
* **Engineering Gap:** In an actual industrial water treatment plant, end-to-end latency includes Modbus/TCP polling cycles, PLC network jitter, sliding window FIFO queue updates, and SCADA alarm dispatch. True real-time streaming viability can only be certified through pilot hardware-in-the-loop (HIL) testbed integration.

---

## 3. Summary of Prohibited Research Claims

To ensure complete adherence to academic ethics during Review 2, the following claims are **strictly prohibited**:
1. ❌ *"R-STNet is state-of-the-art and outperforms all baselines on SWaT."* (False: Isolation Forest is far superior on clean SWaT).
2. ❌ *"The SRM completely solves the sensor fault problem in anomaly detection."* (False: Stuck actuators cannot be diagnosed without control signals, and FPR remains $62.14\%$).
3. ❌ *"R-STNet eliminates all false alarms on SWaT."* (False: It eliminates $14,898$ false alarms, but $24,541$ remain).
4. ❌ *"The seven unrecorded SWaT channels are corrupted sensors."* (False: They are dataset collection artifacts and actuator transitions).
5. ❌ *"R-STNet has been proven real-time in industrial production."* (False: Only in-memory GPU inference latency was benchmarked).
