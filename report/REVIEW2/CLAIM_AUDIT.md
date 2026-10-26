# SCIENTIFIC CLAIM AUDIT AND DISCIPLINARY BOUNDARIES

**Project:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Purpose:** Guarding Against Scientific Overclaims, Literature Inflation, and Premature Generalization  
**Auditor:** Antigravity Research Engineering & Reproducibility Team  
**Audit Date:** October 23, 2026  

---

## 1. Disallowed Claims vs. Supported Scientific Findings

| Disallowed / Unjustified Claim | Why It Is Unjustified | Authorized / Scientifically Defensible Formulation |
| :--- | :--- | :--- |
| ❌ *"R-STNet decisively outperforms all baselines on SWaT."* | **False.** Baseline A (Isolation Forest) achieved $F_1 = 0.7182$ and $FPR = 0.92\%$ on clean SWaT, whereas R-STNet achieved $F_1 = 0.2740$ and $FPR = 62.14\%$. | ✅ *"Within the spatio-temporal deep architecture family, R-STNet significantly outperforms its unweighted counterpart (Baseline C, $F_1 = 0.2171 \to 0.2740$, eliminating 14,898 false alarms), while the classical Isolation Forest baseline remains superior on the clean SWaT benchmark."* |
| ❌ *"This project achieves State-of-the-Art (SOTA) performance."* | **Unsupported.** This project investigates reliability weighting under sensor corruption; claiming SOTA is methodologically unverified and inappropriate for Review 2. | ✅ *"The project investigates the specific hypothesis of whether explicit sensor reliability weighting improves robustness under sensor degradation."* |
| ❌ *"Sensor corruption in multivariate time-series anomaly detection is unexplored."* | **False.** Prior literature (e.g., robust estimation, CAROTS, sensor fault tolerant architectures) has explored sensor noise and corruption. | ✅ *"CAROTS and related robust estimation works provide valuable external research baselines; this work investigates continuous multi-perspective reliability fusion within a spatio-temporal graph framework."* |
| ❌ *"SWaT labels represent sensor-fault ground truth."* | **False.** Official SWaT labels indicate cyber-physical process attack periods, NOT individual sensor instrument faults. | ✅ *"Controlled synthetic corruption experiments on known channels ($E2-E4$) provide the sole ground truth for sensor-level fault localization and reliability separation."* |
| ❌ *"R-STNet is fully real-time."* | **Incomplete.** The measured per-window GPU latency is $0.060\text{ ms}$, but end-to-end streaming ingestion, buffering, and I/O pipeline latency was not benchmarked. | ✅ *"The measured inference latency of $0.060\text{ ms}$ per window demonstrates computational efficiency sufficient for the 10-second sampling period of SWaT, though end-to-end streaming systems require further deployment testing."* |
| ❌ *"The 7 unrecorded SWaT channels are corrupted sensors."* | **Misleading.** The 7 channels (`AIT201`, `MV101`, etc.) reflect unrecorded operational states or discrete actuators in the raw dataset, not broken sensors. | ✅ *"The unrecorded SWaT channels represent dataset-level operational distribution shifts between training and test periods that severely degrade standard reconstruction models."* |

---

## 2. Research Question Hypothesis Assessment

### Primary Research Question:
> *Can explicit sensor-reliability estimation improve the robustness of multivariate time-series anomaly detection when sensor observations are missing, noisy, or corrupted?*

### Authorized Scientific Response:
1. **Evidence Supporting the Hypothesis:**
   * Adding the Sensor Reliability Module (SRM) to the identical spatio-temporal graph backbone (comparing Baseline C vs. Proposed R-STNet) improved Point-wise F1 by **+26.2%** (from $0.2171$ to $0.2740$) on the clean benchmark.
   * SRM eliminated **14,898 false alarms** ($FPR$ decreased from $99.86\%$ to $62.14\%$).
   * Under controlled Gaussian noise and drift corruptions, SRM established a statistically significant reliability separation of **$+0.2856$ to $+0.2946$** between clean and corrupted channels across all evaluated seeds ($42, 101, 2024$), achieving $100\%$ fault detection recall.
2. **Evidence Against / Critical Limitations:**
   * Proposed R-STNet did **not** outperform Baseline A (Isolation Forest, $F_1 = 0.7182$) on clean SWaT.
   * On frozen/stuck actuators (e.g., pumps that normally have near-zero variance), temporal variance penalties cannot distinguish normal idle states from stuck faults without actuator command signals.
   * In the presence of catastrophic single-channel distribution shifts (`AIT201`), graph attention message passing propagates corrupted spatial representations to neighboring nodes, preventing R-STNet from fully eliminating false alarms ($FPR$ remains $62.14\%$).
3. **Synthesis:**
   * The hypothesis is **supported within the deep spatio-temporal architecture comparison**, demonstrating that explicit reliability estimation meaningfully improves robustness against sensor-level disturbances relative to uniform error aggregation, but does not yet establish superiority over static non-parametric tree ensembles on un-pruned industrial benchmarks.
