# Project Overview: Reliability-Aware Anomaly Detection for Multivariate Sensor Data

**Project Title:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Students:** P Naga Venkata Kushal (23BDS0133) & Koteru Tarun (23BDS0103)  
**Faculty Guide:** Dr. Dega Nagaraju  
**Course:** BCSE209P — Machine Learning Lab  
**Academic Year:** 2026  

---

## 1. Problem Statement

Industrial Cyber-Physical Systems (CPS)—including municipal water treatment facilities, smart power grids, and chemical refineries—rely heavily on multivariate sensor arrays for automated supervisory control and safety monitoring. Deep learning models for Multivariate Time-Series Anomaly Detection (MTSAD), such as autoencoders, recurrent networks, and spatio-temporal graph neural networks, have become the standard paradigm for detecting physical cyber-attacks.

However, existing deep MTSAD architectures operate under a critical flaw: **equal observation credulity across all sensor channels**. In real-world industrial environments, physical sensors routinely experience non-adversarial degradation:
- High-frequency electrical noise and EMI interference.
- Mechanical calibration drift and gradual sensor fouling.
- Frozen/stuck sensor readings due to transducer power failure.
- Transient unrecorded operational shifts or actuator mode switches.

Conventional reconstruction models compute anomaly scores by uniformly summing squared reconstruction errors across all channels:
$$S(t) = \frac{1}{M} \sum_{i=1}^M (x_i(t) - \hat{x}_i(t))^2$$
When a single sensor experiences non-adversarial degradation or a severe uncalibrated shift, its quadratic residual spikes catastrophically. The unweighted model interprets this isolated measurement discrepancy as a systemic cyber-physical attack, triggering severe plant-wide false alarms and operator alarm fatigue.

---

## 2. Research Question & Hypothesis

### Primary Research Question
> **Can explicit sensor-reliability estimation improve the robustness of multivariate time-series anomaly detection when sensor observations are missing, noisy, or corrupted?**

### Core Hypothesis
> **Explicit multi-perspective sensor-reliability estimation will significantly reduce false-alarm saturation and improve robustness to controlled sensor degradation within a spatio-temporal deep-learning anomaly-detection architecture.**

---

## 3. Proposed Solution: The R-STNet Framework

To address this challenge, we developed the **Reliability-Aware Spatio-Temporal Network (R-STNet)**.

```
Multivariate Sensor Telemetry X(t) in R^(L x M)
                   |
     +-------------+-------------+
     |                           |
     v                           v
Temporal Backbone            Sensor Reliability
  (Dilated TCN)                Module (SRM)
     |                           |
     v                           +--> 1. Temporal Continuity r^(v)
Spatial Backbone                 +--> 2. Graph Neighborhood Consistency r^(c)
  (Multi-Head GAT)               +--> 3. Predictive Residual Confidence r^(p)
     |                           |
     v                           v
Reconstruction x_hat(t)      Reliability Weights r_i(t) in [0, 1]
     |                           |
     +-------------+-------------+
                   |
                   v
   Reliability-Weighted System Scoring
       S_sys(t) = sum(r_i * e_i) / sum(r_i)
                   |
                   v
         Decision Alarm (S_sys >= tau)
```

### Key Architectural Elements:
1. **Spatiotemporal Neural Backbone ($23,233$ parameters):**
   - *Temporal Modeling:* Causal Dilated Temporal Convolutional Network (TCN) capturing multi-scale temporal dynamics across sliding windows ($L = 30$).
   - *Spatial Modeling:* Multi-head Graph Attention Network (GAT) capturing physical inter-sensor dependencies across an adjacency matrix.
2. **Sensor Reliability Module (SRM):**
   Fuses three complementary perspectives into a dynamic scalar reliability score $r_i(t) \in [0, 1]$:
   - *Temporal Continuity ($r_i^{(v)}$):* Penalizes non-physical rate-of-change spikes.
   - *Graph Consistency ($r_i^{(c)}$):* Penalizes divergence from physical topological neighbors.
   - *Residual Confidence ($r_i^{(p)}$):* Penalizes extreme z-score prediction errors.
3. **Reliability-Weighted Decision Layer:**
   Aggregates channel residuals using dynamic reliability weights, effectively filtering localized sensor noise before it can inflate the plant-wide anomaly score.

---

## 4. Key Empirical Findings (The Balanced Scientific Picture)

Our experiments on the Secure Water Treatment (SWaT) testbed produced a clear, nuanced, and academically honest set of findings:

1. **Within-Architecture Breakthrough Over Baseline C:**
   Under identical backbone parameters ($23,233$) and identical validation-quantile thresholding, Proposed R-STNet achieves:
   - Point-wise F1: **$0.2740$ vs. $0.2171$** (+26.21% relative gain).
   - False Positive Rate: **$62.14\%$ vs. $99.86\%$** (-37.78% relative reduction).
   - False Alarms Avoided: **$14,898$ normal windows** rescued from false emergency flags.
2. **Robustness Under Controlled Corruption:**
   Across multi-seed evaluations ($[42, 101, 2024]$) spanning Gaussian noise ($\sigma \in [0.1, 0.5]$) and linear drift ($\delta \in [0.1, 0.5]$):
   - R-STNet maintains stable detection ($F_1 \approx 0.276$) while Baseline C remains collapsed at $FPR = 99.86\%$.
   - The SRM establishes a statistically significant reliability separation of **$\Delta r = +0.2856\text{ to }+0.2946$** and achieves **$100\%$ fault detection recall** against synthetic corruption masks.
3. **The Clean-Data Baseline Reality (Important Negative Finding):**
   - Classical **Isolation Forest outperforms R-STNet on clean SWaT** ($F_1 = 0.7182, FPR = 0.0092$ vs. $F_1 = 0.2740, FPR = 0.6214$).
   - This occurs because seven channels in SWaT (most notably `AIT201`) were unrecorded/constant during training ($\sigma = 0$), but active during testing ($z > 1,600$). Tree ensembles isolate these univariate shifts near tree roots without distorting the remaining 50 features.
   - While R-STNet successfully downweights `AIT201`, upstream spatial Graph Attention message passing propagates latent contamination into neighboring nodes, leaving an elevated residual FPR of $62.14\%$.

---

## 5. Repository Documentation Roadmap

- **Authoritative Specifications:** [SOURCE_OF_TRUTH.md](file:///d:/Projects/Reliability-Aware-Anomaly-Detection/docs/SOURCE_OF_TRUTH.md)
- **Dataset Attribution & Setup:** [DATASET.md](file:///d:/Projects/Reliability-Aware-Anomaly-Detection/docs/DATASET.md)
- **Formal Methodology:** [METHODOLOGY.md](file:///d:/Projects/Reliability-Aware-Anomaly-Detection/docs/METHODOLOGY.md)
- **Experimental Protocol:** [EXPERIMENTS.md](file:///d:/Projects/Reliability-Aware-Anomaly-Detection/docs/EXPERIMENTS.md)
- **Reproducibility Guides:** [REPRODUCIBILITY.md](file:///d:/Projects/Reliability-Aware-Anomaly-Detection/docs/REPRODUCIBILITY.md)
- **Scientific Limitations:** [LIMITATIONS.md](file:///d:/Projects/Reliability-Aware-Anomaly-Detection/docs/LIMITATIONS.md)
- **Interactive Demo:** [DEMO.md](file:///d:/Projects/Reliability-Aware-Anomaly-Detection/docs/DEMO.md)
- **Results Traceability:** [RESULTS_MANIFEST.md](file:///d:/Projects/Reliability-Aware-Anomaly-Detection/docs/RESULTS_MANIFEST.md)
