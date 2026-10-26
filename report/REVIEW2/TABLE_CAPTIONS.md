# Review 2 Chapter 4: Table Captions and Detailed Scientific Annotations

This document provides complete, publication-grade academic captions for Tables 4.1 through 4.11 in Chapter 4 of the Review 2 Report.

---

### Table 4.1: Experimental Configuration and Environmental Parameters
* **What:** Complete summary of dataset partitions, preprocessing hyperparameters, sliding window parameters, threshold calibration protocol, and computing infrastructure.
* **Dataset:** Secure Water Treatment (SWaT A1/A2, December 2015 testbed) containing 51 physical sensor and actuator channels.
* **Experiment:** Official chronological benchmark evaluation dividing the 7-day normal commissioning run into 80% train ($39,715$ windows) and 20% validation ($9,907$ windows), and utilizing the 4-day attack run as the test evaluation sequence ($44,963$ windows).
* **Comparison:** Details consistent environmental and hyperparameter baselines applied identically across all four evaluated machine learning architectures.
* **Important Interpretation:** Confirms zero data leakage across splits: normalization parameters and decision thresholds $\tau$ are calculated strictly on normal training and validation data, without visibility into test attack labels.

---

### Table 4.2: Model Architecture and Computational Specifications
* **What:** Architectural taxonomy, trainable parameter counts, tensor input dimensions, loss objectives, anomaly scoring formulations, and computational footprints.
* **Dataset:** SWaT A1/A2 testbed ($M=51$ channels, $L=30$ window length).
* **Experiment:** Controlled architectural benchmarking across classical unsupervised learning (Isolation Forest), feedforward deep reconstruction (ST-AE), spatiotemporal graph learning without reliability (RSTNetNoSRM), and the proposed reliability-aware network (R-STNet with SRM).
* **Comparison:** Demonstrates that Baseline C and Proposed R-STNet possess identical neural backbone capacities ($23,233$ parameters), isolating the Sensor Reliability Module as the sole experimental variable.
* **Important Interpretation:** Proposed R-STNet introduces minimal computational overhead ($134.73$ s training vs. $112.89$ s for Baseline C; $0.060$ ms inference latency vs. $0.050$ ms), preserving real-time industrial feasibility.

---

### Table 4.3: Clean SWaT Benchmark Performance Comparison
* **What:** Full quantitative evaluation metrics across core detection, false alarm rates, confusion matrix counts, and threshold-free areas under the curve.
* **Dataset:** Official chronological mixed SWaT test sequence ($44,963$ windows, $5,468$ attack labels, $12.16\%$ positive class rate).
* **Experiment:** Clean baseline benchmark ($E1$) comparing Isolation Forest, Spatio-Temporal Autoencoder, RSTNetNoSRM, and Proposed R-STNet.
* **Comparison:** Contrasts the proposed method against both classical tabular isolation trees and deep neural baselines.
* **Important Interpretation:** Isolation Forest achieves the highest clean performance ($F_1 = 0.7182, FPR = 0.0092$) due to tree partition robustness against unscaled channel shifts. Within deep neural architectures, R-STNet's SRM provides a decisive $+26.21\%$ F1 gain and eliminates $14,898$ false alarms relative to Baseline C ($FPR = 62.14\%$ vs. $99.86\%$).

---

### Table 4.4: Robustness Under Additive Gaussian Noise Corruption ($E2$)
* **What:** Detection performance ($F_1$, precision, recall, FPR) and reliability metrics ($\Delta r$, fault recall) under increasing additive Gaussian noise.
* **Dataset:** SWaT test sequence with synthetic zero-mean Gaussian noise $\mathcal{N}(0, \sigma^2)$ injected into channels $\{3, 14, 28\}$ across severities $\sigma \in \{0.1, 0.3, 0.5\}$.
* **Experiment:** Controlled multi-seed robustness suite ($E2$) evaluated across random seeds $[42, 101, 2024]$.
* **Comparison:** Evaluates degradation trajectories of Baseline C (uniform residual aggregation) versus Proposed R-STNet (reliability-weighted aggregation).
* **Important Interpretation:** Baseline C remains collapsed at $FPR = 99.86\%$. In contrast, R-STNet maintains stable anomaly detection ($F_1 \approx 0.276$) and achieves statistically significant reliability separation ($\Delta r = +0.2856$ to $+0.2890$) with $100\%$ sensor fault detection recall.

---

### Table 4.5: Robustness Under Stuck-Sensor Corruption ($E3$)
* **What:** Anomaly detection stability and fault identification recall when multiple sensor channels freeze at constant values.
* **Dataset:** SWaT test partition with $k \in \{1, 3, 5\}$ sensors held constant ($x_i(t) = x_i(t_{\text{fault}})$) across seeds $[42, 101, 2024]$.
* **Experiment:** Controlled stuck-sensor robustness experiment ($E3$).
* **Comparison:** Contrasts Baseline C and Proposed R-STNet system anomaly metrics under multi-channel sensor freezes.
* **Important Interpretation:** R-STNet maintains system detection stability ($F_1 \approx 0.271 - 0.274$), but sensor fault recall is $0.0\%$. This reveals an essential scientific boundary: stuck sensors on naturally steady or inactive actuator channels do not generate velocity or residual anomalies and cannot be diagnosed without control-command telemetry.

---

### Table 4.6: Robustness Under Dynamic Sensor Drift Corruption ($E4$)
* **What:** System anomaly metrics and reliability separation under continuous linear ramp calibration drifts.
* **Dataset:** SWaT continuous sensor channels subjected to ramp offsets $x_i(t) \pm \delta \cdot \frac{t - t_0}{T}$ across slopes $\delta \in \{0.1, 0.3, 0.5\}$ and seeds $[42, 101, 2024]$.
* **Experiment:** Controlled sensor drift experiment ($E4$).
* **Comparison:** Evaluates ability of SRM to detect creeping sensor bias versus unweighted residual scoring.
* **Important Interpretation:** R-STNet dynamically detects creeping drifts via temporal continuity and graph neighborhood discrepancies, establishing clear reliability separation ($\Delta r = +0.2928$ to $+0.2946$) and achieving $100\%$ fault recall without inducing false system alarms.

---

### Table 4.7: Sensor Reliability Estimation and Fault Localization Metrics
* **What:** Formal separation analysis between clean channel reliability ($\bar{r}_{\text{clean}}$) and corrupted channel reliability ($\bar{r}_{\text{corrupt}}$) alongside fault localization recall and false alarm rates.
* **Dataset:** SWaT evaluation sequence under controlled synthetic corruption masks.
* **Experiment:** Direct validation of the Sensor Reliability Module's ability to decouple localized sensor degradation from plant-wide cyber-physical attacks.
* **Comparison:** Contrasts empirical reliability distributions between intact and perturbed sensor streams.
* **Important Interpretation:** Confirms that the SRM reliably drives corrupted sensor weights toward zero ($r < 0.02$) while preserving intact sensor weights ($r \approx 0.30$), producing a net separation of $\Delta r \approx +0.29$ with zero false fault alarms on healthy channels.

---

### Table 4.8: Sensor Reliability Module (SRM) Perspective Ablation Study
* **What:** Component-wise performance breakdown of the three SRM dimensions: Temporal Continuity ($w_v$), Graph Consistency ($w_c$), and Residual Confidence ($w_p$).
* **Dataset:** SWaT test sequence under identical backbone architectures.
* **Experiment:** 5-condition ablation suite isolating each individual reliability perspective and comparing against the full multi-perspective formulation.
* **Comparison:** Tracks the evolution of $F_1$, precision, recall, FPR, and PR-AUC from No SRM through single-component configurations to Full SRM.
* **Important Interpretation:** Single perspectives produce negligible improvements ($F_1 \approx 0.216 - 0.218$, $FPR > 92\%$). Only Full SRM combines rapid outlier dampening with temporal smoothing and topological validation, achieving $F_1 = 0.2740$ and driving FPR down to $62.14\%$.

---

### Table 4.9: Computational Resource Utilization and Execution Benchmarks
* **What:** Exhaustive resource accounting: trainable parameter count, training wall-clock time, per-window inference latency, theoretical throughput, and peak allocated VRAM.
* **Dataset:** SWaT training ($39,715$ windows) and testing ($44,963$ windows).
* **Experiment:** Hardware benchmarking on an NVIDIA GeForce RTX 3050 Laptop GPU (4 GB) and Intel Core i5 CPU.
* **Comparison:** Contrasts computational footprints across all four benchmarked architectures.
* **Important Interpretation:** R-STNet achieves an inference latency of $0.060$ ms per window ($16,667$ windows/s), five orders of magnitude faster than the SWaT sampling interval ($10,000$ ms), confirming computational feasibility for real-time edge deployment.

---

### Table 4.10: Empirical Failure Cases and Anomaly Score Diagnostic Matrix
* **What:** Detailed diagnostic breakdown of five representative failure modes and boundary edge cases across time, physical sensors, model predictions, and ground-truth states.
* **Dataset:** Specific operational intervals of the SWaT test sequence.
* **Experiment:** Qualitative error analysis investigating unrecorded channel shifts (`AIT201`), synthetic noise suppression, stuck zero-variance actuators, coordinated attacks, and stealthy drift.
* **Comparison:** Juxtaposes the internal scoring behavior of Baseline C against Proposed R-STNet.
* **Important Interpretation:** Demonstrates exactly where R-STNet succeeds (suppressing false alarms caused by unrecorded shifts and sensor degradation) and where it encounters physical boundaries (stealthy sensor-spoofing attacks that mask genuine system anomalies).

---

### Table 4.11: Research-Hypothesis Empirical Evidence Synthesis
* **What:** Structured synthesis mapping each empirical dimension of the experimental investigation directly to the primary research question.
* **Dataset:** SWaT A1/A2 benchmark and synthetic corruption suite.
* **Experiment:** Holistic assessment of all experimental results gathered across Phase 2.
* **Comparison:** Balanced comparison weighing evidence supporting the research hypothesis against empirical counter-evidence and physical limitations.
* **Important Interpretation:** Concludes that the research hypothesis is **supported within deep spatiotemporal neural architectures** (SRM provides superior robustness, false-alarm reduction, and fault isolation relative to RSTNetNoSRM), but is **not supported as a general claim of superiority over classical tree baselines** on clean industrial data.
