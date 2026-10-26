# CHAPTER 5: CONCLUSION

---

## 5.1 Work Carried Out in the Thesis

This thesis investigated the critical vulnerability of equal observation credulity in multivariate time-series anomaly detection (MTSAD) models deployed within industrial Cyber-Physical Systems (CPS). When physical sensors experience non-adversarial degradation—such as high-frequency electrical noise, mechanical calibration drift, transducer freezes, or unrecorded operational shifts—conventional deep reconstruction architectures uniformly aggregate residuals across all channels, misinterpreting localized sensor faults as plant-wide cyber-physical attacks. To address this challenge, we developed the **Reliability-Aware Spatio-Temporal Network (R-STNet)**, an architecture combining a Causal Dilated Temporal Convolutional Network (TCN) and a Multi-Head Spatial Graph Attention Network (GAT) with an auxiliary **Sensor Reliability Module (SRM)**. The SRM dynamically estimates a continuous reliability coefficient $r_i(t) \in [0, 1]$ for each sensor by fusing three orthogonal perspectives: temporal rate-of-change continuity ($r^{(v)}$), graph neighborhood consistency ($r^{(c)}$), and predictive residual confidence ($r^{(p)}$).

The experimental framework was conducted on the official Secure Water Treatment (SWaT) industrial testbed dataset from iTrust SUTD, comprising 51 physical variables (25 continuous sensors and 26 discrete actuators) across six purification stages. Telemetry was downsampled $10\times$ (10-second steps) and partitioned strictly chronologically: 80% of the 7-day normal run for training ($39,715$ windows), 20% of the normal run for validation threshold calibration ($9,907$ windows), and the full 4-day attack period for mixed test evaluation ($44,963$ windows, $5,468$ attack labels). Data was scaled using a leak-free `RobustStandardScaler` with zero-variance protection. Decision thresholds $\tau$ were calibrated strictly at the $95^{\text{th}}$ percentile of validation scores ($q = 0.95$), guaranteeing zero label leakage. In addition to clean benchmarking, we engineered a controlled synthetic corruption suite evaluating additive Gaussian noise ($\sigma \in \{0.1, 0.3, 0.5\}$), stuck sensors ($k \in \{1, 3, 5\}$), and dynamic linear drift ($\delta \in \{0.1, 0.3, 0.5\}$) across multiple random seeds ($[42, 101, 2024]$), completed a 5-condition ablation study, and diagnosed five operational failure modes and boundary intervals.

---

## 5.2 Conclusions from Results and Discussion

The empirical findings from Chapter 4 establish the following validated conclusions:

1. **Isolation Forest Achieved the Best Clean-Data Performance:**  
   On the unperturbed SWaT benchmark, Baseline A (Isolation Forest) achieved the highest overall detection performance ($\text{Point-wise } F_1 = \mathbf{0.7182}$, $\text{Precision} = \mathbf{0.8995}$, $\text{Recall} = 0.5977$, and $\text{FPR} = \mathbf{0.0092}$). Classical random tree partitioning proved inherently robust against dataset-level unrecorded channel shifts because univariate outliers are isolated near tree roots without distorting the orthogonal partitioning of remaining features.
2. **R-STNet Significantly Improved Over RSTNetNoSRM Within Deep Architectures:**  
   Under an identical $23,233$-parameter spatiotemporal backbone, Proposed R-STNet achieved an $F_1$ score of **$0.2740$** compared to Baseline C's **$0.2171$**, representing a **$+26.21\%$ relative improvement**. Precision improved from $0.1218$ to $0.1626$ (+33.50% gain) while maintaining high attack detection sensitivity ($87.13\%$ recall, identifying $4,764$ of $5,468$ attack windows).
3. **The SRM Substantially Mitigated False-Alarm Saturation:**  
   While standard deep baselines (ST-AE and RSTNetNoSRM) collapsed into near-universal false-positive saturation ($\text{FPR} = 100.0\%$ and $99.86\%$, flagging all normal windows), R-STNet reduced the False Positive Rate to **$62.14\%$**. The SRM eliminated **$14,898$ false alarms** (-37.78% relative reduction) and recovered $14,954$ True Negatives.
4. **Controlled Corruption Confirmed Statistically Significant Reliability Decoupling:**  
   Under additive Gaussian noise and linear drift, R-STNet established a statistically significant empirical separation gap of **$\Delta r = +0.2856\text{ to }+0.2946$** between clean channels ($\bar{r} \approx 0.304$) and corrupted channels ($\bar{r} \approx 0.015$). Using a threshold of $r_i < 0.60$, the SRM achieved **$100.0\%$ sensor fault detection recall** against synthetic corruption masks with a $0.00\%$ false alarm rate on clean channels.
5. **Physical Actuator Boundaries Were Identified:**  
   Under stuck-sensor corruption, the SRM registered $0.0\%$ fault recall on inactive backup pumps (`P202`, `P204`). Because inactive actuators naturally rest at zero with zero variance during normal operation, holding them frozen produces zero rate-of-change and zero reconstruction error. Statistical models cannot detect stuck states on inactive actuators without supervisory control telemetry.
6. **SWaT Channel Distribution Shifts Were Diagnosed:**  
   The deep models' baseline collapse was traced to seven unrecorded or constant channels (`MV101`, `AIT201`, `MV201`, `P201`, `P202`, `P204`, `MV303`). In particular, chemical analyzer `AIT201` had zero training variance ($\sigma = 0$), but surged to standardized $z > 1,600$ during test collection. R-STNet's SRM successfully suppressed this unrecorded variable ($r < 0.005$), preventing catastrophic false emergencies.
7. **The Research Hypothesis is Supported Within Deep Spatiotemporal Models, Not Universally:**  
   The empirical evidence supports the hypothesis that explicit reliability estimation improves robustness and reduces false alarms *within deep spatiotemporal neural architectures*. However, the hypothesis cannot be generalized as a claim of universal superiority across all anomaly detection models, as classical Isolation Forest remains superior on clean SWaT.

---

## 5.3 Managerial and Operational Implications

The findings of this thesis offer actionable insights for the design, procurement, and supervisory operation of industrial anomaly-detection and SCADA monitoring systems:

1. **Mitigation of Operator Alarm Fatigue in SCADA Environments:**  
   In industrial process plants, alarm flood and operator fatigue represent severe operational risks; operators routinely acknowledge or disable alarm systems that generate excessive false positives. By eliminating $14,898$ false alarms relative to unweighted deep models, reliability-aware mechanisms can significantly improve operator trust in automated alerting systems.
2. **Dual-Layer Supervisory Decision Support:**  
   Conventional anomaly detectors output an opaque scalar alarm indicating that "something" in the plant is wrong. R-STNet provides a dual-layer operational diagnostic:
   - *System-Level Alarm ($S_{\text{sys}} \ge \tau$):* Alerts operators to coordinated cyber-physical disruptions requiring emergency response.
   - *Sensor-Level Reliability Health Index ($r_i(t) < 0.60$):* Directs instrumentation maintenance technicians to specific degrading physical transmitters (fouled analyzers, noisy flow meters, drifting pressure cells) before catastrophic failure occurs.
3. **Defense Against Non-Adversarial Transients:**  
   Industrial commissioning frequently encounters uncalibrated transmitters or transient process reconfigurations. Rather than requiring emergency retraining or manual channel deletion, reliability-aware scoring dynamically downweights erratic channels at runtime, preserving continuous plant monitoring.
4. **Human-in-the-Loop Operational Positioning:**  
   The identified failure cases (e.g., subtle single-sensor attack masking and stuck inactive actuators) emphasize that automated ML detectors must not operate as autonomous plant shutdown switches. Instead, they should function as intelligent decision-support assistants providing contextualized evidence to human control-room operators and process engineers.

---

## 5.4 Future Scope of the Work

Based on the empirical boundaries and methodological limitations identified during this research, several promising directions are outlined for future work:

1. **Upstream Reliability Gating Before Graph Message Passing:**  
   In the current R-STNet design, the SRM operates at the final decision layer. Consequently, corrupted or shifted input features pass through the spatial Graph Attention layers, partially contaminating latent representations of neighboring nodes. Future research should integrate reliability gating *prior* to graph attention, masking unreliable node embeddings before spatial message passing occurs.
2. **Supervisory Control Command & Actuator Telemetry Fusion:**  
   To resolve the $0.0\%$ stuck-actuator fault recall on inactive backup pumps, future architectures should ingest supervisory control telemetry—specifically comparing PLC output command states (`CMD_OPEN`, `CMD_START`) against sensor feedback states (`VALVE_STATE`, `PUMP_FLOW`). Discrepancies between command and response will enable direct detection of frozen actuators.
3. **Training-Time Dynamic Reliability Masking:**  
   Extending reliability estimation into the training phase via self-supervised masking or noise-injected curriculum learning could encourage the neural backbone to learn invariant representations that do not overfit to zero-variance channels.
4. **Evaluation Across Additional Real-World Industrial Testbeds:**  
   Evaluating R-STNet on additional industrial datasets—such as the Water Distribution (WADI) testbed, the Tennessee Eastman Process (TEP), and the Gas Pipeline ICS dataset—will validate the cross-domain generalizability of the multi-perspective reliability formulation.
5. **Online Adaptive Threshold Calibration:**  
   Investigating extreme value theory (EVT) and adaptive sliding-window quantile calibration could improve threshold responsiveness under non-stationary seasonal or operational mode shifts.
6. **Hardware-in-the-Loop (HIL) Streaming Deployment:**  
   While R-STNet demonstrated an in-memory inference latency of $0.060\text{ ms}$ per window, future work should deploy the model onto physical edge compute modules (e.g., NVIDIA Jetson Orin) interfaced via industrial Modbus/TCP or OPC-UA protocols to evaluate end-to-end streaming jitter in live PLC environments.

---

## 5.5 Academic Formatting Note for Final Submission

In accordance with university thesis guidelines, this chapter is formatted in standard Markdown as an intermediate research document. When compiling the final hard-copy thesis or digital PDF submission, ensure the following typography settings are applied in Microsoft Word / LaTeX:
- **Font Family:** Times New Roman
- **Font Size:** 12 pt for body text; 14 pt bold for section headings; 16 pt bold for chapter title
- **Line Spacing:** Exactly 1.5 line spacing
- **Margins:** 1.25 inches (left margin for binding); 1.0 inch (top, bottom, right margins)
- **Paragraph Alignment:** Justified with 0.5-inch first-line indentation
