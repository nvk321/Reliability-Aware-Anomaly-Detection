# Failure Case Analysis and Anomaly Score Diagnostics

**Project:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Course:** BCSE209P — Machine Learning Lab  
**Date:** October 2026  
**Status:** Comprehensive Qualitative Diagnostic Compendium  

---

## 1. Scientific Motivation

A critical requirement of scientific research in Machine Learning is conducting rigorous qualitative failure analysis. Rather than treating models as black-box metric generators, analyzing specific operational intervals reveals **why models succeed, how they fail, and where physical boundaries lie**.

This document examines **five representative failure cases, boundary conditions, and diagnostic intervals** from the SWaT evaluation suite (Table 4.10):
1. **Case 1:** Deep Model False Alarm Caused by Channel Distribution Shift (`AIT201`).
2. **Case 2:** Successful Sensor Corruption Suppression by R-STNet (Noise on `FIT101` / `LIT101`).
3. **Case 3:** Stuck-Sensor Diagnostic Failure on Naturally Steady Actuators (`P202` / `P204`).
4. **Case 4:** Genuine Coordinated Cyber-Physical Attack Correctly Detected (`LIT301` Tank Overflow).
5. **Case 5:** Stealthy Anomaly Masking by SRM Downweighting (`FIT401` Low-Rate Spoofing).

---

## 2. Case 1: Deep Model False Alarm Caused by Channel Distribution Shift

* **Timestamp / Interval:** Test Windows $t = 1,200 - 1,800$ (Normal Operation Interval).
* **Physical Sensor Involved:** `AIT201` (Chemical Conductivity Analyzer, Stage 2).
* **Ground-Truth State:** Normal Operation ($y = 0$).
* **Model Predictions:**
  - **Baseline C (RSTNetNoSRM):** False Alarm ($S_c(t) = 1,482.3 \gg \tau_c = 12.07$).
  - **Proposed R-STNet:** True Negative ($S_{prop}(t) = 2.14 < \tau_{prop} = 3.14$).
* **Physical & Telemetry Behavior:** During normal training, `AIT201` was offline/unrecorded, outputting empty records that standardized to a constant $z \approx 1.0$ with zero variance. In this test window, the instrument began streaming real telemetry with dynamic chemical oscillations, producing standardized values exceeding $z = 1,600$.
* **Diagnostic Interpretation:**
  - In Baseline C, the squared error $e_{\text{AIT201}} = (1600 - 1)^2 \approx 2.56 \times 10^6$ is summed uniformly into the plant-wide score. This causes $S_c(t)$ to exceed the threshold by over $100\times$, falsely triggering an emergency shutdown.
  - In Proposed R-STNet, the SRM's residual confidence term immediately evaluates $r_{\text{AIT201}}^{(p)}(t) = \exp(-\frac{1}{2} z^2) \to 0.0$. The overall reliability score drops to $r_{\text{AIT201}} < 0.005$. In the weighted sum $\sum r_i e_i / \sum r_i$, the massive error is dampened into a negligible contribution, keeping $S_{prop}(t)$ safely below $\tau_{prop} = 3.14$.
* **Significance:** Demonstrates how R-STNet successfully resolves single-channel commissioning artifacts without human intervention.

---

## 3. Case 2: Successful Sensor Corruption Suppression Under Noise

* **Timestamp / Interval:** Test Windows $t = 2,500 - 2,700$ (Normal Operation with Synthetic Noise).
* **Physical Sensors Involved:** `FIT101` (Flow Meter), `LIT101` (Level Transmitter), `DPIT301` (Differential Pressure).
* **Ground-Truth State:** Normal Operation with Localized Sensor Degradation ($y_{\text{native}} = 0, \mathbf{M}_{\text{synth}} = 1$).
* **Model Predictions:**
  - **Baseline C (RSTNetNoSRM):** False System Alarm ($S_c(t) = 45.2 > \tau_c = 12.07$).
  - **Proposed R-STNet:** True Negative ($S_{prop}(t) = 2.89 < \tau_{prop} = 3.14$).
* **Physical & Telemetry Behavior:** Synthetic high-frequency Gaussian noise ($\sigma = 0.5$) was injected into continuous sensor streams. The physical plant continued operating in a completely stable, nominal regime.
* **Diagnostic Interpretation:**
  - Baseline C interprets the rapid sensor fluctuations as a physical process disruption. The unweighted residual aggregation surges to $45.2$, generating an extended false system alarm (**Figure 4.9**, top panel).
  - Proposed R-STNet's SRM detects the severe temporal velocity violations ($v_{i, t} \gg 3$) on `FIT101` and `LIT101`. Their reliability scores drop from $0.35$ down to $0.015$ (**Figure 4.6**). In the system summation, their noisy contributions are downweighted by $95\%$. The plant-wide anomaly score remains at $2.89 < 3.14$ (**Figure 4.9**, bottom panel).
* **Significance:** Proves the core design premise: localized sensor degradation is isolated from plant-wide emergency alarms.

---

## 4. Case 3: Stuck-Sensor Diagnostic Failure on Naturally Steady Actuators

* **Timestamp / Interval:** Test Windows $t = 14,000 - 14,500$ (Normal Operation with Inactive Backup Pumps).
* **Physical Sensors Involved:** `P202`, `P204` (Stage 2 Chemical Dosing Metering Backup Pumps).
* **Ground-Truth State:** Normal Operation ($y = 0$). Synthetic stuck-sensor fault injected on `P202` and `P204`.
* **Model Predictions:**
  - **Baseline C (RSTNetNoSRM):** True Negative ($S_c(t) < \tau_c$).
  - **Proposed R-STNet:** True Negative ($S_{prop}(t) < \tau_{prop}$).
  - **SRM Fault Diagnosis:** Missed Fault ($r_{\text{P202}} = 0.95$, Fault Recall $= 0.0\%$).
* **Physical & Telemetry Behavior:** The backup dosing pumps `P202` and `P204` are redundant units designed to engage only if primary pumps `P201` or `P203` fail. During normal training and normal testing, they remain in a de-energized, zero-variance state ($0.0$). Synthetic stuck corruption held them frozen at $0.0$.
* **Diagnostic Interpretation:**
  - Because `P202` and `P204` were already steady at $0.0$, holding them constant produced zero temporal rate-of-change ($\Delta x = 0$) and zero reconstruction error ($e = |0 - 0| = 0$).
  - The SRM evaluated $r^{(v)} \approx 1.0$ and $r^{(p)} \approx 1.0$, resulting in a high reliability score ($0.95$). Thus, the SRM failed to detect that the channels were "stuck" ($0.0\%$ fault recall, Table 4.5).
* **Significance & Research Boundary:** This failure exposes an intrinsic limitation of unsupervised statistical anomaly detection: **a statistical module cannot distinguish an inactive device from a frozen device without external supervisory command telemetry (PLC actuator commands vs. feedback state).**

---

## 5. Case 4: Genuine Cyber-Physical Attack Correctly Detected

* **Timestamp / Interval:** Test Windows $t = 28,100 - 28,600$ (SWaT Attack #26).
* **Physical Sensors Involved:** `LIT301` (Stage 3 UF Feed Tank Level), `P301` (UF Feed Pump), `MV301` (UF Inflow Valve).
* **Ground-Truth State:** Cyber-Physical Attack ($y = 1$).
* **Model Predictions:**
  - **Baseline C (RSTNetNoSRM):** True Positive ($S_c(t) = 210.4 \gg \tau_c = 12.07$).
  - **Proposed R-STNet:** True Positive ($S_{prop}(t) = 48.7 \gg \tau_{prop} = 3.14$).
* **Physical & Telemetry Behavior:** The attacker spoofed the level sensor `LIT301` to report a falsely low water level ($z < -2.5$). In response, the automated PLC logic opened valve `MV301` to maximum flow and halted pump `P301`, attempting to fill the tank. In reality, the tank was already full, resulting in physical tank overflow and pump cavitation.
* **Diagnostic Interpretation:**
  - This was a coordinated multi-sensor attack altering the physical mass balance across three instruments.
  - While `LIT301` exhibited an unusual residual, its physical neighbors (`FIT301`, `P301`) simultaneously displayed correlated, non-random deviations.
  - The SRM's graph consistency term $r_i^{(c)}$ recognized that the deviation was coupled across the Stage 3 subsystem. Consequently, `LIT301` was not suppressed to zero ($r_{\text{LIT301}} \approx 0.18$).
  - The residual errors across `LIT301`, `P301`, and `MV301` combined to produce an aggregate score of $48.7$, triggering a clear, unequivocal attack alarm (**Figure 4.2**, red shaded interval).
* **Significance:** Proves that R-STNet does not suppress genuine cyber-physical attacks; it distinguishes coordinated multi-point attacks from isolated sensor glitches.

---

## 6. Case 5: Stealthy Anomaly Masking by SRM Downweighting

* **Timestamp / Interval:** Test Windows $t = 36,200 - 36,450$ (SWaT Attack #33).
* **Physical Sensors Involved:** `FIT401` (Dechlorination Flow Meter), `UV401` (Ultraviolet Dechlorination Unit).
* **Ground-Truth State:** Cyber-Physical Attack ($y = 1$).
* **Model Predictions:**
  - **Baseline C (RSTNetNoSRM):** True Positive ($S_c(t) = 14.2 > \tau_c = 12.07$).
  - **Proposed R-STNet:** False Negative ($S_{prop}(t) = 2.91 < \tau_{prop} = 3.14$).
* **Physical & Telemetry Behavior:** A stealthy, low-rate cyber-attack slightly biased the flow transmitter `FIT401` reading downward by $8\%$ over an extended duration, causing the UV unit to receive slightly less chemical dosing than required without tripping immediate physical alarm thresholds.
* **Diagnostic Interpretation:**
  - The attack manipulated only a single continuous sensor with a subtle, gradual bias.
  - Because only one sensor was modified, it appeared to the SRM as an isolated sensor calibration drift.
  - The SRM evaluated $r_{\text{FIT401}}^{(c)}$ and $r_{\text{FIT401}}^{(p)}$ as slightly degraded ($r \approx 0.12$), downweighting `FIT401`.
  - While this downweighting is beneficial when dealing with true sensor noise, in this specific case it downweighted a *genuine single-point attack*, reducing the aggregate system score from $14.2$ down to $2.91$. Because $2.91 < \tau_{prop} = 3.14$, R-STNet missed the attack, generating a False Negative.
* **Significance & Negative Finding:** This failure reveals the critical **dual-edged nature of reliability weighting**:
  $$\text{Aggressive Sensor Downweighting} \implies \text{Suppression of Sensor Noise} \quad \text{AND} \quad \text{Potential Masking of Stealthy Single-Sensor Attacks}$$
  Documenting this trade-off provides essential scientific balance for Review 2.
