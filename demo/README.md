# R-STNet Interactive Demonstration Dashboard

This directory contains the presentation-level interactive web demonstration for the project **"Reliability-Aware Anomaly Detection for Multivariate Sensor Data"**.

---

## 1. Fast Launch Instructions

Ensure your Python virtual environment is activated and dependencies are installed:

```powershell
# From the repository root
streamlit run demo/app.py
```

The web dashboard will automatically launch in your default browser at:  
👉 `http://localhost:8501`

---

## 2. Dashboard Features & Walkthrough

The demo provides an intuitive, interactive visualization of the core research contribution:

### Tab 1: Live Anomaly Timeline
- **Real-Time Evaluation:** Renders anomaly scores dynamically across the SWaT cyber-physical test sequence.
- **Interactive Thresholding:** Displays the calibrated decision threshold $\tau$ alongside ground-truth attack intervals (red shading).
- **Physical Telemetry:** Plots real-time flow (`FIT101`), level (`LIT101`), and pressure (`DPIT301`) telemetry signals.

### Tab 2: Sensor Reliability Module (SRM) Decomposition
- **Per-Sensor Reliability Bar Chart:** Displays $r_i(t) \in [0, 1]$ across all 51 physical sensors.
- **Perspective Breakdown:** Inspect the individual contributions of:
  - *Temporal Continuity ($r^{(v)}$)*
  - *Graph Neighborhood Consistency ($r^{(c)}$)*
  - *Predictive Residual Confidence ($r^{(p)}$)*
- **Channel Shift Diagnostic:** Visualizes why unrecorded channels (e.g. `AIT201`) are automatically suppressed ($r < 0.01$) without manual filtering.

### Tab 3: Controlled Sensor Corruption Simulator
- **Live Injection:** Select a target sensor (e.g., flow meter `FIT101`), corruption type (Gaussian Noise, Drift, or Stuck Sensor), and severity slider.
- **Side-by-Side Comparison:**
  - *Baseline C (Without SRM):* Shows the anomaly score surge past threshold, triggering a **false system alarm**.
  - *Proposed R-STNet (With SRM):* Shows the SRM immediately depress the corrupted sensor's reliability to $r \approx 0.01$, keeping the aggregate score safely below threshold and **preventing the false alarm**.

### Tab 4: Validated Benchmark & Findings
- Complete audited clean benchmark table (Isolation Forest, ST-AE, RSTNetNoSRM, R-STNet).
- Transparent acknowledgment of Isolation Forest's clean-data superiority.
- Multi-seed robustness summary and physical boundaries.

---

## 3. Offline Notice

> **Scientific Notice:** This application operates on the Secure Water Treatment (SWaT) industrial testbed dataset and controlled offline fault simulations. It is designed for academic evaluation and research demonstration, not live industrial SCADA control.
