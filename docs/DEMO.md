# Interactive Demonstration Architecture and Operator Guide

**Project:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Students:** P Naga Venkata Kushal (23BDS0133) & Koteru Tarun (23BDS0103)  
**Faculty Guide:** Dr. Dega Nagaraju  
**Technology Stack:** Streamlit / PyTorch / Matplotlib / NumPy  
**File Location:** `demo/app.py`  

---

## 1. Demonstration Objective

The primary objective of the demonstration application is to bridge the gap between abstract mathematical formulas and intuitive physical phenomena for evaluators, examiners, and industrial engineers. Specifically, the demo must visually answer:
> **How does R-STNet distinguish localized sensor unreliability from a systemic cyber-physical attack, and how does this prevent false plant shutdowns?**

---

## 2. Interactive Simulation Workflow

```
[User Selects Target Sensor & Corruption Modality (e.g., Noise on FIT101)]
                                |
                                v
               [Synthetic Perturbation Injected at Time t]
                                |
               +----------------+----------------+
               |                                 |
               v                                 v
   [Baseline C: RSTNetNoSRM]            [Proposed R-STNet with SRM]
               |                                 |
               v                                 v
    Uniform Residual Summing             SRM Evaluates Telemetry:
      S_c(t) surges past tau_c             - Rate-of-change spike: r^(v) -> 0
               |                           - Neighbor mismatch: r^(c) -> 0
               v                           - Large residual: r^(p) -> 0
    🚨 FALSE SYSTEM ALARM                Overall Reliability: r_FIT101 -> 0.015
  (Unnecessary Plant Shutdown)                   |
                                                 v
                                        Weighted Residual Aggregation:
                                          Corrupted residual downweighted 95%
                                          S_prop(t) remains below tau_prop
                                                 |
                                                 v
                                        🛡️ FALSE ALARM PREVENTED
                                      (Plant Monitoring Remains Normal)
```

---

## 3. Key Demonstration Modules

### Module 1: Pre-Computed Cache Loading
The app loads preprocessed SWaT arrays (`data/processed/SWaT/`) and PyTorch model checkpoints (`results/checkpoints/`) using `@st.cache_resource` and `@st.cache_data`. This ensures that launching the app takes **under 5 seconds** and inference execution is instantaneous ($0.060\text{ ms}$ per window).

### Module 2: Live Timeline Tracking
Renders dynamic anomaly score timelines across 300-step windows with real-time comparison against calibrated decision thresholds ($\tau_c = 12.07$ vs. $\tau_{prop} = 3.14$). Genuine cyber-physical attack windows from SWaT are highlighted in red shading, visually demonstrating where Baseline C produces false alarms during normal intervals and where R-STNet accurately recovers normal operation.

### Module 3: Dynamic SRM Breakdown
Provides an interactive bar chart of all 51 physical sensors. Evaluators can select specific treatment stages (Stage 1 Raw Water, Stage 2 Dosing, Stage 3 Ultrafiltration) and inspect the three underlying reliability perspectives:
- Temporal rate-of-change continuity ($r^{(v)}$).
- Spatial graph neighborhood consistency ($r^{(c)}$).
- Predictive residual confidence ($r^{(p)}$).

### Module 4: Live Corruption Injector
Allows users to select any sensor (e.g., flow meter `FIT101`), apply Gaussian noise, stuck reading, or linear drift with an interactive severity slider, and witness the immediate side-by-side diagnostic: Baseline C collapsing into an alarm vs. R-STNet downweighting the sensor and preventing false emergency alarms.

---

## 4. Execution Command

```powershell
streamlit run demo/app.py
```

*Prerequisite:* Ensure PyTorch and project dependencies are installed (`pip install -r requirements.txt`).
