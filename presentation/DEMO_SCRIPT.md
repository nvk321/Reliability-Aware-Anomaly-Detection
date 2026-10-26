# Interactive Demo Script & Presentation Walkthrough

**Project Title:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Course:** BCSE209P — Machine Learning Lab  
**Students:** P Naga Venkata Kushal (23BDS0133), Koteru Tarun (23BDS0103)  
**Faculty Guide:** Dr. Dega Nagaraju  
**Demonstration Target Duration:** 3 to 5 Minutes  
**Tool / Dashboard:** Streamlit Interactive Web Application (`demo/app.py`)

---

## 1. Demo Preparation & Setup Checklist

Before initiating the oral defense demonstration:

1. **Activate Virtual Environment:**
   ```bash
   .venv\Scripts\activate
   ```
2. **Launch Streamlit Dashboard:**
   ```bash
   streamlit run demo/app.py
   ```
3. **Open Browser:**
   - URL: `http://localhost:8501`
   - Verify all 4 tabs load smoothly:
     - Tab 1: **SWaT Process & Anomaly Timeline**
     - Tab 2: **Sensor Reliability Module (SRM)**
     - Tab 3: **Live Sensor Corruption Simulation**
     - Tab 4: **Audited Benchmark & Ablation Study**
4. **Display Mode:** Fullscreen browser mode (Press `F11`).

---

## 2. Chronological Walkthrough (3 to 5 Minutes)

### Phase 1: Problem Definition & Motivation (0:00 – 0:30)
- **Speaker Action:** Focus on the dashboard header and introduce the core research question.
- **Spoken Commentary:**
  > *"Respected guide and examiners, in critical cyber-physical systems like water treatment facilities and power grids, multivariate deep-learning models frequently suffer from alarm fatigue. A single failing, drifting, or noisy sensor can cause uniform residual aggregation to trigger false emergency plant shutdowns.*  
  > *Our core research question asks: **Can explicit sensor-reliability estimation improve the robustness of multivariate time-series anomaly detection when sensor observations are missing, noisy, or corrupted?**  
  > *To test this, we developed **R-STNet**, a dual-perspective spatio-temporal architecture with a dedicated Sensor Reliability Module (SRM)."*

---

### Phase 2: SWaT Process & Anomaly Timeline (0:30 – 1:15)
- **Speaker Action:** Navigate to **Tab 1: SWaT Process & Anomaly Timeline**. Select sensor `FIT101` and `LIT301`. Hover over the anomaly score plot.
- **Spoken Commentary:**
  > *"Here we inspect the Secure Water Treatment (SWaT) testbed, comprising 51 sensor and actuator channels spanning six chemical treatment stages.  
  > In the top panel, we display the multi-channel sensor telemetry over the 44,991-window test benchmark, containing 36 physical cyber-attack intervals marked by the red shaded bands.  
  > In the lower panel, R-STNet's reliability-weighted anomaly score tracks system states. When an attack alters physical water levels (such as tank overflows), the anomaly score crosses the calibrated threshold $\tau = 0.5284$, correctly raising alerts."*

---

### Phase 3: R-STNet Architecture & Dual-Perspective SRM (1:15 – 2:00)
- **Speaker Action:** Navigate to **Tab 2: Sensor Reliability Module (SRM)**. Select a clean channel like `FIT101`.
- **Spoken Commentary:**
  > *"Instead of directly summing reconstruction errors, R-STNet evaluates each sensor's health across two statistical perspectives:  
  > First, a **Statistical Range Assessment** ($z$-score clipping and variance checks), and second, a **Dynamic Rate-of-Change Assessment** ($L_1$-norm consecutive first-differences).  
  > On clean operational data, sensors maintain a healthy reliability score of approximately $0.30$ to $0.85$.  
  > When we select `AIT201`, notice that its reliability score drops to $0.05$. As identified in our channel-shift analysis, `AIT201` was unrecorded during normal training and shifted during testing. The SRM dynamically recognizes this unreliability and suppresses its influence on the global anomaly score."*

---

### Phase 4: Live Sensor Corruption Simulation (2:00 – 2:45)
- **Speaker Action:** Navigate to **Tab 3: Live Sensor Corruption Simulation**.
  1. Select Target Channel: `LIT101`.
  2. Select Corruption Type: **Gaussian Noise ($\sigma = 0.50$)**.
  3. Click **Inject Corruption & Re-evaluate**.
- **Spoken Commentary:**
  > *"Now let us test our hypothesis under controlled sensor degradation. We inject zero-mean Gaussian noise ($\sigma = 0.50$) into water level sensor `LIT101` while the physical treatment plant remains completely normal.  
  > In the top plot, you can see the corrupted sensor reading (red) deviating wildly from the ground-truth physical reading (blue).  
  > In a conventional architecture, this extreme deviation would immediately blow up the reconstruction residual and trigger an emergency plant shutdown."*

---

### Phase 5: Reliability Suppression & False Alarm Elimination (2:45 – 3:30)
- **Speaker Action:** Point to the two comparison panels below the corruption simulation in Tab 3.
- **Spoken Commentary:**
  > *"Notice what happens in R-STNet:  
  > As shown in the gauge, the SRM instantly detects the abnormal high-frequency noise jitter, causing `LIT101`'s reliability score to collapse from $0.462$ down to $0.012$—a separation of over $+0.40$!  
  > In the comparison chart on the right:  
  > Under the unweighted model (**Baseline C: RSTNetNoSRM**), the anomaly score spikes to $0.89$, triggering a persistent **FALSE POSITIVE**.  
  > Under **R-STNet**, because the corrupted channel is weighted down by $r_i \approx 0.01$, the system anomaly score stays at $0.21$, safely below the threshold $\tau = 0.5284$. **The false alarm is completely eliminated!**"*

---

### Phase 6: Audited Benchmark & Ablation Comparison (3:30 – 4:15)
- **Speaker Action:** Navigate to **Tab 4: Audited Benchmark & Ablation Study**.
- **Spoken Commentary:**
  > *"Let us examine the audited benchmark results across all 44,991 test windows.  
  > Comparing R-STNet against its direct ablation baseline `RSTNetNoSRM` under the identical 23k-parameter network:  
  > - False Alarm Rate drops from $99.86\%$ down to $62.14\%$.  
  > - Total false alarm windows reduced by **14,898 windows**—a $37.8\%$ false alarm reduction.  
  > - Precision improves from $0.1218$ to $0.1626$, lifting $F_1$ from $0.2171$ to $0.2740$.  
  > - Furthermore, on our controlled corruption benchmark across 9 degradation modes, the SRM maintained $100\%$ synthetic fault recall with an average separation of $+0.29$."*

---

### Phase 7: Negative Findings, Limitations & Conclusion (4:15 – 5:00)
- **Speaker Action:** Highlight the 'Scientific Integrity & Negative Findings' callout in Tab 4.
- **Spoken Commentary:**
  > *"Finally, we present our findings with strict scientific rigor:  
  > First, **Isolation Forest** remains the strongest overall model on clean SWaT data ($F_1 = 0.7182, \text{FPR} = 0.92\%$). We do not claim R-STNet beats Isolation Forest on clean data.  
  > Second, statistical reliability cannot detect stuck inactive actuators: when backup pump `P202` is stuck at zero while already idle, rate-of-change is zero, requiring PLC supervisory logic for detection.  
  > In conclusion, our hypothesis is supported within the spatio-temporal deep-learning paradigm: explicit reliability estimation effectively curbs residual saturation and protects multivariate monitoring against sensor degradation.  
  > Thank you, and we welcome your questions."*

---

## 3. Anticipated Viva / Examiner Questions & Defensible Answers

### Q1: "Why does Isolation Forest outperform R-STNet on clean SWaT data?"
- **Answer:** *"Isolation Forest constructs orthogonal axis-aligned partition trees, making it inherently robust to individual non-stationary sensor shifts (such as `AIT201`). Deep autoencoders, conversely, backpropagate joint spatio-temporal reconstruction loss across all 51 dimensions. When an uninformative or shifted sensor deviates, the reconstruction error ripples across the latent graph, causing residual saturation. While the SRM mitigates this by eliminating 14,898 false alarms, the underlying tree partitioning remains superior for uncorrupted tabular distributions."*

### Q2: "Can your SRM be used upstream of the graph attention layer?"
- **Answer:** *"In the current v1.0 architecture, the SRM weights the reconstruction residual downstream ($s(t) = \sum_i r_i \cdot e_i$). In Section 5.4 (Future Scope), we outline that integrating reliability weights upstream—as attention bias in the GAT adjacency matrix—is a natural next step to prevent corrupted sensor features from corrupting neighbor representations during message passing."*

### Q3: "Did you test this on a live water treatment plant?"
- **Answer:** *"No. This is an offline experimental validation using the official SUTD iTrust SWaT testbed dataset and controlled synthetic corruption injections. We explicitly state in our thesis and demo that this is not a live industrial control-system deployment."*
