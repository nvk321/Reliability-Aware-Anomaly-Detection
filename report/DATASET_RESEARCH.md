# Dataset Research & Preparation Protocol: Secure Water Treatment (SWaT)

**Project:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Review Target:** Review 1 (Phase 1)  
**Primary Testbed Benchmark:** Secure Water Treatment (SWaT) [38], [45]  
**Secondary Testbed Benchmark:** Water Distribution (WADI) [39]  
**Source Institution:** iTrust Centre for Research in Cyber Security, Singapore University of Technology and Design (SUTD)

---

## 1. Domain Overview & Physical Architecture of SWaT

The Secure Water Treatment (SWaT) testbed is a fully operational, scaled-down industrial water purification facility designed by the Singapore University of Technology and Design (SUTD) for research into cyber-physical system (CPS) security, automation safety, and data-driven anomaly detection. Unlike synthetic simulation datasets, SWaT mirrors a real-world municipal water treatment facility with an operational capacity of approximately 5 gallons per minute, producing water compliant with international potable water standards.

The physical facility spans six interconnected, continuous stages denoted **P1 through P6**, governed by dedicated Programmable Logic Controllers (PLCs), motorized actuators, and an array of physical and chemical sensor transmitters:

```
┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐
│  Stage P1 │───>│  Stage P2 │───>│  Stage P3 │───>│  Stage P4 │───>│  Stage P5 │───>│  Stage P6 │
│ Raw Water │    │ Pre-treat │    │ Ultrafilt │    │ Dechlorin │    │  Reverse  │    │ Backwash  │
│  Inflow   │    │ Coagulat  │    │   (UF)    │    │  (UV/NaOCl)│   │  Osmosis  │    │  Storage  │
└───────────┘    └───────────┘    └───────────┘    └───────────┘    └───────────┘    └───────────┘
```

### Physical Subsystem Specifications:
1. **Stage P1 (Raw Water Intake & Dosing):**
   - Pumps raw water from an external supply into a primary storage tank.
   - Key Instruments: `FIT101` (Inflow Electromagnetic Flow Transmitter), `LIT101` (Raw Water Tank Level Differential Pressure Indicator), `P101`/`P102` (Raw Water Transfer Pumps), `MV101` (Motorized Inflow Valve).
2. **Stage P2 (Pre-treatment & Chemical Dosing):**
   - Regulates chemical dosing (coagulant, sodium hypochlorite, acid, and alkali) to induce flocculation and optimize pH/oxidation levels.
   - Key Instruments: `AIT201`–`AIT203` (Water Quality Analyzers: Conductivity, pH, ORP), `FIT201` (Dosing Flow Meter), `P201`–`P206` (Chemical Dosing Metering Pumps).
3. **Stage P3 (Ultrafiltration - UF):**
   - Removes flocculated suspended solids through semi-permeable ultrafiltration membranes.
   - Key Instruments: `DPIT301` (Differential Pressure across UF Membrane), `FIT301` (UF Feed Flow Meter), `LIT301` (UF Feed Tank Level Indicator), `P301`/`P302` (UF Feed Pumps).
4. **Stage P4 (Dechlorination & UV Disinfection):**
   - Eliminates residual chlorine via chemical dechlorination and exposes water to ultraviolet (UV) irradiation to neutralize microbial pathogens.
   - Key Instruments: `AIT401`/`AIT402` (Free Chlorine and ORP Analyzers), `FIT401` (UV Dechlorination Flow Meter), `LIT401` (Dechlorination Buffer Tank Level), `UV401` (Ultraviolet Dechlorination Lamps).
5. **Stage P5 (Reverse Osmosis - RO):**
   - Drives water under high hydraulic pressure through reverse osmosis membranes to remove dissolved inorganic salts and minerals.
   - Key Instruments: `AIT501`–`AIT504` (Conductivity, Silica, and Salt Analyzers), `FIT501`–`FIT504` (Permeate and Concentrate Flow Meters), `PIT501`–`PIT503` (High-Pressure RO Feed Transmitters), `P501`/`P502` (High-Pressure Booster Pumps).
6. **Stage P6 (Clean Water Storage & Backwash):**
   - Stores purified product water and executes cyclic automated backwash flushing to regenerate Stage P3 ultrafiltration membranes.
   - Key Instruments: `FIT601` (Backwash Flow Meter), `P601`–`P603` (Membrane Backwash Pumps).

---

## 2. Multivariate Sensor/Actuator Structure

The SWaT testbed telemetry monitors a total of **$M = 51$ continuous and discrete channels**:
- **Continuous Physical Process Sensors (25 channels):**
  - Continuous float/differential pressure level indicators (`LIT101`, `LIT301`, `LIT401`).
  - Electromagnetic and turbine flow transmitters (`FIT101`, `FIT201`, `FIT301`, `FIT401`, `FIT501`–`FIT504`, `FIT601`).
  - High-pressure transmitters and membrane differential pressure sensors (`PIT501`–`PIT503`, `DPIT301`).
  - Multi-parameter electrochemical water quality analyzers (`AIT201`–`AIT203`, `AIT401`–`AIT402`, `AIT501`–`AIT504`).
- **Discrete Actuators and Supervisory Valves (26 channels):**
  - Motorized modulating and isolation valves (`MV101`, `MV201`, `MV301`–`MV304`, `MV501`–`MV504`).
  - Two-speed and variable-speed motorized centrifugal pumps (`P101`, `P102`, `P201`–`P206`, `P301`, `P302`, `P401`–`P404`, `P501`, `P502`, `P601`–`P603`).
  - Ultraviolet irradiation disinfection lamps (`UV401`).

### The Intrinsic Physical Coupling of SWaT
The continuous physical state of SWaT is tightly governed by first-principles conservation laws:
- **Mass Balance in Tank P1:** $\frac{d}{dt} \text{LIT101} \propto \text{FIT101} - \text{FIT201}$. If pump `P101` is running and motorized valve `MV101` is closed, the fluid level `LIT101` must steadily decline.
- **Hydraulic Transport Continuity:** Flow rate `FIT301` is physically constrained by the operational state of pump `P301` and differential pressure across membrane `DPIT301`.
- **Chemical Equilibrium:** Chemical analyzer `AIT202` (pH) responds directly to the volumetric dosing of chemical pumps `P201` and `P203`.

This strict physical coupling provides the foundational basis for our **Sensor Reliability Module (SRM)**: because physical laws tightly bind correlated channels, an isolated sensor cannot arbitrarily deviate without violating consensus with its physical neighbors.

---

## 3. Official SUTD Dataset Architecture and the Reconstructed Sequence

The official release of the SWaT benchmark corresponds to the **SWaT.A1 & A2 (December 2015)** testbed operational run. A comprehensive technical audit of the raw data files stored in `reference/swat_data/` reveals the exact composition of the benchmark:

### 3.1. Official 7-Day Normal Run (`SWaT_Dataset_Normal_v1.csv`)
- **Duration:** 7 continuous days (168 hours) of uninterrupted nominal operation.
- **Timestamp Coverage:** `22/12/2015 4:00:00 PM` to `28/12/2015 9:59:59 AM`.
- **Sample Count:** Exactly **496,800 rows** sampled at 1 Hz ($y = 0$, 100% normal).
- **Physical Characteristics:** Captures full diurnal cyclic transients, automated backwash cycles, chemical dosing adjustments, and normal steady-state hydraulic oscillations.

### 3.2. Official 4-Day Attack Test Run (`SWaT_Dataset_Attack_v0.csv`)
- **Duration:** 4 continuous days (approximately 96 hours) under active cyber-physical attack testing.
- **Timestamp Coverage:** `28/12/2015 10:00:00 AM` to `02/01/2016 2:59:59 PM`.
- **Sample Count:** Exactly **449,919 rows** sampled at 1 Hz.
- **Attack Composition:** Contains 36 distinct cyber-physical attacks executed across single-stage and multi-stage configurations:
  - *Single-Stage Single-Point (SSSP):* Altering a single sensor reading (e.g., spoofing `LIT101` low to cause tank overflow) or forcing an actuator state.
  - *Single-Stage Multi-Point (SSMP):* Simultaneously manipulating multiple instruments within a single stage.
  - *Multi-Stage Single-Point (MSSP):* Coordinated attacks across adjacent treatment stages (e.g., stages P1 and P3).
  - *Multi-Stage Multi-Point (MSMP):* Complex, stealthy coordinated attacks designed to evade static threshold alarms.
- **Attack Density:** Exactly **54,621 attack seconds** ($\approx 12.14\%$) interspersed chronologically with **395,298 normal operational seconds** ($\approx 87.86\%$).

---

## 4. Crucial Conceptual Distinction: Attack Labels vs. Sensor Fault Ground Truth

A critical contribution of our Phase 1 project audit is establishing the distinction between **cyber-physical attack labels** and **sensor fault ground truth**:

> [!IMPORTANT]
> **The Official Benchmark Ground-Truth Reality:**  
> The ground-truth binary labels ($y_{\text{sys}} \in \{0, 1\}$) provided by iTrust SUTD represent **cyber-physical attack intervals** affecting the plant. They do **NOT** provide ground truth indicating that an individual sensor instrument is physically damaged, uncalibrated, noisy, or faulty!

### Why This Distinction Matters:
1. In a genuine cyber-physical attack (e.g., an attacker forces valve `MV101` closed, causing water level `LIT101` to drop rapidly), the sensor `LIT101` is **functioning perfectly and reporting the true physical state of the plant**. The anomaly is an **attack on the physical system (Case C)**, not a sensor hardware malfunction!
2. Conversely, if a pressure transmitter `DPIT301` experiences high Gaussian noise or drifts due to membrane fouling during a normal period, the official SWaT label is $y=0$ (Normal). A standard detector flags an anomaly, which is counted as a false positive alarm.
3. Therefore, **real-world benchmark datasets alone cannot evaluate whether a model can accurately isolate degraded sensors from healthy ones**.
4. To scientifically resolve this limitation, our project introduces the **Controlled Sensor Corruption Framework (`src/corruption/corruptor.py`)**, which overlays reproducible synthetic sensor failure modes onto the evaluation set and generates an explicit binary ground-truth mask:
   $$M_{\text{fault}}(t, i) = \begin{cases} 1, & \text{if sensor } i \text{ is synthetically degraded at time } t \\ 0, & \text{otherwise} \end{cases}$$
   This dual ground-truth framework enables simultaneous, independent evaluation of plant-wide attack detection ($y_{\text{sys}}$) and instrument-level fault localization ($M_{\text{fault}}$).

---

## 5. Project Preprocessing & Chronological Partitioning Protocol

To eliminate all possibility of data leakage and adhere to the highest scientific standards of time-series benchmarking [40], [41]:

### 5.1. Chronological Partitioning (Project-Specific Split)
Time-series observations violate the independent and identically distributed (i.i.d.) assumption; random train-test splitting or standard K-fold cross-validation causes massive lookahead leakage. We enforce strict chronological partitioning:

```
[================== 7-DAY NORMAL SEQUENCE (496,800 rows) ==================] [====== 4-DAY ATTACK SEQUENCE (449,919 rows) ======]
[---------- 80% TRAIN SPLIT ----------] [------- 20% VAL SPLIT -------] [----------------- TEST SPLIT -----------------]
      397,440 rows (100% Normal)              99,360 rows (100% Normal)               449,919 rows (12.14% Attack Density)
```

- **Training Split (80% Normal):** First 397,440 rows (or 39,744 samples at 10x downsample). Contains pure normal operation ($y=0$). All neural parameters and scalers are trained exclusively on this split.
- **Validation Split (20% Normal):** Next 99,360 rows (or 9,936 samples at 10x downsample). Contains pure normal operation ($y=0$). Used strictly for:
  - Computing normal residual distributions ($\text{Median}(e_i)$ and $\text{MAD}_i$).
  - Determining normal rate-of-change quantiles ($\delta_i^{\max}$).
  - Calibrating SRM evidential fusion weights ($w_v, w_c, w_p, b_r$).
  - Setting decision thresholds ($\tau_{\text{sys}}, \theta_{\text{rel}}$).
  - **No test attack labels are ever referenced during calibration.**
- **Test Split (Official 4-Day Attack Sequence):** Reconstructed chronologically from `normal.csv` and `attack.csv` to form the continuous 449,919-row sequence (44,992 samples at 10x downsample). Evaluated blindly.

### 5.2. Downsampling Protocol
Industrial water processes exhibit hydraulic inertia and settling times measured in minutes. Sampling at 1 Hz produces redundant consecutive observations and extreme computational overhead. Consistent with established literature (GDN [19], MTAD-GAT [20], CAROTS [21]):
- Downsampling by factor **10x** yields a clean 10-second sampling interval (39,744 train, 9,936 val, 44,992 test samples), reducing epoch training time by 10x while fully preserving physical transients.
- Downsampling by factor **5x** (5-second intervals) is supported for full resolution benchmarking.

### 5.3. Leak-Free Scaling
Sensor channels in SWaT span widely differing physical units: flow meters record $0$–$2500$ m$^3$/h, level sensors record $0$–$1200$ mm, conductivity meters record $0$–$1000$ $\mu$S/cm, and actuators are discrete $\{0, 1, 2\}$.
- We employ `RobustStandardScaler`:
  $$\tilde{x}_i(t) = \frac{x_i(t) - \mu_{\text{train}, i}}{\sigma_{\text{train}, i} + \epsilon}$$
- **Variance Protection:** Certain digital backup pumps (e.g., `P102`, `P202`) remain completely off throughout normal training ($\sigma_{\text{train}, i} = 0$). To prevent numerical breakdown (NaNs), our pipeline clamps zero-variance denominators: $\sigma_{\text{train}, i} = 1.0$ if $\sigma_{\text{train}, i} < 10^{-6}$.

### 5.4. Chronological Sliding Window Construction
Individual static time points do not contain sequential temporal context. The input tensor is constructed using a chronological sliding window:
$$\mathbf{W}_t = \begin{bmatrix} x_1(t - L + 1) & \cdots & x_M(t - L + 1) \\ \vdots & \ddots & \vdots \\ x_1(t) & \cdots & x_M(t) \end{bmatrix} \in \mathbb{R}^{L \times M}$$
- Lookback window length: $L = 60$ steps ($\approx 10$ minutes of physical process at 10s downsampling, or 5 minutes at 5s downsampling).
- Stride: $S=1$ during inference and test evaluation; $S=1$ (or $S=5$) during training.
