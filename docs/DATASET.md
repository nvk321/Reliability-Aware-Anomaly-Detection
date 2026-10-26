# SWaT Dataset Documentation, Attribution & Access Guidelines

**Dataset Name:** Secure Water Treatment (SWaT) Testbed Dataset  
**Release Version:** SWaT.A1 & A2 (December 2015 Collection)  
**Provider / Institution:** iTrust, Centre for Research in Cyber Security, Singapore University of Technology and Design (SUTD)  
**Official Portal:** [iTrust Datasets Page](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)  
**License / Terms:** Academic Research Use Only (Subject to SUTD iTrust Data Transfer Agreement)  

---

## 1. Overview and Physical System Description

The Secure Water Treatment (SWaT) dataset is an internationally recognized benchmark for cyber-physical systems (CPS) security research. Developed by the iTrust research center at SUTD, SWaT is an operational scaled-down physical water treatment testbed producing five gallons per minute of purified water.

The facility comprises six interconnected physical stages:
1. **Stage 1 (P1) — Raw Water Intake & Level Control:** Controls raw water inflow via motorized valve `MV101`, monitors tank level with differential level transmitter `LIT101`, and pumps water forward via pump `P101` (with backup `P102`).
2. **Stage 2 (P2) — Chemical Pre-Treatment & Coagulation:** Doses sodium hypochlorite (NaOCl), alum, and polyaluminum chloride to reduce fouling and buffer pH. Monitored via flow meter `FIT201`, chemical analyzers `AIT201`–`AIT203`, and metering pumps `P201`–`P206`.
3. **Stage 3 (P3) — Ultrafiltration (UF):** Removes suspended solids via semi-permeable membranes. Monitored via differential pressure transmitter `DPIT301`, level transmitter `LIT301`, backwash valves `MV301`–`MV304`, and UF feed pumps `P301`–`P302`.
4. **Stage 4 (P4) — Dechlorination & Ultraviolet Disinfection:** Destroys residual chlorine via sodium bisulfite (NaHSO3) dosing and neutralizes microorganisms via ultraviolet (UV) lamp `UV401`. Monitored via analyzers `AIT401`–`AIT402`, level sensor `LIT401`, and dosing pumps `P401`–`P404`.
5. **Stage 5 (P5) — Reverse Osmosis (RO) Desalination:** High-pressure reverse osmosis filtration splitting water into permeate (purified stream) and concentrate (waste brine). Monitored via pressure transmitters `PIT501`–`PIT503`, flux flow meters `FIT501`–`FIT504`, and high-pressure pumps `P501`–`P502`.
6. **Stage 6 (P6) — Storage & Distribution:** Stores treated water in distribution tanks and pumps it to consumer supply lines via pumps `P601`–`P603` and flow meter `FIT601`.

### Monitored Telemetry Variables ($M = 51$)
The testbed logs $51$ physical telemetry signals at a native sampling frequency of 1 Hz:
- **25 Continuous Process Sensors:** Float level transmitters (`LIT`), magnetic flow meters (`FIT`), differential and gauge pressure transmitters (`DPIT`, `PIT`), and multi-parameter electrochemical water quality analyzers (`AIT`).
- **26 Discrete Actuators:** Two-state and three-state motorized valves (`MV`), primary and redundant chemical dosing pumps (`P`), and ultraviolet units (`UV`).

---

## 2. Dataset Collection Phases and Chronology

The December 2015 release encompasses 11 continuous operational days:
- **Phase A1 (Normal Operational Baseline):** Recorded from 22 December 2015 to 28 December 2015 ($7$ consecutive days, $496,800$ continuous records at 1 Hz). During this period, the plant operated under nominal automated closed-loop PLC logic without cyber or physical attacks.
- **Phase A2 (Cyber-Physical Attack Scenarios):** Recorded from 28 December 2015 to 2 January 2016 ($4$ consecutive days, $449,919$ continuous records at 1 Hz). During this period, 36 distinct cyber-physical attacks were executed against physical actuators, PLC controllers, and SCADA communication links. Attacks lasted from a few minutes to several hours, targeting individual sensors (single-point spoofing) and multi-stage control loops (coordinated physical attacks).

---

## 3. Data Partitions Used in This Project

In strict compliance with machine learning best practices and temporal causality, data was partitioned chronologically without random shuffling:

| Partition Name | Source Dataset Segment | Raw Rows | Downsampled Rows ($10\times$) | Sliding Windows ($L=30$) | Anomaly Prevalence ($y=1$) | Methodological Purpose |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Training Split** | First $80\%$ of Phase A1 (Days 1–5.6) | $397,440$ | $39,744$ | $39,715$ | $0$ ($0.0\%$) | Model training (`train.npy`); fitting `RobustStandardScaler`. |
| **Validation Split** | Final $20\%$ of Phase A1 (Days 5.6–7) | $99,360$ | $9,936$ | $9,907$ | $0$ ($0.0\%$) | Threshold calibration ($\tau = \text{Quantile}(S_{\text{val}}, 0.95)$); hyperparameter validation. |
| **Test Split** | Full Phase A2 (Days 8–11) | $449,919$ | $44,992$ | $44,963$ | $5,468$ ($12.16\%$) | Final benchmark evaluation (`test.npy`, `test_labels.npy`). |

---

## 4. Preprocessing and Zero-Variance Handling

1. **Downsampling ($10\times$):** Industrial 1 Hz telemetry contains measurement noise and oversampling redundancy. Telemetry was downsampled to $10$-second intervals, preserving process dynamics while dramatically improving computational efficiency.
2. **Leak-Free Robust Scaling:** Scaler parameters (mean $\mu$ and standard deviation $\sigma$) were fitted **strictly on the 80% training split**.
3. **Zero-Variance Actuator Protection:** Backup actuators (e.g., `P202`, `MV101`) have zero variance during normal training ($\sigma = 0$). To prevent division-by-zero, scaling sets $\sigma^* = 1.0$ if $\sigma < 10^{-6}$.
4. **Sliding Windows ($L=30, S=1$):** Windows spanning 300 seconds (5 minutes) capture temporal dynamics, with labels assigned causally via `point_end` labeling ($y(\mathbf{X}_t) = y_t$).

---

## 5. Controlled Synthetic Sensor Corruption Suite

Because the native SWaT dataset contains cyber-physical attacks but does not label non-adversarial sensor hardware faults, this project introduced a controlled synthetic perturbation suite evaluated across random seeds $[42, 101, 2024]$:
- **Experiment E2 (Additive Gaussian Noise):** $\sigma \in \{0.1, 0.3, 0.5\}$ on channels $\{3, 14, 28\}$.
- **Experiment E3 (Stuck Sensors):** $k \in \{1, 3, 5\}$ simultaneously frozen sensor channels.
- **Experiment E4 (Dynamic Sensor Drift):** Linear ramp drift $\delta \in \{0.1, 0.3, 0.5\}$.

*Important Distinction:* Synthetic corruption masks ($\mathbf{M}_{\text{synth}}$) are strictly distinguished from native SWaT attack labels ($y_{\text{native}}$).

---

## 6. SUTD iTrust Attribution and Terms of Access

### Data Redistribution Policy
In accordance with the Data Transfer Agreement of the iTrust Centre for Research in Cyber Security (SUTD):
> **This repository DOES NOT redistribute the raw SWaT dataset files (`normal.csv`, `attack.csv`).**

Researchers wishing to replicate this work must request the raw SWaT dataset directly from iTrust SUTD via their official portal:  
👉 [https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)

Once downloaded, raw files should be placed in `reference/swat_data/` as documented in `data/README.md`.

### Required Official Citation
Any publication or report utilizing the SWaT testbed dataset must cite the original benchmark publication:

```bibtex
@inproceedings{goh2016dataset,
  author    = {Goh, Jonathan and Adepu, Sridhar and Junejo, Khurum Nazir and Mathur, Aditya},
  title     = {A Dataset to Support Research in the Design of Secure Water Treatment Systems},
  booktitle = {Proceedings of the 11th International Conference on Critical Information Infrastructures Security (CRITIS)},
  series    = {Lecture Notes in Computer Science},
  volume    = {10047},
  pages     = {88--99},
  year      = {2016},
  publisher = {Springer},
  doi       = {10.1007/978-3-319-71368-7_8}
}
```

IEEE Citation Format:
> [1] J. Goh, S. Adepu, K. N. Junejo, and A. Mathur, "A Dataset to Support Research in the Design of Secure Water Treatment Systems," in *Critical Information Infrastructures Security (CRITIS 2016)*, Lecture Notes in Computer Science, vol. 10047, pp. 88–99, Springer, 2016, doi: 10.1007/978-3-319-71368-7_8.
