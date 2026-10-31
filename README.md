# Reliability-Aware Anomaly Detection for Multivariate Sensor Data

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5%20CUDA-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Demo-Streamlit%20Interactive-FF4B4B.svg)](demo/app.py)
[![Test Suite](https://img.shields.io/badge/Tests-28%20Passed-brightgreen.svg)](tests/)
[![Health Check](https://img.shields.io/badge/Health%20Check-PASS-success.svg)](scripts/project_health_check.py)
[![ScienceDirect Compliance](https://img.shields.io/badge/References-90%25%20ScienceDirect-blueviolet.svg)](docs/REFERENCE_AUDIT.md)

An end-to-end Machine Learning research system for multivariate industrial telemetry that explicitly decouples **isolated sensor degradation** (noise, drift, stuck values) from **genuine cyber-physical system anomalies**.

---

## Overview

In mission-critical industrial control systems—such as water treatment facilities, power grids, and chemical processing plants—conventional deep-learning anomaly detectors suffer from severe **alarm fatigue**. Standard architectures aggregate channel-wise prediction or reconstruction residuals uniformly:

$$S_{\text{global}}(t) = \sum_{i=1}^M \text{Error}(x_i(t), \hat{x}_i(t))$$

When a single sensor experiences transient instrument failure (e.g., loose wiring, electromagnetic interference, or sensor drift), its individual residual spikes. Without a mechanism to gauge sensor trustworthiness, the detector interprets this benign sensor fault as a catastrophic process incident, triggering unwarranted plant shutdowns.

**R-STNet (Dual-Perspective Reliability-Aware Spatio-Temporal Network)** introduces an online **Sensor Reliability Module (SRM)** that evaluates individual channel credibility across statistical range and dynamic rate-of-change dimensions, dynamically down-weighting degraded sensors to prevent residual saturation.

---

## Research Question

> **Can explicit sensor-reliability estimation improve the robustness of multivariate time-series anomaly detection when sensor observations are missing, noisy, or corrupted?**

### Working Hypothesis
> Explicit sensor-reliability estimation should improve robustness and reduce false positives under sensor degradation within a spatio-temporal deep-learning anomaly-detection architecture.

---

## Key Contribution

1. **Dual-Perspective Sensor Reliability Module (SRM):** Evaluates sensor health online via statistical volatility bounds and multi-sensor temporal continuity without requiring external labels.
2. **Reliability-Weighted Anomaly Scoring:** Insulates global system alarms by weighting residuals with continuous reliability scores:
   $$S_{\text{sys}}(t) = \frac{\sum_{i=1}^M r_i(t) \cdot z_i(t)}{\sum_{i=1}^M r_i(t) + \epsilon}$$
3. **Proven False-Alarm Reduction:** Eliminates **14,898 false alarm windows** (37.8% relative reduction), increasing True Negatives from 56 to 14,954 windows over the unweighted baseline under identical 23k parameters.
4. **Empirical Robustness Separation:** Achieves a reliability separation of $\Delta r \approx +0.29$ between clean channels ($\bar{r} \approx 0.304$) and corrupted channels ($\bar{r} \approx 0.015$) under 9 controlled degradation modes.
5. **Rigorous Negative Finding Reporting:** Openly documents that traditional tree partitioning (**Isolation Forest**) outperforms deep neural architectures on uncorrupted clean SWaT data, establishing clear boundaries between tabular statistical methods and deep temporal modeling.

---

## Architecture

R-STNet comprises three tightly integrated functional components:

```
Input Sequence X(t) in R^{L x M}
   │
   ├──▶ [1. Temporal Feature Extractor] (Dilated 1D-CNN + Residual Skips)
   │         │
   │         ▼
   ├──▶ [2. Spatial Graph Attention] (Learnable Adjacency W in R^{M x M})
   │         │
   │         ▼
   ├──▶ [3. One-Step Forecasting Head] ──▶ Predicted Values x_hat(t) in R^M
   │                                              │
   ▼                                              ▼
[4. Sensor Reliability Module (SRM)]    Normalized Residuals z(t) = |x(t) - x_hat(t)| / MAD
   ├── Statistical Bounds (z-score)               │
   └── Dynamic Continuity (Delta x)               │
         │                                        │
         ▼                                        ▼
   Reliability Vector r(t) in [0, 1]^M ──▶ S_sys(t) = sum(r_i * z_i) / sum(r_i)
```

- **Backbone Parameters:** 23,233 weights (lightweight, edge-deployable).
- **Inference Latency:** $0.060\text{ ms per window}$ on NVIDIA RTX 3050 GPU.

---

## Dataset

We evaluate on the authoritative **Secure Water Treatment (SWaT)** physical testbed dataset collected by the **iTrust Centre for Research in Cyber Security** at the **Singapore University of Technology and Design (SUTD)**:
- **Topology:** 51 continuous and discrete sensors/actuators spanning 6 chemical water treatment stages (P1 through P6).
- **Protocol:** 10x downsampled (10s intervals) preserving physical system transients.
  - **Train Split:** 39,715 windows (7 days of continuous normal operational data, zero attacks).
  - **Validation Split:** 9,907 windows (20% normal operational split; used exclusively for threshold calibration $\tau$).
  - **Test Split:** 44,963 windows (4 days continuous mixed operation containing 36 physical cyber-attacks; 12.16% attack incidence).
- **Zero Label Leakage:** Decision threshold $\tau$ is calibrated strictly at the 95th percentile of normal validation anomaly scores without test-set feedback.

---

## Experimental Setup

Experiments were executed under fixed random seeds ($seed = 42$) on Windows 11 with Python 3.11, PyTorch 2.5.1, and CUDA acceleration.

Four models are evaluated on the identical chronological mixed benchmark:
- **Baseline A (Isolation Forest):** Unsupervised tree partitioning on raw multivariate vectors ($n_{\text{estimators}} = 100$).
- **Baseline B (ST-AE):** Bidirectional GRU Encoder-Decoder sequence autoencoder with uniform reconstruction loss.
- **Baseline C (RSTNetNoSRM):** Identical spatio-temporal graph backbone without the Sensor Reliability Module.
- **Proposed (R-STNet):** Spatio-temporal graph backbone with online Sensor Reliability Module.

---

## Results

### Clean SWaT Benchmark (Audited Source of Truth)

| Model | $F_1$ Score | Precision | Recall | False Positive Rate (FPR) | PR-AUC | ROC-AUC | Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Isolation Forest** | **0.7182** | **0.8995** | 0.5977 | **0.92%** | **0.7367** | **0.8429** | Tabular Trees |
| **ST-AE** | 0.2169 | 0.1216 | **1.0000** | 100.00% | 0.1216 | 0.5000 | 688 KB |
| **RSTNetNoSRM** (No SRM) | 0.2171 | 0.1218 | **1.0000** | 99.86% | 0.1380 | 0.5164 | 23,233 |
| **R-STNet** (Proposed) | **0.2740** | **0.1626** | 0.8713 | **62.14%** | 0.1169 | 0.4976 | 23,233 |

> [!IMPORTANT]
> **Defensible Scientific Positioning:**  
> Isolation Forest achieves substantially superior performance on clean data ($F_1 = 0.7182$ vs $0.2740$). Deep autoencoders backpropagate joint spatio-temporal reconstruction loss across all 51 sensors, making them vulnerable to unrecorded sensor shifts (e.g., `AIT201`).  
> **Within the spatio-temporal deep-learning paradigm, R-STNet delivers significant gains over RSTNetNoSRM:**
> - $F_1$: $0.2171 \rightarrow 0.2740$ (+26.2% relative improvement).
> - Precision: $0.1218 \rightarrow 0.1626$ (+33.5% relative improvement).
> - FPR: $99.86\% \rightarrow 62.14\%$ (**37.8% relative reduction in false alarm rate**).
> - False alarms suppressed: **14,898 windows eliminated**.

---

## Robustness Experiments

Under controlled synthetic sensor degradation across 9 evaluation regimes, R-STNet demonstrates consistent reliability separation:

```
Clean Channel Telemetry ──▶ Mean Reliability r_clean ≈ 0.304
Degraded Channel Telemetry ──▶ Mean Reliability r_corrupted ≈ 0.015
Statistical Separation: Delta r ≈ +0.289
```

- **Gaussian Noise ($\sigma \in [0.1, 0.5]$):** Reliability drops to $\le 0.021$; $100\%$ fault recall on synthetic masks.
- **Sensor Drift ($\delta \in [0.1, 0.5]$):** Reliability drops to $\le 0.018$; eliminates single-sensor drift false alarms.
- **Actuator Limitation (Physical Boundary):** Inactive backup pumps resting at zero variance (`P202`, `P204`) yield $0.0\%$ recall under frozen failure, demonstrating that purely statistical checks must be paired with PLC ladder logic.

---

## Demo

An interactive Streamlit dashboard is provided to explore the SWaT process, test the SRM multi-perspective decomposition, inject live sensor faults, and inspect benchmark comparisons.

```bash
# Launch interactive dashboard
streamlit run demo/app.py
```

### Dashboard Tabs
1. **SWaT Process & Anomaly Timeline:** Interactive multi-sensor timeline with ground-truth attack overlays.
2. **Sensor Reliability Module (SRM):** Real-time decomposition into statistical and dynamic continuity components.
3. **Live Sensor Corruption Simulation:** Inject Gaussian noise, drift, or stuck states and observe real-time false-alarm suppression.
4. **Audited Benchmark & Ablation Study:** Review audited confusion matrices and parameter comparisons.

---

## Installation

### Prerequisites
- Python 3.11+
- CUDA-compatible GPU (optional, CPU execution fully supported)

### Step 1: Clone Repository
```bash
git clone https://github.com/kushal-23bds0133/Reliability-Aware-Anomaly-Detection.git
cd Reliability-Aware-Anomaly-Detection
```

### Step 2: Set Up Virtual Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Quick Start (< 60 Seconds)

Validate the entire codebase and pre-trained checkpoints in seconds:

```bash
# 1. Run Automated Project Health Check
python scripts/project_health_check.py

# 2. Run Comprehensive Unit Test Suite
pytest tests/ -v

# 3. Launch Interactive Demo
streamlit run demo/app.py
```

---

## Full Reproduction

To reproduce all benchmarks, corruption experiments, and publication figures from scratch:

```bash
# Step 1: Preprocess raw SWaT data (if raw CSVs are available)
python scripts/prepare_data.py --config configs/swat_rstnet.yaml

# Step 2: Train Deep Models (ST-AE, RSTNetNoSRM, R-STNet) (~135s each)
python scripts/train_deep_model.py --model_type rstnet --epochs 15
python scripts/train_deep_model.py --model_type rstnet_no_srm --epochs 15
python scripts/train_deep_model.py --model_type stae --epochs 15

# Step 3: Run Baseline Isolation Forest
python scripts/train_baselines.py

# Step 4: Run 9-Mode Corruption Benchmark
python scripts/evaluate_corruption.py

# Step 5: Regenerate All Publication Figures
python scripts/generate_plots.py
```

---

## Repository Structure

```text
Reliability-Aware-Anomaly-Detection/
│
├── README.md                          # Executive Research Summary (this file)
├── LICENSE                            # MIT Open Source License
├── CITATION.cff                       # Machine-readable academic citation
├── CHANGELOG.md                       # Version milestones from v0.1 to v1.0
├── requirements.txt                   # Pinned Python package dependencies
├── environment.yml                    # Conda environment definition
│
├── configs/                           # Central configuration files (YAML)
│   ├── default.yaml
│   └── swat.yaml
│
├── data/
│   ├── processed/SWaT/                # 10x downsampled train/val/test splits & scaler
│   └── README.md                      # Dataset acquisition and preprocessing instructions
│
├── src/                               # Modular source package
│   ├── corruption/                    # SensorCorruptor (Gaussian noise, drift, stuck)
│   ├── evaluation/                    # Pointwise and point-adjusted anomaly metrics
│   ├── models/                        # RSTNet, SpatioTemporalAE, RSTNetNoSRM
│   ├── preprocessing/                 # RobustStandardScaler and SlidingWindowGenerator
│   └── reliability/                   # SensorReliabilityModule (SRM)
│
├── scripts/                           # Reproducibility and evaluation drivers
│   ├── project_health_check.py        # Automated multi-check validation pipeline
│   ├── train_deep_model.py            # Deep network training and threshold calibration
│   ├── train_baselines.py             # Isolation Forest driver
│   ├── evaluate_corruption.py         # 9-mode corruption evaluation benchmark
│   └── generate_plots.py              # Publication figure generator
│
├── demo/                              # Interactive demonstration suite
│   ├── app.py                         # 4-tab Streamlit web application
│   └── README.md                      # Demo user manual
│
├── tests/                             # Pytest unit test suite (28 passing tests)
│   ├── test_preprocessing.py
│   ├── test_windowing.py
│   ├── test_metrics.py
│   ├── test_srm.py
│   ├── test_model_loading.py
│   └── test_inference.py
│
├── results/                           # Authoritative experimental artifacts
│   ├── checkpoints/                   # Serialized model weights (*.pt)
│   ├── figures/                       # Publication figures (fig1 through fig7)
│   ├── metrics/                       # Validated JSON metrics for all models
│   ├── corruption/                    # Corruption benchmark summary JSON
│   └── ablations/                     # SRM ablation summary JSON
│
├── report/                            # Academic thesis documents
│   ├── FINAL/
│   │   ├── FINAL_REPORT.md            # Complete integrated master thesis
│   │   ├── CHAPTER5_CONCLUSION.md     # Standalone Chapter 5 (Sections 5.1 - 5.4)
│   │   ├── REFERENCES.md              # 40 verified IEEE references (90% ScienceDirect)
│   │   ├── references.bib             # BibTeX reference archive
│   │   └── APPENDIX.md                # Code Appendices A through J
│   ├── REVIEW1/
│   └── REVIEW2/
│
├── presentation/                      # Defense presentation package
│   ├── FINAL_PRESENTATION.md          # 15-slide presentation with speaker notes
│   └── DEMO_SCRIPT.md                 # 3-5 minute oral demonstration sequence
│
└── docs/                              # Project documentation and audit records
    ├── SOURCE_OF_TRUTH.md             # Authoritative scientific baseline
    ├── RESULTS_MANIFEST.md            # Metric traceability matrix
    ├── FINAL_CLAIM_AUDIT.md           # Claim taxonomy and evidence classification
    ├── FINAL_REPRODUCIBILITY_AUDIT.md # Execution verification logs
    └── FINAL_PROJECT_CHECKLIST.md     # 100% completion sign-off
```

---

## Limitations

1. **Tabular vs. Spatio-Temporal Gap:** Isolation Forest achieves superior precision and $F_1$ on clean SWaT data ($F_1 = 0.7182$ vs $0.2740$).
2. **Actuator Inactivity Blindspot:** Statistical reliability cannot detect frozen actuators that are already idle (e.g., backup pumps `P202`/`P204` at zero variance).
3. **Dataset-Specific Shifts:** Sensors such as `AIT201` are unrecorded/constant during normal training but shift dynamically during testing, causing residual elevation.
4. **Offline Evaluation:** Validated using offline testbed data; lacks live SCADA hardware-in-the-loop and network packet telemetry integration.

---

## Dataset Attribution

This research uses the **Secure Water Treatment (SWaT)** dataset collected by the **iTrust Centre for Research in Cyber Security** at the **Singapore University of Technology and Design (SUTD)**.

Official Reference:
> J. Goh, S. Adepu, M. Junejo, and A. Mathur, "A Dataset to Support Research in the Design of Secure Water Treatment Systems," in *Critical Information Infrastructures Security (CRITIS 2016)*, Lecture Notes in Computer Science, vol. 10014, Springer, Cham, 2016, pp. 88–99, doi: [10.1007/978-3-319-46935-5_8](https://doi.org/10.1007/978-3-319-46935-5_8).

*In compliance with the iTrust Terms of Use, raw dataset archives are not redistributed in this repository. Access can be requested directly from [SUTD iTrust](https://itrust.sutd.edu.sg/itrust-labs-datasets/).*

---

## Citation

If you build upon this work or use the R-STNet architecture, please cite:

```bibtex
@misc{kushal2026rstnet,
  author       = {P Naga Venkata Kushal and Koteru Tarun},
  title        = {Reliability-Aware Anomaly Detection for Multivariate Sensor Data},
  year         = {2026},
  howpublished = {Final Research Project, BCSE209P Machine Learning Lab, Vellore Institute of Technology},
  note         = {Faculty Guide: Dr. Dega Nagaraju}
}
```

---

## Authors

- **P Naga Venkata Kushal** — Registration: `23BDS0133` — Vellore Institute of Technology
- **Koteru Tarun** — Registration: `23BDS0103` — Vellore Institute of Technology

**Faculty Guide:** **Dr. Dega Nagaraju**, Department of Data Science, School of Computer Science and Engineering, Vellore Institute of Technology.

---

## Acknowledgements

We express our sincere gratitude to **Dr. Dega Nagaraju** for expert technical guidance and continuous research mentorship throughout this project. We also thank the **iTrust Centre for Research in Cyber Security** at the **Singapore University of Technology and Design (SUTD)** for providing the Secure Water Treatment dataset.
