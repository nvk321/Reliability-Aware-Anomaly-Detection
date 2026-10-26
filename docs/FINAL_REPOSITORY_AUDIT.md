# Final Comprehensive Repository Audit

**Project Title:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Students:** P Naga Venkata Kushal (23BDS0133) & Koteru Tarun (23BDS0103)  
**Faculty Guide:** Dr. Dega Nagaraju  
**Course:** BCSE209P — Machine Learning Lab  
**Date of Audit:** October 2026  
**Auditor:** Lead ML Research Engineer & Reproducibility Auditor  
**Status:** Certified Final Repository Audit  

---

## 1. Executive Summary of Audit

An exhaustive technical, scientific, and architectural audit of the entire `Reliability-Aware-Anomaly-Detection` repository was conducted to transition the project from its experimental Review 2 state into a fully finalized, publication-grade, reproducible, and GitHub-ready research artifact.

The repository possesses a well-structured, modular machine learning pipeline with zero test-label leakage and preserved experimental checkpoints. However, prior iterations left fragmented documentation, temporary caches, missing standardized tests, and lacked an interactive demonstration interface. This document records the complete findings of the audit and establishes the roadmap for the final release.

---

## 2. Inventory of What Currently Exists

### 2.1 Source Code (`src/`)
- `src/data/swat_loader.py`: Downsampling, chronological ordering, and leak-free train/val/test splitting for SWaT Dec 2015 testbed.
- `src/preprocessing/scaler.py`: `RobustStandardScaler` with zero-variance protection ($\epsilon = 10^{-6}$).
- `src/preprocessing/windowing.py`: `SlidingWindowGenerator` ($L=30, S=1$) with causal `point_end` labeling.
- `src/models/baselines/isolation_forest.py`: Baseline A Isolation Forest wrapper (100 trees, $n=256$).
- `src/models/baselines/reconstruction_ae.py`: Baseline B Spatio-Temporal Convolutional Autoencoder ($170,643$ parameters).
- `src/models/baselines/rstnet_no_srm.py`: Baseline C Dilated TCN + Spatial Multi-Head Graph Attention with uniform residual scoring ($23,233$ parameters).
- `src/models/rstnet.py`: Proposed R-STNet combining TCN-GAT with dynamic Sensor Reliability Module ($23,233$ parameters).
- `src/reliability/sensor_reliability.py`: Multi-perspective Sensor Reliability Module (SRM) computing temporal continuity ($r^{(v)}$), graph neighborhood consistency ($r^{(c)}$), and predictive residual confidence ($r^{(p)}$).
- `src/corruption/corruptor.py`: Multi-modal sensor corruption engine (Gaussian noise, stuck sensor, linear drift).
- `src/evaluation/metrics.py`: Point-wise F1, Precision, Recall, FPR, Confusion Matrix, Point-Adjusted F1, PR-AUC, and ROC-AUC.
- `src/anomaly_detection/discriminator.py`: Multi-perspective fault vs. anomaly discriminator.

### 2.2 Execution Scripts (`scripts/`)
- `scripts/preprocess_swat.py`: Pipeline entrypoint to generate processed arrays.
- `scripts/train_baseline.py`: Trains Isolation Forest and records clean benchmark JSON.
- `scripts/train_deep_model.py`: Trains ST-AE, RSTNetNoSRM, and R-STNet for 15 epochs and calibrates thresholds.
- `scripts/evaluate_corruption.py`: Evaluates multi-seed corruption experiments ($E2-E4$).
- `scripts/run_ablations_and_generate_plots.py`: Runs 5-condition ablation and primary publication figures.
- `scripts/generate_review2_supplementary_plots.py`: Generates confusion matrices, bar charts, loss curves, and channel-shift diagnostics.
- `scripts/smoke_test.py`: Fast sanity test on dummy synthetic tensors.

### 2.3 Experimental Results & Checkpoints
- `results/checkpoints/`:
  - `baseline_stae_swat.pt` ($688\text{ KB}$)
  - `baseline_rstnet_no_srm_swat.pt` ($100\text{ KB}$)
  - `proposed_rstnet_swat.pt` ($101\text{ KB}$)
- `results/metrics/`: Authoritative clean evaluation JSON files for all four models.
- `results/corruption/`: Multi-seed corruption summary JSON (`corruption_benchmark_summary.json`).
- `results/ablations/`: Ablation summary JSON (`ablation_summary.json`).
- `results/plots/`: High-resolution figures across anomaly scores, reliability, corruption, and ablations.

### 2.4 Reports and Academic Documentation
- `report/REVIEW1/`: Review 1 documentation (Introduction, Literature Review, Methodology).
- `report/REVIEW2/`: Complete Review 2 deliverable, including Chapter 4 (`CHAPTER4_RESULTS_AND_DISCUSSION.md`), master dossier (`REVIEW2_MASTER.md`), 11 empirical tables (`TABLES.md`), 12 publication figures (`figures/`), captions, and specialized audits (`CLAIM_AUDIT.md`, `DATA_LEAKAGE_AUDIT.md`, `THRESHOLD_AUDIT.md`, `SWAT_CHANNEL_SHIFT_ANALYSIS.md`).

---

## 3. What Is Correct and Experimentally Validated

1. **Experimental Ground Truth:** All four models were evaluated under identical protocols; metric JSON files match reported figures and tables.
2. **Zero-Leakage Preprocessing:** `RobustStandardScaler` is fitted strictly on the $80\%$ normal training split. Decision thresholds $\tau$ are calibrated strictly on the $20\%$ normal validation split at the $95^{\text{th}}$ percentile ($q = 0.95$). No test labels or corruption masks leak into model training or calibration.
3. **Model Checkpoint Integrity:** All three PyTorch checkpoints in `results/checkpoints/` are intact, lightweight ($< 1\text{ MB}$ total), load without errors, and accurately reconstruct validated metric evaluations.
4. **The Research Finding:** The experimental finding is solid and honest:
   - R-STNet significantly improves upon Baseline C within the deep spatio-temporal architecture ($F_1: 0.2171 \to 0.2740$, $FPR: 99.86\% \to 62.14\%$, eliminating $14,898$ false alarms).
   - Baseline A (Isolation Forest) remains superior on clean SWaT ($F_1 = 0.7182, FPR = 0.0092$), which is acknowledged transparently.

---

## 4. What Is Outdated or Inconsistent

1. **Root `README.md`:** Currently contains preliminary scaffolding instructions and outdated project notes from earlier milestones; lacks unified benchmark tables, interactive demo instructions, and final architectural diagrams.
2. **Documentation Spread across `docs/` vs. `report/`:** Multiple overlapping status documents exist (`docs/PROJECT_STATUS.md`, `docs/DECISIONS.md`, `report/REVIEW1_MASTER.md`, `report/REVIEW2/REVIEW2_MASTER.md`). A single, unambiguous `docs/SOURCE_OF_TRUTH.md` is required.
3. **University Final Thesis Deliverable Missing:** While Review 1 (Chapters 1–3) and Review 2 (Chapter 4) are completed, the final unified thesis (`report/FINAL/FINAL_REPORT.md`), Chapter 5 (*Conclusion* with Sections 5.1–5.4), verified IEEE References (with $\ge 90\%$ ScienceDirect indexing), and Appendices A–J have not yet been compiled.

---

## 5. What Is Duplicated

1. **Checkpoints Directory:** `models/checkpoints/` contains only `.gitkeep`, while the real checkpoints reside in `results/checkpoints/`. Standardize on `results/checkpoints/` and symlink/document clearly.
2. **Figures Mirroring:** Figures were generated in `results/plots/` and copied into `report/REVIEW2/figures/`. This duplication was intentional for Review 2 self-containment, but must be cataloged in `docs/RESULTS_MANIFEST.md` to prevent confusion.
3. **Redundant Metric Files:** `smoke_test_report.json` and temporary validation outputs reside in `results/metrics/`.

---

## 6. What Should Be Preserved (Frozen Artifacts)

- All PyTorch checkpoints in `results/checkpoints/`.
- All JSON files in `results/metrics/`, `results/corruption/`, and `results/ablations/`.
- All 12 publication-grade figures in `report/REVIEW2/figures/`.
- All verified analytical chapters in `report/REVIEW1/` and `report/REVIEW2/`.
- Preprocessed arrays in `data/processed/SWaT/` (`train.npy`, `val.npy`, `test.npy`, `test_labels.npy`, `scaler_params.json`).

---

## 7. What Should Be Cleaned or Removed

- Remove Python bytecode caches (`__pycache__/`, `*.pyc`).
- Remove temporary test logs and scratch files from `.system_generated/` and `logs/`.
- Ensure `.gitignore` prevents accidentally committing `reference/swat_data/` raw CSVs (over 400 MB) or virtual environment binaries.
- Ensure no hardcoded Windows drive paths (`d:\Projects\...` or `C:\Users\...`) exist in portable source code or configuration files.

---

## 8. What Must Be Added for Final Release

1. **Single Source of Truth:** `docs/SOURCE_OF_TRUTH.md` and `docs/RESULTS_MANIFEST.md`.
2. **Official Dataset Documentation:** `docs/DATASET.md` verifying official SUTD iTrust SWaT citations, access terms, and attribution.
3. **Interactive Demonstration:** Presentation-level Streamlit web app in `demo/app.py` supporting both Fast Demo Mode and interactive sensor corruption injection.
4. **Master Final Report Package:**
   - `report/FINAL/FINAL_REPORT.md` (complete integrated Chapters 1 to 5).
   - `report/FINAL/CHAPTER5_CONCLUSION.md` (Sections 5.1 to 5.4 conforming to university formatting guidelines).
   - `report/FINAL/REFERENCES.md` and `report/FINAL/references.bib` (35–40+ verified IEEE references, $\ge 90\%$ ScienceDirect).
   - `report/FINAL/APPENDIX.md` (Appendices A to J with organized code excerpts).
   - `docs/REFERENCE_AUDIT.md`.
5. **Presentation Assets:** `presentation/FINAL_PRESENTATION.md` (10–15 slides with speaker notes) and `presentation/DEMO_SCRIPT.md`.
6. **Comprehensive Test Suite & Health Check:** `scripts/project_health_check.py` and 6 targeted unit tests in `tests/`.
7. **Production GitHub Release Files:** Professional root `README.md`, `LICENSE` (MIT), `CITATION.cff`, and `CHANGELOG.md`.

---

## 9. Reproducibility Assessment

- **Environment:** Requires Python 3.10+, PyTorch 2.6.0+cu124 (or CPU), NumPy, Scipy, Scikit-Learn, Matplotlib, Streamlit.
- **Two Execution Modes Required:**
  - *Mode A: Fast Demo Mode:* Instant evaluation using pre-computed checkpoints and processed cache ($< 60$ seconds).
  - *Mode B: Full Reproduction Mode:* Step-by-step pipeline from raw CSV preprocessing to multi-seed corruption benchmarking.
- **Portability Check:** Source files utilize relative imports and `os.path.join`; no OS-specific hardcoded absolute paths exist in `src/`.

---

## 10. Final Recommended Repository Structure

```text
Reliability-Aware-Anomaly-Detection/
│
├── README.md                          <-- Comprehensive, publication-grade research README
├── LICENSE                            <-- MIT License
├── CITATION.cff                       <-- Official Citation File Format metadata
├── CHANGELOG.md                       <-- Version progression (v0.1 to v1.0-final)
├── .gitignore                         <-- Rigorous git ignore excluding raw data and caches
├── requirements.txt                   <-- Audited pip dependency specifications
├── environment.yml                    <-- Conda environment definition
│
├── configs/
│   ├── base.yaml                      <-- General pipeline configurations
│   └── swat_rstnet.yaml               <-- Model and training hyperparameters
│
├── data/
│   ├── README.md                      <-- Data acquisition instructions
│   └── processed/
│       └── README.md                  <-- Processed cache metadata & scaler documentation
│
├── src/
│   ├── data/                          <-- SWaT data loader and chronological splitters
│   ├── preprocessing/                 <-- RobustStandardScaler & SlidingWindowGenerator
│   ├── models/                        <-- Baseline architectures (IF, ST-AE, RSTNetNoSRM) and R-STNet
│   ├── reliability/                   <-- Multi-perspective Sensor Reliability Module (SRM)
│   ├── corruption/                    <-- Controlled sensor corruption engine (Noise, Stuck, Drift)
│   ├── evaluation/                    <-- Industrial anomaly detection and point-adjusted metrics
│   ├── anomaly_detection/             <-- Fault vs Anomaly discriminator
│   └── visualization/                 <-- Plotting helpers
│
├── scripts/
│   ├── preprocess_swat.py             <-- Preprocessing pipeline
│   ├── train_baseline.py              <-- Trains Isolation Forest
│   ├── train_deep_model.py            <-- Trains ST-AE, RSTNetNoSRM, and R-STNet
│   ├── evaluate_corruption.py         <-- Runs multi-seed corruption experiments
│   ├── run_ablations_and_generate_plots.py <-- Ablation runner and publication figures
│   ├── generate_review2_supplementary_plots.py <-- Diagnostic and confusion matrix figures
│   └── project_health_check.py        <-- Automated verification and health check
│
├── demo/
│   ├── app.py                         <-- Interactive Streamlit demonstration dashboard
│   ├── README.md                      <-- Demo usage and live simulation instructions
│   └── assets/                        <-- Visual assets and architecture diagrams
│
├── tests/                             <-- Unit and integration test suite
│   ├── test_preprocessing.py
│   ├── test_windowing.py
│   ├── test_metrics.py
│   ├── test_srm.py
│   ├── test_model_loading.py
│   └── test_inference.py
│
├── results/
│   ├── checkpoints/                   <-- Trained PyTorch models (< 1 MB total)
│   ├── metrics/                       <-- Clean benchmark JSON metric records
│   ├── corruption/                    <-- Multi-seed corruption benchmark results
│   ├── ablations/                     <-- 5-condition ablation JSON outputs
│   └── plots/                         <-- Publication-grade PNG figures
│
├── report/
│   ├── REVIEW1/                       <-- Review 1 report documents (Chapters 1 to 3)
│   ├── REVIEW2/                       <-- Review 2 report documents (Chapter 4, Tables, Figures)
│   └── FINAL/                         <-- University final thesis package (Chapters 1-5, References, Appendix)
│       ├── FINAL_REPORT.md
│       ├── CHAPTER5_CONCLUSION.md
│       ├── REFERENCES.md
│       ├── references.bib
│       └── APPENDIX.md
│
├── presentation/
│   ├── FINAL_PRESENTATION.md          <-- 15-Slide defense presentation with speaker notes
│   ├── DEMO_SCRIPT.md                 <-- 3-5 minute live demonstration walkthrough
│   └── figures/                       <-- Presentation figures
│
└── docs/
    ├── FINAL_REPOSITORY_AUDIT.md      <-- This comprehensive audit document
    ├── SOURCE_OF_TRUTH.md             <-- Authoritative project parameters, models, and numbers
    ├── DATASET.md                     <-- SWaT testbed, SUTD iTrust attribution and terms
    ├── PROJECT_OVERVIEW.md            <-- Research scope, problem statement, and architecture
    ├── METHODOLOGY.md                 <-- Formal mathematical formulation of R-STNet and SRM
    ├── EXPERIMENTS.md                 <-- Protocol, baselines, and multi-seed corruption suites
    ├── REPRODUCIBILITY.md             <-- Step-by-step Fast Demo & Full Replication guides
    ├── DEMO.md                        <-- Interactive demonstration documentation
    ├── LIMITATIONS.md                 <-- 7 Fundamental scientific limitations & boundaries
    ├── RESULTS_MANIFEST.md            <-- End-to-end result provenance and artifact mapping
    ├── FINAL_CLAIM_AUDIT.md           <-- Verification status of all academic claims
    ├── FINAL_REPRODUCIBILITY_AUDIT.md <-- Test execution results and verification matrix
    ├── REFERENCE_AUDIT.md             <-- 40 IEEE references, ScienceDirect index audit
    └── FINAL_PROJECT_CHECKLIST.md     <-- Master quality, scientific, and release checklist
```

---

*Audit Complete. Proceeding to Phase 2: Freezing Validated Results & Establishing Single Source of Truth.*
