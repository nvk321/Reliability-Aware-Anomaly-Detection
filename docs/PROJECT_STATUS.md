# Project Status: Reliability-Aware Anomaly Detection for Multivariate Sensor Data

**Single Source of Truth for Project Progress**  
*Last Updated: 2026-10-21*

---

## 1. Project Phase

**Current Phase:** `PHASE 1: Evaluation Audit & Baseline A Rerun`  
**Status:** `AUDIT COMPLETE / AWAITING USER APPROVAL FOR BASELINE B & C` (Stop condition reached)

---

## 2. Completed Work (Phase 0)

| Task / Item | Status | Verification Detail |
| :--- | :---: | :--- |
| Workspace Inspection | **VALIDATED** | Verified workspace root, `.git` repository, and `reference/` directory. |
| Reference Architecture Analysis | **VALIDATED** | Inspected `reference/CAROTS-master` (ICML 2025), `reference/OmniAnomaly-master`, and `reference/RANSynCoders-main`. |
| Raw Reference Data Audit | **VALIDATED** | Confirmed existing offline reference data: `reference/swat_data` (426 MB merged, 402 MB normal, 15 MB attack) and `reference/wadi_data` (493 MB 14days, 109 MB attack). |
| Reference Directory Alias | **VALIDATED** | Created NTFS junction `reference/CAROTS -> reference/CAROTS-master` to guarantee path compatibility. |
| Hardware / Compute Audit | **VALIDATED** | NVIDIA GeForce RTX 3050 Laptop GPU (4096 MiB VRAM), Driver 592.82, CUDA 13.1 compatibility confirmed. |
| Python Runtime Audit | **VALIDATED** | Installed dedicated virtual environment `.venv/` using Python 3.11.0 (`py -3.11`). Verified clean invocation. |
| Modular Directory Scaffolding | **VALIDATED** | Created `data/{raw,processed,corrupted}/{SWaT,WADI}`, `src/{data,preprocessing,reliability,models,anomaly_detection,evaluation,corruption,visualization}`, `configs`, `models/checkpoints`, `results/{metrics,plots,checkpoints}`, `scripts`, `tests`, `docs/references`. |
| Version Control Rules | **VALIDATED** | Created `.gitignore` strictly excluding datasets, checkpoints, weights, and caches while preserving structure. |
| Literature Review & Synthesis | **VALIDATED** | Conducted comprehensive comparative review of 8 anomaly detection and sensor fault paradigms (2019–2026). |
| Architecture Selection | **VALIDATED** | Formulated **R-STNet** (Dual-Perspective Reliability-Aware Spatio-Temporal Network) addressing PPT limitations. |
| Evaluation Protocol Audit | **VALIDATED** | Identified positive-only bias in preliminary evaluation; reconstructed authentic chronological train/val/test splits. |
| SWaT Dataset Structure Reconstruction | **VALIDATED** | Isolated official 7-day normal run (496,800 rows) and reconstructed 4-day attack run (449,919 rows, 12.15% anomaly rate). |
| Leak-Free Preprocessing Pipeline | **VALIDATED** | `RobustStandardScaler` fit strictly on 80% train split. Output saved in `data/processed/SWaT/`. |
| Hardware & Smoke Test Verification | **VALIDATED** | RTX 3050 Laptop GPU verified via PyTorch CUDA. Peak VRAM measured at 73.81 MB (reserved 86 MB). |
| Baseline A Re-run & Validation | **VALIDATED** | Isolation Forest re-evaluated on corrected mixed test set. Threshold calibrated on validation normal data ($\tau=0.63587$). |

---

## 3. Current Work

- Finalizing initialization documentation:
  - `docs/ARCHITECTURE.md` (Architectural contract)
  - `docs/DECISIONS.md` (Architecture Decision Records)
  - `docs/RESEARCH_NOTES.md` (Literature review & taxonomy)
  - `docs/DATA_SETUP.md` (Dataset specification & preparation pipeline)
  - `docs/EXPERIMENT_PLAN.md` (Controlled corruption matrix & hypotheses)
  - `docs/REPRODUCIBILITY.md` (Runtime specs, seeds, hardware limits)
  - `docs/KNOWN_ISSUES.md` (Risk log and edge cases)
  - `docs/references/technical_references.md` (Academic attribution)

---

## 4. Next Implementation Milestones (Roadmap)

1. **PHASE 1: Environment & Dependency Finalization**
   - Install pinned dependencies in `.venv` (PyTorch with CUDA 12.1/12.4 wheel, NumPy 1.x, Pandas, Scikit-learn).
   - Verify GPU acceleration via PyTorch `torch.cuda.is_available()`.
2. **PHASE 2: Baseline Reproduction & Reference Study**
   - Study `reference/CAROTS` execution pipeline on SWaT without altering reference code.
   - Record baseline execution protocol and benchmark metric conventions.
3. **PHASE 3: Dataset Ingestion & Preprocessing Pipeline**
   - Build `src/data/swat_loader.py` and `src/preprocessing/scaler.py`.
   - Ingest clean SWaT into `data/processed/SWaT/` (train normal, validation normal, test attack).
4. **PHASE 4: Classical & ML Baselines**
   - Implement Baseline 1: Classical (Isolation Forest on raw/differenced features).
   - Implement Baseline 2: Standard Autoencoder / Reconstruction Model.
   - Implement Baseline 3: Reference Contrastive / Causal Model (CAROTS).
5. **PHASE 5: Sensor Reliability Module (SRM) Prototype**
   - Implement 3-way reliability estimators: Temporal Continuity, Neighborhood Cross-Sensor Agreement, Residual Z-Score.
   - Implement calibrated logistic reliability aggregator ($r_i(t) \in [0, 1]$).
6. **PHASE 6: Proposed Architecture (R-STNet)**
   - Spatio-temporal backbone (Temporal Conv / GRU + Graph Attention).
   - Dual-head output: Sensor Reliability Vector $\mathbf{r}_t$ + Reliability-Weighted System Anomaly Score $S_{\text{sys}}(t)$.
   - Sensor fault vs. system anomaly decision logic.
7. **PHASE 7: Controlled Sensor Corruption Framework**
   - Build `src/corruption/corruptor.py` supporting 9 deterministic corruption modes across 4 severity tiers (5%, 10%, 20%, 30%).
   - Store generated datasets in `data/corrupted/SWaT/` with `corruption_metadata.json`.
8. **PHASE 8–10: Experiments, Ablations, & Robustness Validation**
   - Execute 10 controlled corruption experiments comparing Baselines vs. Proposed R-STNet.
   - Perform ablation studies (No SRM vs. Temporal SRM vs. Full Spatio-Temporal SRM).
   - Generate machine-readable result files in `results/metrics/`.
9. **PHASE 11–12: Explainability, Visualization, & Reporting**
   - Intrinsic explanation layer (sensor-level trust attribution, edge violation).
   - Interactive diagnostic dashboard.

---

## 5. Architectural Status

- **Status:** **LOCKED (Architectural Contract Established)**
- **Selected Method:** **R-STNet (Dual-Perspective Reliability-Aware Spatio-Temporal Network)**
- **Baseline Models:**
  1. Classical: Isolation Forest / Robust PCA
  2. Deep Reconstruction: Spatial-Temporal Autoencoder (ST-AE)
  3. External Reference: CAROTS (Kim et al., ICML 2025)
- **Key Invariant:** The architecture cannot be casually changed without a documented ADR entry in `docs/DECISIONS.md`.

---

## 6. Dataset Status

- **Primary (SWaT):** Raw files located in `reference/swat_data/`. Preprocessing pipeline planned for `data/processed/SWaT/`.
- **Secondary (WADI):** Raw files located in `reference/wadi_data/`. Scheduled for external validation in Phase 8.
- **PSM / SMD:** Documented in `docs/DATA_SETUP.md`. Deferred until primary SWaT benchmarks are validated.

---

## 7. Known Blockers & Issues

- None at present. Virtual environment established with Python 3.11.0; 4GB VRAM GPU detected and accommodated in model design.
