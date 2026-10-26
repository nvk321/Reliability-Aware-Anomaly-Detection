# Changelog

All notable changes to the **Reliability-Aware Anomaly Detection for Multivariate Sensor Data** research project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0-final] - 2026-10-23

### Added
- **Final Master Thesis Compilation:** Unified academic report (`report/FINAL/FINAL_REPORT.md`) containing Chapters 1 through 5, References, and Appendices A through J.
- **Chapter 5 Conclusion:** Standard university-mandated structure including Work Carried Out (5.1), Conclusions from Results & Discussion (5.2), Managerial Implications (5.3), and Future Scope (5.4).
- **Comprehensive Literature Bibliography:** 40 verified IEEE-formatted references (`report/FINAL/REFERENCES.md`, `report/FINAL/references.bib`) featuring exactly 36 ScienceDirect / Elsevier journal articles (90.0% ScienceDirect compliance).
- **Interactive Streamlit Web Application:** Presentation-grade 4-tab interactive dashboard (`demo/app.py`) for live sensor corruption injection, reliability decomposition, process monitoring, and benchmark auditing.
- **Project Health Check Pipeline:** Automated validation utility (`scripts/project_health_check.py`) testing environment, dataset, checkpoints, inference, metrics, and demo assets (`PROJECT HEALTH CHECK: PASS`).
- **Comprehensive Unit Test Suite:** 28 unit tests across 6 modules (`tests/test_preprocessing.py`, `tests/test_windowing.py`, `tests/test_metrics.py`, `tests/test_srm.py`, `tests/test_model_loading.py`, `tests/test_inference.py`) with 100% pass rate.
- **Formal Audit Package:** `docs/FINAL_REPOSITORY_AUDIT.md`, `docs/SOURCE_OF_TRUTH.md`, `docs/RESULTS_MANIFEST.md`, `docs/FINAL_CLAIM_AUDIT.md`, `docs/FINAL_REPRODUCIBILITY_AUDIT.md`, `docs/FINAL_PROJECT_CHECKLIST.md`.
- **Presentation Assets:** 15 defense slides (`presentation/FINAL_PRESENTATION.md`) and a chronological 3–5 minute oral demonstration script (`presentation/DEMO_SCRIPT.md`).
- **Release Documentation:** Pinned `environment.yml`, updated `requirements.txt`, `LICENSE` (MIT), and `CITATION.cff`.

### Changed
- Refactored `src/reliability/sensor_reliability.py` to support full serialization/deserialization (`to_dict` and `from_dict`) including topology neighbor weights and continuous masks.
- Updated root `README.md` into an executive ML research repository format with clear architecture diagrams, audited metric comparisons, and quick-start reproduction instructions.

### Fixed
- Calibrated decision thresholds and scaler serialization alignment in unit test fixtures.
- Corrected path references for processed SWaT data arrays and ablation summaries.

---

## [0.2.0-review2] - 2026-10-22

### Added
- **Phase 2 Benchmark Execution:** Validated training of Baseline A (Isolation Forest), Baseline B (ST-AE), Baseline C (RSTNetNoSRM), and Proposed R-STNet.
- **Controlled Corruption Suite:** 9 synthetic degradation modes across Gaussian noise ($\sigma \in [0.1, 0.5]$), sensor drift ($\delta \in [0.1, 0.5]$), and stuck sensors (1, 3, 5 channels).
- **Ablation Study:** Direct empirical quantification isolating the Sensor Reliability Module under identical 23k-parameter network backbones.
- **Negative Finding Documentation:** Detailed disclosure of Isolation Forest clean superiority and actuator dead-band detection boundaries.
- **Channel Shift Analysis:** Formal analysis of unrecorded sensors during training (`AIT201`, `MV101`, `P202`, etc.).

---

## [0.1.0-review1] - 2026-10-21

### Added
- Project initialization and scaffolding.
- Baseline A (Isolation Forest) preliminary implementation.
- SUTD iTrust Secure Water Treatment (SWaT) data preprocessing pipeline and 10x downsampling.
- Spatio-temporal graph network skeleton and sliding window generator.
