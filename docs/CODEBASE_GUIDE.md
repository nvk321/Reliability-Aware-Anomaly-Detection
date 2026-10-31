# Comprehensive Codebase Guide: Reliability-Aware Anomaly Detection (R-STNet)

**Students:** P Naga Venkata Kushal (23BDS0133) & Koteru Tarun (23BDS0103)  
**Faculty Guide:** Dr. Dega Nagaraju  
**Course:** BCSE209P — Machine Learning Lab  
**Repository:** [https://github.com/nvk321/Reliability-Aware-Anomaly-Detection](https://github.com/nvk321/Reliability-Aware-Anomaly-Detection)  
**Release Tag:** `v1.0.0-final`

---

## 1. Start Here: Reading Order for a New Evaluator

If you are opening this repository for the first time, follow this 5-step onboarding path:
1. **[README.md](file:///d:/Projects/Reliability-Aware-Anomaly-Detection/README.md):** High-level summary of the research question, core contributions, results summary, and quickstart commands.
2. **[docs/PIPELINE.md](file:///d:/Projects/Reliability-Aware-Anomaly-Detection/docs/PIPELINE.md):** The end-to-end canonical pipeline diagram from raw CSVs to Streamlit visualization.
3. **[docs/ARCHITECTURE.md](file:///d:/Projects/Reliability-Aware-Anomaly-Detection/docs/ARCHITECTURE.md):** Mathematical formulation of the 1D TCN, Top-$k$ Graph Attention, and Sensor Reliability Module (SRM).
4. **[notebooks/01_dataset_exploration.ipynb](file:///d:/Projects/Reliability-Aware-Anomaly-Detection/notebooks/01_dataset_exploration.ipynb):** Hands-on visual exploration of the SWaT testbed and physical telemetry.
5. **[demo/app.py](file:///d:/Projects/Reliability-Aware-Anomaly-Detection/demo/app.py):** Interactive 9-tab presentation dashboard visualizing live inference, reliability degradation, and benchmark audits.

---

## 2. Research Pipeline: Where Each Stage Lives

| Pipeline Stage | Primary Directory | Core Script / Entry Point | Description |
| :--- | :--- | :--- | :--- |
| **Ingestion & Split** | `src/data/` | `scripts/preprocess_swat.py` | Sanitizes 51 channels, downsamples 10x, splits chronological train/val/test. |
| **Scaling & Windows** | `src/preprocessing/` | `src/preprocessing/scaler.py` | Leak-free `RobustStandardScaler` and `SlidingWindowGenerator` ($L=30$). |
| **Baselines** | `src/models/baselines/` | `scripts/train_baseline.py` | Isolation Forest, ST-AE, and RSTNetNoSRM (primary ablation). |
| **Proposed R-STNet** | `src/models/` | `src/models/rstnet.py` | 23k-parameter network with integrated `SensorReliabilityModule`. |
| **Reliability (SRM)** | `src/reliability/` | `src/reliability/sensor_reliability.py` | Triangulates Temporal Continuity, Graph Consistency, Residual Confidence. |
| **State Engine** | `src/anomaly_detection/` | `src/anomaly_detection/discriminator.py` | 4-way operational state logic (Normal, Sensor Fault, Cyber-Attack, Dual). |
| **Metrics & Eval** | `src/evaluation/` | `src/evaluation/metrics.py` | Point-wise F1, Precision, Recall, FPR, PR-AUC, ROC-AUC, fault recall. |
| **Corruption Suite** | `src/corruption/` | `scripts/evaluate_corruption.py` | 9-regime multi-seed perturbation testbed ($E2-E4$). |
| **Ablation Studies** | `scripts/` | `scripts/run_ablations_and_generate_plots.py` | Evaluates 5 systematic ablations isolating SRM signals. |
| **Interactive Demo** | `demo/` | `demo/app.py` | Streamlit 9-tab dashboard for live viva demonstration. |

---

## 3. Data Loading & Preprocessing

- **`src/data/swat_loader.py`:**
  - Class `SWaTLoader`: Reads raw SWaT CSV files, cleans column whitespace, drops extraneous columns, downsamples 10x (10s intervals), and creates chronological 80/20 train/validation splits on normal data.
- **`src/preprocessing/scaler.py`:**
  - Class `RobustStandardScaler`: Fits $\mu$ and $\sigma$ strictly on the normal training split. Features zero-variance protection ($\sigma_i < 10^{-8} \implies \sigma_i = 1.0$), preventing runtime division-by-zero on inactive backup actuators (e.g. `P202`).
- **`src/preprocessing/windowing.py`:**
  - Class `SlidingWindowGenerator`: Transforms 2D multivariate time-series into 3D sliding windows $(N, L=30, M=51)$ with stride 1. Enforces `point_end` labeling protocol to avoid future leakage.

---

## 4. Models: Architecture Implementations

All neural models are implemented in PyTorch under `src/models/`:
- **`src/models/rstnet.py`:**
  - Class `RSTNet`: The proposed architecture. Contains dilated 1D temporal convolutions per sensor, `SensorGraphAttention`, forecasting MLP head, and an integrated `SensorReliabilityModule`. Total trainable parameters: **23,233**.
  - Method `evaluate_step(window)`: Performs one-step inference, normalized residual calculation, SRM reliability estimation, and reliability-weighted anomaly aggregation.
- **`src/models/baselines/rstnet_no_srm.py`:**
  - Class `RSTNetNoSRM` (Baseline C): The exact same 23k-parameter temporal + graph backbone as R-STNet, but sums residuals uniformly without reliability weights. Serves as the primary controlled ablation baseline.
- **`src/models/baselines/isolation_forest.py`:**
  - Class `IsolationForestBaseline` (Baseline A): Classical tree-based ensemble fitted on normal training data. Decision threshold $\tau$ calibrated on validation data.
- **`src/models/baselines/reconstruction_ae.py`:**
  - Class `SpatioTemporalAE` (Baseline B): Bidirectional GRU encoder-decoder autoencoder reconstructing sequence observations without graph or reliability mechanisms.

---

## 5. Reliability: Sensor Reliability Module (SRM)

- **`src/reliability/sensor_reliability.py`:**
  - Class `SensorReliabilityModule`: Computes online per-sensor trustworthiness $r_i(t) \in [0, 1]$ via three complementary physical perspectives:
    1. `compute_temporal_continuity(window)`: Evaluates rate-of-change against validation quantile limits and applies variance checks to flag stuck sensors.
    2. `compute_neighborhood_consistency(x)`: Evaluates physical agreement with correlated neighbors using learned graph weights $W_{ji}$.
    3. `compute_residual_confidence(x, x_hat)`: Converts standardized prediction residuals $z_i(t)$ into tail probabilities under normal operation.
    4. `estimate_reliability(window, pred)`: Fuses the three perspectives via calibrated logit fusion.
    5. `compute_weighted_system_anomaly(z, r)`: Computes reliability-weighted system score:
       $$S_{\text{sys}}(t) = \frac{\sum_i r_i(t) z_i(t)}{\sum_i r_i(t) + \epsilon}$$

---

## 6. Evaluation: Metrics & Decision Thresholds

- **`src/evaluation/metrics.py`:**
  - `compute_anomaly_metrics(y_true, y_scores, threshold)`: Computes point-wise Precision, Recall, F1, PR-AUC, ROC-AUC, FPR, and full confusion matrix.
  - `compute_sensor_fault_metrics(mask, pred_idx, flags)`: Computes sensor fault detection recall and localization accuracy against synthetic corruption ground-truth masks.
  - `point_adjustment_eval(y_true, y_pred)`: Optional Point Adjustment (PA) computation for direct comparison with external published benchmarks. Clearly separated from raw metrics to prevent academic inflation.

---

## 7. Experiments: Controlled Corruption & Ablations

- **`src/corruption/corruptor.py`:**
  - Class `SensorCorruptor`: Deterministically injects 9 controlled non-adversarial failure modes:
    - Gaussian noise (high electrical jitter)
    - Calibration drift (linear ramp)
    - Stuck sensor (constant frozen reading)
    - Single-sensor and multi-sensor permutations
  - Creates a synthetic binary mask `sensor_fault_mask[t, i]` providing ground truth for sensor-level fault localization.
- **`scripts/evaluate_corruption.py`:**
  - Driver script executing multi-seed ($[42, 101, 2024]$) evaluations across 9 corruption regimes. Computes mean F1, mean FPR, reliability separation $\Delta r$, and sensor fault identification accuracy.
- **`scripts/run_ablations_and_generate_plots.py`:**
  - Evaluates 5 systematic ablations (Ablation 1: No SRM, Ablation 2: No Temporal, Ablation 3: No Residual, Ablation 4: No Graph, Ablation 5: Full SRM).

---

## 8. Authoritative Frozen Results

Authoritative results are stored as version-controlled JSON files in `results/`:
- **`results/metrics/baseline_isolation_forest_swat.json`:** Clean SWaT metrics for Isolation Forest ($F_1 = 0.7182, \text{FPR} = 0.92\%$).
- **`results/metrics/baseline_stae_swat.json`:** Clean SWaT metrics for ST-AE ($F_1 = 0.2169$).
- **`results/metrics/baseline_rstnet_no_srm_swat.json`:** Clean SWaT metrics for Baseline C ($F_1 = 0.2171, \text{FPR} = 99.86\%$).
- **`results/metrics/proposed_rstnet_swat.json`:** Clean SWaT metrics for Proposed R-STNet ($F_1 = 0.2740, \text{FPR} = 62.14\%$).
- **`results/corruption/corruption_benchmark_summary.json`:** 9-regime multi-seed robustness results.
- **`results/ablations/ablation_summary.json`:** SRM component ablation performance metrics.
- **`results/metrics/smoke_test_report.json`:** Hardware profile and execution verification.

---

## 9. Interactive Streamlit Demo

- **`demo/app.py`:**
  - Standalone presentation dashboard organized into 9 tabs matching the thesis narrative.
  - Loads frozen checkpoints from `results/checkpoints/` and preprocessed test windows from `data/processed/SWaT/`.
  - Supports preset attack scenario scrubbing, real-time threshold comparison, live sensor corruption injection, and automated viva summary reporting.
  - Run via: `streamlit run demo/app.py`

---

## 10. Automated Tests: What Each File Verifies

The test suite contains **31 automated tests** passing in pytest:
- **`tests/test_corruption.py` (2 tests):** Verifies all 9 corruption modes execute and maintain exact multi-seed reproducibility.
- **`tests/test_inference.py` (4 tests):** Verifies forward pass shapes and outputs for R-STNet, RSTNetNoSRM, ST-AE, and `evaluate_step()` diagnostics.
- **`tests/test_metrics.py` (4 tests):** Verifies metric formulas (F1, Precision, Recall, FPR), confusion matrices, point adjustment protocols, and sensor fault metrics.
- **`tests/test_model_loading.py` (3 tests):** Verifies checkpoint file existence, `state_dict` key alignment, and decision threshold validity.
- **`tests/test_preprocessing.py` (3 tests):** Verifies scaling on normal data, zero-variance handling, and parameter persistence.
- **`tests/test_reliability.py` (3 tests):** Verifies SRM behavior on clean data, stuck sensor penalization, and false alarm discounting.
- **`tests/test_scaler.py` (3 tests):** Verifies zero-variance actuator handling and JSON serialization.
- **`tests/test_srm.py` (4 tests):** Verifies SRM high reliability on nominal data, stuck sensor degradation, noise/drift separation, and state serialization.
- **`tests/test_windowing.py` (2 tests):** Verifies sequential sliding window shapes $(N, L, M)$ and label aggregation modes.
- **`tests/test_notebooks.py` (3 tests):** Verifies all 5 canonical notebooks exist, conform to nbformat v4 schema, and import modularly from `src.*`.

---

## 11. Thesis Documentation Structure

- **`report/FINAL/FINAL_REPORT.md`:** The complete master research report.
- **`report/FINAL/CHAPTER5_CONCLUSION.md`:** Chapter 5 conclusion and future work.
- **`report/FINAL/APPENDIX.md`:** Mathematical proofs, hyperparameter tables, and testbed sensor mappings.
- **`report/FINAL/REFERENCES.md`:** Complete IEEE-formatted bibliography.
- **`report/REVIEW2/`:** Detailed review phase documentation including `CHAPTER4_RESULTS_AND_DISCUSSION.md`, `ROBUSTNESS_ANALYSIS.md`, and `LIMITATIONS.md`.

---

## 12. Recommended Viva Voce Presentation Order

During the viva voce examination, navigate the codebase in this order:
1. Open **`README.md`** &rarr; Explain project motivation, student details, and research questions.
2. Open **`src/data/swat_loader.py`** &rarr; Explain SWaT testbed stages P1-P6 and 10x downsampling rationale.
3. Open **`src/preprocessing/scaler.py`** &rarr; Explain leak-free zero-variance scaling.
4. Open **`src/models/rstnet.py`** &rarr; Explain 1D TCN + Graph Attention + Forecasting Head.
5. Open **`src/reliability/sensor_reliability.py`** &rarr; Explain the 3 SRM perspectives ($v_i, c_i, p_i$).
6. Open **`src/anomaly_detection/discriminator.py`** &rarr; Explain the 4-way operational state classification engine.
7. Open **`notebooks/03_baseline_evaluation.ipynb`** &rarr; Honestly explain clean benchmark results (Isolation Forest vs. Deep Baselines).
8. Open **`notebooks/05_robustness_and_ablation.ipynb`** &rarr; Explain corruption robustness and false alarm suppression.
9. Launch **`streamlit run demo/app.py`** &rarr; Walk through the 9 tabs and demonstrate live sensor corruption suppression.
10. Open **`docs/LIMITATIONS.md`** &rarr; Defend research boundaries and negative findings with academic integrity.
