# Final Claim Audit and Scientific Verification

**Project Title:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Course:** BCSE209P — Machine Learning Lab  
**Students:** P Naga Venkata Kushal (23BDS0133), Koteru Tarun (23BDS0103)  
**Faculty Guide:** Dr. Dega Nagaraju  
**Audit Date:** October 2026  
**Audited Target Release:** v1.0-final  

---

## 1. Audit Taxonomy & Classification Schema

Every major academic claim within the thesis (`report/FINAL/FINAL_REPORT.md`), slides, README, and documentation is audited against empirical artifacts and classified under one of six categories:

1. **`[VERIFIED BY EXPERIMENT]`**: Supported by reproducible quantitative evidence in `results/`.
2. **`[CODE-DERIVED]`**: Inherited directly from mathematical formulas or tensor operations in `src/`.
3. **`[LITERATURE-SUPPORTED]`**: Established in peer-reviewed scientific literature with verified IEEE citations.
4. **`[INTERPRETATION]`**: Legitimate scientific induction or causal reasoning bounded by experimental parameters.
5. **`[PROPOSED FUTURE WORK]`**: Explicitly demarcated research trajectory; not claimed as completed.
6. **`[NOT SUPPORTED]`**: Unsubstantiated or exaggerated assertion strictly forbidden and purged.

---

## 2. Forbidden Claims Verification Matrix

| Claim Statement | Status | Project Position / Disciplinary Guardrail |
| :--- | :---: | :--- |
| **"R-STNet achieves State-of-the-Art (SOTA) performance on SWaT"** | **PURGED / FORBIDDEN** | **Zero claims of SOTA.** Explicitly documented that Isolation Forest outperforms deep architectures on clean SWaT. |
| **"R-STNet is the best overall anomaly detection model"** | **PURGED / FORBIDDEN** | **Rejected.** Clean benchmark establishes Isolation Forest ($F_1 = 0.7182$) superior to R-STNet ($F_1 = 0.2740$). |
| **"R-STNet provides universal robustness against any corruption"** | **PURGED / FORBIDDEN** | **Rejected.** Robustness is demonstrated specifically under 9 evaluated synthetic degradation profiles. |
| **"The model perfectly detects all industrial sensor faults"** | **PURGED / FORBIDDEN** | **Rejected.** $100\%$ recall holds on tested synthetic masks; stuck inactive actuators (`P202`, `P204`) yield $0.0\%$ recall. |
| **"The SRM eliminates zero false alarms entirely"** | **PURGED / FORBIDDEN** | **Rejected.** R-STNet reduces false alarm windows by $14,898$ ($37.8\%$), but residual FPR remains at $62.14\%$. |
| **"The system is validated for live industrial plant deployment"** | **PURGED / FORBIDDEN** | **Rejected.** Offline testbed evaluation only; lacks live SCADA integration and PLC hardware-in-the-loop validation. |
| **"The method operates as a real-time SCADA security firewall"** | **PURGED / FORBIDDEN** | **Rejected.** Latency is benchmarked offline ($0.060\text{ ms/window}$), but network telemetry transport is out of scope. |

---

## 3. Detailed Audit of Core Academic Claims

### 3.1 Baseline Benchmark & Clean Performance
- **Claim 1.1:** *"Isolation Forest achieves the highest clean-benchmark performance on downsampled SWaT ($F_1 = 0.7182, \text{Precision} = 0.8995, \text{Recall} = 0.5977, \text{FPR} = 0.92\%$) due to orthogonal axis-aligned partitioning."*
  - **Classification:** `[VERIFIED BY EXPERIMENT]` & `[INTERPRETATION]`
  - **Evidence:** `results/metrics/baseline_isolation_forest_swat.json`. Validated zero false-alarm saturation on uncorrupted data.
- **Claim 1.2:** *"Standard Spatio-Temporal Autoencoder (ST-AE) suffers from severe residual saturation on SWaT ($F_1 = 0.2169, \text{Precision} = 0.1216, \text{Recall} = 1.0000, \text{FPR} = 100.00\%$) when evaluated on mixed chronological test data without label leakage."*
  - **Classification:** `[VERIFIED BY EXPERIMENT]`
  - **Evidence:** `results/metrics/baseline_stae_swat.json`. Identifies vulnerability of continuous reconstruction error aggregation under unrecorded sensor shifts.

### 3.2 Ablation & SRM Contribution
- **Claim 2.1:** *"Proposed R-STNet outperforms its direct ablation counterpart RSTNetNoSRM across all precision-related metrics under identical 23k parameters ($F_1: 0.2171 \rightarrow 0.2740$, Precision: $0.1218 \rightarrow 0.1626$, FPR: $99.86\% \rightarrow 62.14\%$)."*
  - **Classification:** `[VERIFIED BY EXPERIMENT]`
  - **Evidence:** `results/metrics/proposed_rstnet_swat.json`, `results/metrics/baseline_rstnet_no_srm_swat.json`, `results/ablations/ablation_summary.json`.
- **Claim 2.2:** *"The Sensor Reliability Module (SRM) successfully suppresses 14,898 false alarm windows (37.8% relative false alarm reduction) and increases True Negatives from 56 to 14,954 windows."*
  - **Classification:** `[VERIFIED BY EXPERIMENT]`
  - **Evidence:** Confusion matrix entries in `results/metrics/proposed_rstnet_swat.json` ($TN=14954, FP=24541$) versus `results/metrics/baseline_rstnet_no_srm_swat.json` ($TN=56, FP=39439$).

### 3.3 Sensor Reliability & Degradation Separation
- **Claim 3.1:** *"Under controlled synthetic sensor degradation (Gaussian noise $\sigma \in [0.1, 0.5]$ and linear drift $\delta \in [0.1, 0.5]$), the SRM produces a statistically significant reliability separation of $\Delta r \approx +0.29$ between clean channels ($\bar{r}_{\text{clean}} \approx 0.304$) and corrupted channels ($\bar{r}_{\text{corrupted}} \approx 0.015$)."*
  - **Classification:** `[VERIFIED BY EXPERIMENT]`
  - **Evidence:** `results/corruption/corruption_benchmark_summary.json` and Figure 4.2 (`results/figures/fig2_clean_vs_corrupted_reliability.png`).
- **Claim 3.2:** *"The SRM achieved 100% sensor fault detection recall across the evaluated synthetic corruption masks."*
  - **Classification:** `[VERIFIED BY EXPERIMENT]`
  - **Evidence:** `results/corruption/corruption_benchmark_summary.json` (`fault_detection_recall: 1.0000`). Bounded strictly to synthetic corruption protocol.

### 3.4 Physical & Dataset Boundaries
- **Claim 4.1:** *"Purely statistical reliability estimation fails on naturally inactive, zero-variance actuators (such as backup pumps P202 and P204), yielding 0.0% fault recall when stuck at zero."*
  - **Classification:** `[VERIFIED BY EXPERIMENT]` & `[INTERPRETATION]`
  - **Evidence:** Detailed in Section 4.5 and Section 5.2. Confirmed by `test_scaler.py` and physical SWaT telemetry characteristics.
- **Claim 4.2:** *"Seven channels (including AIT201) exhibit severe distribution shift between training and testing, acting as constant/uninformative channels during normal training but shifting dynamically during testing."*
  - **Classification:** `[VERIFIED BY EXPERIMENT]` & `[CODE-DERIVED]`
  - **Evidence:** `docs/SWAT_CHANNEL_SHIFT_ANALYSIS.md` and Figure 4.7 (`results/figures/fig7_channel_shift_ait201.png`). Care taken not to mislabel shifts as physical sensor faults.

### 3.5 Methodological & Future Work Claims
- **Claim 5.1:** *"Threshold calibration is strictly conducted at the 95th percentile of normal validation anomaly scores with zero test label leakage."*
  - **Classification:** `[CODE-DERIVED]` & `[VERIFIED BY EXPERIMENT]`
  - **Evidence:** `scripts/train_deep_model.py` lines 14-15, 290-300; `results/metrics/proposed_rstnet_swat.json`.
- **Claim 5.2:** *"Future architectures should explore upstream reliability gating directly inside graph attention adjacency computation and incorporate PLC supervisory ladder logic."*
  - **Classification:** `[PROPOSED FUTURE WORK]`
  - **Evidence:** Section 5.4 of `report/FINAL/CHAPTER5_CONCLUSION.md`. Explicitly framed as future directions.

---

## 4. Conclusion of Claim Audit

All claims in the final project deliverables have been audited and verified. No forbidden, exaggerated, or unfalsifiable claims exist in the thesis, presentation, or codebase. The research contribution is rigorously defended within its empirical scope.
