# RAW RESULTS INDEX AND ARTIFACT MANIFEST

This manifest provides full provenance and traceability for all experimental results, metric files, model checkpoints, and generated figures incorporated into the Review 2 Report (**Chapter 4: Results and Discussion**).

---

## 1. Experimental Metrics Artifacts

| Artifact | Original Location | Copied Review 2 Location | Experiment / Model | Used in Chapter 4? | Authoritative Source Description |
| :--- | :--- | :--- | :--- | :---: | :--- |
| `baseline_isolation_forest_swat.json` | `results/metrics/` | `report/REVIEW2/metrics/` | Baseline A: Isolation Forest | **YES** (Table 4.3, Fig 4.1) | Clean SWaT mixed test evaluation (100 estimators, $\tau=0.63587$, Point-wise $F_1=0.7182$, $FPR=0.0092$). |
| `baseline_stae_swat.json` | `results/metrics/` | `report/REVIEW2/metrics/` | Baseline B: SpatioTemporal Autoencoder | **YES** (Table 4.3, Fig 4.1, 4.4, 4.5) | 15 epochs Conv1D-Linear AE ($\tau=0.08922$, Point-wise $F_1=0.2169$, $FPR=1.0000$, 170k params). |
| `baseline_rstnet_no_srm_swat.json` | `results/metrics/` | `report/REVIEW2/metrics/` | Baseline C: RSTNet without SRM | **YES** (Table 4.3, Fig 4.1, 4.4, 4.5, 4.6) | 15 epochs Spatio-Temporal Graph Network with uniform residual scoring ($\tau=12.07062$, Point-wise $F_1=0.2171$, $FPR=0.9986$, 23k params). |
| `proposed_rstnet_swat.json` | `results/metrics/` | `report/REVIEW2/metrics/` | Proposed R-STNet (with SRM) | **YES** (Table 4.3, Fig 4.1, 4.4, 4.5, 4.6) | 15 epochs Spatio-Temporal Graph Network with Sensor Reliability Module ($\tau=3.14008$, Point-wise $F_1=0.2740$, $FPR=0.6214$, 23k params). |

---

## 2. Robustness and Ablation Artifacts

| Artifact | Original Location | Copied Review 2 Location | Experiment / Scope | Used in Chapter 4? | Authoritative Source Description |
| :--- | :--- | :--- | :--- | :---: | :--- |
| `corruption_benchmark_summary.json` | `results/corruption/` | `report/REVIEW2/source_results/` | Controlled Corruption Suite ($E1-E4$) | **YES** (Tables 4.4–4.7, Figs 4.8–4.12) | 10 corruption scenarios (Clean, Gaussian Noise, Stuck Sensor, Sensor Drift) across random seeds $[42, 101, 2024]$ recording F1, FPR, reliability separation, and localization. |
| `ablation_summary.json` | `results/ablations/` | `report/REVIEW2/source_results/` | SRM Perspective Ablation (1 to 5) | **YES** (Table 4.8, Fig 4.13) | 5 ablation configurations isolating No SRM, Temporal Continuity only, Graph Neighborhood only, Residual Z-Score only, and Full SRM. |

---

## 3. Publication Figures Manifest

| Figure File | Original Source | Review 2 Location | Primary Subject / Scientific Question Addressed | Chapter 4 Mapping |
| :--- | :--- | :--- | :--- | :---: |
| `fig4_1_model_comparison_barchart.png` | Generated | `report/REVIEW2/figures/` | Clean-data benchmark comparison across all 4 models (F1, Precision, Recall, PR-AUC). | **Figure 4.1** |
| `system_anomaly_scores_timeline.png` | `results/plots/anomaly_scores/` | `report/REVIEW2/figures/` | Anomaly score timeline comparing Baseline C vs R-STNet against true SWaT attack intervals. | **Figure 4.2** |
| `fig4_4_confusion_matrices.png` | Generated | `report/REVIEW2/figures/` | Confusion matrices comparing TP, FP, TN, FN across all 4 models. | **Figure 4.3** |
| `fig4_5_training_loss_curves.png` | Generated | `report/REVIEW2/figures/` | Training MSE loss convergence across 15 epochs on normal training data. | **Figure 4.4** |
| `reliability_score_vs_time.png` | `results/plots/reliability/` | `report/REVIEW2/figures/` | Dynamic reliability trajectories $r_i(t)$ over time for normal vs shifted sensor channels. | **Figure 4.5** |
| `corrupted_vs_clean_reliability.png` | `results/plots/reliability/` | `report/REVIEW2/figures/` | Empirical separation between clean ($r \approx 0.30$) and corrupted channels ($r \approx 0.017$). | **Figure 4.6** |
| `f1_vs_corruption_severity.png` | `results/plots/corruption/` | `report/REVIEW2/figures/` | System anomaly F1 curves as Gaussian noise severity increases from $\sigma=0.0$ to $0.5$. | **Figure 4.7** |
| `fpr_vs_corruption_severity.png` | `results/plots/corruption/` | `report/REVIEW2/figures/` | System False Positive Rate (FPR) curves verifying R-STNet's false-alarm suppression. | **Figure 4.8** |
| `baseline_c_vs_rstnet_under_corruption.png` | `results/plots/corruption/` | `report/REVIEW2/figures/` | Side-by-side anomaly score timeline under sensor corruption: Baseline C explodes vs R-STNet suppresses. | **Figure 4.9** |
| `sensor_fault_id_f1_vs_severity.png` | `results/plots/corruption/` | `report/REVIEW2/figures/` | Fault detection recall (100%) and localization accuracy under increasing corruption severity. | **Figure 4.10** |
| `srm_ablation_comparison.png` | `results/plots/ablations/` | `report/REVIEW2/figures/` | Bar chart comparing F1 and FPR across all 5 SRM ablation configurations. | **Figure 4.11** |
| `fig4_12_channel_shift_diagnostic.png` | Generated | `report/REVIEW2/figures/` | Time-series diagnostic of SWaT channel AIT201: constant in normal train vs dynamic in test. | **Figure 4.12** |

---

## 4. Deep Model Checkpoint Verification

| Checkpoint File | Location | Size (Bytes) | Model Class | Parameters | Verified Integrity |
| :--- | :--- | :---: | :--- | :---: | :---: |
| `baseline_stae_swat.pt` | `results/checkpoints/` | 687,990 | `SpatioTemporalAE` | 170,643 | **VERIFIED** |
| `baseline_rstnet_no_srm_swat.pt` | `results/checkpoints/` | 99,540 | `RSTNetNoSRM` | 23,233 | **VERIFIED** |
| `proposed_rstnet_swat.pt` | `results/checkpoints/` | 100,502 | `RSTNet` (with SRM) | 23,233 | **VERIFIED** |

*All artifacts preserved in original locations and mirrored non-destructively in `report/REVIEW2/`.*
