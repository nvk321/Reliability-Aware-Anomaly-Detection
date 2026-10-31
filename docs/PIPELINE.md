# Authoritative Research Pipeline: Reliability-Aware Anomaly Detection (R-STNet)

**Project:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Course:** BCSE209P — Machine Learning Lab  
**Students:** P Naga Venkata Kushal (23BDS0133) & Koteru Tarun (23BDS0103)  
**Faculty Guide:** Dr. Dega Nagaraju  
**Repository:** [https://github.com/nvk321/Reliability-Aware-Anomaly-Detection](https://github.com/nvk321/Reliability-Aware-Anomaly-Detection)  
**Release:** `v1.0.0-final`

---

## 1. Executive Summary & Canonical Flow

The project enforces **one single authoritative research pipeline**. There are no competing or deprecated pipelines. Every experiment, result table, figure, and Streamlit view traces directly back to this canonical sequence.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             SWaT RAW DATASET (CSV)                               │
│  496,800 train rows (7 days normal) | 449,919 test rows (4 days with 36 attacks)  │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼ [scripts/prepare_data.py]
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION & 10x DOWNSAMPLING                         │
│  - Column sanitization (51 sensor/actuator channels)                             │
│  - 10x temporal downsampling (10s physical intervals)                            │
│  - Chronological 80/20 train/validation split (y=0, 100% normal)                 │
│  - Outputs: train.npy (39,744), val.npy (9,936), test.npy (44,992)               │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼ [src/preprocessing/scaler.py]
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   LEAK-FREE ZERO-VARIANCE ROBUST SCALING                         │
│  - Mean mu_i and scale sigma_i computed STRICTLY on normal training split        │
│  - Zero-variance protection: sigma_i clamped to 1.0 if sigma_i < 10^(-8)         │
│  - Validation and test transformed blindly using frozen train parameters         │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼ [src/preprocessing/windowing.py]
┌──────────────────────────────────────────────────────────────────────────────────┐
│                       SLIDING WINDOW GENERATION (3D TENSORS)                     │
│  - Window length L = 30 (300 seconds = 5 minutes historical context)             │
│  - Stride S = 1 (complete sequential coverage without data loss)                 │
│  - Label protocol: point_end (window label equals ground truth at step t)       │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
┌──────────────────────────────────────┐    ┌──────────────────────────────────────┐
│       BASELINE MODEL TRAINING        │    │       PROPOSED R-STNet TRAINING      │
│  - Baseline A: Isolation Forest      │    │  - Dilated 1D Temporal Convolutions  │
│    (scripts/train_baselines.py)      │    │  - Top-k Graph Attention (GAT)       │
│  - Baseline B: ST-AE                 │    │  - One-step ahead forecasting head   │
│  - Baseline C: RSTNetNoSRM (Ablation)│    │  - AdamW, lr=1e-3, 15 epochs         │
└──────────────────┬───────────────────┘    └──────────────────┬───────────────────┘
                   │                                           │
                   │                                           ▼ [src/reliability/]
                   │                        ┌──────────────────────────────────────┐
                   │                        │      SRM VALIDATION CALIBRATION      │
                   │                        │  - Calibrated on normal val windows  │
                   │                        │  - Rate-of-change limits (max_delta) │
                   │                        │  - Topological neighbor weights (W)  │
                   │                        │  - Residual MAD statistics           │
                   │                        └──────────────────┬───────────────────┘
                   │                                           │
                   └─────────────────────┬─────────────────────┘
                                         │
                                         ▼ [Validation Quantile Protocol]
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   DECISION THRESHOLD CALIBRATION (LEAK-FREE)                     │
│  - Decision threshold tau calibrated strictly on pure normal validation split    │
│  - R-STNet threshold: tau = 3.1401 (99th percentile of normal val scores)        │
│  - Baseline C threshold: tau = 12.0706                                           │
│  - ZERO test labels accessed during threshold tuning                             │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼ [src/anomaly_detection/discriminator.py]
┌──────────────────────────────────────────────────────────────────────────────────┐
│               ONLINE INFERENCE & 4-WAY OPERATIONAL CLASSIFICATION                │
│  - Forecast: x_hat(t)                                                            │
│  - Normalized residual: z_i(t) = |x_i(t) - x_hat_i(t)| / (MAD_i + eps)           │
│  - Online reliability vector: r_i(t) in [0, 1]                                   │
│  - Weighted system anomaly score: S_sys(t) = sum(r_i * z_i) / (sum(r_i) + eps)   │
│  - Sensor fault score: S_fault(t) = max_i [ (1 - r_i) * z_i ]                    │
│  - Operational states: Case A (Normal), Case B (Sensor Fault),                   │
│                       Case C (System Attack), Case D (Dual Fault)               │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                   ┌─────────────────────┼─────────────────────┐
                   ▼                     ▼                     ▼
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│   CLEAN EVALUATION   │  │ CORRUPTION BENCHMARK │  │  ABLATION EXPERIMENTS│
│  - Point-wise F1     │  │  - 9 failure modes   │  │  - Ablation 1 (No SRM│
│  - Precision, Recall │  │  - Multi-seed (3x)   │  │  - -Temporal         │
│  - FPR, PR-AUC       │  │  - Delta r separation│  │  - -Residual         │
│  - results/metrics/  │  │  - results/corruption│  │  - -Graph, Full SRM  │
└──────────────────────┘  └──────────────────────┘  └──────────────────────┘
                   │                     │                     │
                   └─────────────────────┼─────────────────────┘
                                         │
                                         ▼ [demo/app.py]
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         INTERACTIVE STREAMLIT DEMO                               │
│  - Loads frozen checkpoints and preprocessed evaluation arrays                   │
│  - 9 structured tabs matching thesis narrative from overview to limitations      │
│  - Interactive telemetry scrubbing, fault simulation, and audited comparison     │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Stage-by-Stage Implementation Mapping

| Pipeline Stage | Python Driver Script | Core Module | Primary Input | Authoritative Output |
| :--- | :--- | :--- | :--- | :--- |
| **1. Ingestion** | `scripts/preprocess_swat.py` | `src/data/swat_loader.py` | `reference/swat_data/*.csv` | `data/processed/SWaT/*.npy` |
| **2. Scaling** | `scripts/preprocess_swat.py` | `src/preprocessing/scaler.py` | `train.npy` (y=0) | `scaler_params.json` |
| **3. Windowing** | `src/preprocessing/windowing.py` | `SlidingWindowGenerator` | `(T, 51)` 2D array | `(N, 30, 51)` 3D tensor |
| **4. Deep Training** | `scripts/train_deep_model.py` | `src/models/rstnet.py` | `train.npy`, `val.npy` | `results/checkpoints/*.pt` |
| **5. Baselines** | `scripts/train_baseline.py` | `src/models/baselines/` | `train.npy`, `val.npy` | `results/metrics/baseline_*.json` |
| **6. SRM Calibration**| `scripts/train_deep_model.py` | `src/reliability/sensor_reliability.py` | `val.npy` windows | Calibrated SRM state dict |
| **7. Inference** | `src/models/rstnet.py` | `RSTNet.evaluate_step()` | Window `(30, 51)` | $S_{\text{sys}}(t), r(t), S_{\text{fault}}$ |
| **8. Corruption** | `scripts/evaluate_corruption.py` | `src/corruption/corruptor.py` | `test.npy` windows | `results/corruption/*.json` |
| **9. Ablations** | `scripts/run_ablations_and_generate_plots.py` | `SensorReliabilityModule` | `test.npy` windows | `results/ablations/*.json` |
| **10. Demonstration**| `streamlit run demo/app.py` | `demo/app.py` | Checkpoints + Processed | Interactive Web Dashboard |

---

## 3. Strict Scientific Freeze & Integrity Rules

1. **No Test Leakage:** Decision thresholds $\tau$ are computed strictly as the 99th percentile of pure normal validation scores ($y = 0$). Test labels are never accessed during tuning.
2. **Standardized Downsampling:** All models and benchmarks utilize the identical 10x downsampled splits ($39,744$ train, $9,936$ val, $44,963$ test windows).
3. **Primary Metric Ground Truth:** Raw point-wise F1, Precision, Recall, PR-AUC, and FPR are the sole authoritative metrics. Point Adjustment (PA) is reported strictly as an external reference.
4. **Honest Baseline Reporting:** Isolation Forest achieves $F_1 = 0.7182$ on clean SWaT, outperforming deep reconstruction baselines ($F_1 = 0.2171$). The proposed contribution is explicit robustness under sensor corruption (+26.2% F1 over Baseline C, 14,898 false alarms eliminated), never a false claim of universal clean-benchmark superiority.
