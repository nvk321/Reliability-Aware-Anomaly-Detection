# Reproducibility Guide and Experimental Replication Package

**Project:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Course:** BCSE209P — Machine Learning Lab  
**Date:** October 2026  
**Status:** Certified Zero-Leakage Reproducible Framework  

---

## 1. Overview and Environment Requirements

This replication package enables full, deterministic reproduction of all empirical findings, metric JSON files, model checkpoints, and publication-quality figures presented in Review 2 Chapter 4.

### 1.1 Hardware and OS Requirements
- **Operating System:** Windows 10/11 64-bit or Linux (Ubuntu 20.04+).
- **GPU:** NVIDIA GPU with $\ge 4\text{ GB}$ VRAM (CUDA 12.1+ support). Tested on NVIDIA GeForce RTX 3050 Laptop GPU.
- **CPU:** Intel/AMD x86_64 CPU with $\ge 4$ cores.
- **System Memory:** Minimum $8\text{ GB}$ RAM ($16\text{ GB}$ recommended).
- **Storage:** $\ge 5\text{ GB}$ free disk space for raw and processed datasets.

### 1.2 Software Dependencies
Replication requires Python 3.10+ and the virtual environment at `.venv`:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install numpy scipy scikit-learn matplotlib
```

Exact audited environment versions:
- `torch`: `2.6.0+cu124`
- `numpy`: `2.2.3`
- `scikit-learn`: `1.6.1`
- `scipy`: `1.15.2`
- `matplotlib`: `3.10.1`

---

## 2. Determinism and Random Seed Specifications

To guarantee mathematical reproducibility across hardware platforms:
- **Primary Training & Clean Benchmark Seed:** `42`
- **Multi-Seed Robustness Evaluation Seeds:** `[42, 101, 2024]`
- **PyTorch Determinism:** CUDA benchmarking flags configured:
  ```python
  torch.manual_seed(seed)
  np.random.seed(seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed_all(seed)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
  ```

---

## 3. End-to-End Pipeline Replication Instructions

### Step 1: Preprocess the SWaT Dataset
Reconstruct the official chronological train, validation, and mixed test partitions with leak-free scaling:
```powershell
.venv\Scripts\python.exe scripts/preprocess_swat.py --raw_dir reference/swat_data --processed_dir data/processed/SWaT --downsample_rate 10 --train_ratio 0.80
```
*Outputs generated in `data/processed/SWaT/`:*
- `train.npy`: Shape $(39744, 51)$, 100% normal.
- `val.npy`: Shape $(9936, 51)$, 100% normal.
- `test.npy`: Shape $(44992, 51)$, chronological mixed test.
- `test_labels.npy`: Shape $(44992,)$, $5,468$ attack labels ($12.16\%$).
- `scaler_params.json`: Robust standard scaler parameters with zero-variance protection.

---

### Step 2: Train Baseline A (Isolation Forest)
Train the non-parametric tree ensemble on pure normal training data and evaluate on mixed test data:
```powershell
.venv\Scripts\python.exe scripts/train_baseline.py --model isolation_forest --data_dir data/processed/SWaT --output_dir results/metrics
```
*Metric Output:* `results/metrics/baseline_isolation_forest_swat.json` ($F_1 = 0.7182, FPR = 0.0092$).

---

### Step 3: Train Deep Models (ST-AE, RSTNetNoSRM, Proposed R-STNet)
Train each deep architecture for 15 epochs on normal training data and calibrate thresholds on normal validation data:

```powershell
# Baseline B: Spatio-Temporal Autoencoder
.venv\Scripts\python.exe scripts/train_deep_model.py --model stae --epochs 15 --batch_size 64 --lr 0.001 --seed 42

# Baseline C: RSTNet without SRM
.venv\Scripts\python.exe scripts/train_deep_model.py --model rstnet_no_srm --epochs 15 --batch_size 64 --lr 0.0005 --seed 42

# Proposed Model: R-STNet (with SRM)
.venv\Scripts\python.exe scripts/train_deep_model.py --model rstnet --epochs 15 --batch_size 64 --lr 0.0005 --seed 42
```
*Checkpoints generated in `results/checkpoints/`:*
- `baseline_stae_swat.pt` ($688\text{ KB}$, 170k params).
- `baseline_rstnet_no_srm_swat.pt` ($100\text{ KB}$, 23k params).
- `proposed_rstnet_swat.pt` ($101\text{ KB}$, 23k params).

*Metrics generated in `results/metrics/`:*
- `baseline_stae_swat.json` ($F_1 = 0.2169, FPR = 1.0000$).
- `baseline_rstnet_no_srm_swat.json` ($F_1 = 0.2171, FPR = 0.9986$).
- `proposed_rstnet_swat.json` ($F_1 = 0.2740, FPR = 0.6214$).

---

### Step 4: Execute Multi-Seed Controlled Robustness Suite
Evaluate all four models across Gaussian noise ($E2$), stuck sensors ($E3$), and drift ($E4$) across seeds $[42, 101, 2024]$:
```powershell
.venv\Scripts\python.exe scripts/evaluate_corruption.py --data_dir data/processed/SWaT --checkpoint_dir results/checkpoints --output_dir results/corruption
```
*Output generated:* `results/corruption/corruption_benchmark_summary.json` ($18.8\text{ KB}$ containing all 10 corruption scenarios and multi-seed variances).

---

### Step 5: Execute Ablations and Generate All 12 Figures
Run the 5-condition SRM ablation suite and render all 12 publication-quality plots at 300 DPI:
```powershell
# Run ablations and primary figures (Figs 4.2, 4.5 - 4.11)
.venv\Scripts\python.exe scripts/run_ablations_and_generate_plots.py

# Run supplementary benchmark figures (Figs 4.1, 4.3, 4.4, 4.12)
.venv\Scripts\python.exe scripts/generate_review2_supplementary_plots.py
```
*Outputs generated in `report/REVIEW2/figures/`:*
All 12 PNG figures certified at 300 DPI.

---

## 4. Verification Checksums and Artifact Verification

All core artifacts are mirrored in `report/REVIEW2/`:
- Metrics: `report/REVIEW2/metrics/`
- Tables: `report/REVIEW2/TABLES.md`
- Figures: `report/REVIEW2/figures/`
- Full Manifest: `report/REVIEW2/RAW_RESULTS_INDEX.md`

Re-executing the commands above will deterministically reproduce the exact numerical metrics reported in Chapter 4 within numerical float precision ($\pm 10^{-4}$).
