# Reproducibility Guide and Execution Modes

**Project:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Students:** P Naga Venkata Kushal (23BDS0133) & Koteru Tarun (23BDS0103)  
**Faculty Guide:** Dr. Dega Nagaraju  
**Status:** Certified Zero-Leakage Reproducibility Guide  

---

## 1. Environment and Hardware Specifications

All reported experiments were executed on an edge-representative industrial workstation:
- **Operating System:** Microsoft Windows 11 Enterprise (64-bit). Also verified on Ubuntu 22.04 LTS.
- **Python Version:** Python 3.11.9 (compatible with 3.10+).
- **Deep Learning Framework:** PyTorch 2.6.0+cu124 (CUDA 12.4).
- **GPU Accelerator:** NVIDIA GeForce RTX 3050 Laptop GPU ($4,096\text{ MB}$ VRAM, $2,048$ CUDA Cores).
- **CPU:** Intel Core i5 (12 virtual threads @ $2.50\text{ GHz}$).
- **Host RAM:** $16.0\text{ GB}$ DDR4.

### Virtual Environment Setup
```powershell
# 1. Clone repository
git clone https://github.com/kushal23bds0133/Reliability-Aware-Anomaly-Detection.git
cd Reliability-Aware-Anomaly-Detection

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate   # Linux/macOS: source .venv/bin/activate

# 3. Install PyTorch with CUDA 12.4 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Install project dependencies
pip install -r requirements.txt
```

---

## 2. Execution Mode A: Fast Demo Mode (< 60 Seconds)

**Target Audience:** Evaluators, faculty examiners, and reviewers who want to immediately inspect the working model, visualize anomaly detection timelines, simulate sensor corruption, and verify benchmark numbers without retraining deep networks.

Fast Demo Mode utilizes the pre-computed, lightweight model checkpoints ($< 1\text{ MB}$ total) located in `results/checkpoints/` and preprocessed test data in `data/processed/SWaT/`.

### Option A1: Run the Interactive Streamlit Demo
Launch the interactive web application to visually explore the plant status, inject synthetic sensor corruption, and inspect the SRM's dynamic downweighting in real time:
```powershell
streamlit run demo/app.py
```
*Expected Startup Time:* $\approx 3$ to $5$ seconds. Opens at `http://localhost:8501`.

### Option A2: Run the Project Health Check & Fast Evaluation
Run the automated validation script to verify all checkpoints, environment dependencies, and test inference:
```powershell
python scripts/project_health_check.py
```
*Expected Runtime:* $\approx 15$ seconds. Produces a complete diagnostic pass across all components.

---

## 3. Execution Mode B: Full Reproduction Mode (From Scratch)

**Target Audience:** Researchers wishing to completely rebuild the processed datasets from raw SWaT CSVs, train all four models from random initialization, execute the multi-seed corruption benchmark, run the 5-condition ablation suite, and re-render all 12 publication figures.

### Step 1: Preprocess the Raw SWaT Dataset
Reconstruct the official chronological train, validation, and mixed test splits with leak-free robust scaling:
```powershell
python scripts/preprocess_swat.py --raw_dir reference/swat_data --processed_dir data/processed/SWaT --downsample_rate 10 --train_ratio 0.80
```
- *Measured Runtime:* $\approx 45$ seconds.
- *Outputs Generated:* `train.npy` ($39,744 \times 51$), `val.npy` ($9,936 \times 51$), `test.npy` ($44,992 \times 51$), `test_labels.npy`, `scaler_params.json`.

---

### Step 2: Train Baseline A (Isolation Forest)
Train the classical 100-tree ensemble on normal training data and evaluate on mixed test data:
```powershell
python scripts/train_baseline.py --model isolation_forest --data_dir data/processed/SWaT --output_dir results/metrics
```
- *Measured Runtime:* **12.40 seconds** (CPU).
- *Output:* `results/metrics/baseline_isolation_forest_swat.json` ($F_1 = 0.7182, FPR = 0.0092$).

---

### Step 3: Train Deep Learning Models (15 Epochs Each)
Train each deep architecture on normal data and calibrate decision thresholds $\tau$ at the $95^{\text{th}}$ percentile of validation scores:

```powershell
# 1. Baseline B: Spatio-Temporal Autoencoder (ST-AE)
python scripts/train_deep_model.py --model stae --epochs 15 --batch_size 64 --lr 0.001 --seed 42
# Measured Runtime: 112.35 seconds (7.49 s/epoch)
# Output: results/checkpoints/baseline_stae_swat.pt (688 KB)

# 2. Baseline C: RSTNet without SRM (RSTNetNoSRM)
python scripts/train_deep_model.py --model rstnet_no_srm --epochs 15 --batch_size 64 --lr 0.0005 --seed 42
# Measured Runtime: 112.89 seconds (7.53 s/epoch)
# Output: results/checkpoints/baseline_rstnet_no_srm_swat.pt (100 KB)

# 3. Proposed Model: R-STNet (with SRM)
python scripts/train_deep_model.py --model rstnet --epochs 15 --batch_size 64 --lr 0.0005 --seed 42
# Measured Runtime: 134.73 seconds (8.98 s/epoch)
# Output: results/checkpoints/proposed_rstnet_swat.pt (101 KB)
```
- *Total Deep Training Wall-Clock Time:* $\approx 360$ seconds ($6$ minutes).

---

### Step 4: Execute Multi-Seed Controlled Corruption Suite
Evaluate all four models across Gaussian noise ($E2$), stuck sensors ($E3$), and dynamic drift ($E4$) across random seeds $[42, 101, 2024]$:
```powershell
python scripts/evaluate_corruption.py --data_dir data/processed/SWaT --checkpoint_dir results/checkpoints --output_dir results/corruption
```
- *Measured Runtime:* $\approx 180$ seconds ($3$ minutes).
- *Output:* `results/corruption/corruption_benchmark_summary.json` ($18.8\text{ KB}$).

---

### Step 5: Execute Ablations and Generate All 12 Figures
Run the 5-condition ablation suite and render all 12 publication-quality plots at 300 DPI:
```powershell
# Run ablations and primary figures (Figs 4.2, 4.5 - 4.11)
python scripts/run_ablations_and_generate_plots.py

# Run supplementary diagnostic figures (Figs 4.1, 4.3, 4.4, 4.12)
python scripts/generate_review2_supplementary_plots.py
```
- *Measured Runtime:* $\approx 120$ seconds ($2$ minutes).
- *Outputs Generated:* All 12 high-resolution figures in `report/REVIEW2/figures/` and `results/plots/`.

---

## 4. Total Pipeline Reproduction Runtime Summary

| Pipeline Stage | Executed Command | Measured Execution Time | Hardware |
| :--- | :--- | :---: | :--- |
| **Stage 1: Preprocessing** | `scripts/preprocess_swat.py` | $45\text{ s}$ | Host CPU & SSD |
| **Stage 2: Isolation Forest** | `scripts/train_baseline.py` | $\mathbf{12.40\text{ s}}$ | Host CPU (12 cores) |
| **Stage 3: Deep Models** | `scripts/train_deep_model.py` ($\times 3$) | $\mathbf{359.97\text{ s}}$ ($6.0\text{ min}$) | NVIDIA RTX 3050 Laptop GPU |
| **Stage 4: Corruption Suite**| `scripts/evaluate_corruption.py` | $\mathbf{180.00\text{ s}}$ ($3.0\text{ min}$) | NVIDIA RTX 3050 Laptop GPU |
| **Stage 5: Ablations & Plots**| `run_ablations...py` + `generate...py` | $\mathbf{120.00\text{ s}}$ ($2.0\text{ min}$) | NVIDIA RTX 3050 Laptop GPU |
| **Full Reproduction Total** | End-to-end execution from raw CSV | $\mathbf{\approx 12\text{ minutes}}$ | Complete Rebuild |

---

## 5. Automated Unit Tests

Execute the standardized test suite to verify module shapes, tensor slicing, and reliability functions:
```powershell
pytest tests/ -v
```
All unit tests execute in $< 10$ seconds.
