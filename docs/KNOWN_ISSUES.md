# Known Issues, Edge Cases, and Risk Register

**Project:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Last Updated:** 2026-10-21  

---

## 1. Data Parsing & Schema Gotchas

### Issue 1.1: Leading Whitespace in SWaT Column Headers
- **Severity:** High (Causes `KeyError` or missing column exceptions).
- **Description:**  
  Raw SWaT files (`normal.csv`, `attack.csv`, `merged.csv`) contain leading/trailing whitespaces in column names (e.g., `' Timestamp'`, `' FIT101'`, `' Normal/Attack'`).
- **Mitigation:**  
  In `src/data/swat_loader.py`, execute:
  ```python
  df.columns = [col.strip() for col in df.columns]
  ```

### Issue 1.2: Zero-Variance Channels in Normal Operation
- **Severity:** Medium (Causes division by zero / NaNs during normalization).
- **Description:**  
  Certain digital actuators and backup pumps (e.g. `P102`, `P202`, `P302`, `UV401`) remain constantly OFF or ON throughout the normal 7-day training period in SWaT. Their sample variance is exactly zero ($\sigma_i = 0$).
- **Mitigation:**  
  Use a robust scaling function with an explicit variance check:
  ```python
  std = np.std(x, axis=0)
  std[std < 1e-6] = 1.0  # Prevent divide-by-zero, preserving constant value
  ```

### Issue 1.3: Label String Formatting
- **Severity:** Medium.
- **Description:**  
  Labels in `attack.csv` are strings (`'Normal'`, `'Attack'`, `'A T T A C K'`).
- **Mitigation:**  
  Normalize label mapping explicitly:
  ```python
  labels = (df['Normal/Attack'].astype(str).str.strip().str.upper() != 'NORMAL').astype(int)
  ```

---

## 2. Modeling & Hardware Risks

### Issue 2.1: 4GB GPU VRAM Out-of-Memory (OOM)
- **Severity:** High.
- **Risk:**  
  Attempting to use dense multi-head self-attention with long sequence windows ($L \ge 128$) or batch sizes ($B \ge 128$) on the RTX 3050 Laptop GPU (4096 MiB VRAM) will trigger PyTorch CUDA OOM.
- **Mitigation:**  
  - Fixed hyperparameter ceiling: $L = 60$, $B = 64$.
  - Lightweight Dilated Convolutions / Bi-GRU backbone rather than massive multi-layer Transformers.
  - Automatic fallback to CPU mode if GPU memory allocation exceeds threshold.

### Issue 2.2: Point Adjustment Metric Distortion
- **Severity:** High (Scientific Validity).
- **Description:**  
  Many anomaly detection papers use the "Point Adjustment" (PA) protocol (if any point in an anomalous segment is flagged, the entire segment is counted as TP). Recent literature (Kim et al., 2022; Garg et al., 2021) has proven that PA dramatically inflates F1 scores even for random noise detectors.
- **Mitigation:**  
  - Report **raw point-wise F1, Precision, and Recall** as the primary ground truth.
  - Report PR-AUC (Precision-Recall Area Under Curve), which is completely threshold- and adjustment-independent.
  - If point-adjusted metrics are computed for direct comparison with external baselines, they are explicitly labeled as `(PA)` alongside the raw metrics.

---

## 3. Reliability Calibration Gotchas

### Issue 3.1: Feedback Loops in Reliability Weighting
- **Severity:** Medium.
- **Risk:**  
  If the model predicts sensor values based on corrupted neighbors, corruption can cascade across the graph.
- **Mitigation:**  
  - Graph attention weights are bounded.
  - The reliability estimation relies on a three-way triangulation (Temporal Volatility + Graph Neighbors + Residual Z-Score), so a sensor cannot fool the system simply by pulling down an adjacent channel.
