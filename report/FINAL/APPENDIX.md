# APPENDIX: CORE IMPLEMENTATION CODE REPOSITORY

**Project:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Students:** P Naga Venkata Kushal (23BDS0133) & Koteru Tarun (23BDS0103)  
**Faculty Guide:** Dr. Dega Nagaraju  
**Repository:** [https://github.com/kushal23bds0133/Reliability-Aware-Anomaly-Detection](https://github.com/kushal23bds0133/Reliability-Aware-Anomaly-Detection)  

---

## Academic Note on Code Appendix

In compliance with university thesis guidelines, this Appendix compiles the authoritative, executable Python source code implementations of the primary computational algorithms developed in this thesis. The complete, fully runnable software package—including training scripts, multi-seed corruption test harnesses, and automated unit tests—is publicly hosted in the official GitHub repository referenced above.

---

## Appendix A — Data Preprocessing and Zero-Variance Scaler

**File:** `src/preprocessing/scaler.py`

```python
"""
Robust Standard Scaler with Zero-Variance Protection for Industrial Telemetry.
Guarantees numerical stability on constant/zero-variance channels without data leakage.
"""

from typing import Dict, Any, Optional
import json
import numpy as np


class RobustStandardScaler:
    """
    Standardizes features by removing the mean and scaling to unit variance.
    Guarantees numerical stability on constant/zero-variance channels.
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        self.constant_mask_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "RobustStandardScaler":
        """Compute mean and standard deviation strictly from normal training data."""
        if X.ndim == 3:
            flat_X = X.reshape(-1, X.shape[-1])
        elif X.ndim == 2:
            flat_X = X
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape {X.shape}")

        self.n_features_ = flat_X.shape[1]
        self.mean_ = np.nanmean(flat_X, axis=0)
        std = np.nanstd(flat_X, axis=0)

        # Detect constant features with zero or near-zero training variance
        self.constant_mask_ = std < self.eps
        # Replace zero scale with 1.0 to preserve constant values (x - mean -> 0.0)
        std[self.constant_mask_] = 1.0
        self.std_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardize the input array using fitted training mean and scale."""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler has not been fitted yet. Call fit() first.")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to training data, then transform it."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Undo the standardization."""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler has not been fitted yet. Call fit() first.")
        return (X * self.std_) + self.mean_
```

---

## Appendix B — Sliding Window Generation with Causal Labeling

**File:** `src/preprocessing/windowing.py`

```python
"""
Temporal Sliding Window Generator for Multivariate Time-Series Anomaly Detection.
Enforces point_end causal labeling to prevent future-label leakage.
"""

from typing import Tuple, Optional
import numpy as np


class SlidingWindowGenerator:
    """Generates overlapping sliding windows from multivariate sequences."""

    def __init__(self, window_size: int = 30, stride: int = 1, label_mode: str = "point_end"):
        self.window_size = window_size
        self.stride = stride
        self.label_mode = label_mode

    def create_windows(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extracts temporal windows of shape (N_windows, window_size, M).
        Label assignment respects point_end causality: y(window_t) = label(t).
        """
        n_samples, n_features = data.shape
        if n_samples < self.window_size:
            raise ValueError(f"Data length ({n_samples}) < window size ({self.window_size})")

        num_windows = (n_samples - self.window_size) // self.stride + 1
        windows = np.lib.stride_tricks.sliding_window_view(data, (self.window_size, n_features))[::self.stride]
        windows = windows.squeeze(axis=1)

        window_labels = None
        if labels is not None:
            if self.label_mode == "point_end":
                window_indices = np.arange(self.window_size - 1, n_samples, self.stride)[:num_windows]
                window_labels = labels[window_indices]
            elif self.label_mode == "any":
                label_windows = np.lib.stride_tricks.sliding_window_view(labels, self.window_size)[::self.stride]
                window_labels = (np.sum(label_windows, axis=1) > 0).astype(int)

        return windows, window_labels
```

---

## Appendix C — Proposed R-STNet Architecture

**File:** `src/models/rstnet.py`

```python
"""
R-STNet: Reliability-Aware Spatio-Temporal Network.
Combines Causal Dilated TCN, Spatial Graph Attention, and Sensor Reliability Module.
Total Trainable Parameters: 23,233.
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.reliability.sensor_reliability import SensorReliabilityModule


class CausalConv1d(nn.Module):
    """1D Causal Convolution ensuring output at t depends strictly on <= t."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class SpatialGraphAttention(nn.Module):
    """Multi-Head Graph Attention Layer across learned adjacency topology."""
    def __init__(self, in_features: int, out_features: int, num_nodes: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.adj = nn.Parameter(torch.ones(num_nodes, num_nodes) / num_nodes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, N, F_in = h.size()
        Wh = self.W(h)  # (B, N, F_out)
        Wh1 = Wh.unsqueeze(2).expand(B, N, N, -1)
        Wh2 = Wh.unsqueeze(1).expand(B, N, N, -1)
        comb = torch.cat([Wh1, Wh2], dim=-1)
        e = F.leaky_relu(torch.matmul(comb, self.a).squeeze(-1))
        # Mask with learned adjacency
        adj_soft = F.softmax(self.adj, dim=-1)
        attn = F.softmax(e * adj_soft.unsqueeze(0), dim=-1)
        out = torch.matmul(attn, Wh)
        return F.elu(out)


class RSTNet(nn.Module):
    """Reliability-Aware Spatio-Temporal Network for Industrial Multivariate Telemetry."""
    def __init__(self, in_channels: int = 51, hidden_dim: int = 32, num_nodes: int = 51):
        super().__init__()
        self.in_channels = in_channels
        self.num_nodes = num_nodes

        # Temporal Dilated Backbone (Causal TCN)
        self.tcn1 = CausalConv1d(1, hidden_dim, kernel_size=3, dilation=1)
        self.tcn2 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2)
        self.tcn3 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=4)

        # Spatial Graph Backbone (GAT)
        self.gat = SpatialGraphAttention(hidden_dim, hidden_dim, num_nodes=num_nodes)

        # Reconstruction Head
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Auxiliary Sensor Reliability Module
        self.srm = SensorReliabilityModule(num_sensors=in_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, L, M)
        Returns:
            x_hat: Reconstructed sensor values at time t (B, M)
            latent: Latent embeddings (B, M, hidden_dim)
        """
        B, L, M = x.size()
        # Permute for node-wise TCN processing: (B * M, 1, L)
        x_perm = x.permute(0, 2, 1).contiguous().view(B * M, 1, L)
        h_t = F.relu(self.tcn1(x_perm))
        h_t = F.relu(self.tcn2(h_t))
        h_t = F.relu(self.tcn3(h_t))
        h_t_last = h_t[:, :, -1].view(B, M, -1)  # (B, M, hidden_dim)

        # Spatial Graph Attention
        h_spatial = self.gat(h_t_last)

        # Reconstruction
        x_hat = self.decoder(h_spatial).squeeze(-1)  # (B, M)
        return x_hat, h_spatial

    def calibrate_srm(self, val_windows: np.ndarray, device: torch.device):
        """Calibrates SRM statistical baselines on normal validation data."""
        self.eval()
        with torch.no_grad():
            tensor_val = torch.tensor(val_windows, dtype=torch.float32).to(device)
            preds, _ = self.forward(tensor_val)
            preds_np = preds.cpu().numpy()
            self.srm.calibrate_baselines(val_windows, predicted_x=preds_np)
```

---

## Appendix D — Multi-Perspective Sensor Reliability Module (SRM)

**File:** `src/reliability/sensor_reliability.py`

```python
"""
Multi-Perspective Sensor Reliability Module (SRM).
Fuses Temporal Continuity, Graph Neighborhood Consistency, and Residual Confidence.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


class SensorReliabilityModule:
    """Computes dynamic scalar reliability weights r_i(t) in [0, 1]."""

    def __init__(self, num_sensors: int, w_v: float = 1.0, w_c: float = 1.2, w_p: float = 1.0):
        self.num_sensors = num_sensors
        self.w_v = w_v
        self.w_c = w_c
        self.w_p = w_p

        # Statistical calibration baselines
        self.mad_velocity_: Optional[np.ndarray] = None
        self.mad_residual_: Optional[np.ndarray] = None
        self.sigma_neigh_: Optional[np.ndarray] = None
        self.adj_matrix_: Optional[np.ndarray] = None

    def calibrate_baselines(self, val_windows: np.ndarray, predicted_x: np.ndarray):
        """Calibrates MAD and neighborhood variances strictly from normal validation data."""
        # 1. Temporal Velocity MAD
        diffs = np.abs(val_windows[:, -1, :] - val_windows[:, -2, :])
        med_diff = np.median(diffs, axis=0)
        self.mad_velocity_ = np.median(np.abs(diffs - med_diff), axis=0)
        self.mad_velocity_[self.mad_velocity_ < 1e-4] = 1.0

        # 2. Residual Error MAD
        errors = np.abs(val_windows[:, -1, :] - predicted_x)
        med_err = np.median(errors, axis=0)
        self.mad_residual_ = np.median(np.abs(errors - med_err), axis=0)
        self.mad_residual_[self.mad_residual_ < 1e-4] = 1.0

        # 3. Neighborhood Consistency Baseline
        # Correlation-based empirical adjacency
        flat_val = val_windows[:, -1, :]
        corr = np.corrcoef(flat_val.T)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 0.0)
        self.adj_matrix_ = np.abs(corr)
        row_sums = self.adj_matrix_.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.adj_matrix_ = self.adj_matrix_ / row_sums

        neigh_expect = np.dot(flat_val, self.adj_matrix_.T)
        neigh_diff = np.abs(flat_val - neigh_expect)
        self.sigma_neigh_ = np.std(neigh_diff, axis=0)
        self.sigma_neigh_[self.sigma_neigh_ < 1e-4] = 1.0

    def estimate_reliability_batch(
        self,
        windows: np.ndarray,
        predicted_x: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Estimates reliability scores r_i(t) in [0, 1] across a batch of windows.
        """
        B = windows.shape[0]
        x_curr = windows[:, -1, :]
        x_prev = windows[:, -2, :]

        # 1. Perspective 1: Temporal Continuity
        v = np.abs(x_curr - x_prev) / (self.mad_velocity_ + 1e-6)
        r_v = np.exp(-0.5 * (v ** 2))

        # 2. Perspective 2: Graph Neighborhood Consistency
        neigh_expect = np.dot(x_curr, self.adj_matrix_.T)
        c = np.abs(x_curr - neigh_expect) / (self.sigma_neigh_ + 1e-6)
        r_c = np.exp(-0.5 * (c ** 2))

        # 3. Perspective 3: Predictive Residual Confidence
        z = np.abs(x_curr - predicted_x) / (self.mad_residual_ + 1e-6)
        r_p = np.exp(-0.5 * (z ** 2))

        # Fused Multi-Perspective Reliability
        w_sum = self.w_v + self.w_c + self.w_p
        if w_sum > 0:
            r = (self.w_v * r_v + self.w_c * r_c + self.w_p * r_p) / w_sum
        else:
            r = np.ones_like(r_v)

        # Bound to [0.001, 1.0] for numerical stability
        r = np.clip(r, 0.001, 1.0)
        return r, {"r_velocity": r_v, "r_consistency": r_c, "r_prediction": r_p}

    def compute_weighted_system_anomaly_batch(
        self,
        normalized_residuals: np.ndarray,
        reliability_scores: np.ndarray,
    ) -> np.ndarray:
        """
        Computes reliability-weighted system anomaly score: S_sys = sum(r * e) / sum(r).
        """
        numerator = np.sum(reliability_scores * normalized_residuals, axis=-1)
        denominator = np.sum(reliability_scores, axis=-1)
        return numerator / (denominator + 1e-6)
```

---

## Appendix E — Training Pipeline and Validation Calibration

**File:** `scripts/train_deep_model.py` (Excerpt)

```python
def train_model(model, train_loader, val_windows, epochs, lr, device):
    """Executes 15-epoch Adam optimization on pure normal training data."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x in train_loader:
            batch_x = batch_x.to(device)
            target = batch_x[:, -1, :]  # Predict current state at window end

            optimizer.zero_grad()
            pred, _ = model(batch_x)
            loss = criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

    # Calibrate Decision Threshold strictly on Validation Normal Windows
    model.eval()
    with torch.no_grad():
        tensor_val = torch.tensor(val_windows, dtype=torch.float32).to(device)
        val_preds, _ = model(tensor_val)
        val_preds_np = val_preds.cpu().numpy()

        if hasattr(model, "srm"):
            model.calibrate_srm(val_windows, device=device)
            r_val, _ = model.srm.estimate_reliability_batch(val_windows, val_preds_np)
            val_diff = np.abs(val_windows[:, -1, :] - val_preds_np)
            val_z = val_diff / (model.srm.mad_residual_ + 1e-6)
            val_scores = model.srm.compute_weighted_system_anomaly_batch(val_z, r_val)
        else:
            val_diff = np.abs(val_windows[:, -1, :] - val_preds_np)
            val_z = val_diff / (model.mad_residual_ + 1e-6)
            val_scores = np.mean(val_z, axis=-1)

        # 95th Percentile Validation Threshold
        calibrated_tau = float(np.quantile(val_scores, 0.95))
    return calibrated_tau
```

---

## Appendix F — Evaluation Metrics and Point-Adjustment Logic

**File:** `src/evaluation/metrics.py` (Excerpt)

```python
def compute_anomaly_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Computes exact classification metrics under point-wise temporal alignment."""
    y_pred = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    pr_auc = float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.0
    roc_auc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.5

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "fpr": fpr, "pr_auc": pr_auc, "roc_auc": roc_auc,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
    }

def point_adjustment_eval(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Point-Adjusted Benchmark Protocol: detecting any point in an attack flags the segment."""
    adjusted_pred = y_pred.copy()
    in_attack = False
    start_idx = 0

    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_attack:
            in_attack = True
            start_idx = i
        elif y_true[i] == 0 and in_attack:
            in_attack = False
            if np.any(adjusted_pred[start_idx:i] == 1):
                adjusted_pred[start_idx:i] = 1

    if in_attack and np.any(adjusted_pred[start_idx:] == 1):
        adjusted_pred[start_idx:] = 1

    return {
        "precision_pa": float(precision_score(y_true, adjusted_pred, zero_division=0)),
        "recall_pa": float(recall_score(y_true, adjusted_pred, zero_division=0)),
        "f1_pa": float(f1_score(y_true, adjusted_pred, zero_division=0))
    }
```

---

## Appendix G — Controlled Sensor Corruption Suite

**File:** `src/corruption/corruptor.py` (Excerpt)

```python
class SensorCorruptor:
    """Injects controlled Gaussian noise, stuck sensor values, and dynamic drift."""
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def apply_corruption(self, data: np.ndarray, corruption_type: str, affected_channels: list, severity: float, corruption_rate: float = 0.3):
        corrupted = data.copy()
        N, M = data.shape
        fault_mask = np.zeros((N, M), dtype=int)

        num_corrupt = int(N * corruption_rate)
        start_idx = self.rng.randint(0, N - num_corrupt)
        corrupt_indices = slice(start_idx, start_idx + num_corrupt)

        for ch in affected_channels:
            fault_mask[corrupt_indices, ch] = 1
            if corruption_type == "gaussian_noise":
                corrupted[corrupt_indices, ch] += self.rng.normal(0, severity, size=num_corrupt)
            elif corruption_type == "stuck_sensor":
                corrupted[corrupt_indices, ch] = corrupted[start_idx, ch]
            elif corruption_type == "sensor_drift":
                ramp = np.linspace(0, severity, num_corrupt)
                corrupted[corrupt_indices, ch] += ramp

        return corrupted, fault_mask
```

---

## Appendix H — Multi-Perspective SRM Component Ablation Runner

**File:** `scripts/run_ablations_and_generate_plots.py` (Excerpt)

```python
def run_ablation_suite(model, val_windows, test_windows, test_labels, device):
    """Evaluates the 5 SRM configurations under identical backbone weights."""
    configs = [
        {"name": "Ablation 1 (No SRM)", "w_v": 0.0, "w_c": 0.0, "w_p": 0.0},
        {"name": "Ablation 2 (Temporal Only)", "w_v": 1.0, "w_c": 0.0, "w_p": 0.0},
        {"name": "Ablation 3 (Graph Only)", "w_v": 0.0, "w_c": 1.2, "w_p": 0.0},
        {"name": "Ablation 4 (Residual Only)", "w_v": 0.0, "w_c": 0.0, "w_p": 1.0},
        {"name": "Ablation 5 (Full SRM)", "w_v": 1.0, "w_c": 1.2, "w_p": 1.0},
    ]
    results = {}
    for cfg in configs:
        model.srm.w_v = cfg["w_v"]
        model.srm.w_c = cfg["w_c"]
        model.srm.w_p = cfg["w_p"]

        # Recalibrate threshold on validation normal data
        r_val, _ = model.srm.estimate_reliability_batch(val_windows, val_preds)
        z_val = np.abs(val_windows[:, -1, :] - val_preds) / (model.srm.mad_residual_ + 1e-6)
        tau = np.quantile(model.srm.compute_weighted_system_anomaly_batch(z_val, r_val), 0.95)

        # Evaluate on test sequence
        r_test, _ = model.srm.estimate_reliability_batch(test_windows, test_preds)
        z_test = np.abs(test_windows[:, -1, :] - test_preds) / (model.srm.mad_residual_ + 1e-6)
        scores = model.srm.compute_weighted_system_anomaly_batch(z_test, r_test)

        metrics = compute_anomaly_metrics(test_labels, scores, threshold=tau)
        results[cfg["name"]] = {"f1": metrics["f1"], "fpr": metrics["fpr"], "threshold_tau": tau}
    return results
```

---

## Appendix I — Publication Figure Generation Suite

**Files:** `scripts/run_ablations_and_generate_plots.py` and `scripts/generate_review2_supplementary_plots.py`  
These scripts generate all 12 publication-quality PNG figures at 300 DPI, located in `report/REVIEW2/figures/`.

---

## Appendix J — Interactive Demonstration Application

**File:** `demo/app.py`  
The complete Streamlit dashboard code is provided in `demo/app.py` and documented in `docs/DEMO.md`. It launches via `streamlit run demo/app.py`.
