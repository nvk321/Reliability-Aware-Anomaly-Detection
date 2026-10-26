"""
Proposed Architecture: R-STNet (Dual-Perspective Reliability-Aware Spatio-Temporal Network).

Integrates:
1. Spatio-Temporal Graph Attention Backbone (TCN + Sensor Graph Attention).
2. Online Sensor Reliability Module (SRM) (Temporal Continuity + Graph Neighborhood + Residual Z-Score).
3. Reliability-Weighted System Anomaly Scoring:
       S_sys(t) = sum_i( r_i(t) * z_i(t) ) / ( sum_i( r_i(t) ) + eps )
4. Sensor Fault Scoring:
       S_fault(t) = max_i [ (1 - r_i(t)) * z_i(t) ]

Decouples isolated instrument failures from genuine cyber-physical attacks.
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.baselines.rstnet_no_srm import SensorGraphAttention
from src.reliability.sensor_reliability import SensorReliabilityModule


class RSTNet(nn.Module):
    """
    R-STNet: Spatio-Temporal Graph Network with integrated Sensor Reliability Module.
    """

    def __init__(
        self,
        in_channels: int = 51,
        hidden_dim: int = 64,
        node_dim: int = 32,
        top_k: int = 15,
        temporal_kernel: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # 1. Temporal Feature Extractor: 1D Convolutions per sensor
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=hidden_dim,
            kernel_size=temporal_kernel,
            padding="same",
        )
        self.conv2 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=temporal_kernel,
            dilation=2,
            padding="same",
        )
        self.temporal_fc = nn.Linear(hidden_dim, node_dim)
        self.dropout = nn.Dropout(dropout)

        # 2. Graph Attention Dependency Module
        self.graph_attention = SensorGraphAttention(
            in_channels=in_channels, node_dim=node_dim, top_k=top_k
        )

        # 3. Output Forecasting / Reconstruction Head
        self.out_head = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # 4. Sensor Reliability Module (calibrated post-training on validation split)
        self.srm = SensorReliabilityModule(num_sensors=in_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input window of shape (batch_size, seq_len, in_channels)

        Returns:
            x_pred: One-step prediction / reconstruction: (batch_size, in_channels)
            adj_weights: Learned adjacency matrix: (in_channels, in_channels)
        """
        B, L, M = x.shape

        # Reshape to (B * M, 1, L) so each sensor channel gets its own temporal conv embedding
        x_reshaped = x.permute(0, 2, 1).reshape(B * M, 1, L)
        h_conv = F.relu(self.conv1(x_reshaped))
        h_conv = F.relu(self.conv2(h_conv)) + h_conv  # Residual skip
        h_temp = self.dropout(h_conv[:, :, -1])       # (B * M, hidden_dim)
        h_temp_nodes = self.temporal_fc(h_temp).reshape(B, M, -1)  # (B, M, node_dim)

        # Spatial Graph Attention
        h_spatial_nodes, adj_weights = self.graph_attention(h_temp_nodes)

        # Joint node features: (B, M, 2 * node_dim)
        h_joint = torch.cat([h_temp_nodes, h_spatial_nodes], dim=-1)

        # Predicted channel values: (B, M)
        x_pred = self.out_head(h_joint).squeeze(-1)

        return x_pred, adj_weights

    def calibrate_srm(
        self,
        val_windows: np.ndarray,
        device: torch.device = torch.device("cpu"),
    ) -> "RSTNet":
        """
        Calibrates the Sensor Reliability Module on clean normal validation windows.
        val_windows: (N, L, M) numpy array.
        """
        self.eval()
        with torch.no_grad():
            tensor_val = torch.tensor(val_windows, dtype=torch.float32, device=device)
            val_preds_list = []
            batch_size = 128
            for i in range(0, len(val_windows), batch_size):
                batch = tensor_val[i:i + batch_size]
                preds, adj = self.forward(batch)
                val_preds_list.append(preds.cpu().numpy())
            val_preds = np.concatenate(val_preds_list, axis=0)
            adj_np = adj.cpu().numpy()

        val_current = val_windows[:, -1, :]
        self.srm.fit_calibration(
            val_data=val_current,
            val_predictions=val_preds,
            adjacency_matrix=adj_np,
        )
        return self

    def evaluate_step(
        self,
        window_np: np.ndarray,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, Any]:
        """
        Evaluates a single temporal window at inference time.
        window_np: (L, M) array.

        Returns diagnostic dictionary containing:
            - system_score S_sys(t)
            - reliability vector r(t)
            - most suspicious sensor
            - per-sensor residuals
        """
        self.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(window_np[np.newaxis, ...], dtype=torch.float32, device=device)
            x_pred, adj_weights = self.forward(x_tensor)
            pred_np = x_pred.squeeze(0).cpu().numpy()

        current_x = window_np[-1]
        raw_error = np.abs(current_x - pred_np)

        # Normalized residual z_i
        if self.srm.mad_residual_ is not None:
            z = raw_error / (self.srm.mad_residual_ + 1e-6)
        else:
            z = raw_error

        # Estimate composite reliability vector
        r, components = self.srm.estimate_reliability(window=window_np, predicted_x=pred_np)

        # Reliability-weighted system anomaly score
        s_sys = self.srm.compute_weighted_system_anomaly(residuals=z, reliability=r)

        # Sensor fault score
        fault_weights = (1.0 - r) * z
        s_fault = float(np.max(fault_weights))
        most_faulty_ch = int(np.argmax(fault_weights))

        return {
            "system_score": s_sys,
            "sensor_fault_score": s_fault,
            "most_suspicious_sensor": most_faulty_ch,
            "reliability_vector": r,
            "normalized_residuals": z,
            "predicted_x": pred_np,
            "components": components,
        }
