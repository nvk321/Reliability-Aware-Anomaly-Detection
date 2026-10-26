"""
Baseline C: Spatio-Temporal Graph Backbone WITHOUT Sensor Reliability Module (SRM).

Identical neural backbone to proposed R-STNet (Temporal Convolutions + Graph Attention),
but computes UNWEIGHTED anomaly scores across all channels.
Serves as the primary ablation model to isolate the exact contribution of the SRM.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SensorGraphAttention(nn.Module):
    """
    Learns directed adjacency weights and aggregates feature messages
    across correlated sensor nodes.
    """

    def __init__(self, in_channels: int, node_dim: int = 32, top_k: int = 15):
        super().__init__()
        self.in_channels = in_channels
        self.top_k = min(top_k, in_channels - 1)

        # Learnable node embeddings for graph structure learning
        self.node_embed1 = nn.Parameter(torch.randn(in_channels, node_dim))
        self.node_embed2 = nn.Parameter(torch.randn(in_channels, node_dim))
        nn.init.xavier_uniform_(self.node_embed1)
        nn.init.xavier_uniform_(self.node_embed2)

        self.lin_spatial = nn.Linear(node_dim, node_dim)

    def forward(self, h_temporal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_temporal: (batch_size, in_channels, feat_dim) node feature tensor.

        Returns:
            h_spatial: (batch_size, in_channels, feat_dim) spatially aggregated features.
            adj_weights: (in_channels, in_channels) learned graph adjacency weights.
        """
        # Compute normalized similarity matrix between node embeddings
        raw_adj = F.relu(torch.mm(self.node_embed1, self.node_embed2.t()))
        
        # Zero out self-loops (a sensor does not check redundancy against itself)
        mask = torch.eye(self.in_channels, device=raw_adj.device).bool()
        raw_adj = raw_adj.masked_fill(mask, 0.0)

        # Top-k sparsification to retain strongest physical relationships
        if self.top_k > 0:
            topk_vals, topk_indices = torch.topk(raw_adj, self.top_k, dim=-1)
            sparse_adj = torch.zeros_like(raw_adj)
            sparse_adj.scatter_(-1, topk_indices, topk_vals)
        else:
            sparse_adj = raw_adj

        # Softmax normalization over incoming edges
        adj_weights = F.softmax(sparse_adj + 1e-6, dim=-1)

        # Message passing: h_spatial = adj_weights * h_temporal
        # h_temporal: (B, M, D), adj_weights: (M, M)
        h_spatial = torch.einsum("ij,bjd->bid", adj_weights, h_temporal)
        return h_spatial, adj_weights


class RSTNetNoSRM(nn.Module):
    """
    Spatio-Temporal Graph Network without Sensor Reliability Module.
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input window of shape (batch_size, seq_len, in_channels)

        Returns:
            x_pred: One-step prediction / reconstruction for each sensor at time t: (batch_size, in_channels)
            adj_weights: Learned adjacency matrix (in_channels, in_channels)
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

        # Concatenate temporal and spatial node representations: (B, M, 2 * node_dim)
        h_joint = torch.cat([h_temp_nodes, h_spatial_nodes], dim=-1)

        # Predict target values for each sensor: (B, M)
        x_pred = self.out_head(h_joint).squeeze(-1)

        return x_pred, adj_weights

    def calibrate_mad(
        self,
        val_windows: np.ndarray,
        device: torch.device = torch.device("cpu"),
    ) -> "RSTNetNoSRM":
        """
        Calibrates per-channel median and MAD of prediction errors on normal validation split.
        val_windows: (N, L, M) array
        """
        self.eval()
        with torch.no_grad():
            tensor_val = torch.tensor(val_windows, dtype=torch.float32, device=device)
            val_preds_list = []
            batch_size = 128
            for i in range(0, len(val_windows), batch_size):
                batch = tensor_val[i : i + batch_size]
                preds, _ = self.forward(batch)
                val_preds_list.append(preds.cpu().numpy())
            val_preds = np.concatenate(val_preds_list, axis=0)

        val_current = val_windows[:, -1, :]
        res = np.abs(val_current - val_preds)
        median_res = np.median(res, axis=0)
        mad_res = np.median(np.abs(res - median_res), axis=0)
        mad_res[mad_res < 1e-4] = 1.0
        self.mad_residual_ = mad_res
        return self

    def compute_anomaly_score(
        self,
        x: torch.Tensor,
        mad_res: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes UNWEIGHTED anomaly scores across all channels.

        Returns:
            per_sensor_error: (batch_size, in_channels)
            system_score: (batch_size,) unweighted mean
            adj_weights: (in_channels, in_channels)
        """
        self.eval()
        with torch.no_grad():
            x_pred, adj_weights = self.forward(x)
            current_x = x[:, -1, :]
            diff = torch.abs(current_x - x_pred)
            if mad_res is not None:
                per_sensor_error = diff / (mad_res + 1e-6)
            elif self.mad_residual_ is not None:
                mad_t = torch.tensor(self.mad_residual_, dtype=diff.dtype, device=diff.device)
                per_sensor_error = diff / (mad_t + 1e-6)
            else:
                per_sensor_error = diff

            # UNWEIGHTED uniform mean across channels (Baseline C)
            system_score = torch.mean(per_sensor_error, dim=-1)

        return per_sensor_error, system_score, adj_weights
