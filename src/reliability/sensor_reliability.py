"""
Sensor Reliability Module (SRM) for Multivariate Sensor Systems.

Estimates a continuous trustworthiness score r_i(t) in [0, 1] for each sensor
by triangulating three complementary, physically motivated signals:
1. Temporal Continuity (Rate-of-change bounds and stuck-sensor variance check).
2. Cross-Sensor Neighborhood Consistency (Agreement with physically linked sensors).
3. Standardized Residual Confidence (Tail probability under normal distribution).

Includes systematic calibration on validation data to prevent arbitrary constants.
"""

from typing import Dict, Any, Optional, Tuple
import json
import numpy as np


class SensorReliabilityModule:
    """
    Online per-sensor reliability estimator with calibrated fusion.
    """

    def __init__(
        self,
        num_sensors: int,
        stuck_window: int = 15,
        stuck_var_threshold: float = 1e-4,
        max_rate_quantile: float = 0.999,
    ):
        """
        Args:
            num_sensors: Number of multivariate channels M.
            stuck_window: Lookback steps to detect dead-band / frozen sensor.
            stuck_var_threshold: Variance threshold below which sensor is flagged stuck.
            max_rate_quantile: Quantile on validation set for maximum normal rate of change.
        """
        self.num_sensors = num_sensors
        self.stuck_window = stuck_window
        self.stuck_var_threshold = stuck_var_threshold
        self.max_rate_quantile = max_rate_quantile

        # Calibration parameters (learned from validation split)
        self.max_delta_: Optional[np.ndarray] = None
        self.median_residual_: Optional[np.ndarray] = None
        self.mad_residual_: Optional[np.ndarray] = None
        self.adjacency_: Optional[np.ndarray] = None
        self.neighbor_weights_: Optional[np.ndarray] = None

        # Fusion weights: w_v (volatility), w_c (cross-sensor), w_p (residual)
        self.w_v = 1.0
        self.w_c = 1.2
        self.w_p = 1.0
        self.bias = 0.0

    def fit_calibration(
        self,
        val_data: np.ndarray,
        val_predictions: Optional[np.ndarray] = None,
        adjacency_matrix: Optional[np.ndarray] = None,
    ) -> "SensorReliabilityModule":
        """
        Calibrates normal operating baselines on clean validation split.

        Args:
            val_data: Array of shape (N, M) from normal validation set.
            val_predictions: Array of shape (N, M) from model predictions on val_data.
            adjacency_matrix: Optional graph adjacency (M, M) between sensors.
        """
        N, M = val_data.shape
        assert M == self.num_sensors, f"Expected {self.num_sensors} sensors, got {M}"

        # 1. Normal temporal rate of change: |x_t - x_{t-1}|
        deltas = np.abs(np.diff(val_data, axis=0))
        self.max_delta_ = np.quantile(deltas, self.max_rate_quantile, axis=0)
        self.max_delta_[self.max_delta_ < 1e-4] = 1.0

        # Identify continuous/dynamic sensors vs naturally static discrete actuators
        self.continuous_mask_ = (np.var(val_data, axis=0) > 1e-3)

        # 2. Estimate or record cross-sensor adjacency
        if adjacency_matrix is not None:
            adj = np.copy(adjacency_matrix)
        else:
            # Empirical correlation-based adjacency matrix if graph model not passed
            with np.errstate(divide="ignore", invalid="ignore"):
                corr = np.nan_to_num(np.abs(np.corrcoef(val_data, rowvar=False)), nan=0.0)
            np.fill_diagonal(corr, 0.0)
            adj = corr

        # Normalize neighbor weights for reconstruction: sum_j W_ji = 1
        row_sums = np.sum(adj, axis=0, keepdims=True)
        row_sums[row_sums < 1e-6] = 1.0
        self.adjacency_ = adj
        self.neighbor_weights_ = adj / row_sums

        # 3. Normal residual statistics (Median Absolute Deviation)
        if val_predictions is not None:
            res = np.abs(val_data - val_predictions)
        else:
            # Neighborhood residual baseline
            val_neighbor_pred = val_data @ self.neighbor_weights_
            res = np.abs(val_data - val_neighbor_pred)

        self.median_residual_ = np.median(res, axis=0)
        self.mad_residual_ = np.median(np.abs(res - self.median_residual_), axis=0)
        self.mad_residual_[self.mad_residual_ < 1e-4] = 1.0

        return self

    def compute_temporal_continuity(self, window: np.ndarray) -> np.ndarray:
        """
        Computes temporal continuity score v_i in [0, 1] for current step.
        window: (L, M) array of recent observations ending at time t.
        """
        L, M = window.shape
        if self.max_delta_ is None:
            return np.ones(M, dtype=np.float32)

        # Rate of change at latest step
        if L >= 2:
            step_delta = np.abs(window[-1] - window[-2])
            delta_ratio = step_delta / self.max_delta_
            # Exponential decay for abrupt spike violations
            rate_score = np.exp(-0.5 * np.maximum(0.0, delta_ratio - 1.0) ** 2)
        else:
            rate_score = np.ones(M, dtype=np.float32)

        # Stuck sensor check: variance over lookback window (only for naturally dynamic sensors)
        lookback = min(L, self.stuck_window)
        recent_var = np.var(window[-lookback:], axis=0)
        continuous = getattr(self, "continuous_mask_", np.ones(M, dtype=bool))
        is_stuck = (recent_var < self.stuck_var_threshold) & continuous
        stuck_penalty = np.where(is_stuck, 0.1, 1.0)

        return np.clip(rate_score * stuck_penalty, 0.01, 0.99).astype(np.float32)

    def compute_neighborhood_consistency(self, current_x: np.ndarray) -> np.ndarray:
        """
        Computes agreement between sensor i and its topological neighbors: c_i in [0, 1].
        current_x: (M,) vector of observed sensor readings at time t.
        """
        if self.neighbor_weights_ is None or self.mad_residual_ is None:
            return np.ones(self.num_sensors, dtype=np.float32)

        # Expected value from correlated neighbors: x_hat_i = sum_j W_ji * x_j
        x_hat_neighbors = current_x @ self.neighbor_weights_
        dev = np.abs(current_x - x_hat_neighbors)
        scaled_dev = dev / (2.5 * self.mad_residual_)
        consistency = np.exp(-scaled_dev)
        return np.clip(consistency, 0.01, 0.99).astype(np.float32)

    def compute_residual_confidence(
        self,
        current_x: np.ndarray,
        predicted_x: np.ndarray,
    ) -> np.ndarray:
        """
        Computes residual tail confidence p_i in [0, 1].
        current_x: (M,) observed values.
        predicted_x: (M,) model predictions.
        """
        if self.median_residual_ is None or self.mad_residual_ is None:
            return np.ones(self.num_sensors, dtype=np.float32)

        error = np.abs(current_x - predicted_x)
        z = (error - self.median_residual_) / (self.mad_residual_ + 1e-6)
        z_clamped = np.maximum(0.0, z)
        # Exponential tail probability
        tail_prob = np.exp(-0.5 * (z_clamped / 2.0) ** 2)
        return np.clip(tail_prob, 0.01, 0.99).astype(np.float32)

    def estimate_reliability(
        self,
        window: np.ndarray,
        predicted_x: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Estimates composite reliability vector r(t) in [0, 1]^M for latest step.

        Args:
            window: (L, M) array of recent sensor readings.
            predicted_x: (M,) model prediction for latest step, if available.

        Returns:
            r: Composite reliability vector (M,) where 1.0 = fully trustworthy, 0.0 = faulty.
            components: Dictionary containing individual component scores (v, c, p).
        """
        current_x = window[-1]

        v = self.compute_temporal_continuity(window)
        c = self.compute_neighborhood_consistency(current_x)

        if predicted_x is not None:
            p = self.compute_residual_confidence(current_x, predicted_x)
        else:
            p = c.copy()

        # Calibrated logit fusion
        def logit(val):
            eps = 1e-5
            clipped = np.clip(val, eps, 1.0 - eps)
            return np.log(clipped / (1.0 - clipped))

        fusion_logit = (
            self.w_v * logit(v) +
            self.w_c * logit(c) +
            self.w_p * logit(p) +
            self.bias
        )
        # Standard sigmoid
        r = 1.0 / (1.0 + np.exp(-fusion_logit))
        r = np.clip(r, 0.0, 1.0).astype(np.float32)

        components = {
            "temporal_continuity": v,
            "neighborhood_consistency": c,
            "residual_confidence": p,
        }
        return r, components

    def estimate_reliability_batch(
        self,
        windows: np.ndarray,
        predicted_x: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Vectorized batch estimation of composite reliability: (B, M).
        windows: (B, L, M)
        predicted_x: (B, M) or None
        """
        B, L, M = windows.shape
        current_x = windows[:, -1, :]

        # 1. Temporal Continuity Batch
        if self.max_delta_ is not None and L >= 2:
            step_delta = np.abs(windows[:, -1, :] - windows[:, -2, :])
            delta_ratio = step_delta / self.max_delta_
            rate_score = np.exp(-0.5 * np.maximum(0.0, delta_ratio - 1.0) ** 2)
            lookback = min(L, self.stuck_window)
            recent_var = np.var(windows[:, -lookback:, :], axis=1)
            continuous = getattr(self, "continuous_mask_", np.ones(M, dtype=bool))
            is_stuck = (recent_var < self.stuck_var_threshold) & continuous[np.newaxis, :]
            stuck_penalty = np.where(is_stuck, 0.1, 1.0)
            v = np.clip(rate_score * stuck_penalty, 0.01, 0.99).astype(np.float32)
        else:
            v = np.ones((B, M), dtype=np.float32)

        # 2. Neighborhood Consistency Batch
        if self.neighbor_weights_ is not None and self.mad_residual_ is not None:
            x_hat_neighbors = current_x @ self.neighbor_weights_
            dev = np.abs(current_x - x_hat_neighbors)
            scaled_dev = dev / (2.5 * self.mad_residual_)
            c = np.clip(np.exp(-scaled_dev), 0.01, 0.99).astype(np.float32)
        else:
            c = np.ones((B, M), dtype=np.float32)

        # 3. Residual Confidence Batch
        if predicted_x is not None and self.median_residual_ is not None and self.mad_residual_ is not None:
            error = np.abs(current_x - predicted_x)
            z = (error - self.median_residual_) / (self.mad_residual_ + 1e-6)
            z_clamped = np.maximum(0.0, z)
            p = np.clip(np.exp(-0.5 * (z_clamped / 2.0) ** 2), 0.01, 0.99).astype(np.float32)
        else:
            p = c.copy()

        # Logit fusion
        eps = 1e-5
        def logit_np(val):
            clipped = np.clip(val, eps, 1.0 - eps)
            return np.log(clipped / (1.0 - clipped))

        fusion_logit = (
            self.w_v * logit_np(v) +
            self.w_c * logit_np(c) +
            self.w_p * logit_np(p) +
            self.bias
        )
        r = 1.0 / (1.0 + np.exp(-fusion_logit))
        r = np.clip(r, 0.0, 1.0).astype(np.float32)

        components = {
            "temporal_continuity": v,
            "neighborhood_consistency": c,
            "residual_confidence": p,
        }
        return r, components

    def compute_weighted_system_anomaly(
        self,
        residuals: np.ndarray,
        reliability: np.ndarray,
        eps: float = 1e-5,
    ) -> float:
        """
        Computes reliability-weighted system anomaly score:
            S_sys(t) = sum_i( r_i(t) * z_i(t) ) / ( sum_i( r_i(t) ) + eps )
        """
        return float(np.sum(reliability * residuals) / (np.sum(reliability) + eps))

    def compute_weighted_system_anomaly_batch(
        self,
        residuals: np.ndarray,
        reliability: np.ndarray,
        eps: float = 1e-5,
    ) -> np.ndarray:
        """
        Batch reliability-weighted system anomaly score:
        residuals: (B, M), reliability: (B, M) -> (B,)
        """
        num = np.sum(reliability * residuals, axis=-1)
        den = np.sum(reliability, axis=-1) + eps
        return (num / den).astype(np.float64)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize calibration state to dictionary."""
        return {
            "num_sensors": self.num_sensors,
            "stuck_window": self.stuck_window,
            "stuck_var_threshold": self.stuck_var_threshold,
            "w_v": self.w_v,
            "w_c": self.w_c,
            "w_p": self.w_p,
            "bias": self.bias,
            "max_delta": self.max_delta_.tolist() if self.max_delta_ is not None else None,
            "median_residual": self.median_residual_.tolist() if self.median_residual_ is not None else None,
            "mad_residual": self.mad_residual_.tolist() if self.mad_residual_ is not None else None,
            "continuous_mask": getattr(self, "continuous_mask_", None).tolist() if getattr(self, "continuous_mask_", None) is not None else None,
            "adjacency": self.adjacency_.tolist() if self.adjacency_ is not None else None,
            "neighbor_weights": self.neighbor_weights_.tolist() if self.neighbor_weights_ is not None else None,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SensorReliabilityModule":
        """Deserialize SensorReliabilityModule instance from dictionary."""
        srm = cls(
            num_sensors=d["num_sensors"],
            stuck_window=d.get("stuck_window", 15),
            stuck_var_threshold=d.get("stuck_var_threshold", 1e-4),
        )
        srm.w_v = d.get("w_v", 1.0)
        srm.w_c = d.get("w_c", 1.2)
        srm.w_p = d.get("w_p", 1.0)
        srm.bias = d.get("bias", 0.0)
        if d.get("max_delta") is not None:
            srm.max_delta_ = np.array(d["max_delta"], dtype=np.float32)
        if d.get("median_residual") is not None:
            srm.median_residual_ = np.array(d["median_residual"], dtype=np.float32)
        if d.get("mad_residual") is not None:
            srm.mad_residual_ = np.array(d["mad_residual"], dtype=np.float32)
        if d.get("continuous_mask") is not None:
            srm.continuous_mask_ = np.array(d["continuous_mask"], dtype=bool)
        if d.get("adjacency") is not None:
            srm.adjacency_ = np.array(d["adjacency"], dtype=np.float32)
        if d.get("neighbor_weights") is not None:
            srm.neighbor_weights_ = np.array(d["neighbor_weights"], dtype=np.float32)
        return srm
