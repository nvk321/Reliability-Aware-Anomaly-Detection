"""
Sliding Window Generator for Multivariate Time-Series Anomaly Detection.

Constructs overlapping or non-overlapping temporal windows of shape:
    (batch_size, window_size, num_features)
from contiguous time-series arrays. Fits strictly chronologically to prevent
future leakage.
"""

from typing import Tuple, Optional
import numpy as np


class SlidingWindowGenerator:
    """
    Generates temporal windows W_t = [X_{t - L + 1}, ..., X_t]
    and corresponding labels / forecasting targets.
    """

    def __init__(
        self,
        window_size: int = 60,
        stride: int = 1,
        label_mode: str = "point_end",  # "point_end", "window_any", or "next_step"
    ):
        """
        Args:
            window_size: Sequence length L (number of time steps per window).
            stride: Stride between successive windows.
            label_mode:
                - "point_end": Label corresponding to the last time-step of the window X_t.
                - "window_any": 1 if any time-step in the window is anomalous, else 0.
        """
        self.window_size = window_size
        self.stride = stride
        self.label_mode = label_mode

    def create_windows(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Converts 2D array (N, M) into 3D window tensor (num_windows, L, M).

        Args:
            data: Continuous sensor array of shape (N, M).
            labels: Binary labels of shape (N,) or None.

        Returns:
            X_windows: Array of shape (num_windows, L, M)
            y_windows: Array of shape (num_windows,) if labels provided, else None.
        """
        N, M = data.shape
        if N < self.window_size:
            raise ValueError(
                f"Data length {N} is shorter than window_size {self.window_size}"
            )

        num_windows = (N - self.window_size) // self.stride + 1
        
        # Use strided sliding window view for high memory and compute efficiency
        sub_shape = (num_windows, self.window_size, M)
        sub_strides = (data.strides[0] * self.stride, data.strides[0], data.strides[1])
        X_windows = np.lib.stride_tricks.as_strided(
            data, shape=sub_shape, strides=sub_strides
        ).copy()

        y_windows = None
        if labels is not None:
            if len(labels) != N:
                raise ValueError(
                    f"Label length {len(labels)} does not match data length {N}"
                )
            y_windows = np.zeros(num_windows, dtype=np.int32)
            for i in range(num_windows):
                start = i * self.stride
                end = start + self.window_size
                if self.label_mode == "point_end":
                    # Label is the state at the current observation step (end of window)
                    y_windows[i] = int(labels[end - 1])
                elif self.label_mode == "window_any":
                    # Anomaly present if any step is attack
                    y_windows[i] = int(np.any(labels[start:end] == 1))
                else:
                    raise ValueError(f"Unknown label_mode: {self.label_mode}")

        return X_windows, y_windows
