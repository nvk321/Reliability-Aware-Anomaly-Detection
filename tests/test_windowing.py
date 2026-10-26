"""
Unit tests for SlidingWindowGenerator.
"""

import numpy as np
import pytest

from src.preprocessing.windowing import SlidingWindowGenerator


def test_windowing_shape():
    """Test standard sliding window shape generation."""
    N, M = 100, 10
    data = np.arange(N * M, dtype=np.float32).reshape(N, M)
    labels = np.zeros(N, dtype=np.int32)
    labels[50:60] = 1  # Attack interval

    win_gen = SlidingWindowGenerator(window_size=20, stride=2, label_mode="point_end")
    X_win, y_win = win_gen.create_windows(data, labels)

    expected_windows = (100 - 20) // 2 + 1
    assert X_win.shape == (expected_windows, 20, 10)
    assert y_win.shape == (expected_windows,)
    # Verify the last element in window matches point_end label
    assert y_win[0] == labels[19]


def test_windowing_any_mode():
    """Test window_any labeling mode."""
    N, M = 50, 4
    data = np.zeros((N, M), dtype=np.float32)
    labels = np.zeros(N, dtype=np.int32)
    labels[15] = 1  # Single anomalous point

    win_gen = SlidingWindowGenerator(window_size=10, stride=1, label_mode="window_any")
    X_win, y_win = win_gen.create_windows(data, labels)

    # Window containing index 15 must be labeled 1
    # Index 6 covers [6..15], so end-1 is 15 -> should be 1
    assert y_win[6] == 1
    assert y_win[0] == 0  # [0..9] contains no anomaly
