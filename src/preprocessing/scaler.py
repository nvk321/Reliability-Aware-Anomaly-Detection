"""
Robust Standard Scaler with Zero-Variance Protection for Multivariate Time-Series.

Ensures no division-by-zero or NaN values when constant sensors/actuators
(e.g., backup pumps or inactive valves) are encountered in normal operation.
Fits strictly on training data to prevent data leakage.
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
        """
        Compute mean and standard deviation from normal training data.
        
        Args:
            X: Array of shape (N, M) or (N, L, M)
        """
        if X.ndim == 3:
            # Flatten time dimension for feature-wise statistics: (N * L, M)
            flat_X = X.reshape(-1, X.shape[-1])
        elif X.ndim == 2:
            flat_X = X
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape {X.shape}")

        self.n_features_ = flat_X.shape[1]
        self.mean_ = np.nanmean(flat_X, axis=0)
        std = np.nanstd(flat_X, axis=0)

        # Detect constant features with zero or near-zero variance
        self.constant_mask_ = std < self.eps
        # Replace zero scale with 1.0 to preserve constant values (x - mean -> 0)
        std[self.constant_mask_] = 1.0
        self.std_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize the input array using fitted mean and scale.
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler has not been fitted yet. Call fit() first.")
        
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Undo the standardization."""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler has not been fitted yet. Call fit() first.")
        
        return (X * self.std_) + self.mean_

    def to_dict(self) -> Dict[str, Any]:
        """Serialize scaler parameters to dictionary."""
        return {
            "eps": self.eps,
            "n_features": self.n_features_,
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "std": self.std_.tolist() if self.std_ is not None else None,
            "constant_mask": self.constant_mask_.tolist() if self.constant_mask_ is not None else None,
        }

    def save_json(self, filepath: str) -> None:
        """Save scaler parameters to JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobustStandardScaler":
        """Reconstruct scaler from dictionary."""
        scaler = cls(eps=data.get("eps", 1e-6))
        scaler.n_features_ = data.get("n_features")
        if data.get("mean") is not None:
            scaler.mean_ = np.array(data["mean"], dtype=np.float32)
        if data.get("std") is not None:
            scaler.std_ = np.array(data["std"], dtype=np.float32)
        if data.get("constant_mask") is not None:
            scaler.constant_mask_ = np.array(data["constant_mask"], dtype=bool)
        return scaler

    @classmethod
    def load_json(cls, filepath: str) -> "RobustStandardScaler":
        """Load scaler from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
