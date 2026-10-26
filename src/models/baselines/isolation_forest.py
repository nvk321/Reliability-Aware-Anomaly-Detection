"""
Baseline A: Isolation Forest Anomaly Detector.

Standard classical baseline for multivariate anomaly detection.
Fits strictly on normal training data and scores test instances based on path length.
"""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationForestBaseline:
    """
    Isolation Forest anomaly detection baseline.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.05,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X_train: np.ndarray) -> "IsolationForestBaseline":
        """
        Fits on clean normal training data.
        X_train: (N, M) or (N, L, M).
        """
        if X_train.ndim == 3:
            # Flatten window dimension: (N, L * M)
            N, L, M = X_train.shape
            X_flat = X_train.reshape(N, L * M)
        else:
            X_flat = X_train

        self.model.fit(X_flat)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Returns continuous anomaly score where higher values mean MORE anomalous.
        In sklearn, score_samples returns opposite of anomaly score (lower is more anomalous).
        We invert it so that higher = more anomalous.
        """
        if X.ndim == 3:
            N, L, M = X.shape
            X_flat = X.reshape(N, L * M)
        else:
            X_flat = X

        raw_scores = self.model.score_samples(X_flat)
        # Invert so higher score = more anomalous
        anomaly_scores = -raw_scores
        # Normalize to [0, 1] for comparability
        min_s, max_s = np.min(anomaly_scores), np.max(anomaly_scores)
        if max_s > min_s:
            norm_scores = (anomaly_scores - min_s) / (max_s - min_s)
        else:
            norm_scores = np.zeros_like(anomaly_scores)
        return norm_scores

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Returns binary predictions (1 = anomaly, 0 = normal).
        """
        scores = self.score_samples(X)
        if threshold is None:
            threshold = float(np.quantile(scores, 1.0 - self.contamination))
        return (scores >= threshold).astype(np.int32)
