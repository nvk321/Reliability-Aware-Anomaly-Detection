"""
Four-Way Operational State Discrimination Engine for Reliability-Aware Anomaly Detection.

Classifies each time-step into four mutually exclusive operational states:
- CASE A: NORMAL (S_sys < tau_sys AND min_i r_i >= theta_rel)
- CASE B: SENSOR FAULT (S_sys < tau_sys AND min_i r_i < theta_rel)
- CASE C: SYSTEM ANOMALY (S_sys >= tau_sys AND mean_i r_i >= theta_rel)
- CASE D: SYSTEM ANOMALY + SENSOR DEGRADATION (S_sys >= tau_sys AND mean_i r_i < theta_rel)

Calibrates thresholds (tau_sys, theta_rel) systematically on the validation split.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np


class OperationalState:
    NORMAL = "NORMAL"                                      # Case A
    SENSOR_FAULT = "SENSOR_FAULT"                          # Case B
    SYSTEM_ANOMALY = "SYSTEM_ANOMALY"                      # Case C
    SYSTEM_ANOMALY_DEGRADED = "SYSTEM_ANOMALY_DEGRADED"    # Case D


class StateDiscriminator:
    """
    Decouples sensor instrument faults from cyber-physical system anomalies.
    """

    def __init__(
        self,
        system_quantile: float = 0.99,
        reliability_threshold: float = 0.60,
    ):
        """
        Args:
            system_quantile: Validation quantile for setting system anomaly threshold tau_sys.
            reliability_threshold: Cutoff theta_rel below which a sensor is deemed untrusted.
        """
        self.system_quantile = system_quantile
        self.reliability_threshold = reliability_threshold
        self.tau_sys_: Optional[float] = None
        self.theta_rel_: float = reliability_threshold

    def fit_thresholds(
        self,
        val_system_scores: np.ndarray,
        val_reliability_scores: Optional[np.ndarray] = None,
    ) -> "StateDiscriminator":
        """
        Calibrates tau_sys and theta_rel on clean normal validation data.

        Args:
            val_system_scores: 1D array of system anomaly scores S_sys on normal val split.
            val_reliability_scores: 2D array (N, M) of sensor reliability scores on normal val split.
        """
        # Set system anomaly threshold as the (1 - quantile) extreme of normal behavior
        self.tau_sys_ = float(np.quantile(val_system_scores, self.system_quantile))

        # If validation reliability is provided, calibrate theta_rel to normal lower tail
        if val_reliability_scores is not None:
            min_normal_rel = np.min(val_reliability_scores, axis=1)
            # Threshold set below 99% of normal sensor reliability
            self.theta_rel_ = float(max(0.40, np.quantile(min_normal_rel, 0.01)))
        else:
            self.theta_rel_ = self.reliability_threshold

        return self

    def classify_state(
        self,
        system_score: float,
        reliability_vector: np.ndarray,
        sensor_residuals: Optional[np.ndarray] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Classifies operational state at a single time step.

        Args:
            system_score: S_sys(t) scalar.
            reliability_vector: (M,) per-sensor reliability r_i(t) in [0, 1].
            sensor_residuals: Optional (M,) per-sensor residual errors z_i(t).

        Returns:
            state: One of OperationalState constants (NORMAL, SENSOR_FAULT, SYSTEM_ANOMALY, SYSTEM_ANOMALY_DEGRADED).
            diagnostics: Dictionary containing most suspicious sensor, fault score, etc.
        """
        if self.tau_sys_ is None:
            raise RuntimeError("Discriminator must be fitted before classification.")

        min_rel = float(np.min(reliability_vector))
        mean_rel = float(np.mean(reliability_vector))
        is_system_anomaly = system_score >= self.tau_sys_

        # Pinpoint the most degraded sensor
        most_faulty_idx = int(np.argmin(reliability_vector))
        fault_score = float(1.0 - min_rel)

        if not is_system_anomaly and min_rel >= self.theta_rel_:
            state = OperationalState.NORMAL
        elif not is_system_anomaly and min_rel < self.theta_rel_:
            state = OperationalState.SENSOR_FAULT
        elif is_system_anomaly and mean_rel >= self.theta_rel_:
            state = OperationalState.SYSTEM_ANOMALY
        else:
            state = OperationalState.SYSTEM_ANOMALY_DEGRADED

        diagnostics = {
            "state": state,
            "system_score": float(system_score),
            "tau_sys": float(self.tau_sys_),
            "min_reliability": min_rel,
            "mean_reliability": mean_rel,
            "theta_rel": float(self.theta_rel_),
            "most_suspicious_sensor_idx": most_faulty_idx,
            "fault_score": fault_score,
            "is_false_alarm_suppressed": (state == OperationalState.SENSOR_FAULT),
        }

        return state, diagnostics

    def batch_classify(
        self,
        system_scores: np.ndarray,
        reliability_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized classification across an entire evaluation sequence.

        Args:
            system_scores: (N,) array of S_sys(t).
            reliability_matrix: (N, M) matrix of r_i(t).

        Returns:
            system_anomalies: Binary array (N,), 1 = system attack/anomaly (Cases C and D).
            sensor_faults: Binary array (N,), 1 = sensor degradation (Cases B and D).
            most_suspicious_sensors: Array of shape (N,) indicating sensor index with lowest r_i.
        """
        if self.tau_sys_ is None:
            raise RuntimeError("Discriminator must be fitted before classification.")

        min_rels = np.min(reliability_matrix, axis=1)
        mean_rels = np.mean(reliability_matrix, axis=1)
        most_suspicious = np.argmin(reliability_matrix, axis=1)

        is_sys = system_scores >= self.tau_sys_
        is_sensor_untrusted = min_rels < self.theta_rel_

        # System anomaly flags: Case C or Case D
        system_anomaly_flags = is_sys.astype(np.int32)
        # Sensor fault flags: Case B or Case D
        sensor_fault_flags = is_sensor_untrusted.astype(np.int32)

        return system_anomaly_flags, sensor_fault_flags, most_suspicious
