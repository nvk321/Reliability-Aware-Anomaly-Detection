"""
Controlled Sensor Corruption Framework for Multivariate Time-Series.

Generates reproducible, deterministic synthetic sensor corruptions across
9 operational failure modes, while preserving raw source data and generating
explicit ground-truth masks:
    sensor_fault_mask[t, i] == 1 if sensor i is corrupted at time t.

This enables rigorous quantitative evaluation of:
1. Sensor-level fault identification accuracy (ground-truth available via mask).
2. False-alarm suppression rate under sensor degradation.
3. Decoupling of synthetic sensor faults from real system cyber-attacks.
"""

from typing import Dict, List, Tuple, Any, Optional
import json
import numpy as np


class SensorCorruptor:
    """
    Applies controlled, reproducible sensor corruptions to evaluation data.
    """

    SUPPORTED_TYPES = [
        "missing",
        "gaussian_noise",
        "spikes",
        "stuck_constant",
        "drift",
        "intermittent",
        "single_sensor",
        "multi_sensor",
        "mixed",
    ]

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def apply_corruption(
        self,
        data: np.ndarray,
        corruption_type: str,
        affected_channels: Optional[List[int]] = None,
        severity: float = 0.2,
        corruption_rate: float = 0.1,
        channel_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Applies synthetic corruption to data array.

        Args:
            data: Clean test array of shape (N, M).
            corruption_type: One of SUPPORTED_TYPES.
            affected_channels: List of column indices to corrupt.
                               If None, randomly selected based on corruption type.
            severity: Magnitude multiplier (e.g. noise std, drift slope, spike magnitude).
            corruption_rate: Fraction of time steps affected (e.g. 0.05, 0.10, 0.20, 0.30).
            channel_names: Optional string names of the M sensors.

        Returns:
            corrupted_data: Copy of data with corruption applied.
            sensor_fault_mask: Binary array of shape (N, M), 1 = corrupted.
            metadata: Structured dictionary describing the exact corruption.
        """
        if corruption_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unknown corruption_type '{corruption_type}'. Supported: {self.SUPPORTED_TYPES}"
            )

        N, M = data.shape
        corrupted = data.copy()
        mask = np.zeros((N, M), dtype=np.int32)

        # Determine target channels
        if affected_channels is None:
            if corruption_type == "single_sensor":
                target_channels = [int(self.rng.integers(0, M))]
            elif corruption_type == "multi_sensor":
                num_k = min(3, max(2, int(M * 0.1)))
                target_channels = self.rng.choice(M, size=num_k, replace=False).tolist()
            else:
                num_k = max(1, int(M * 0.15))
                target_channels = self.rng.choice(M, size=num_k, replace=False).tolist()
        else:
            target_channels = affected_channels

        # Per-channel standard deviation for physical scaling
        channel_stds = np.std(data, axis=0)
        channel_stds[channel_stds < 1e-4] = 1.0

        affected_ranges = []

        if corruption_type == "missing":
            # Continuous blocks of missingness (forward-filled or zero-imputed)
            block_len = max(5, int(N * corruption_rate / 10))
            num_blocks = max(1, int((N * corruption_rate) / block_len))
            for ch in target_channels:
                for _ in range(num_blocks):
                    start = int(self.rng.integers(0, max(1, N - block_len)))
                    end = min(N, start + block_len)
                    # Forward-fill previous value or zero out
                    corrupted[start:end, ch] = corrupted[max(0, start - 1), ch] if start > 0 else 0.0
                    mask[start:end, ch] = 1
                    affected_ranges.append({"channel": ch, "start": start, "end": end})

        elif corruption_type == "gaussian_noise":
            # Additive white Gaussian noise
            for ch in target_channels:
                noise_time_mask = self.rng.random(N) < corruption_rate
                noise = self.rng.normal(0, severity * channel_stds[ch], size=N)
                corrupted[noise_time_mask, ch] += noise[noise_time_mask]
                mask[noise_time_mask, ch] = 1

        elif corruption_type == "spikes":
            # Sharp impulse deviations
            num_spikes = max(1, int(N * corruption_rate * 0.1))
            for ch in target_channels:
                spike_indices = self.rng.choice(N, size=num_spikes, replace=False)
                signs = self.rng.choice([-1.0, 1.0], size=num_spikes)
                corrupted[spike_indices, ch] += signs * (severity * 3.0 * channel_stds[ch])
                mask[spike_indices, ch] = 1

        elif corruption_type == "stuck_constant":
            # Sensor value freezes at fixed reading
            duration = max(20, int(N * corruption_rate))
            for ch in target_channels:
                start = int(self.rng.integers(0, max(1, N - duration)))
                end = min(N, start + duration)
                frozen_val = corrupted[start, ch]
                corrupted[start:end, ch] = frozen_val
                mask[start:end, ch] = 1
                affected_ranges.append({"channel": ch, "start": start, "end": end})

        elif corruption_type == "drift":
            # Linear ramp offset delta_x(t) = alpha * (t - start)
            duration = max(30, int(N * corruption_rate))
            for ch in target_channels:
                start = int(self.rng.integers(0, max(1, N - duration)))
                end = min(N, start + duration)
                ramp = np.linspace(0, severity * 2.5 * channel_stds[ch], end - start)
                corrupted[start:end, ch] += ramp
                mask[start:end, ch] = 1
                affected_ranges.append({"channel": ch, "start": start, "end": end})

        elif corruption_type == "intermittent":
            # Bursts of noise alternating with normal periods
            period = max(10, int(N * 0.05))
            for ch in target_channels:
                for start in range(0, N, period * 2):
                    end = min(N, start + period)
                    noise = self.rng.normal(0, severity * channel_stds[ch], size=end - start)
                    corrupted[start:end, ch] += noise
                    mask[start:end, ch] = 1

        elif corruption_type == "single_sensor":
            # Severe drift + noise on a single designated channel
            ch = target_channels[0]
            duration = max(50, int(N * corruption_rate))
            start = int(self.rng.integers(0, max(1, N - duration)))
            end = min(N, start + duration)
            ramp = np.linspace(0, severity * 2.0 * channel_stds[ch], end - start)
            noise = self.rng.normal(0, 0.2 * channel_stds[ch], size=end - start)
            corrupted[start:end, ch] += (ramp + noise)
            mask[start:end, ch] = 1
            affected_ranges.append({"channel": ch, "start": start, "end": end})

        elif corruption_type == "multi_sensor":
            # Drift on first sensor, stuck on second sensor
            for idx, ch in enumerate(target_channels):
                duration = max(40, int(N * corruption_rate))
                start = int(self.rng.integers(0, max(1, N - duration)))
                end = min(N, start + duration)
                if idx % 2 == 0:
                    corrupted[start:end, ch] += np.linspace(0, severity * channel_stds[ch], end - start)
                else:
                    corrupted[start:end, ch] = corrupted[start, ch]
                mask[start:end, ch] = 1
                affected_ranges.append({"channel": ch, "start": start, "end": end})

        elif corruption_type == "mixed":
            # Combined noise, spikes, and missingness across subset
            for ch in target_channels:
                # Add background noise
                corrupted[:, ch] += self.rng.normal(0, severity * 0.3 * channel_stds[ch], size=N)
                mask[:, ch] = 1
                # Add sporadic spikes
                spike_idx = self.rng.choice(N, size=max(1, int(N * 0.005)), replace=False)
                corrupted[spike_idx, ch] += severity * 2.0 * channel_stds[ch]

        affected_channel_names = (
            [channel_names[c] for c in target_channels]
            if channel_names is not None
            else [f"sensor_{c}" for c in target_channels]
        )

        metadata = {
            "corruption_type": corruption_type,
            "severity": float(severity),
            "corruption_rate": float(corruption_rate),
            "random_seed": int(self.seed),
            "target_channels": target_channels,
            "target_channel_names": affected_channel_names,
            "affected_ranges": affected_ranges,
            "total_corrupted_points": int(np.sum(mask)),
            "fraction_corrupted": float(np.mean(mask)),
        }

        return corrupted, mask, metadata
