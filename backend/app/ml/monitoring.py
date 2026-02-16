"""Runtime drift monitoring for inference confidence distributions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

try:
    from scipy.stats import ks_2samp

    SCIPY_AVAILABLE = True
except Exception:  # noqa: BLE001
    SCIPY_AVAILABLE = False


@dataclass
class DriftCheckResult:
    """Result payload for one drift-check attempt."""

    checked: bool
    drift_detected: bool
    samples: int
    reason: str
    p_value: float | None = None
    statistic: float | None = None
    reference_mean: float | None = None
    current_mean: float | None = None


class DriftDetector:
    """Detect distribution drift via KS test (or mean-shift fallback)."""

    def __init__(
        self,
        *,
        window_size: int = 1000,
        check_every: int = 100,
        min_samples: int = 200,
        p_value_threshold: float = 0.05,
        mean_shift_threshold: float = 0.12,
        reference_confidences: list[float] | np.ndarray | None = None,
    ) -> None:
        self.window_size = max(64, int(window_size))
        self.check_every = max(1, int(check_every))
        self.min_samples = max(32, int(min_samples))
        self.p_value_threshold = float(np.clip(p_value_threshold, 1e-6, 1.0))
        self.mean_shift_threshold = float(np.clip(mean_shift_threshold, 0.0, 1.0))

        self._buffer: deque[float] = deque(maxlen=self.window_size)
        self._observed_count = 0

        self._reference: np.ndarray | None = None
        if reference_confidences is not None:
            self.set_reference(reference_confidences)

    def set_reference(self, values: list[float] | np.ndarray) -> None:
        """Set fixed reference distribution for future checks."""
        arr = np.asarray(values, dtype=np.float32)
        arr = np.clip(arr, 0.0, 1.0)
        if arr.size < 2:
            raise ValueError("Reference distribution requires at least 2 samples")
        self._reference = arr

    def record(self, confidence: float) -> DriftCheckResult:
        """Record one confidence sample and run a periodic drift check."""
        normalized = float(np.clip(confidence, 0.0, 1.0))
        self._buffer.append(normalized)
        self._observed_count += 1

        sample_count = len(self._buffer)
        if sample_count < self.min_samples:
            return DriftCheckResult(
                checked=False,
                drift_detected=False,
                samples=sample_count,
                reason="insufficient_samples",
            )

        if self._reference is None:
            self._reference = np.asarray(self._buffer, dtype=np.float32)
            return DriftCheckResult(
                checked=False,
                drift_detected=False,
                samples=sample_count,
                reason="reference_initialized",
            )

        if self._observed_count % self.check_every != 0:
            return DriftCheckResult(
                checked=False,
                drift_detected=False,
                samples=sample_count,
                reason="interval_not_reached",
            )

        current = np.asarray(self._buffer, dtype=np.float32)
        reference_mean = float(np.mean(self._reference))
        current_mean = float(np.mean(current))

        if SCIPY_AVAILABLE and self._reference.size >= 2 and current.size >= 2:
            statistic, p_value = ks_2samp(self._reference, current, method="auto")
            drift_detected = float(p_value) < self.p_value_threshold
            return DriftCheckResult(
                checked=True,
                drift_detected=bool(drift_detected),
                samples=sample_count,
                reason="ks_test",
                p_value=float(p_value),
                statistic=float(statistic),
                reference_mean=reference_mean,
                current_mean=current_mean,
            )

        mean_shift = abs(current_mean - reference_mean)
        drift_detected = mean_shift >= self.mean_shift_threshold
        return DriftCheckResult(
            checked=True,
            drift_detected=bool(drift_detected),
            samples=sample_count,
            reason="mean_shift_fallback",
            statistic=float(mean_shift),
            reference_mean=reference_mean,
            current_mean=current_mean,
        )

    def snapshot(self) -> dict[str, float | int | bool]:
        """Expose internal state for debugging."""
        reference_size = int(self._reference.size) if self._reference is not None else 0
        return {
            "buffer_size": int(len(self._buffer)),
            "window_size": int(self.window_size),
            "observed_count": int(self._observed_count),
            "has_reference": bool(self._reference is not None),
            "reference_size": reference_size,
            "check_every": int(self.check_every),
            "min_samples": int(self.min_samples),
        }
