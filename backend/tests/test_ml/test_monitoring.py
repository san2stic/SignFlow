"""Tests for confidence drift monitoring."""

from __future__ import annotations

from app.ml.monitoring import DriftDetector


def test_drift_detector_initializes_reference_then_checks() -> None:
    """Detector should initialize reference first, then perform periodic checks."""
    detector = DriftDetector(
        window_size=256,
        check_every=10,
        min_samples=50,
    )

    result = None
    for _ in range(60):
        result = detector.record(0.82)

    assert result is not None
    assert detector.snapshot()["has_reference"] is True

    result = detector.record(0.80)
    assert result.checked is False
    assert result.reason in {"interval_not_reached", "insufficient_samples", "reference_initialized"}


def test_drift_detector_flags_large_distribution_shift() -> None:
    """Large confidence shift should eventually trigger drift detection."""
    detector = DriftDetector(
        window_size=400,
        check_every=20,
        min_samples=100,
        p_value_threshold=0.05,
        mean_shift_threshold=0.05,
        reference_confidences=[0.92] * 400,
    )

    drift_detected = False
    for _ in range(200):
        result = detector.record(0.12)
        if result.checked and result.drift_detected:
            drift_detected = True
            break

    assert drift_detected is True
