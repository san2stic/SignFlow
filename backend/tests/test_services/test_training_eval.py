"""Training evaluation helper tests (calibration, thresholds, deployment gate)."""

from __future__ import annotations

import numpy as np

from app.services.training_service import TrainingService


def _mean_nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    scaled = logits / max(1e-6, float(temperature))
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    probs = np.exp(scaled)
    probs = probs / np.clip(np.sum(probs, axis=1, keepdims=True), 1e-9, None)
    picked = probs[np.arange(labels.shape[0]), labels]
    return float(-np.mean(np.log(np.clip(picked, 1e-9, None))))


def test_fit_temperature_reduces_nll() -> None:
    """Fitted temperature should not worsen NLL on calibration samples."""
    logits = np.array(
        [
            [5.0, -1.0],
            [4.5, -0.5],
            [4.0, -0.2],
            [1.2, 0.9],
            [0.9, 1.1],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 0, 1, 1], dtype=np.int64)

    before = _mean_nll(logits, labels, temperature=1.0)
    temperature = TrainingService._fit_temperature(logits, labels)
    after = _mean_nll(logits, labels, temperature=temperature)

    assert 0.5 <= temperature <= 10.0
    assert after <= before + 1e-6


def test_predict_with_thresholds_can_route_to_none() -> None:
    """Low-confidence predictions should map to [NONE] with class thresholds."""
    probs = np.array(
        [
            [0.60, 0.40],
            [0.45, 0.55],
        ],
        dtype=np.float32,
    )
    predictions = TrainingService._predict_with_thresholds(
        probs,
        class_labels=["[NONE]", "lsfb_bonjour"],
        class_thresholds={"[NONE]": 0.5, "lsfb_bonjour": 0.8},
    )
    assert predictions.tolist() == [0, 0]


def test_deployment_gate_requires_all_thresholds() -> None:
    """Deployment gate should fail when one metric is under threshold."""
    config = {
        "macro_f1_gate": 0.82,
        "target_sign_f1_gate": 0.85,
        "open_set_fpr_gate": 0.05,
        "latency_p95_ms_gate": 120.0,
    }
    ok = TrainingService._passes_deployment_gate(
        mode="few-shot",
        config=config,
        macro_f1=0.9,
        target_sign_f1=0.9,
        open_set_fpr=0.02,
        latency_p95_ms=80.0,
    )
    not_ok = TrainingService._passes_deployment_gate(
        mode="few-shot",
        config=config,
        macro_f1=0.9,
        target_sign_f1=0.8,
        open_set_fpr=0.02,
        latency_p95_ms=80.0,
    )

    assert ok is True
    assert not_ok is False
