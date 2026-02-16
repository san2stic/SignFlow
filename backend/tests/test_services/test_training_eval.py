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


def test_compute_ece_returns_expected_value() -> None:
    """ECE helper should compute weighted calibration gap across bins."""
    probs = np.array(
        [
            [0.90, 0.10],
            [0.90, 0.10],
            [0.60, 0.40],
            [0.40, 0.60],
        ],
        dtype=np.float32,
    )
    y_true = np.array([0, 1, 1, 0], dtype=np.int64)

    ece = TrainingService._compute_ece(probs=probs, y_true=y_true, n_bins=2)

    assert ece is not None
    assert abs(ece - 0.5) < 1e-6


def test_eval_report_contains_new_quality_fields() -> None:
    """Evaluation report should include ECE, confusion matrix and class-wise diagnostics."""
    y_true = np.array([0, 1, 1, 0, 1], dtype=np.int64)
    y_pred = np.array([0, 1, 0, 0, 1], dtype=np.int64)
    probs = np.array(
        [
            [0.90, 0.10],
            [0.10, 0.90],
            [0.60, 0.40],
            [0.55, 0.45],
            [0.20, 0.80],
        ],
        dtype=np.float32,
    )

    report = TrainingService._build_eval_report(
        y_true=y_true,
        y_pred=y_pred,
        probs=probs,
        class_labels=["[NONE]", "lsfb_bonjour"],
        target_label="lsfb_bonjour",
    )

    assert "ece" in report
    assert "confusion_matrix" in report
    assert "per_class_metrics" in report
    assert "weakest_classes" in report
    confusion = report["confusion_matrix"]
    assert isinstance(confusion, dict)
    assert confusion["labels"] == ["[NONE]", "lsfb_bonjour"]
    assert confusion["support"] == [2, 3]
    assert confusion["matrix"] == [[2, 0], [1, 2]]
    per_class = report["per_class_metrics"]
    assert isinstance(per_class, list)
    assert len(per_class) == 2


def test_deployment_gate_rejects_high_ece() -> None:
    """Deployment gate should reject models that exceed configured ECE threshold."""
    config = {
        "macro_f1_gate": 0.82,
        "target_sign_f1_gate": 0.85,
        "open_set_fpr_gate": 0.05,
        "latency_p95_ms_gate": 120.0,
        "ece_gate": 0.10,
    }

    ok = TrainingService._passes_deployment_gate(
        mode="few-shot",
        config=config,
        macro_f1=0.9,
        target_sign_f1=0.9,
        open_set_fpr=0.02,
        latency_p95_ms=80.0,
        ece=0.08,
    )
    not_ok = TrainingService._passes_deployment_gate(
        mode="few-shot",
        config=config,
        macro_f1=0.9,
        target_sign_f1=0.9,
        open_set_fpr=0.02,
        latency_p95_ms=80.0,
        ece=0.15,
    )

    assert ok is True
    assert not_ok is False
