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
    assert "brier_score" in report
    assert "precision_macro" in report
    assert "recall_macro" in report
    assert "f1_weighted" in report
    assert "roc_auc_ovr_macro" in report
    assert "pr_auc_macro" in report
    assert "confusion_matrix" in report
    assert "confusion_matrix_full" in report
    assert "per_class_metrics" in report
    assert "weakest_classes" in report
    assert "cv_summary" in report
    assert "interpretability" in report
    confusion = report["confusion_matrix"]
    assert isinstance(confusion, dict)
    assert confusion["labels"] == ["[NONE]", "lsfb_bonjour"]
    assert confusion["support"] == [2, 3]
    assert confusion["matrix"] == [[2, 0], [1, 2]]
    per_class = report["per_class_metrics"]
    assert isinstance(per_class, list)
    assert len(per_class) == 2
    full_confusion = report["confusion_matrix_full"]
    assert full_confusion["labels"] == ["[NONE]", "lsfb_bonjour"]
    assert full_confusion["matrix"] == [[2, 0], [1, 2]]
    assert full_confusion["normalized"] == [[1.0, 0.0], [0.3333, 0.6667]]


def test_eval_report_single_class_marks_unavailable_metrics() -> None:
    """Single-class eval should set unsupported metrics to None with explicit reasons."""
    y_true = np.array([0, 0, 0], dtype=np.int64)
    y_pred = np.array([0, 0, 0], dtype=np.int64)
    probs = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)

    report = TrainingService._build_eval_report(
        y_true=y_true,
        y_pred=y_pred,
        probs=probs,
        class_labels=["only_class"],
        target_label="only_class",
    )

    assert report["roc_auc_ovr_macro"] is None
    assert report["pr_auc_macro"] is None
    assert report["brier_score"] is None
    assert report["metric_unavailable_reasons"]["roc_auc_ovr_macro"] == "requires_at_least_two_classes"
    assert report["metric_unavailable_reasons"]["pr_auc_macro"] == "requires_at_least_two_classes"


def test_threshold_learning_honors_cost_objective() -> None:
    """Cost-driven thresholding should bias toward low-FP decisions."""
    probs = np.array(
        [
            [0.95, 0.05],
            [0.82, 0.18],
            [0.65, 0.35],
            [0.45, 0.55],
            [0.25, 0.75],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 0, 1, 1], dtype=np.int64)

    thresholds_fbeta = TrainingService._learn_class_thresholds(
        probs,
        labels,
        ["a", "b"],
        objective="fbeta",
        beta=2.0,
    )
    thresholds_cost = TrainingService._learn_class_thresholds(
        probs,
        labels,
        ["a", "b"],
        objective="cost",
        fp_cost=4.0,
        fn_cost=1.0,
    )

    assert thresholds_cost["b"] >= thresholds_fbeta["b"]


def test_cv_summary_fallback_when_stratification_is_insufficient() -> None:
    """CV helper should fallback to holdout when class supports are too small for k-fold."""
    probs = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
        ],
        dtype=np.float32,
    )
    y_true = np.array([0, 0, 1], dtype=np.int64)

    cv = TrainingService._compute_cv_summary(
        probs=probs,
        y_true=y_true,
        class_labels=["a", "b"],
        k_folds=5,
    )

    assert cv["mode"] == "holdout_fallback"
    assert cv["k_folds_used"] == 1
    assert cv["f1_macro_mean"] is not None
    assert len(cv["folds"]) == 1


def test_cv_summary_marks_unavailable_on_single_class() -> None:
    """CV helper should expose explicit unavailable reason on single-class labels."""
    probs = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
    y_true = np.array([0, 0, 0], dtype=np.int64)

    cv = TrainingService._compute_cv_summary(
        probs=probs,
        y_true=y_true,
        class_labels=["only"],
        k_folds=3,
    )

    assert cv["mode"] == "unavailable"
    assert cv["unavailable_reason"] == "single_class_ground_truth"


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
