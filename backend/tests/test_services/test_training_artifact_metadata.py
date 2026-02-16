"""Artifact metadata structure tests for training outputs."""

from __future__ import annotations

from app.services.training_service import TrainingService


def test_build_artifact_metadata_contains_eval_report_quality_fields() -> None:
    """Persisted artifact metadata should carry enriched eval_report fields."""
    eval_report = {
        "macro_f1": 0.91,
        "target_sign_f1": 0.89,
        "open_set_fpr": 0.03,
        "ece": 0.08,
        "confusion_matrix": {
            "labels": ["[NONE]", "lsfb_bonjour"],
            "matrix": [[5, 1], [0, 4]],
            "support": [6, 4],
            "truncated": False,
        },
        "per_class_metrics": [
            {
                "label": "lsfb_bonjour",
                "index": 1,
                "precision": 0.8,
                "recall": 1.0,
                "f1": 0.8889,
                "support": 4,
            }
        ],
        "weakest_classes": [
            {
                "label": "lsfb_bonjour",
                "index": 1,
                "precision": 0.8,
                "recall": 1.0,
                "f1": 0.8889,
                "support": 4,
            }
        ],
        "latency_p95_ms": 42.5,
        "deployment_gate_passed": True,
    }

    metadata = TrainingService._build_artifact_metadata(
        class_thresholds={"[NONE]": 0.7, "lsfb_bonjour": 0.82},
        calibration_temperature=1.07,
        eval_report=eval_report,
        threshold_config={"objective": "fbeta", "beta": 1.5, "fp_cost": 1.0, "fn_cost": 1.5},
        calibration_details={"enabled": True, "method": "temperature_scaling"},
        training_diagnostics={"divergence_detected": False},
    )

    assert "class_thresholds" in metadata
    assert "calibration" in metadata
    assert "eval_report" in metadata
    assert metadata["eval_report"]["ece"] == 0.08
    assert "confusion_matrix" in metadata["eval_report"]
    assert "per_class_metrics" in metadata["eval_report"]
    assert "weakest_classes" in metadata["eval_report"]
    assert metadata["calibration"]["method"] == "temperature_scaling"
    assert metadata["thresholding"]["objective"] == "fbeta"
    assert metadata["training_diagnostics"]["divergence_detected"] is False
