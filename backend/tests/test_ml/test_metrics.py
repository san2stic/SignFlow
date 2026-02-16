"""Tests for inference metrics collector."""

from __future__ import annotations

from app.ml.metrics import InferenceMetricsCollector


def test_metrics_collector_records_and_renders() -> None:
    """Collector should record key events and expose Prometheus text payload."""
    collector = InferenceMetricsCollector()

    collector.record_inference(
        model_version="v10",
        sign="bonjour",
        confidence=0.91,
        latency_seconds=0.034,
    )
    collector.record_torchserve_error(reason="timeout")
    collector.record_drift_alert(kind="confidence")
    collector.record_routing_decision(route="canary")
    collector.record_shadow_comparison(route="canary", disagreed=True)

    payload, content_type = collector.render_latest()
    text = payload.decode("utf-8")

    assert "signflow_inference_total" in text
    assert "signflow_inference_latency_seconds" in text
    assert "signflow_prediction_confidence" in text
    assert "signflow_torchserve_errors_total" in text
    assert "signflow_drift_alerts_total" in text
    assert "signflow_routing_total" in text
    assert "signflow_shadow_comparisons_total" in text
    assert "text/plain" in content_type
