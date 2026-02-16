"""Monitoring endpoint tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.ml.metrics import get_metrics_collector


def test_metrics_endpoint_exposes_prometheus_payload() -> None:
    """`/metrics` should return a Prometheus-compatible text payload."""
    collector = get_metrics_collector()
    collector.record_inference(
        model_version="v-metrics-test",
        sign="bonjour",
        confidence=0.88,
        latency_seconds=0.02,
    )

    with TestClient(app) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers.get("content-type", "")
    assert "signflow_inference_total" in response.text
