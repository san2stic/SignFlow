"""Active-learning integration tests for translate websocket and queue endpoints."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi.testclient import TestClient

import app.api.translate as translate_api
from app.main import app


@dataclass
class _Prediction:
    prediction: str
    confidence: float
    alternatives: list[dict[str, float]]
    sentence_buffer: str
    is_sentence_complete: bool
    decision_diagnostics: dict[str, object] | None = None


class _UncertainPipeline:
    def spawn_session(self) -> _UncertainPipeline:
        return self

    def reset(self) -> None:
        return None

    def process_frame(self, _payload: dict) -> _Prediction:
        return _Prediction(
            prediction="NONE",
            confidence=0.51,
            alternatives=[{"sign": "lsfb_bonjour", "confidence": 0.49}],
            sentence_buffer="",
            is_sentence_complete=False,
            decision_diagnostics={"status": "rejected"},
        )


def _payload() -> dict:
    return {
        "timestamp": 1705312200.123,
        "frame_idx": 42,
        "hands": {"left": [[0.1, 0.1, 0.0]] * 21, "right": [[0.1, 0.1, 0.0]] * 21},
        "pose": [[0.2, 0.2, 0.0]] * 33,
    }


def test_translate_stream_queues_uncertain_samples_for_active_learning(monkeypatch) -> None:
    """Uncertain websocket predictions should feed active-learning queue and endpoints."""
    settings = translate_api.get_settings()
    monkeypatch.setattr(settings, "active_learning_enabled", True)
    monkeypatch.setattr(settings, "active_learning_strategy", "margin")
    monkeypatch.setattr(settings, "active_learning_min_uncertainty", 0.5)
    monkeypatch.setattr(settings, "active_learning_max_queue", 100)
    monkeypatch.setattr(settings, "active_learning_top_n", 25)
    monkeypatch.setattr(settings, "active_learning_cooldown_seconds", 0.0)
    monkeypatch.setattr(settings, "shadow_mode_enabled", False)
    monkeypatch.setattr(settings, "canary_percentage", 0.0)

    monkeypatch.setattr(translate_api, "_global_model_router", None)
    monkeypatch.setattr(translate_api, "_global_model_router_config", None)
    monkeypatch.setattr(translate_api, "_global_active_learning_queue", None)
    monkeypatch.setattr(translate_api, "_global_active_learning_config", None)
    monkeypatch.setattr(translate_api, "get_or_create_pipeline", lambda: _UncertainPipeline())

    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/translate/stream") as websocket:
            websocket.send_json(_payload())
            stream_response = websocket.receive_json()

        assert stream_response["active_learning"]["queued"] is True
        sample_id = stream_response["active_learning"]["sample_id"]

        queue_response = client.get("/api/v1/translate/active-learning/queue?limit=10")
        assert queue_response.status_code == 200
        queue_payload = queue_response.json()
        assert queue_payload["enabled"] is True
        assert queue_payload["queue_size"] >= 1
        assert any(item["id"] == sample_id for item in queue_payload["items"])

        resolve_response = client.post(f"/api/v1/translate/active-learning/queue/{sample_id}/resolve")
        assert resolve_response.status_code == 200
        resolved_payload = resolve_response.json()
        assert resolved_payload["resolved"] is True
        assert resolved_payload["sample"]["id"] == sample_id


def test_active_learning_queue_endpoint_returns_disabled_when_feature_flag_off(monkeypatch) -> None:
    """Queue endpoint should return disabled state when feature flag is off."""
    settings = translate_api.get_settings()
    monkeypatch.setattr(settings, "active_learning_enabled", False)

    with TestClient(app) as client:
        response = client.get("/api/v1/translate/active-learning/queue")

    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is False
    assert payload["queue_size"] == 0
    assert payload["items"] == []
