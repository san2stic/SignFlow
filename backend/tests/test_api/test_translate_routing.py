"""Routing and shadow-mode integration tests for translate websocket."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

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


class _PrimaryPipeline:
    model_version = "v_primary"

    def __init__(self) -> None:
        self.calls = 0

    def process_frame(self, _payload: dict) -> _Prediction:
        self.calls += 1
        return _Prediction(
            prediction="bonjour",
            confidence=0.92,
            alternatives=[],
            sentence_buffer="bonjour",
            is_sentence_complete=False,
            decision_diagnostics={"status": "accepted"},
        )


class _ShadowPipeline:
    model_version = "v_shadow"

    def __init__(self) -> None:
        self.calls = 0

    def process_frame(self, _payload: dict) -> _Prediction:
        self.calls += 1
        return _Prediction(
            prediction="merci",
            confidence=0.89,
            alternatives=[],
            sentence_buffer="",
            is_sentence_complete=False,
            decision_diagnostics={"status": "accepted"},
        )


class _FakeRouter:
    def __init__(self, primary: _PrimaryPipeline, shadow: _ShadowPipeline | None, route: str) -> None:
        self.primary = primary
        self.shadow = shadow
        self.route = route

    def spawn_sessions(self):
        return SimpleNamespace(
            primary=self.primary,
            shadow=self.shadow,
            route=self.route,
        )


class _FakeShadowEvaluator:
    def __init__(self) -> None:
        self.calls = 0

    def compare(self, *, primary, shadow):
        self.calls += 1
        return SimpleNamespace(
            disagreed=primary.prediction != shadow.prediction,
            high_confidence_disagreement=True,
            primary_prediction=primary.prediction,
            shadow_prediction=shadow.prediction,
            confidence_gap=abs(primary.confidence - shadow.confidence),
        )


def _payload() -> dict:
    return {
        "timestamp": 0.0,
        "frame_idx": 1,
        "hands": {"left": [[0.1, 0.1, 0.0]] * 21, "right": [[0.1, 0.1, 0.0]] * 21},
        "pose": [[0.2, 0.2, 0.0]] * 33,
    }


def test_translate_stream_uses_router_and_executes_shadow_mode(monkeypatch) -> None:
    """Translate stream should route via model router and execute shadow inference silently."""
    primary = _PrimaryPipeline()
    shadow = _ShadowPipeline()
    evaluator = _FakeShadowEvaluator()
    settings = translate_api.get_settings()

    monkeypatch.setattr(settings, "use_torchserve", False)
    monkeypatch.setattr(settings, "shadow_mode_enabled", True)
    monkeypatch.setattr(settings, "inference_metrics_enabled", True)
    monkeypatch.setattr(
        translate_api,
        "get_or_create_model_router",
        lambda: _FakeRouter(primary=primary, shadow=shadow, route="canary"),
    )
    monkeypatch.setattr(
        translate_api,
        "get_or_create_shadow_evaluator",
        lambda: evaluator,
    )

    with TestClient(app) as test_client:
        with test_client.websocket_connect("/api/v1/translate/stream") as websocket:
            websocket.send_json(_payload())
            response = websocket.receive_json()

    assert response["prediction"] == "bonjour"
    assert response["confidence"] == 0.92
    assert primary.calls == 1
    assert shadow.calls == 1
    assert evaluator.calls == 1
