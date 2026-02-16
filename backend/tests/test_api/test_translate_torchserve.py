"""TorchServe integration tests for translate websocket."""

from __future__ import annotations

from dataclasses import dataclass

import httpx
import numpy as np
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


class _TorchServePipeline:
    def __init__(self, fallback_label: str = "local_fallback") -> None:
        self.fallback_label = fallback_label
        self.async_calls = 0

    def spawn_session(self) -> _TorchServePipeline:
        return self

    def reset(self) -> None:
        self.async_calls = 0

    async def process_frame_async(self, _payload: dict, *, infer_window_async):
        self.async_calls += 1
        window = np.zeros((64, 469), dtype=np.float32)
        label, confidence, alternatives = await infer_window_async(window)
        return _Prediction(
            prediction=label,
            confidence=confidence,
            alternatives=alternatives,
            sentence_buffer=label,
            is_sentence_complete=False,
            decision_diagnostics={"status": "async_path"},
        )

    def process_frame(self, _payload: dict) -> _Prediction:
        return _Prediction(
            prediction=self.fallback_label,
            confidence=0.42,
            alternatives=[],
            sentence_buffer=self.fallback_label,
            is_sentence_complete=False,
            decision_diagnostics={"status": "sync_path"},
        )

    def _infer_window(self, _window):
        return self.fallback_label, 0.42, []


class _HealthyTorchServeClient:
    def __init__(self) -> None:
        self.calls = 0

    async def predict(self, _window):
        self.calls += 1
        return "bonjour", 0.93, [{"sign": "salut", "confidence": 0.5}]


class _FailingTorchServeClient:
    async def predict(self, _window):
        raise httpx.TimeoutException("torchserve timeout")


def _payload() -> dict:
    return {
        "timestamp": 0.0,
        "frame_idx": 1,
        "hands": {"left": [[0.1, 0.1, 0.0]] * 21, "right": [[0.1, 0.1, 0.0]] * 21},
        "pose": [[0.2, 0.2, 0.0]] * 33,
    }


def test_translate_stream_uses_torchserve_when_enabled(monkeypatch) -> None:
    """When TorchServe flag is enabled, websocket should route inference through client."""
    pipeline = _TorchServePipeline()
    client = _HealthyTorchServeClient()
    settings = translate_api.get_settings()

    monkeypatch.setattr(settings, "use_torchserve", True)
    monkeypatch.setattr(settings, "torchserve_url", "http://test:8080")
    monkeypatch.setattr(settings, "torchserve_timeout_ms", 1000)
    monkeypatch.setattr(translate_api, "get_or_create_pipeline", lambda: pipeline)
    monkeypatch.setattr(translate_api, "get_or_create_torchserve_client", lambda: client)

    with TestClient(app) as test_client:
        with test_client.websocket_connect("/api/v1/translate/stream") as websocket:
            websocket.send_json(_payload())
            response = websocket.receive_json()

    assert client.calls == 1
    assert pipeline.async_calls == 1
    assert response["prediction"] == "bonjour"
    assert response["confidence"] == 0.93
    assert response["decision_diagnostics"]["status"] == "async_path"


def test_translate_stream_falls_back_to_local_on_torchserve_error(monkeypatch) -> None:
    """TorchServe runtime errors should fallback to local inference without failing stream."""
    pipeline = _TorchServePipeline(fallback_label="fallback_ok")
    settings = translate_api.get_settings()

    monkeypatch.setattr(settings, "use_torchserve", True)
    monkeypatch.setattr(settings, "torchserve_url", "http://test:8080")
    monkeypatch.setattr(settings, "torchserve_timeout_ms", 1000)
    monkeypatch.setattr(translate_api, "get_or_create_pipeline", lambda: pipeline)
    monkeypatch.setattr(
        translate_api,
        "get_or_create_torchserve_client",
        lambda: _FailingTorchServeClient(),
    )

    with TestClient(app) as test_client:
        with test_client.websocket_connect("/api/v1/translate/stream") as websocket:
            websocket.send_json(_payload())
            response = websocket.receive_json()

    assert response["prediction"] == "fallback_ok"
    assert response["confidence"] == 0.42
    assert response["decision_diagnostics"]["status"] == "async_path"
