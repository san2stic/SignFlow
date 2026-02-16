"""Usage-count integration tests for translate websocket."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import select

import app.api.translate as translate_api
from app.database import SessionLocal
from app.main import app
from app.models.sign import Sign


@dataclass
class _Prediction:
    prediction: str
    confidence: float
    alternatives: list[dict[str, float]]
    sentence_buffer: str
    is_sentence_complete: bool


class _FakePipeline:
    def __init__(self, slug: str) -> None:
        self.slug = slug
        self.calls = 0

    def spawn_session(self) -> _FakePipeline:
        return self

    def reset(self) -> None:
        self.calls = 0

    def process_frame(self, _payload: dict) -> _Prediction:
        self.calls += 1
        if self.calls == 1:
            sentence = self.slug
        elif self.calls == 2:
            sentence = self.slug
        else:
            sentence = f"{self.slug} {self.slug}"
        return _Prediction(
            prediction=self.slug,
            confidence=0.94,
            alternatives=[],
            sentence_buffer=sentence,
            is_sentence_complete=False,
        )


class _PerSessionPipeline:
    def __init__(self, slug: str) -> None:
        self.slug = slug
        self.calls = 0

    def reset(self) -> None:
        self.calls = 0

    def process_frame(self, _payload: dict) -> _Prediction:
        self.calls += 1
        sentence = " ".join([self.slug] * self.calls)
        return _Prediction(
            prediction=self.slug,
            confidence=0.91,
            alternatives=[],
            sentence_buffer=sentence,
            is_sentence_complete=False,
        )


class _PerSessionPipelineTemplate:
    def __init__(self, slug: str) -> None:
        self.slug = slug
        self.spawn_calls = 0
        self.sessions: list[_PerSessionPipeline] = []

    def spawn_session(self) -> _PerSessionPipeline:
        self.spawn_calls += 1
        pipeline = _PerSessionPipeline(slug=self.slug)
        self.sessions.append(pipeline)
        return pipeline


def test_translate_stream_increments_usage_once_per_new_token(monkeypatch) -> None:
    """Usage count must increment only when sentence buffer gains new tokens."""
    suffix = uuid4().hex[:8]
    slug = f"lsfb_usage_{suffix}"

    with SessionLocal() as db:
        sign = Sign(
            name=f"Usage {suffix}",
            slug=slug,
            description="Usage test",
            tags=[],
            variants=[],
            usage_count=0,
        )
        db.add(sign)
        db.commit()
        sign_id = sign.id

    fake_pipeline = _FakePipeline(slug=slug)
    monkeypatch.setattr(translate_api, "get_or_create_pipeline", lambda: fake_pipeline)

    payload = {
        "timestamp": 0.0,
        "frame_idx": 1,
        "hands": {"left": [[0.1, 0.1, 0.0]] * 21, "right": [[0.1, 0.1, 0.0]] * 21},
        "pose": [[0.2, 0.2, 0.0]] * 33,
    }

    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/translate/stream") as websocket:
            for _ in range(3):
                websocket.send_json(payload)
                _ = websocket.receive_json()

    with SessionLocal() as db:
        refreshed = db.scalar(select(Sign).where(Sign.id == sign_id))
        assert refreshed is not None
        # 1st message emits first token, 2nd emits no new token, 3rd emits second token.
        assert refreshed.usage_count == 2


def test_translate_stream_isolates_state_between_two_sessions(monkeypatch) -> None:
    """Each websocket must get its own inference state to prevent cross-session leakage."""
    suffix = uuid4().hex[:8]
    slug = f"lsfb_isolation_{suffix}"
    template = _PerSessionPipelineTemplate(slug=slug)
    monkeypatch.setattr(translate_api, "get_or_create_pipeline", lambda: template)

    payload = {
        "timestamp": 0.0,
        "frame_idx": 1,
        "hands": {"left": [[0.1, 0.1, 0.0]] * 21, "right": [[0.1, 0.1, 0.0]] * 21},
        "pose": [[0.2, 0.2, 0.0]] * 33,
    }

    with TestClient(app) as client:
        with (
            client.websocket_connect("/api/v1/translate/stream") as websocket_one,
            client.websocket_connect("/api/v1/translate/stream") as websocket_two,
        ):
            websocket_one.send_json(payload)
            response_one_first = websocket_one.receive_json()

            websocket_two.send_json(payload)
            response_two_first = websocket_two.receive_json()

            websocket_one.send_json(payload)
            response_one_second = websocket_one.receive_json()

            websocket_two.send_json(payload)
            response_two_second = websocket_two.receive_json()

    assert template.spawn_calls == 2
    assert len(template.sessions) == 2
    assert response_one_first["sentence_buffer"] == slug
    assert response_two_first["sentence_buffer"] == slug
    assert response_one_second["sentence_buffer"] == f"{slug} {slug}"
    assert response_two_second["sentence_buffer"] == f"{slug} {slug}"
    assert [session.calls for session in template.sessions] == [2, 2]
