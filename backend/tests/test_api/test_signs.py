"""Smoke test for sign CRUD operations."""

from __future__ import annotations

import io
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.config import get_settings
from app.database import SessionLocal
from app.main import app
from app.models.sign import Sign
from app.models.video import Video
from app.services import sign_service as sign_service_module
from app.services.search_service import SearchBackendUnavailable


def test_sign_crud_smoke() -> None:
    """Create and fetch one sign through API."""
    with TestClient(app) as client:
        suffix = uuid4().hex[:8]
        payload = {
            "name": f"Bonjour Test {suffix}",
            "description": "Description",
            "tags": ["demo"],
            "variants": [],
            "related_signs": []
        }
        create_res = client.post("/api/v1/signs", json=payload)
        assert create_res.status_code == 201

        sign_id = create_res.json()["id"]
        get_res = client.get(f"/api/v1/signs/{sign_id}")
        assert get_res.status_code == 200
        assert get_res.json()["name"] == f"Bonjour Test {suffix}"


def test_upload_video_rejects_invalid_file() -> None:
    """Video upload should return 400 for unsupported extension/content."""
    with TestClient(app) as client:
        suffix = uuid4().hex[:8]
        create_res = client.post(
            "/api/v1/signs",
            json={
                "name": f"Upload Target {suffix}",
                "description": "Target",
                "tags": ["demo"],
                "variants": [],
                "related_signs": [],
            },
        )
        assert create_res.status_code == 201
        sign_id = create_res.json()["id"]

        response = client.post(
            f"/api/v1/signs/{sign_id}/videos",
            files={"file": ("sample.txt", io.BytesIO(b"not a video"), "text/plain")},
            data={"type": "training"},
        )

    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"].lower()


def test_list_sign_videos_normalizes_legacy_video_type() -> None:
    """List endpoint should not fail on legacy type values stored in DB."""
    suffix = uuid4().hex[:8]
    legacy_video_id = str(uuid4())

    with SessionLocal() as db:
        sign = Sign(
            name=f"Legacy Video Target {suffix}",
            slug=f"legacy-video-target-{suffix}",
            description="Target",
            tags=[],
            variants=[],
        )
        db.add(sign)
        db.flush()
        sign_id = sign.id

        db.add(
            Video(
                id=legacy_video_id,
                sign_id=sign_id,
                file_path="/tmp/legacy.mp4",
                type="legacy",
                landmarks_extracted=False,
                landmarks_path=None,
            )
        )
        db.commit()

    with TestClient(app) as client:
        response = client.get(f"/api/v1/signs/{sign_id}/videos")

    assert response.status_code == 200
    payload = response.json()
    legacy_item = next((item for item in payload if item["id"] == legacy_video_id), None)
    assert legacy_item is not None
    assert legacy_item["type"] == "reference"

    with SessionLocal() as db:
        sign = db.scalar(select(Sign).where(Sign.id == sign_id))
        if sign:
            db.delete(sign)
            db.commit()


def test_get_sign_backlinks_returns_referencing_signs() -> None:
    """Backlinks endpoint should list incoming related signs."""
    suffix = uuid4().hex[:8]
    with SessionLocal() as db:
        target = Sign(
            name=f"Target {suffix}",
            slug=f"target-{suffix}",
            description="Target",
            tags=[],
            variants=[],
        )
        source = Sign(
            name=f"Source {suffix}",
            slug=f"source-{suffix}",
            description="Source",
            tags=[],
            variants=[],
        )
        db.add(target)
        db.add(source)
        db.flush()
        source.related_signs.append(target)
        db.commit()
        target_id = target.id
        source_id = source.id

    with TestClient(app) as client:
        response = client.get(f"/api/v1/signs/{target_id}/backlinks")

    assert response.status_code == 200
    payload = response.json()
    assert payload["sign_id"] == target_id
    assert any(item["id"] == source_id for item in payload["backlinks"])

    with SessionLocal() as db:
        source = db.scalar(select(Sign).where(Sign.id == source_id))
        target = db.scalar(select(Sign).where(Sign.id == target_id))
        if source:
            db.delete(source)
        if target:
            db.delete(target)
        db.commit()


def test_sign_search_uses_elasticsearch_result_order(monkeypatch) -> None:
    """When ES backend is active, API should preserve ES hit order."""
    settings = get_settings()
    monkeypatch.setattr(settings, "search_backend", "elasticsearch")
    monkeypatch.setattr(settings, "elasticsearch_fail_open", True)
    monkeypatch.setattr(sign_service_module.search_service, "index_sign", lambda *_args, **_kwargs: None)

    suffix = uuid4().hex[:8]
    with TestClient(app) as client:
        first_res = client.post(
            "/api/v1/signs",
            json={
                "name": f"Bonjour Premier {suffix}",
                "description": "First",
                "tags": ["demo"],
                "variants": [],
                "related_signs": [],
            },
        )
        assert first_res.status_code == 201
        first_id = first_res.json()["id"]

        second_res = client.post(
            "/api/v1/signs",
            json={
                "name": f"Bonjour Second {suffix}",
                "description": "Second",
                "tags": ["demo"],
                "variants": [],
                "related_signs": [],
            },
        )
        assert second_res.status_code == 201
        second_id = second_res.json()["id"]

        monkeypatch.setattr(
            sign_service_module.search_service,
            "search_sign_ids",
            lambda **_: ([second_id, first_id], 2),
        )

        response = client.get(f"/api/v1/signs?search=bonjor-{suffix}")

    assert response.status_code == 200
    payload = response.json()
    assert [item["id"] for item in payload["items"]] == [second_id, first_id]
    assert payload["total"] == 2


def test_sign_search_fallbacks_to_sql_when_elasticsearch_fails(monkeypatch) -> None:
    """Fail-open mode should fallback to SQL search when ES query fails."""
    settings = get_settings()
    monkeypatch.setattr(settings, "search_backend", "elasticsearch")
    monkeypatch.setattr(settings, "elasticsearch_fail_open", True)
    monkeypatch.setattr(sign_service_module.search_service, "index_sign", lambda *_args, **_kwargs: None)

    suffix = uuid4().hex[:8]
    with TestClient(app) as client:
        create_res = client.post(
            "/api/v1/signs",
            json={
                "name": f"Bonjour Fallback {suffix}",
                "description": "Fallback target",
                "tags": ["demo"],
                "variants": [],
                "related_signs": [],
            },
        )
        assert create_res.status_code == 201

        def raise_unavailable(**_):
            raise SearchBackendUnavailable("down")

        monkeypatch.setattr(sign_service_module.search_service, "search_sign_ids", raise_unavailable)

        response = client.get(f"/api/v1/signs?search=Fallback {suffix}")

    assert response.status_code == 200
    payload = response.json()
    assert any(f"Bonjour Fallback {suffix}" == item["name"] for item in payload["items"])


def test_sign_search_typo_uses_elasticsearch_candidates(monkeypatch) -> None:
    """Typo query should still return expected sign when ES resolves it."""
    settings = get_settings()
    monkeypatch.setattr(settings, "search_backend", "elasticsearch")
    monkeypatch.setattr(settings, "elasticsearch_fail_open", True)
    monkeypatch.setattr(sign_service_module.search_service, "index_sign", lambda *_args, **_kwargs: None)

    suffix = uuid4().hex[:8]
    with TestClient(app) as client:
        create_res = client.post(
            "/api/v1/signs",
            json={
                "name": f"Bonjour Typo {suffix}",
                "description": "Typo target",
                "tags": ["demo"],
                "variants": [],
                "related_signs": [],
            },
        )
        assert create_res.status_code == 201
        sign_id = create_res.json()["id"]

        monkeypatch.setattr(
            sign_service_module.search_service,
            "search_sign_ids",
            lambda **_: ([sign_id], 1),
        )

        response = client.get(f"/api/v1/signs?search=bonjor {suffix}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["items"][0]["id"] == sign_id
