"""Smoke test for sign CRUD operations."""

from __future__ import annotations

import io
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.database import SessionLocal
from app.main import app
from app.models.sign import Sign


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
