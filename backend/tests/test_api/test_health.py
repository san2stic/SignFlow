"""Basic API health tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_healthcheck() -> None:
    """Health endpoint should return an ok status."""
    with TestClient(app) as client:
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_healthcheck_legacy_alias() -> None:
    """Legacy health endpoint should remain available for compatibility."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
