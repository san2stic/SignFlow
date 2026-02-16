"""Search administration API tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import search as search_api
from app.config import get_settings
from app.main import app
from app.services.search_service import SearchBackendUnavailable


def test_reindex_search_success_when_elasticsearch_enabled(monkeypatch) -> None:
    """Endpoint should return reindex metrics when Elasticsearch backend is active."""
    settings = get_settings()
    monkeypatch.setattr(settings, "search_backend", "elasticsearch")

    payload = {
        "backend": "elasticsearch",
        "index": "signflow-signs",
        "indexed": 42,
        "failed": 0,
        "duration_ms": 123,
        "timestamp": "2026-02-16T12:00:00Z",
    }
    monkeypatch.setattr(search_api.search_service, "reindex_all", lambda _db: payload)

    with TestClient(app) as client:
        response = client.post("/api/v1/search/reindex")

    assert response.status_code == 200
    assert response.json() == payload


def test_reindex_search_rejects_non_elasticsearch_backend(monkeypatch) -> None:
    """Endpoint should return 400 when search backend is not Elasticsearch."""
    settings = get_settings()
    monkeypatch.setattr(settings, "search_backend", "sql")

    with TestClient(app) as client:
        response = client.post("/api/v1/search/reindex")

    assert response.status_code == 400
    assert response.json()["detail"] == "Search backend is not Elasticsearch"


def test_reindex_search_returns_503_when_backend_unavailable(monkeypatch) -> None:
    """Endpoint should return 503 when Elasticsearch is unavailable."""
    settings = get_settings()
    monkeypatch.setattr(settings, "search_backend", "elasticsearch")

    def raise_unavailable(_db):
        raise SearchBackendUnavailable("down")

    monkeypatch.setattr(search_api.search_service, "reindex_all", raise_unavailable)

    with TestClient(app) as client:
        response = client.post("/api/v1/search/reindex")

    assert response.status_code == 503
    assert response.json()["detail"] == "Search backend unavailable"
