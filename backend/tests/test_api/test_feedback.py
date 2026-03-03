"""Integration tests for the feedback corrections REST API.

Endpoints under test:
  POST   /api/v1/feedback/corrections
  GET    /api/v1/feedback/corrections
  GET    /api/v1/feedback/stats
  DELETE /api/v1/feedback/corrections/{id}
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

# Ensure feedback is enabled before app settings are imported.
os.environ.setdefault("FEEDBACK_ENABLED", "true")

from app.main import app
from app.database import SessionLocal
from app.models.feedback import FeedbackCorrection

BASE = "/api/v1/feedback"


@pytest.fixture()
def client():
    """Return a TestClient bound to the FastAPI app."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _enable_feedback(monkeypatch):
    """Force `feedback_enabled=True` on the cached settings object for every test."""
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "feedback_enabled", True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insert_correction(
    *,
    predicted_sign: str = "BONJOUR",
    corrected_sign: str = "AU-REVOIR",
    status: str = "pending",
) -> int:
    """Directly insert a FeedbackCorrection row and return its id."""
    with SessionLocal() as db:
        corr = FeedbackCorrection(
            predicted_sign=predicted_sign,
            corrected_sign=corrected_sign,
            status=status,
        )
        db.add(corr)
        db.commit()
        db.refresh(corr)
        return corr.id


# ---------------------------------------------------------------------------
# POST /corrections — soumission valide (cas 1)
# ---------------------------------------------------------------------------

def test_submit_correction_returns_201_and_expected_fields(client, monkeypatch):
    """A valid payload must persist the correction and return the full response body."""
    import app.api.feedback as feedback_api

    monkeypatch.setattr(
        feedback_api.feedback_service,
        "check_and_trigger_training",
        lambda db, corrected_sign: False,
    )

    payload = {
        "predicted_sign": "BONJOUR",
        "corrected_sign": "AU-REVOIR",
        "confidence": 0.72,
        "session_id": "test-session-001",
    }
    resp = client.post(f"{BASE}/corrections", json=payload)

    assert resp.status_code == 201, resp.text
    body = resp.json()

    assert body["predicted_sign"] == "BONJOUR"
    assert body["corrected_sign"] == "AU-REVOIR"
    assert abs(body["confidence"] - 0.72) < 1e-6
    assert body["session_id"] == "test-session-001"
    assert body["status"] == "pending"
    assert body["trigger_training"] is False
    assert "id" in body
    assert "created_at" in body


def test_submit_correction_without_optional_fields(client, monkeypatch):
    """Only required fields must be accepted; optional fields default to null."""
    import app.api.feedback as feedback_api

    monkeypatch.setattr(
        feedback_api.feedback_service,
        "check_and_trigger_training",
        lambda db, corrected_sign: False,
    )

    payload = {
        "predicted_sign": "MERCI",
        "corrected_sign": "PARDON",
    }
    resp = client.post(f"{BASE}/corrections", json=payload)

    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["confidence"] is None
    assert body["session_id"] is None
    assert body["landmarks_path"] is None


# ---------------------------------------------------------------------------
# POST /corrections — validations Pydantic (champs manquants → 422) (cas 2)
# ---------------------------------------------------------------------------

def test_submit_correction_missing_predicted_sign_returns_422(client):
    """Missing required `predicted_sign` field must yield a 422 Unprocessable Entity."""
    payload = {"corrected_sign": "AU-REVOIR"}
    resp = client.post(f"{BASE}/corrections", json=payload)
    assert resp.status_code == 422


def test_submit_correction_missing_corrected_sign_returns_422(client):
    """Missing required `corrected_sign` field must yield a 422 Unprocessable Entity."""
    payload = {"predicted_sign": "BONJOUR"}
    resp = client.post(f"{BASE}/corrections", json=payload)
    assert resp.status_code == 422


def test_submit_correction_empty_predicted_sign_returns_422(client):
    """Empty string for `predicted_sign` must yield 422 (min_length=1)."""
    payload = {"predicted_sign": "", "corrected_sign": "AU-REVOIR"}
    resp = client.post(f"{BASE}/corrections", json=payload)
    assert resp.status_code == 422


def test_submit_correction_confidence_out_of_range_returns_422(client):
    """Confidence value > 1.0 must be rejected with 422."""
    payload = {
        "predicted_sign": "BONJOUR",
        "corrected_sign": "AU-REVOIR",
        "confidence": 1.5,
    }
    resp = client.post(f"{BASE}/corrections", json=payload)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /corrections — liste vide au départ (cas 3)
# ---------------------------------------------------------------------------

def test_list_corrections_empty_database_returns_empty_list(client):
    """GET /corrections on a clean DB must return HTTP 200 and an empty array."""
    resp = client.get(f"{BASE}/corrections")
    assert resp.status_code == 200
    assert resp.json() == []


# ---------------------------------------------------------------------------
# GET /corrections — après insertion (cas 4)
# ---------------------------------------------------------------------------

def test_list_corrections_returns_inserted_record(client):
    """After direct DB insertion the GET endpoint must return the correction."""
    correction_id = _insert_correction(
        predicted_sign="MERCI",
        corrected_sign="PARDON",
    )

    resp = client.get(f"{BASE}/corrections")
    assert resp.status_code == 200

    items = resp.json()
    assert len(items) == 1
    assert items[0]["id"] == correction_id
    assert items[0]["predicted_sign"] == "MERCI"
    assert items[0]["corrected_sign"] == "PARDON"
    assert items[0]["status"] == "pending"


def test_list_corrections_status_filter(client):
    """The `status` query param must filter results correctly."""
    _insert_correction(predicted_sign="A", corrected_sign="B", status="pending")
    _insert_correction(predicted_sign="C", corrected_sign="D", status="ignored")

    resp_pending = client.get(f"{BASE}/corrections?status=pending")
    assert resp_pending.status_code == 200
    pending_items = resp_pending.json()
    assert len(pending_items) == 1
    assert pending_items[0]["status"] == "pending"

    resp_ignored = client.get(f"{BASE}/corrections?status=ignored")
    assert resp_ignored.status_code == 200
    ignored_items = resp_ignored.json()
    assert len(ignored_items) == 1
    assert ignored_items[0]["status"] == "ignored"


# ---------------------------------------------------------------------------
# GET /stats — retourne une liste (même vide) (cas 5)
# ---------------------------------------------------------------------------

def test_get_stats_empty_database_returns_empty_list(client):
    """GET /stats on a clean DB must return 200 and an empty list."""
    resp = client.get(f"{BASE}/stats")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_stats_after_insertions_returns_aggregates(client):
    """Stats endpoint must aggregate counts per corrected sign."""
    _insert_correction(predicted_sign="A", corrected_sign="BONJOUR", status="pending")
    _insert_correction(predicted_sign="B", corrected_sign="BONJOUR", status="pending")
    _insert_correction(predicted_sign="C", corrected_sign="BONJOUR", status="trained")
    _insert_correction(predicted_sign="D", corrected_sign="MERCI", status="pending")

    resp = client.get(f"{BASE}/stats")
    assert resp.status_code == 200

    stats = {item["sign_name"]: item for item in resp.json()}
    assert "BONJOUR" in stats
    assert stats["BONJOUR"]["correction_count"] == 3
    assert stats["BONJOUR"]["pending_count"] == 2
    assert "MERCI" in stats
    assert stats["MERCI"]["correction_count"] == 1
    assert stats["MERCI"]["pending_count"] == 1


# ---------------------------------------------------------------------------
# DELETE /corrections/{id} — marque ignored → 204 (cas 6)
# ---------------------------------------------------------------------------

def test_delete_correction_marks_ignored_and_returns_204(client):
    """DELETE must soft-delete (status=ignored) and return 204 No Content."""
    correction_id = _insert_correction(
        predicted_sign="BONJOUR",
        corrected_sign="AU-REVOIR",
        status="pending",
    )

    resp = client.delete(f"{BASE}/corrections/{correction_id}")
    assert resp.status_code == 204

    # Verify the status was updated in the DB.
    with SessionLocal() as db:
        row = db.get(FeedbackCorrection, correction_id)
        assert row is not None
        assert row.status == "ignored"


# ---------------------------------------------------------------------------
# DELETE /corrections/{id} — id inexistant → 404 (cas 7)
# ---------------------------------------------------------------------------

def test_delete_nonexistent_correction_returns_404(client):
    """Attempting to delete a correction with a non-existent ID must return 404."""
    resp = client.delete(f"{BASE}/corrections/99999")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Feedback disabled (403)
# ---------------------------------------------------------------------------

def test_submit_correction_when_feedback_disabled_returns_403(client, monkeypatch):
    """When `feedback_enabled=False`, POST must return 403 Forbidden."""
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "feedback_enabled", False)

    payload = {"predicted_sign": "BONJOUR", "corrected_sign": "AU-REVOIR"}
    resp = client.post(f"{BASE}/corrections", json=payload)
    assert resp.status_code == 403
