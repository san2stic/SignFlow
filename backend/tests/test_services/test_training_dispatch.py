"""Training dispatch mode tests (Celery vs thread fallback)."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import SessionLocal
from app.models.training import TrainingSession as TrainingSessionModel
from app.schemas.training import TrainingSessionCreate
from app.services.training_service import training_service


def _delete_session(db: Session, session_id: str) -> None:
    session = db.get(TrainingSessionModel, session_id)
    if session:
        db.delete(session)
        db.commit()


def test_dispatch_uses_thread_when_celery_disabled(monkeypatch) -> None:
    """If Celery is disabled, service should dispatch on local thread."""
    settings = get_settings()
    monkeypatch.setattr(settings, "training_use_celery", False)

    calls = {"thread": 0, "celery": 0}

    def fake_thread_dispatch(_session_id: str) -> None:
        calls["thread"] += 1

    def fake_celery_dispatch(_session_id: str) -> bool:
        calls["celery"] += 1
        return True

    monkeypatch.setattr(training_service, "_launch_background_worker", fake_thread_dispatch)
    monkeypatch.setattr(training_service, "_enqueue_celery_job", fake_celery_dispatch)

    with SessionLocal() as db:
        payload = TrainingSessionCreate(mode="full-retrain")
        session = training_service.create_session(db, payload)
        created_id = session.id

    assert calls["thread"] == 1
    assert calls["celery"] == 0

    with SessionLocal() as cleanup_db:
        _delete_session(cleanup_db, created_id)


def test_dispatch_uses_celery_when_enabled(monkeypatch) -> None:
    """If Celery dispatch succeeds, thread fallback should not run."""
    settings = get_settings()
    monkeypatch.setattr(settings, "training_use_celery", True)

    calls = {"thread": 0, "celery": 0}

    def fake_thread_dispatch(_session_id: str) -> None:
        calls["thread"] += 1

    def fake_celery_dispatch(_session_id: str) -> bool:
        calls["celery"] += 1
        return True

    monkeypatch.setattr(training_service, "_launch_background_worker", fake_thread_dispatch)
    monkeypatch.setattr(training_service, "_enqueue_celery_job", fake_celery_dispatch)

    with SessionLocal() as db:
        payload = TrainingSessionCreate(mode="full-retrain")
        session = training_service.create_session(db, payload)
        created_id = session.id

    assert calls["thread"] == 0
    assert calls["celery"] == 1

    with SessionLocal() as cleanup_db:
        _delete_session(cleanup_db, created_id)


def test_dispatch_falls_back_to_thread_on_celery_error(monkeypatch) -> None:
    """If Celery dispatch fails, local thread fallback should execute."""
    settings = get_settings()
    monkeypatch.setattr(settings, "training_use_celery", True)

    calls = {"thread": 0, "celery": 0}

    def fake_thread_dispatch(_session_id: str) -> None:
        calls["thread"] += 1

    def fake_celery_dispatch(_session_id: str) -> bool:
        calls["celery"] += 1
        return False

    monkeypatch.setattr(training_service, "_launch_background_worker", fake_thread_dispatch)
    monkeypatch.setattr(training_service, "_enqueue_celery_job", fake_celery_dispatch)

    with SessionLocal() as db:
        payload = TrainingSessionCreate(mode="full-retrain")
        session = training_service.create_session(db, payload)
        created_id = session.id

    assert calls["thread"] == 1
    assert calls["celery"] == 1

    with SessionLocal() as cleanup_db:
        _delete_session(cleanup_db, created_id)
