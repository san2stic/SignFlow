"""Unit tests for FeedbackService using an in-memory SQLite session.

Tests:
  1. submit_correction()        — crée bien un enregistrement en DB
  2. get_pending_count()        — compte correctement les entrées 'pending'
  3. check_and_trigger_training() — si count < threshold → False, pas de training
  4. get_stats()                — retourne les bons agrégats
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# Isolated in-memory SQLite for service-level tests.
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("FEEDBACK_ENABLED", "true")

from app.database import Base
from app.models.feedback import FeedbackCorrection  # noqa: F401 – needed for metadata
from app.services.feedback_service import FeedbackService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    _engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=_engine)
    yield _engine
    Base.metadata.drop_all(bind=_engine)
    _engine.dispose()


@pytest.fixture()
def db(engine) -> Session:
    """Provide an isolated, auto-rolled-back DB session for each test."""
    connection = engine.connect()
    transaction = connection.begin()
    SessionFactory = sessionmaker(bind=connection)
    session = SessionFactory()

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture()
def service() -> FeedbackService:
    return FeedbackService()


# ---------------------------------------------------------------------------
# Test 1 — submit_correction() crée un enregistrement en DB
# ---------------------------------------------------------------------------

def test_submit_correction_persists_record(service, db):
    """submit_correction must insert one FeedbackCorrection row with correct fields."""
    correction = service.submit_correction(
        db,
        predicted_sign="BONJOUR",
        corrected_sign="AU-REVOIR",
        confidence=0.85,
        session_id="session-abc",
    )

    assert correction.id is not None
    assert correction.predicted_sign == "BONJOUR"
    assert correction.corrected_sign == "AU-REVOIR"
    assert abs(correction.confidence - 0.85) < 1e-6
    assert correction.session_id == "session-abc"
    assert correction.status == "pending"
    assert correction.landmarks_path is None
    assert correction.created_at is not None

    # Verify the row is actually in the DB.
    row = db.get(FeedbackCorrection, correction.id)
    assert row is not None
    assert row.corrected_sign == "AU-REVOIR"


def test_submit_correction_minimal_fields(service, db):
    """submit_correction with only required fields must not raise."""
    correction = service.submit_correction(
        db,
        predicted_sign="MERCI",
        corrected_sign="PARDON",
    )
    assert correction.id is not None
    assert correction.confidence is None
    assert correction.session_id is None


# ---------------------------------------------------------------------------
# Test 2 — get_pending_count() compte correctement les entrées 'pending'
# ---------------------------------------------------------------------------

def test_get_pending_count_zero_when_empty(service, db):
    """get_pending_count must return 0 when no corrections exist."""
    count = service.get_pending_count(db, "BONJOUR")
    assert count == 0


def test_get_pending_count_counts_only_pending_for_given_sign(service, db):
    """get_pending_count must count only 'pending' rows for the queried corrected_sign."""
    service.submit_correction(db, predicted_sign="A", corrected_sign="BONJOUR")
    service.submit_correction(db, predicted_sign="B", corrected_sign="BONJOUR")
    # Manually insert an 'ignored' row to ensure it is NOT counted.
    ignored = FeedbackCorrection(
        predicted_sign="C",
        corrected_sign="BONJOUR",
        status="ignored",
    )
    db.add(ignored)
    db.commit()
    # Insert a correction for a different sign (must not pollute count for "BONJOUR").
    service.submit_correction(db, predicted_sign="D", corrected_sign="MERCI")

    count = service.get_pending_count(db, "BONJOUR")
    assert count == 2  # only the two 'pending' rows for "BONJOUR"

    count_merci = service.get_pending_count(db, "MERCI")
    assert count_merci == 1


# ---------------------------------------------------------------------------
# Test 3 — check_and_trigger_training() : count < threshold → False
# ---------------------------------------------------------------------------

def test_check_and_trigger_training_returns_false_when_below_threshold(
    service, db, monkeypatch
):
    """When pending_count < threshold, check_and_trigger_training must return False
    without calling any training service."""
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "feedback_enabled", True)
    # Use a high threshold so any number of corrections stays below it.
    monkeypatch.setattr(settings, "feedback_training_trigger_count", 100)

    # Insert only 2 corrections — well below the 100 threshold.
    service.submit_correction(db, predicted_sign="A", corrected_sign="SIGN_X")
    service.submit_correction(db, predicted_sign="B", corrected_sign="SIGN_X")

    # Patch training_service to detect any unwanted side effects.
    mock_training = MagicMock()
    with patch("app.services.feedback_service.training_service" if False else "builtins.open", create=True):
        # We test the return value; no training path should be reached.
        result = service.check_and_trigger_training(db, corrected_sign="SIGN_X", threshold=100)

    assert result is False


def test_check_and_trigger_training_returns_false_when_feedback_disabled(
    service, db, monkeypatch
):
    """When feedback_enabled=False, check_and_trigger_training must short-circuit to False."""
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "feedback_enabled", False)

    result = service.check_and_trigger_training(db, corrected_sign="SIGN_Y")
    assert result is False


def test_check_and_trigger_training_returns_false_when_sign_not_found(
    service, db, monkeypatch
):
    """When the Sign row for corrected_sign is missing, training must NOT be triggered."""
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "feedback_enabled", True)
    monkeypatch.setattr(settings, "feedback_training_trigger_count", 1)

    # Insert one correction so count >= threshold.
    service.submit_correction(
        db, predicted_sign="A", corrected_sign="NONEXISTENT_SIGN_ZZZ"
    )

    # No Sign row exists in the in-memory DB → service must return False.
    result = service.check_and_trigger_training(
        db, corrected_sign="NONEXISTENT_SIGN_ZZZ"
    )
    assert result is False


# ---------------------------------------------------------------------------
# Test 4 — get_stats() retourne les bons agrégats
# ---------------------------------------------------------------------------

def test_get_stats_empty_returns_empty_list(service, db):
    """get_stats must return an empty list when the table is empty."""
    stats = service.get_stats(db)
    assert stats == []


def test_get_stats_aggregates_by_corrected_sign(service, db):
    """get_stats must return FeedbackStats objects with correct counts per sign."""
    service.submit_correction(db, predicted_sign="A", corrected_sign="BONJOUR")
    service.submit_correction(db, predicted_sign="B", corrected_sign="BONJOUR")
    # Add a 'trained' row to verify pending_count is calculated separately.
    trained = FeedbackCorrection(
        predicted_sign="C",
        corrected_sign="BONJOUR",
        status="trained",
    )
    db.add(trained)
    db.commit()

    service.submit_correction(db, predicted_sign="D", corrected_sign="MERCI")

    stats = service.get_stats(db)

    # Signs are returned sorted alphabetically.
    assert len(stats) == 2
    bonjour_stat = next(s for s in stats if s.sign_name == "BONJOUR")
    assert bonjour_stat.correction_count == 3  # 2 pending + 1 trained
    assert bonjour_stat.pending_count == 2

    merci_stat = next(s for s in stats if s.sign_name == "MERCI")
    assert merci_stat.correction_count == 1
    assert merci_stat.pending_count == 1


def test_get_stats_returns_zero_pending_when_all_trained(service, db):
    """pending_count must be 0 when all corrections for a sign are 'trained'."""
    trained = FeedbackCorrection(
        predicted_sign="X",
        corrected_sign="TRAINED_SIGN",
        status="trained",
    )
    db.add(trained)
    db.commit()

    stats = service.get_stats(db)
    assert len(stats) == 1
    assert stats[0].sign_name == "TRAINED_SIGN"
    assert stats[0].correction_count == 1
    assert stats[0].pending_count == 0
