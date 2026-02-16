"""Model API behavior tests."""

from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

import app.api.models as models_api
import app.services.model_service as model_service_module
from app.database import SessionLocal
from app.main import app
from app.models.model_version import ModelVersion
from app.models.training import TrainingSession


def _create_training_session(db: Session) -> TrainingSession:
    session = TrainingSession(
        mode="full-retrain",
        status="completed",
        progress=100.0,
        config={"epochs": 1, "learning_rate": 1e-4, "augmentation": False},
    )
    db.add(session)
    db.flush()
    return session


def _create_model(db: Session, training_session_id: str, *, is_active: bool) -> ModelVersion:
    model = ModelVersion(
        version=f"v-test-{uuid4().hex[:8]}",
        is_active=is_active,
        num_classes=2,
        accuracy=0.9,
        class_labels=["lsfb_bonjour", "lsfb_merci"],
        training_session_id=training_session_id,
        file_path="/tmp/nonexistent-model.pt",
        file_size_mb=1.0,
        parent_version=None,
    )
    db.add(model)
    db.flush()
    return model


def test_activate_model_reloads_pipeline(monkeypatch) -> None:
    """Activating a model should schedule pipeline reload."""
    calls = {"count": 0}

    def fake_reload_pipeline() -> None:
        calls["count"] += 1

    monkeypatch.setattr(models_api, "reload_pipeline", fake_reload_pipeline)

    with SessionLocal() as db:
        for model in db.scalars(select(ModelVersion)).all():
            model.is_active = False
        training = _create_training_session(db)
        target = _create_model(db, training.id, is_active=False)
        db.commit()
        model_id = target.id

    with TestClient(app) as client:
        response = client.post(f"/api/v1/models/{model_id}/activate")

    assert response.status_code == 200
    assert calls["count"] == 1


def test_active_endpoint_ignores_new_inactive_models() -> None:
    """Newest inactive model should not become active until explicit activation."""
    with SessionLocal() as db:
        for model in db.scalars(select(ModelVersion)).all():
            model.is_active = False

        first_training = _create_training_session(db)
        active_model = _create_model(db, first_training.id, is_active=True)

        second_training = _create_training_session(db)
        inactive_model = _create_model(db, second_training.id, is_active=False)
        db.commit()
        active_model_id = active_model.id
        inactive_version = inactive_model.version

    with TestClient(app) as client:
        response = client.get("/api/v1/models/active")

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == active_model_id
    assert payload["version"] != inactive_version


def test_activate_model_attempts_registry_promotion(monkeypatch) -> None:
    """Activation should attempt registry production promotion when metadata is present."""
    calls = {"promote": 0}

    class _FakeRegistry:
        def promote_to_production(self, *, registry_version: str):
            calls["promote"] += 1
            return {"promoted": True, "registry_version": registry_version}

    monkeypatch.setattr(
        model_service_module,
        "create_default_registry",
        lambda **_kwargs: _FakeRegistry(),
    )

    with SessionLocal() as db:
        for model in db.scalars(select(ModelVersion)).all():
            model.is_active = False
        training = _create_training_session(db)
        target = _create_model(db, training.id, is_active=False)
        target.artifact_metadata = {
            "registry": {"registry_version": "9", "model_name": "signflow-registry"}
        }
        db.commit()
        model_id = target.id

    with TestClient(app) as client:
        response = client.post(f"/api/v1/models/{model_id}/activate")

    assert response.status_code == 200
    assert calls["promote"] == 1
