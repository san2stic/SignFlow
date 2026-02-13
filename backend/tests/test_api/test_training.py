"""Training API behavior tests."""

from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.database import SessionLocal
from app.main import app
from app.models.model_version import ModelVersion
from app.models.training import TrainingSession as TrainingSessionModel


def test_few_shot_requires_sign_id() -> None:
    """few-shot sessions must provide sign_id."""
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/training/sessions",
            json={
                "mode": "few-shot",
                "config": {"epochs": 5, "learning_rate": 1e-4, "augmentation": True},
            },
        )

    assert response.status_code == 422


def test_deploy_training_session_activates_model_when_ready() -> None:
    """Deploy endpoint should activate model produced by a completed ready session."""
    ready_version = f"v-ready-model-{uuid4().hex[:8]}"
    with SessionLocal() as db:
        baseline = TrainingSessionModel(
            mode="full-retrain",
            status="completed",
            progress=100.0,
            config={"epochs": 1, "learning_rate": 1e-4, "augmentation": False, "min_deploy_accuracy": 0.85},
            metrics={"loss": 0.9, "accuracy": 0.1, "val_accuracy": 0.1},
        )
        db.add(baseline)
        db.flush()
        db.add(
            ModelVersion(
                version=f"v-active-{uuid4().hex[:8]}",
                is_active=True,
                num_classes=2,
                accuracy=0.1,
                class_labels=["a", "b"],
                training_session_id=baseline.id,
                file_path="/tmp/model-active.pt",
                file_size_mb=1.0,
            )
        )

        session = TrainingSessionModel(
            mode="few-shot",
            status="completed",
            progress=100.0,
            config={"epochs": 5, "learning_rate": 1e-4, "augmentation": True, "min_deploy_accuracy": 0.85},
            metrics={
                "loss": 0.2,
                "accuracy": 0.9,
                "val_accuracy": 0.9,
                "current_epoch": 5,
                "deployment_ready": True,
                "deploy_threshold": 0.85,
                "final_val_accuracy": 0.9,
                "recommended_next_action": "deploy",
            },
            model_version_produced=ready_version,
        )
        db.add(session)
        db.flush()
        db.add(
            ModelVersion(
                version=ready_version,
                is_active=False,
                num_classes=3,
                accuracy=0.9,
                class_labels=["a", "b", "c"],
                training_session_id=session.id,
                file_path="/tmp/model-ready.pt",
                file_size_mb=1.2,
            )
        )
        db.commit()
        session_id = session.id

    with TestClient(app) as client:
        response = client.post(f"/api/v1/training/sessions/{session_id}/deploy")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "deployed"
    assert payload["version"] == ready_version

    with SessionLocal() as db:
        active_versions = db.scalars(select(ModelVersion).where(ModelVersion.is_active.is_(True))).all()
        assert len(active_versions) == 1
        assert active_versions[0].version == ready_version


def test_deploy_training_session_blocked_when_not_ready() -> None:
    """Deploy endpoint should reject sessions below deployment threshold."""
    not_ready_version = f"v-not-ready-model-{uuid4().hex[:8]}"
    with SessionLocal() as db:
        session = TrainingSessionModel(
            mode="few-shot",
            status="completed",
            progress=100.0,
            config={"epochs": 5, "learning_rate": 1e-4, "augmentation": True, "min_deploy_accuracy": 0.85},
            metrics={
                "loss": 0.4,
                "accuracy": 0.7,
                "val_accuracy": 0.72,
                "current_epoch": 5,
                "deployment_ready": False,
                "deploy_threshold": 0.85,
                "final_val_accuracy": 0.72,
                "recommended_next_action": "collect_more_examples",
            },
            model_version_produced=not_ready_version,
        )
        db.add(session)
        db.flush()
        db.add(
            ModelVersion(
                version=not_ready_version,
                is_active=False,
                num_classes=3,
                accuracy=0.72,
                class_labels=["a", "b", "c"],
                training_session_id=session.id,
                file_path="/tmp/model-not-ready.pt",
                file_size_mb=1.1,
            )
        )
        db.commit()
        session_id = session.id

    with TestClient(app) as client:
        response = client.post(f"/api/v1/training/sessions/{session_id}/deploy")

    assert response.status_code == 409
    assert "threshold" in response.json()["detail"].lower()
