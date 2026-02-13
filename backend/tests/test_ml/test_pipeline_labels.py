"""Inference pipeline label-loading tests."""

from __future__ import annotations

from uuid import uuid4

from sqlalchemy import select

from app.api.translate import get_or_create_pipeline, reload_pipeline
from app.database import SessionLocal
from app.models.model_version import ModelVersion
from app.models.training import TrainingSession


def test_pipeline_uses_model_class_labels_when_available() -> None:
    """Pipeline should prefer class_labels from the active model metadata."""
    reload_pipeline()

    with SessionLocal() as db:
        for model in db.scalars(select(ModelVersion)).all():
            model.is_active = False

        training = TrainingSession(
            mode="full-retrain",
            status="completed",
            progress=100.0,
            config={"epochs": 1, "learning_rate": 1e-4, "augmentation": False},
        )
        db.add(training)
        db.flush()

        db.add(
            ModelVersion(
                version=f"v-labels-{uuid4().hex[:8]}",
                is_active=True,
                num_classes=2,
                accuracy=0.95,
                class_labels=["lsfb_bonjour", "lsfb_merci"],
                training_session_id=training.id,
                file_path="/tmp/nonexistent-model.pt",
                file_size_mb=1.2,
                parent_version=None,
            )
        )
        db.commit()

    pipeline = get_or_create_pipeline()
    assert pipeline.labels[:3] == ["NONE", "lsfb_bonjour", "lsfb_merci"]

    reload_pipeline()
