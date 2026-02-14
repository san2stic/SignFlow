"""Inference pipeline label-loading tests."""

from __future__ import annotations

from uuid import uuid4

from sqlalchemy import select
import torch

from app.api.translate import get_or_create_pipeline, reload_pipeline
from app.database import SessionLocal
from app.ml.model import SignTransformer
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
    assert pipeline.labels[:2] == ["lsfb_bonjour", "lsfb_merci"]

    reload_pipeline()


def test_pipeline_uses_checkpoint_sequence_length(tmp_path) -> None:
    """Translate pipeline should use sequence_length from checkpoint config when available."""
    reload_pipeline()
    model_file = tmp_path / "model_seq96.pt"
    model = SignTransformer(num_features=469, num_classes=2, use_multiscale_stem=False, use_cosine_head=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": 2,
            "num_features": 469,
            "d_model": model.d_model,
            "nhead": model.nhead,
            "num_layers": model.num_layers,
            "class_labels": ["[NONE]", "lsfb_bonjour"],
            "config": {"sequence_length": 96},
        },
        model_file,
    )

    with SessionLocal() as db:
        for model_row in db.scalars(select(ModelVersion)).all():
            model_row.is_active = False

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
                version=f"v-seq-{uuid4().hex[:8]}",
                is_active=True,
                num_classes=2,
                accuracy=0.91,
                class_labels=["[NONE]", "lsfb_bonjour"],
                training_session_id=training.id,
                file_path=str(model_file),
                file_size_mb=1.0,
                parent_version=None,
            )
        )
        db.commit()

    pipeline = get_or_create_pipeline()
    assert pipeline.seq_len == 96
    reload_pipeline()
