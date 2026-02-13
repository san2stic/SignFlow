"""Statistics API endpoint tests."""

from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.database import SessionLocal
from app.main import app
from app.models.model_version import ModelVersion
from app.models.sign import Sign
from app.models.training import TrainingSession


def test_accuracy_history_endpoint_returns_model_series() -> None:
    """Accuracy history should include version, accuracy and active flag."""
    with SessionLocal() as db:
        training = TrainingSession(
            mode="full-retrain",
            status="completed",
            progress=100.0,
            config={"epochs": 1, "learning_rate": 1e-4, "augmentation": False},
        )
        db.add(training)
        db.flush()

        version = f"v-history-{uuid4().hex[:8]}"
        model = ModelVersion(
            version=version,
            is_active=False,
            num_classes=3,
            accuracy=0.88,
            class_labels=["a", "b", "c"],
            training_session_id=training.id,
            file_path="/tmp/model-history.pt",
            file_size_mb=1.2,
        )
        db.add(model)
        db.commit()

    with TestClient(app) as client:
        response = client.get("/api/v1/stats/accuracy-history?limit=5")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert any(item["version"] == version for item in payload)


def test_signs_per_category_endpoint_groups_categories() -> None:
    """Category distribution endpoint should aggregate sign counts."""
    suffix = uuid4().hex[:8]
    with SessionLocal() as db:
        first = Sign(
            name=f"Category A {suffix}",
            slug=f"cat-a-{suffix}",
            category="salutations",
            tags=[],
            variants=[],
        )
        second = Sign(
            name=f"Category B {suffix}",
            slug=f"cat-b-{suffix}",
            category="salutations",
            tags=[],
            variants=[],
        )
        third = Sign(
            name=f"Category C {suffix}",
            slug=f"cat-c-{suffix}",
            category=None,
            tags=[],
            variants=[],
        )
        db.add_all([first, second, third])
        db.commit()
        created_ids = [first.id, second.id, third.id]

    with TestClient(app) as client:
        response = client.get("/api/v1/stats/signs-per-category")

    assert response.status_code == 200
    payload = response.json()
    salutations = next(item for item in payload if item["category"] == "salutations")
    assert salutations["count"] >= 2
    assert any(item["category"] == "uncategorized" for item in payload)

    with SessionLocal() as db:
        for sign_id in created_ids:
            sign = db.scalar(select(Sign).where(Sign.id == sign_id))
            if sign:
                db.delete(sign)
        db.commit()
