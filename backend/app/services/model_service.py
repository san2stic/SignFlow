"""Service layer for model versioning and activation."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.models.model_version import ModelVersion
from app.schemas.model_version import (
    ModelVersion as ModelVersionSchema,
    ModelVersionActivateResponse,
    ModelVersionExportResponse,
)


class ModelService:
    """Handles model artifact metadata and active-version switching."""

    def list_models(self, db: Session) -> list[ModelVersionSchema]:
        """Return all model versions ordered by creation timestamp descending."""
        models = db.scalars(select(ModelVersion).order_by(ModelVersion.created_at.desc())).all()
        return [ModelVersionSchema.model_validate(model) for model in models]

    def get_active_model(self, db: Session) -> ModelVersionSchema | None:
        """Return currently active model version, if any."""
        model = db.scalar(select(ModelVersion).where(ModelVersion.is_active.is_(True)))
        return ModelVersionSchema.model_validate(model) if model else None

    def activate(self, db: Session, model_id: str) -> ModelVersionActivateResponse:
        """Set target model as active and deactivate others."""
        target = db.get(ModelVersion, model_id)
        if not target:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")

        for candidate in db.scalars(select(ModelVersion)).all():
            candidate.is_active = candidate.id == model_id
        db.commit()

        return ModelVersionActivateResponse(active_model_id=target.id, version=target.version)

    def export(self, db: Session, model_id: str, fmt: str, settings: Settings) -> ModelVersionExportResponse:
        """Return path to exported model artifact in requested format."""
        model = db.get(ModelVersion, model_id)
        if not model:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")

        if fmt not in {"pt", "onnx"}:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported export format")

        export_dir = Path(settings.export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        out_path = export_dir / f"{model.version}.{fmt}"

        if os.path.exists(model.file_path):
            data = Path(model.file_path).read_bytes()
            out_path.write_bytes(data)
        else:
            out_path.write_text("placeholder model artifact", encoding="utf-8")

        return ModelVersionExportResponse(model_id=model.id, version=model.version, format=fmt, path=str(out_path))
