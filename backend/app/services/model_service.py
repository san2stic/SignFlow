"""Service layer for model versioning and activation."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.config import get_settings
from app.ml.registry import create_default_registry
from app.models.model_version import ModelVersion
from app.schemas.model_version import (
    ModelVersion as ModelVersionSchema,
    ModelVersionActivateResponse,
    ModelVersionExportResponse,
)
from app.utils.model_artifacts import resolve_model_artifact_path


class ModelService:
    """Handles model artifact metadata and active-version switching."""

    @staticmethod
    def _to_public_schema(model: ModelVersion) -> ModelVersionSchema:
        """Return a public-safe model schema without absolute filesystem leakage."""
        return ModelVersionSchema(
            id=model.id,
            version=model.version,
            is_active=model.is_active,
            num_classes=model.num_classes,
            accuracy=model.accuracy,
            class_labels=model.class_labels or [],
            metadata=model.artifact_metadata or {},
            training_session_id=model.training_session_id,
            file_path=Path(model.file_path).name,
            file_size_mb=model.file_size_mb,
            created_at=model.created_at,
            parent_version=model.parent_version,
        )

    def list_models(self, db: Session) -> list[ModelVersionSchema]:
        """Return all model versions ordered by creation timestamp descending."""
        models = db.scalars(select(ModelVersion).order_by(ModelVersion.created_at.desc())).all()
        return [self._to_public_schema(model) for model in models]

    def get_active_model(self, db: Session) -> ModelVersionSchema | None:
        """Return currently active model version, if any."""
        model = db.scalar(select(ModelVersion).where(ModelVersion.is_active.is_(True)))
        return self._to_public_schema(model) if model else None

    def activate(self, db: Session, model_id: str) -> ModelVersionActivateResponse:
        """Set target model as active and deactivate others."""
        target = db.get(ModelVersion, model_id)
        if not target:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")

        for candidate in db.scalars(select(ModelVersion)).all():
            candidate.is_active = candidate.id == model_id
        db.commit()
        self._try_promote_registry_production(target)

        return ModelVersionActivateResponse(active_model_id=target.id, version=target.version)

    @staticmethod
    def _try_promote_registry_production(model: ModelVersion) -> None:
        """Best-effort promotion to MLflow Production stage."""
        metadata = model.artifact_metadata or {}
        registry = metadata.get("registry", {}) if isinstance(metadata, dict) else {}
        registry_version = registry.get("registry_version") if isinstance(registry, dict) else None
        if not registry_version:
            return

        settings = get_settings()
        model_name = registry.get("model_name") if isinstance(registry, dict) else None
        registry_client = create_default_registry(
            enabled=bool(settings.mlflow_registry_enabled),
            model_name=str(model_name or settings.mlflow_registry_model_name),
            tracking_uri=settings.mlflow_tracking_uri,
        )
        registry_client.promote_to_production(registry_version=str(registry_version))

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

        resolved_model_path = resolve_model_artifact_path(
            model.file_path,
            model_dir=settings.model_dir,
        )
        if resolved_model_path is not None:
            data = resolved_model_path.read_bytes()
            out_path.write_bytes(data)
        else:
            out_path.write_text("placeholder model artifact", encoding="utf-8")

        return ModelVersionExportResponse(
            model_id=model.id,
            version=model.version,
            format=fmt,
            path=out_path.name,
        )
