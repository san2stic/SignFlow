"""REST endpoints for model version listing, activation, and export."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import get_app_settings, get_db
from app.api.translate import reload_pipeline
from app.config import Settings
from app.schemas.model_version import ModelVersion, ModelVersionActivateResponse, ModelVersionExportResponse
from app.services.model_service import ModelService

router = APIRouter()
model_service = ModelService()


@router.get("", response_model=list[ModelVersion])
def list_models(db: Session = Depends(get_db)) -> list[ModelVersion]:
    """List all model versions."""
    return model_service.list_models(db)


@router.get("/active", response_model=Optional[ModelVersion])
def active_model(db: Session = Depends(get_db)) -> Optional[ModelVersion]:
    """Get currently active model version."""
    return model_service.get_active_model(db)


@router.post("/{model_id}/activate", response_model=ModelVersionActivateResponse)
def activate_model(model_id: str, db: Session = Depends(get_db)) -> ModelVersionActivateResponse:
    """Activate chosen model version for production inference."""
    response = model_service.activate(db, model_id)
    reload_pipeline()
    return response


@router.get("/{model_id}/export", response_model=ModelVersionExportResponse)
def export_model(
    model_id: str,
    format: str = Query(default="pt"),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
) -> ModelVersionExportResponse:
    """Export one model artifact in .pt or .onnx format."""
    return model_service.export(db, model_id=model_id, fmt=format, settings=settings)
