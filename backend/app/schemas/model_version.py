"""Pydantic schemas for model version operations."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelVersion(BaseModel):
    """Public model version schema."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    version: str
    is_active: bool
    num_classes: int
    accuracy: float
    class_labels: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    training_session_id: UUID
    file_path: str
    file_size_mb: float
    created_at: datetime
    parent_version: str | None = None


class ModelVersionActivateResponse(BaseModel):
    """Response returned when activating a model version."""

    active_model_id: UUID
    version: str


class ModelVersionExportResponse(BaseModel):
    """Response returned for model export requests."""

    model_id: UUID
    version: str
    format: str
    path: str
