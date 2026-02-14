"""Pydantic schemas for sign videos."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class VideoCreateMetadata(BaseModel):
    """Optional metadata sent during upload."""

    duration_ms: int = Field(default=0, ge=0)
    fps: int = Field(default=30, ge=1, le=120)
    resolution: str = Field(default="640x480")


class Video(BaseModel):
    """Public video schema exposed by API."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    sign_id: UUID
    file_path: str
    thumbnail_path: str | None
    duration_ms: int
    fps: int
    resolution: str
    type: Literal["training", "reference", "example"]
    landmarks_extracted: bool
    landmarks_path: str | None
    detection_rate: float
    quality_score: float
    is_trainable: bool
    landmark_feature_dim: int
    created_at: datetime
