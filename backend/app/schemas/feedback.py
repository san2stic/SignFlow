"""Pydantic schemas for user feedback / prediction corrections."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class FeedbackCreate(BaseModel):
    """Payload for submitting a prediction correction."""

    predicted_sign: str = Field(..., min_length=1, max_length=200, description="Sign predicted by the model (incorrect)")
    corrected_sign: str = Field(..., min_length=1, max_length=200, description="Correct sign label provided by the user")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0, description="Confidence of the wrong prediction")
    session_id: str | None = Field(default=None, max_length=200, description="WebSocket session identifier")
    landmarks: list | None = Field(default=None, description="Optional serialised landmark frames for retraining")


class FeedbackResponse(BaseModel):
    """Full feedback correction response."""

    id: int
    predicted_sign: str
    corrected_sign: str
    confidence: float | None
    landmarks_path: str | None
    session_id: str | None
    status: Literal["pending", "trained", "ignored"]
    created_at: datetime
    trained_at: datetime | None
    trigger_training: bool = False

    model_config = {"from_attributes": True}


class FeedbackStats(BaseModel):
    """Per-sign correction statistics."""

    sign_name: str
    correction_count: int
    pending_count: int
