"""Pydantic schemas for training orchestration."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class TrainingConfig(BaseModel):
    """Training hyperparameters and augmentation flags."""

    epochs: int = Field(default=50, ge=1, le=500)
    learning_rate: float = Field(default=1e-4, gt=0, le=1)
    augmentation: bool = True
    min_deploy_accuracy: float = Field(default=0.85, ge=0.0, le=1.0)


class TrainingMetrics(BaseModel):
    """Session metrics at current progress state."""

    loss: float = 0.0
    accuracy: float = 0.0
    val_accuracy: float = 0.0


class TrainingSessionCreate(BaseModel):
    """Payload for starting a training session."""

    sign_id: UUID | None = None
    mode: Literal["few-shot", "full-retrain"]
    config: TrainingConfig = TrainingConfig()

    @model_validator(mode="after")
    def validate_sign_id_for_few_shot(self) -> "TrainingSessionCreate":
        """few-shot training requires a target sign id."""
        if self.mode == "few-shot" and self.sign_id is None:
            raise ValueError("sign_id is required when mode is 'few-shot'")
        return self


class TrainingSession(BaseModel):
    """Training session state returned by API."""

    id: UUID
    sign_id: UUID | None
    mode: Literal["few-shot", "full-retrain"]
    status: Literal["queued", "preprocessing", "training", "validating", "completed", "failed"]
    progress: float
    config: TrainingConfig
    metrics: TrainingMetrics | None = None
    model_version_produced: str | None = None
    deployment_ready: bool = False
    deploy_threshold: float = 0.85
    final_val_accuracy: float | None = None
    recommended_next_action: Literal["deploy", "collect_more_examples", "wait", "review_error"] = "wait"
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
