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
    sequence_length: int = Field(default=64, ge=8, le=256)
    early_stopping_patience: int = Field(default=15, ge=1, le=100)
    early_stopping_min_delta: float = Field(default=1e-4, ge=0, le=1)
    weight_decay: float = Field(default=0.05, ge=0, le=1)
    classifier_lr_multiplier: float = Field(default=2.0, ge=0.1, le=20)
    label_smoothing: float = Field(default=0.1, ge=0.0, le=0.4)
    warmup_epochs: int = Field(default=3, ge=0, le=100)
    use_mixup: bool = True
    mixup_alpha: float = Field(default=0.3, ge=0.0, le=1.0)
    use_ema: bool = True
    ema_decay: float = Field(default=0.995, ge=0.8, le=0.9999)
    gradient_accumulation_steps: int = Field(default=1, ge=1, le=16)
    temporal_mask_prob: float = Field(default=0.15, ge=0.0, le=1.0)
    temporal_mask_span_ratio: float = Field(default=0.2, ge=0.0, le=0.8)
    use_amp: bool = True
    amp_dtype: Literal["float16", "bfloat16"] = "float16"
    use_swa: bool = True
    swa_start_ratio: float = Field(default=0.75, ge=0.1, le=0.95)
    swa_lr: float | None = Field(default=None, gt=0, le=1)
    freeze_until_layer: int = Field(default=2, ge=0, le=8)
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
