"""REST endpoints for user feedback / prediction corrections."""

from __future__ import annotations

from typing import Literal, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import enforce_rate_limit, enforce_write_rate_limit, get_db
from app.config import get_settings
from app.models.feedback import FeedbackCorrection
from app.schemas.feedback import FeedbackCreate, FeedbackResponse, FeedbackStats
from app.services.feedback_service import feedback_service

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.post(
    "/corrections",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(enforce_rate_limit), Depends(enforce_write_rate_limit)],
)
def submit_correction(
    body: FeedbackCreate,
    db: Session = Depends(get_db),
) -> FeedbackResponse:
    """Submit a user correction for a model prediction."""
    settings = get_settings()
    if not settings.feedback_enabled:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Feedback is disabled")

    correction = feedback_service.submit_correction(
        db,
        predicted_sign=body.predicted_sign,
        corrected_sign=body.corrected_sign,
        confidence=body.confidence,
        session_id=body.session_id,
        landmarks_data=body.landmarks,
    )

    training_triggered = feedback_service.check_and_trigger_training(
        db, corrected_sign=body.corrected_sign
    )

    return FeedbackResponse(
        id=correction.id,
        predicted_sign=correction.predicted_sign,
        corrected_sign=correction.corrected_sign,
        confidence=correction.confidence,
        landmarks_path=correction.landmarks_path,
        session_id=correction.session_id,
        status=correction.status,  # type: ignore[arg-type]
        created_at=correction.created_at,
        trained_at=correction.trained_at,
        trigger_training=training_triggered,
    )


@router.get(
    "/corrections",
    response_model=list[FeedbackResponse],
    dependencies=[Depends(enforce_rate_limit)],
)
def list_corrections(
    status_filter: Optional[Literal["pending", "trained", "ignored"]] = Query(
        default=None, alias="status"
    ),
    limit: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list[FeedbackResponse]:
    """List stored corrections, optionally filtered by status."""
    query = select(FeedbackCorrection).order_by(FeedbackCorrection.created_at.desc()).limit(limit)
    if status_filter is not None:
        query = query.where(FeedbackCorrection.status == status_filter)

    corrections = db.scalars(query).all()

    return [
        FeedbackResponse(
            id=c.id,
            predicted_sign=c.predicted_sign,
            corrected_sign=c.corrected_sign,
            confidence=c.confidence,
            landmarks_path=c.landmarks_path,
            session_id=c.session_id,
            status=c.status,  # type: ignore[arg-type]
            created_at=c.created_at,
            trained_at=c.trained_at,
            trigger_training=False,
        )
        for c in corrections
    ]


@router.get(
    "/stats",
    response_model=list[FeedbackStats],
    dependencies=[Depends(enforce_rate_limit)],
)
def get_stats(
    db: Session = Depends(get_db),
) -> list[FeedbackStats]:
    """Return per-sign correction statistics."""
    return feedback_service.get_stats(db)


@router.delete(
    "/corrections/{correction_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(enforce_write_rate_limit)],
)
def delete_correction(
    correction_id: int,
    db: Session = Depends(get_db),
) -> None:
    """Mark a correction as ignored (soft-delete)."""
    correction = db.get(FeedbackCorrection, correction_id)
    if correction is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Correction not found")

    correction.status = "ignored"
    db.commit()
