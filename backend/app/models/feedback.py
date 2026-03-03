"""SQLAlchemy model for user feedback / prediction corrections."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.sql import func

from app.database import Base


class FeedbackCorrection(Base):
    """Stores user corrections to model predictions for progressive few-shot retraining."""

    __tablename__ = "feedback_corrections"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Prediction pair
    predicted_sign = Column(String(200), nullable=False, index=True)
    corrected_sign = Column(String(200), nullable=False, index=True)

    # Optional metadata from the inference context
    confidence = Column(Float, nullable=True)
    landmarks_path = Column(String(500), nullable=True)
    session_id = Column(String(200), nullable=True, index=True)

    # Lifecycle status: pending | trained | ignored
    status = Column(String(20), nullable=False, default="pending", index=True)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    trained_at = Column(DateTime(timezone=True), nullable=True)
