"""SQLAlchemy model for versioned ML artifacts."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from app.database import Base

JSONType = JSON().with_variant(JSONB, "postgresql")


class ModelVersion(Base):
    """Tracks a model artifact with activation state and lineage."""

    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    version: Mapped[str] = mapped_column(String(24), unique=True, index=True, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    num_classes: Mapped[int] = mapped_column(default=1)
    accuracy: Mapped[float] = mapped_column(default=0.0)
    class_labels: Mapped[list[str]] = mapped_column(JSONType, default=list)
    artifact_metadata: Mapped[dict] = mapped_column("metadata", JSONType, default=dict, deferred=True)

    training_session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("training_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    file_size_mb: Mapped[float] = mapped_column(default=0.0)

    parent_version: Mapped[Optional[str]] = mapped_column(String(24), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
