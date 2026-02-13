"""SQLAlchemy model for sign videos and extracted landmarks."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Video(Base):
    """Uploaded training/reference/example video metadata."""

    __tablename__ = "videos"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    sign_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("signs.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )

    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    thumbnail_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    fps: Mapped[int] = mapped_column(Integer, default=30)
    resolution: Mapped[str] = mapped_column(String(32), default="640x480")
    type: Mapped[str] = mapped_column(String(32), default="reference", index=True)

    landmarks_extracted: Mapped[bool] = mapped_column(default=False)
    landmarks_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    sign = relationship("Sign", back_populates="videos")
