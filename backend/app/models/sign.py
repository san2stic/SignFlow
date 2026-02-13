"""SQLAlchemy model for sign dictionary entries and relationships."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Table, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.database import Base

JSONType = JSON().with_variant(JSONB, "postgresql")


sign_relations = Table(
    "sign_relations",
    Base.metadata,
    Column("source_sign_id", String(36), ForeignKey("signs.id", ondelete="CASCADE"), primary_key=True),
    Column("target_sign_id", String(36), ForeignKey("signs.id", ondelete="CASCADE"), primary_key=True),
)


class Sign(Base):
    """Sign dictionary entry with markdown notes and graph relations."""

    __tablename__ = "signs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(120), nullable=False, unique=True)
    slug: Mapped[str] = mapped_column(String(140), nullable=False, unique=True, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    category: Mapped[Optional[str]] = mapped_column(String(80), nullable=True, index=True)

    tags: Mapped[list[str]] = mapped_column(JSONType, default=list)
    variants: Mapped[list[str]] = mapped_column(JSONType, default=list)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    usage_count: Mapped[int] = mapped_column(Integer, default=0)
    video_count: Mapped[int] = mapped_column(Integer, default=0)
    training_sample_count: Mapped[int] = mapped_column(Integer, default=0)
    accuracy: Mapped[Optional[float]] = mapped_column(nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    videos = relationship("Video", back_populates="sign", cascade="all, delete-orphan")

    related_signs = relationship(
        "Sign",
        secondary=sign_relations,
        primaryjoin=id == sign_relations.c.source_sign_id,
        secondaryjoin=id == sign_relations.c.target_sign_id,
        lazy="selectin",
    )
