"""SQLAlchemy models for the Studio annotation workflow."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.database import Base

JSONType = JSON().with_variant(JSONB, "postgresql")


class AnnotationSession(Base):
    """Session d'annotation d'un corpus de vidéos."""

    __tablename__ = "annotation_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="active", nullable=False, index=True)
    # status values: "active" | "completed" | "archived"

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relations
    annotations: Mapped[list["VideoAnnotation"]] = relationship(
        "VideoAnnotation", back_populates="session", cascade="all, delete-orphan"
    )
    grammar_annotations: Mapped[list["GrammarAnnotation"]] = relationship(
        "GrammarAnnotation", back_populates="session", cascade="all, delete-orphan"
    )
    session_videos: Mapped[list["AnnotationSessionVideo"]] = relationship(
        "AnnotationSessionVideo", back_populates="session", cascade="all, delete-orphan"
    )


class AnnotationSessionVideo(Base):
    """Table d'association entre sessions d'annotation et vidéos."""

    __tablename__ = "annotation_session_videos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("annotation_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    video_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    added_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    session: Mapped["AnnotationSession"] = relationship(
        "AnnotationSession", back_populates="session_videos"
    )
    video = relationship("Video")


class VideoAnnotation(Base):
    """Annotation d'une vidéo — mapping temporel signe/frame."""

    __tablename__ = "video_annotations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    session_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("annotation_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sign_label: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    start_frame: Mapped[int] = mapped_column(Integer, nullable=False)
    end_frame: Mapped[int] = mapped_column(Integer, nullable=False)
    start_time_ms: Mapped[float] = mapped_column(Float, nullable=False)
    end_time_ms: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    annotator_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    nmm_tags: Mapped[Optional[dict]] = mapped_column(JSONType, nullable=True)
    # nmm_tags example: {"polar_question": true, "negation": false, "eyebrow_raise": true}

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    session: Mapped["AnnotationSession"] = relationship(
        "AnnotationSession", back_populates="annotations"
    )
    video = relationship("Video")


class GrammarAnnotation(Base):
    """Annotation grammaticale d'une phrase signée complète."""

    __tablename__ = "grammar_annotations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("annotation_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    video_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sign_sequence: Mapped[list] = mapped_column(JSONType, default=list, nullable=False)
    # sign_sequence example: [{"label": "JE", "start": 0, "end": 30}, ...]
    french_translation: Mapped[str] = mapped_column(Text, nullable=False)
    grammar_tags: Mapped[Optional[dict]] = mapped_column(JSONType, nullable=True)
    # grammar_tags: BIO CRF tags
    annotator_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    session: Mapped["AnnotationSession"] = relationship(
        "AnnotationSession", back_populates="grammar_annotations"
    )
    video = relationship("Video")
    annotator = relationship("User")
