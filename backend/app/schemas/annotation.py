"""Pydantic schemas for Studio annotation workflow."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# AnnotationSession
# ---------------------------------------------------------------------------


class AnnotationSessionCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=256)
    description: Optional[str] = None
    status: str = Field("active", pattern="^(active|completed|archived)$")


class AnnotationSessionUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=256)
    description: Optional[str] = None
    status: Optional[str] = Field(None, pattern="^(active|completed|archived)$")


class AnnotationSessionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: Optional[str]
    status: str
    created_at: datetime
    updated_at: datetime


class AnnotationSessionWithStats(AnnotationSessionRead):
    """Lecture enrichie avec compteurs stats."""

    video_count: int = 0
    annotation_count: int = 0
    verified_count: int = 0
    coverage_percent: float = 0.0  # % de vidéos ayant au moins une annotation vérifiée


# ---------------------------------------------------------------------------
# VideoAnnotation
# ---------------------------------------------------------------------------


class NMMTags(BaseModel):
    """Marqueurs non-manuels (NMM) sur la face."""

    polar_question: Optional[bool] = None
    wh_question: Optional[bool] = None
    negation: Optional[bool] = None
    eyebrow_raise: Optional[bool] = None
    eyebrow_furrow: Optional[bool] = None
    head_nod: Optional[bool] = None
    head_shake: Optional[bool] = None
    mouth_gesture: Optional[str] = None  # description libre


class VideoAnnotationCreate(BaseModel):
    sign_label: str = Field(..., min_length=1, max_length=120)
    start_frame: int = Field(..., ge=0)
    end_frame: int = Field(..., ge=0)
    start_time_ms: float = Field(..., ge=0)
    end_time_ms: float = Field(..., ge=0)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    is_verified: bool = False
    annotator_notes: Optional[str] = None
    nmm_tags: Optional[dict[str, Any]] = None


class VideoAnnotationUpdate(BaseModel):
    sign_label: Optional[str] = Field(None, min_length=1, max_length=120)
    start_frame: Optional[int] = Field(None, ge=0)
    end_frame: Optional[int] = Field(None, ge=0)
    start_time_ms: Optional[float] = Field(None, ge=0)
    end_time_ms: Optional[float] = Field(None, ge=0)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    is_verified: Optional[bool] = None
    annotator_notes: Optional[str] = None
    nmm_tags: Optional[dict[str, Any]] = None


class VideoAnnotationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    video_id: str
    session_id: int
    sign_label: str
    start_frame: int
    end_frame: int
    start_time_ms: float
    end_time_ms: float
    confidence: Optional[float]
    is_verified: bool
    annotator_notes: Optional[str]
    nmm_tags: Optional[dict[str, Any]]
    created_at: datetime


# ---------------------------------------------------------------------------
# GrammarAnnotation
# ---------------------------------------------------------------------------


class SignSequenceItem(BaseModel):
    label: str
    start: int  # frame index
    end: int  # frame index
    start_ms: Optional[float] = None
    end_ms: Optional[float] = None


class GrammarAnnotationCreate(BaseModel):
    session_id: int
    video_id: str
    sign_sequence: list[SignSequenceItem] = Field(default_factory=list)
    french_translation: str = Field(..., min_length=1)
    grammar_tags: Optional[dict[str, Any]] = None
    annotator_id: Optional[int] = None


class GrammarAnnotationUpdate(BaseModel):
    sign_sequence: Optional[list[SignSequenceItem]] = None
    french_translation: Optional[str] = None
    grammar_tags: Optional[dict[str, Any]] = None


class GrammarAnnotationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: int
    video_id: str
    sign_sequence: list[Any]
    french_translation: str
    grammar_tags: Optional[dict[str, Any]]
    annotator_id: Optional[int]
    created_at: datetime


# ---------------------------------------------------------------------------
# Bulk import / export
# ---------------------------------------------------------------------------


class BulkAnnotationImport(BaseModel):
    """Import JSON ou ELAN d'annotations vidéo."""

    annotations: list[VideoAnnotationCreate]


class AnnotationExport(BaseModel):
    """Payload d'export d'une session d'annotation."""

    session: AnnotationSessionRead
    annotations: list[VideoAnnotationRead]
    grammar_annotations: list[GrammarAnnotationRead]


# ---------------------------------------------------------------------------
# Studio stats
# ---------------------------------------------------------------------------


class StudioStats(BaseModel):
    total_sessions: int
    active_sessions: int
    total_videos_annotated: int
    total_annotations: int
    verified_annotations: int
    total_grammar_annotations: int
