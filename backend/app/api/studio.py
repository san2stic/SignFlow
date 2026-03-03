"""Studio API — annotation sessions, video annotations, grammar annotations, export."""

from __future__ import annotations

import csv
import io
import json
import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import enforce_rate_limit, enforce_write_rate_limit, get_db
from app.models.annotation import (
    AnnotationSession,
    AnnotationSessionVideo,
    GrammarAnnotation,
    VideoAnnotation,
)
from app.models.video import Video
from app.schemas.annotation import (
    AnnotationExport,
    AnnotationSessionCreate,
    AnnotationSessionRead,
    AnnotationSessionUpdate,
    AnnotationSessionWithStats,
    BulkAnnotationImport,
    GrammarAnnotationCreate,
    GrammarAnnotationRead,
    GrammarAnnotationUpdate,
    StudioStats,
    VideoAnnotationCreate,
    VideoAnnotationRead,
    VideoAnnotationUpdate,
)
from app.services.media_service import MediaService

logger = logging.getLogger("signflow.studio")

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_session_or_404(session_id: int, db: Session) -> AnnotationSession:
    obj = db.get(AnnotationSession, session_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Annotation session not found")
    return obj


def _get_annotation_or_404(annotation_id: int, db: Session) -> VideoAnnotation:
    obj = db.get(VideoAnnotation, annotation_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Annotation not found")
    return obj


def _build_session_stats(session_id: int, db: Session) -> dict[str, Any]:
    """Compute stats counters for an annotation session."""
    video_count = db.scalar(
        select(func.count()).where(
            AnnotationSessionVideo.session_id == session_id
        )
    ) or 0
    annotation_count = db.scalar(
        select(func.count()).where(VideoAnnotation.session_id == session_id)
    ) or 0
    verified_count = db.scalar(
        select(func.count()).where(
            VideoAnnotation.session_id == session_id,
            VideoAnnotation.is_verified == True,  # noqa: E712
        )
    ) or 0
    # coverage = % of session videos that have at least one verified annotation
    if video_count > 0:
        covered = db.scalar(
            select(func.count(func.distinct(VideoAnnotation.video_id))).where(
                VideoAnnotation.session_id == session_id,
                VideoAnnotation.is_verified == True,  # noqa: E712
            )
        ) or 0
        coverage = round(covered / video_count * 100, 1)
    else:
        coverage = 0.0
    return {
        "video_count": video_count,
        "annotation_count": annotation_count,
        "verified_count": verified_count,
        "coverage_percent": coverage,
    }


def _session_with_stats(obj: AnnotationSession, db: Session) -> AnnotationSessionWithStats:
    stats = _build_session_stats(obj.id, db)
    return AnnotationSessionWithStats(
        id=obj.id,
        name=obj.name,
        description=obj.description,
        status=obj.status,
        created_at=obj.created_at,
        updated_at=obj.updated_at,
        **stats,
    )


# ---------------------------------------------------------------------------
# Session endpoints
# ---------------------------------------------------------------------------


@router.get("/sessions", response_model=list[AnnotationSessionWithStats], tags=["studio"])
def list_sessions(
    status_filter: Optional[str] = Query(None, alias="status"),
    db: Session = Depends(get_db),
    _rl: None = Depends(enforce_rate_limit),
) -> list[AnnotationSessionWithStats]:
    """List all annotation sessions with stats."""
    q = select(AnnotationSession).order_by(AnnotationSession.created_at.desc())
    if status_filter:
        q = q.where(AnnotationSession.status == status_filter)
    sessions = db.scalars(q).all()
    return [_session_with_stats(s, db) for s in sessions]


@router.post(
    "/sessions",
    response_model=AnnotationSessionRead,
    status_code=status.HTTP_201_CREATED,
    tags=["studio"],
)
def create_session(
    payload: AnnotationSessionCreate,
    db: Session = Depends(get_db),
    _rl: None = Depends(enforce_write_rate_limit),
) -> AnnotationSessionRead:
    """Create a new annotation session."""
    session = AnnotationSession(
        name=payload.name,
        description=payload.description,
        status=payload.status,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return AnnotationSessionRead.model_validate(session)


@router.get("/sessions/{session_id}", response_model=AnnotationSessionWithStats, tags=["studio"])
def get_session(
    session_id: int,
    db: Session = Depends(get_db),
    _rl: None = Depends(enforce_rate_limit),
) -> AnnotationSessionWithStats:
    """Get a single annotation session with stats."""
    obj = _get_session_or_404(session_id, db)
    return _session_with_stats(obj, db)


@router.patch("/sessions/{session_id}", response_model=AnnotationSessionRead, tags=["studio"])
def update_session(
    session_id: int,
    payload: AnnotationSessionUpdate,
    db: Session = Depends(get_db),
    _rl: None = Depends(enforce_write_rate_limit),
) -> AnnotationSessionRead:
    """Patch an annotation session (name, description, status)."""
    obj = _get_session_or_404(session_id, db)
    data = payload.model_dump(exclude_unset=True)
    for key, value in data.items():
        setattr(obj, key, value)
    obj.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(obj)
    return AnnotationSessionRead.model_validate(obj)


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["studio"])
def delete_session(
    session_id: int,
    db: Session = Depends(get_db),
    _rl: None = Depends(enforce_write_rate_limit),
) -> Response:
    """Delete an annotation session and all child records (cascade)."""
    obj = _get_session_or_404(session_id, db)
    db.delete(obj)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------------------
# Videos inside a session
# ---------------------------------------------------------------------------


class VideoWithAnnotationsRead(VideoAnnotationRead if False else object):
    """Thin wrapper to avoid circular import — inline schema."""
    pass


from pydantic import BaseModel, ConfigDict  # noqa: E402


class VideoInSessionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    file_path: str
    thumbnail_path: Optional[str]
    duration_ms: int
    fps: int
    resolution: str
    landmarks_extracted: bool
    detection_rate: float
    quality_score: float
    sign_id: Optional[str]
    annotation_count: int = 0
    verified_count: int = 0


@router.post(
    "/sessions/{session_id}/videos",
    response_model=list[VideoInSessionRead],
    status_code=status.HTTP_201_CREATED,
    tags=["studio"],
)
async def add_videos_to_session(
    session_id: int,
    files: list[UploadFile],
    db: Session = Depends(get_db),
    _rl: None = Depends(enforce_write_rate_limit),
) -> list[VideoInSessionRead]:
    """Upload one or more videos and attach them to the session."""
    obj = _get_session_or_404(session_id, db)
    media_svc = MediaService(db)
    results = []
    for file in files:
        video = await media_svc.save_uploaded_video(file, video_type="annotation")
        db.add(AnnotationSessionVideo(session_id=obj.id, video_id=video.id))
        db.commit()
        results.append(_video_in_session(video, session_id, db))
    return results


@router.get("/sessions/{session_id}/videos", response_model=list[VideoInSessionRead], tags=["studio"])
def list_session_videos(
    session_id: int,
    db: Session = Depends(get_db),
    _rl: None = Depends(enforce_rate_limit),
) -> list[VideoInSessionRead]:
    """List all videos attached to a session with annotation stats."""
    _get_session_or_404(session_id, db)
    join_rows = db.scalars(
        select(AnnotationSessionVideo.video_id).where(
            AnnotationSessionVideo.session_id == session_id
        )
    ).all()
    video_ids = list(join_rows)
    if not video_ids:
        return []
    videos = db.scalars(select(Video).where(Video.id.in_(video_ids))).all()
    return [_video_in_session(v, session_id, db) for v in videos]


def _video_in_session(video: Video, session_id: int, db: Session) -> VideoInSessionRead:
    ann_count = db.scalar(
        select(func.count()).where(
            VideoAnnotation.video_id == video.id,
            VideoAnnotation.session_id == session_id,
        )
    ) or 0
    ver_count = db.scalar(
        select(func.count()).where(
            VideoAnnotation.video_id == video.id,
            VideoAnnotation.session_id == session_id,
            VideoAnnotation.is_verified == True,  # noqa: E712
        )
    ) or 0
    return VideoInSessionRead(
        id=video.id,
        file_path=video.file_path,
        thumbnail_path=video.thumbnail_path,
        duration_ms=video.duration_ms,
        fps=video.fps,
        resolution=video.resolution,
        landmarks_extracted=video.landmarks_extracted,
        detection_rate=video.detection_rate,
        quality_score=video.quality_score,
        sign_id=video.sign_id,
        annotation_count=ann_count,
        verified_count=ver_count,
    )


# ---------------------------------------------------------------------------
# Video annotations
# ---------------------------------------------------------------------------


@router.get(
    "/videos/{video_id}/annotations",
    response_model=list[VideoAnnotationRead],
    tags=["studio"],
)
def list_video_annotations(
    video_id: str,
    session_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    _rl: None = Depends(enforce_rate_limit),
) -> list[VideoAnnotationRead]:
    """List annotations for a video, optionally filtered by session."""
    q = select(VideoAnnotation).where(VideoAnnotation.video_id == video_id)
    if session_id is not None:
        q = q.where(VideoAnnotation.session_id == session_id)
    rows = db.scalars(q.order_by(VideoAnnotation.start_time_ms)).all()
    return [VideoAnnotationRead.model_validate(r) for r in rows]


@router.post(
    "/videos/{video_id}/annotations",
    response_model=VideoAnnotationRead,
    status_code=status.HTTP_201_CREATED,
    tags=["studio"],
)
def create_video_annotation(
    video_id: str,
    session_id: int,
    payload: VideoAnnotationCreate,
    db: Session = Depends(get_db),
    _rl: None = Depends(enforce_write_rate_limit),
) -> VideoAnnotationRead:
    """Create a single annotation on a video."""
    if not db.get(Video, video_id):
        raise HTTPException(status_code=404, detail="Video not found")
    _get_session_or_404(session_id, db)
    annotation = VideoAnnotation(
        video_id=video_id,
        session_id=session_id,
        **payload.model_dump(),
    )
    db.add(annotation)
    db.commit()
    db.refresh(annotation)
    return VideoAnnotationRead.model_validate(annotation)


@router.put(
    "/annotations/{annotation_id}",
    response_model=VideoAnnotationRead,
    tags=["studio"],
)
def update_video_annotation(
    annotation_id: int,
    payload: VideoAnnotationUpdate,
    db: Session = Depends(get_db),
    _rl: None = Depends(enforce_write_rate_limit),
) -> VideoAnnotationRead:
    """Update an annotation."""
    ann = _get_annotation_or_404(annotation_id, db)
    data = payload.model_dump(exclude_unset=True)
    for key, value in data.items():
        setattr(ann, key, value)
    db.commit()
    db.refresh(ann)
    return VideoAnnotationRead.model_validate(ann)


@router.delete(
    "/annotations/{annotation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["studio"],
)
def delete_video_annotation(
    annotation_id: int,
    db: Session = Depends(get_db),
    _rl: None = Depends(enforce_write_rate_limit),
) -> Response:
    """Delete an annotation."""
    ann = _get_annotation_or_404(annotation_id, db)
    db.delete(ann)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/videos/{video_id}/annotations/bulk",
    response_model=list[VideoAnnotationRead],
    status_code=status.HTTP_201_CREATED,
    tags=["studio"],
)
def bulk_create_annotations(
    video_id: str,
    session_id: int,
    payload: BulkAnnotationImport,
    db: Session = Depends(get_db),
    _rl: None = Depends(enforce_write_rate_limit),
) -> list[VideoAnnotationRead]:
    """Bulk import annotations (JSON / ELAN-style)."""
    if not db.get(Video, video_id):
        raise HTTPException(status_code=404, detail="Video not found")
    _get_session_or_404(session_id, db)
    created = []
    for item in payload.annotations:
        ann = VideoAnnotation(video_id=video_id, session_id=session_id, **item.model_dump())
        db.add(ann)
        created.append(ann)
    db.commit()
    for ann in created:
        db.refresh(ann)
    return [VideoAnnotationRead.model_validate(a) for a in created]


@router.post(
    "/videos/{video_id}/annotations/auto-suggest",
    response_model=list[VideoAnnotationRead],
    tags=["studio"],
)
def auto_suggest_annotations(
    video_id: str,
    session_id: int,
    db: Session = Depends(get_db),
    _rl: None = Depends(enforce_write_rate_limit),
) -> list[VideoAnnotationRead]:
    """
    Use the loaded ML pipeline to suggest sign annotations on a video.
    Returns empty list if no model is loaded.
    """
    video = db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    _get_session_or_404(session_id, db)

    # Try to use the loaded pipeline if landmarks are available
    try:
        from app.ml.pipeline import get_pipeline  # type: ignore

        pipeline = get_pipeline()
        if pipeline is None or not video.landmarks_extracted or not video.landmarks_path:
            return []

        import numpy as np

        landmarks = np.load(video.landmarks_path, allow_pickle=False)
        fps = video.fps or 30
        frame_count = landmarks.shape[0]
        suggestions: list[VideoAnnotation] = []

        # Run sliding-window prediction via pipeline internals
        try:
            preds = pipeline.suggest_segments(landmarks)  # type: ignore[attr-defined]
            for pred in preds:
                start_frame = int(pred.get("start_frame", 0))
                end_frame = int(pred.get("end_frame", frame_count - 1))
                ann = VideoAnnotation(
                    video_id=video_id,
                    session_id=session_id,
                    sign_label=str(pred.get("label", "UNKNOWN")),
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_time_ms=round(start_frame / fps * 1000, 2),
                    end_time_ms=round(end_frame / fps * 1000, 2),
                    confidence=float(pred.get("confidence", 0.0)),
                    is_verified=False,
                    annotator_notes="auto-suggest",
                )
                db.add(ann)
                suggestions.append(ann)
            if suggestions:
                db.commit()
                for a in suggestions:
                    db.refresh(a)
        except AttributeError:
            # pipeline.suggest_segments not implemented — return empty
            return []

        return [VideoAnnotationRead.model_validate(a) for a in suggestions]

    except Exception as exc:  # pragma: no cover
        logger.warning("Auto-suggest failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Grammar annotations
# ---------------------------------------------------------------------------


@router.get(
    "/sessions/{session_id}/grammar",
    response_model=list[GrammarAnnotationRead],
    tags=["studio"],
)
def list_grammar_annotations(
    session_id: int,
    db: Session = Depends(get_db),
    _rl: None = Depends(enforce_rate_limit),
) -> list[GrammarAnnotationRead]:
    """List grammar annotations for a session."""
    _get_session_or_404(session_id, db)
    rows = db.scalars(
        select(GrammarAnnotation)
        .where(GrammarAnnotation.session_id == session_id)
        .order_by(GrammarAnnotation.created_at)
    ).all()
    return [GrammarAnnotationRead.model_validate(r) for r in rows]


@router.post(
    "/grammar",
    response_model=GrammarAnnotationRead,
    status_code=status.HTTP_201_CREATED,
    tags=["studio"],
)
def create_grammar_annotation(
    payload: GrammarAnnotationCreate,
    db: Session = Depends(get_db),
    _rl: None = Depends(enforce_write_rate_limit),
) -> GrammarAnnotationRead:
    """Create a grammar annotation."""
    _get_session_or_404(payload.session_id, db)
    if not db.get(Video, payload.video_id):
        raise HTTPException(status_code=404, detail="Video not found")
    ga = GrammarAnnotation(
        session_id=payload.session_id,
        video_id=payload.video_id,
        sign_sequence=[item.model_dump() for item in payload.sign_sequence],
        french_translation=payload.french_translation,
        grammar_tags=payload.grammar_tags,
        annotator_id=payload.annotator_id,
    )
    db.add(ga)
    db.commit()
    db.refresh(ga)
    return GrammarAnnotationRead.model_validate(ga)


@router.put(
    "/grammar/{grammar_id}",
    response_model=GrammarAnnotationRead,
    tags=["studio"],
)
def update_grammar_annotation(
    grammar_id: int,
    payload: GrammarAnnotationUpdate,
    db: Session = Depends(get_db),
    _rl: None = Depends(enforce_write_rate_limit),
) -> GrammarAnnotationRead:
    """Update a grammar annotation."""
    ga = db.get(GrammarAnnotation, grammar_id)
    if not ga:
        raise HTTPException(status_code=404, detail="Grammar annotation not found")
    data = payload.model_dump(exclude_unset=True)
    if "sign_sequence" in data and data["sign_sequence"] is not None:
        data["sign_sequence"] = [item.model_dump() for item in payload.sign_sequence]  # type: ignore
    for key, value in data.items():
        setattr(ga, key, value)
    db.commit()
    db.refresh(ga)
    return GrammarAnnotationRead.model_validate(ga)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


@router.get("/sessions/{session_id}/export", tags=["studio"])
def export_session(
    session_id: int,
    format: str = Query("json", pattern="^(json|csv|elan)$"),
    db: Session = Depends(get_db),
    _rl: None = Depends(enforce_rate_limit),
) -> StreamingResponse:
    """Export a session's annotations in json, csv, or ELAN-like format."""
    obj = _get_session_or_404(session_id, db)
    annotations = db.scalars(
        select(VideoAnnotation).where(VideoAnnotation.session_id == session_id)
    ).all()
    grammar_annotations = db.scalars(
        select(GrammarAnnotation).where(GrammarAnnotation.session_id == session_id)
    ).all()

    if format == "json":
        payload = AnnotationExport(
            session=AnnotationSessionRead.model_validate(obj),
            annotations=[VideoAnnotationRead.model_validate(a) for a in annotations],
            grammar_annotations=[GrammarAnnotationRead.model_validate(g) for g in grammar_annotations],
        )
        content = payload.model_dump_json(indent=2)
        return StreamingResponse(
            io.StringIO(content),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="session_{session_id}.json"'},
        )

    elif format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            ["id", "video_id", "sign_label", "start_frame", "end_frame",
             "start_time_ms", "end_time_ms", "confidence", "is_verified", "notes"]
        )
        for a in annotations:
            writer.writerow(
                [a.id, a.video_id, a.sign_label, a.start_frame, a.end_frame,
                 a.start_time_ms, a.end_time_ms, a.confidence, a.is_verified, a.annotator_notes]
            )
        output.seek(0)
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="session_{session_id}.csv"'},
        )

    elif format == "elan":
        # Minimal ELAN-like XML
        lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<ANNOTATION_DOCUMENT>"]
        lines.append(f'  <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds"/>')
        lines.append("  <TIME_ORDER>")
        ts_id = 1
        time_slots: list[str] = []
        tiers: list[str] = []
        for a in annotations:
            ts_start = f"ts{ts_id}"
            ts_end = f"ts{ts_id + 1}"
            time_slots.append(f'    <TIME_SLOT TIME_SLOT_ID="{ts_start}" TIME_VALUE="{int(a.start_time_ms)}"/>')
            time_slots.append(f'    <TIME_SLOT TIME_SLOT_ID="{ts_end}" TIME_VALUE="{int(a.end_time_ms)}"/>')
            tiers.append(
                f'  <TIER TIER_ID="SIGNS" VIDEO="{a.video_id}">\n'
                f'    <ANNOTATION>\n'
                f'      <ALIGNABLE_ANNOTATION ANNOTATION_ID="a{a.id}" TIME_SLOT_REF1="{ts_start}" TIME_SLOT_REF2="{ts_end}">\n'
                f'        <ANNOTATION_VALUE>{a.sign_label}</ANNOTATION_VALUE>\n'
                f'      </ALIGNABLE_ANNOTATION>\n'
                f'    </ANNOTATION>\n'
                f'  </TIER>'
            )
            ts_id += 2
        lines.extend(time_slots)
        lines.append("  </TIME_ORDER>")
        lines.extend(tiers)
        lines.append("</ANNOTATION_DOCUMENT>")
        content = "\n".join(lines)
        return StreamingResponse(
            io.StringIO(content),
            media_type="application/xml",
            headers={"Content-Disposition": f'attachment; filename="session_{session_id}.eaf"'},
        )

    raise HTTPException(status_code=400, detail="Unsupported export format")  # pragma: no cover


# ---------------------------------------------------------------------------
# Studio global stats
# ---------------------------------------------------------------------------


@router.get("/stats", response_model=StudioStats, tags=["studio"])
def get_studio_stats(
    db: Session = Depends(get_db),
    _rl: None = Depends(enforce_rate_limit),
) -> StudioStats:
    """Return global studio statistics."""
    total_sessions = db.scalar(select(func.count()).select_from(AnnotationSession)) or 0
    active_sessions = db.scalar(
        select(func.count()).where(AnnotationSession.status == "active")
    ) or 0
    total_videos_annotated = db.scalar(
        select(func.count(func.distinct(VideoAnnotation.video_id)))
    ) or 0
    total_annotations = db.scalar(select(func.count()).select_from(VideoAnnotation)) or 0
    verified_annotations = db.scalar(
        select(func.count()).where(VideoAnnotation.is_verified == True)  # noqa: E712
    ) or 0
    total_grammar = db.scalar(select(func.count()).select_from(GrammarAnnotation)) or 0
    return StudioStats(
        total_sessions=total_sessions,
        active_sessions=active_sessions,
        total_videos_annotated=total_videos_annotated,
        total_annotations=total_annotations,
        verified_annotations=verified_annotations,
        total_grammar_annotations=total_grammar,
    )
