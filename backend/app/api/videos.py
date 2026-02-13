"""Video labeling endpoints with ML-based suggestions."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy import update
from sqlalchemy.orm import Session

from app.api.deps import enforce_rate_limit, enforce_write_rate_limit, get_db
from app.models.video import Video
from app.ml.similarity import find_similar_videos

logger = structlog.get_logger(__name__)

router = APIRouter()


# Pydantic schemas
class VideoResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    sign_id: Optional[str]
    file_path: str
    thumbnail_path: Optional[str]
    resolution: str
    landmarks_path: Optional[str]
    landmarks_extracted: bool
    duration_ms: int
    fps: int
    created_at: datetime


class UnlabeledVideosResponse(BaseModel):
    items: list[VideoResponse]
    total: int


class LabelRequest(BaseModel):
    sign_id: str


class BulkLabelRequest(BaseModel):
    video_ids: list[str]
    sign_id: str


class SuggestionResponse(BaseModel):
    id: str
    sign_id: Optional[str]
    similarity_score: float
    file_path: str
    thumbnail_path: Optional[str]
    resolution: str
    landmarks_path: Optional[str]
    landmarks_extracted: bool
    duration_ms: int
    fps: int
    created_at: datetime


class SuggestionsResponse(BaseModel):
    target_video_id: str
    suggestions: list[SuggestionResponse]


def _public_media_path(video_id: str) -> str:
    """Build API stream path without exposing local filesystem paths."""
    return f"/api/v1/media/{video_id}/stream"


def _to_video_response(video: Video) -> VideoResponse:
    """Serialize ORM video object into public-safe response shape."""
    return VideoResponse(
        id=video.id,
        sign_id=video.sign_id,
        file_path=_public_media_path(video.id),
        thumbnail_path=None,
        resolution=video.resolution,
        landmarks_path=None,
        landmarks_extracted=video.landmarks_extracted,
        duration_ms=video.duration_ms,
        fps=video.fps,
        created_at=video.created_at,
    )


# Endpoints
@router.get("/unlabeled", response_model=UnlabeledVideosResponse, dependencies=[Depends(enforce_rate_limit)])
def get_unlabeled_videos(db: Session = Depends(get_db)):
    """
    List all videos without sign_id (unlabeled videos).

    Returns:
        UnlabeledVideosResponse: List of unlabeled videos with total count
    """
    videos = db.query(Video).filter(Video.sign_id.is_(None)).all()

    return UnlabeledVideosResponse(
        items=[_to_video_response(v) for v in videos],
        total=len(videos)
    )


@router.patch("/{video_id}/label", response_model=VideoResponse, dependencies=[Depends(enforce_write_rate_limit)])
def label_video(
    video_id: str,
    request: LabelRequest,
    db: Session = Depends(get_db)
):
    """
    Associate a video with a sign (label it).

    Args:
        video_id: Video ID to label
        request: Contains sign_id to assign

    Returns:
        VideoResponse: Updated video

    Raises:
        HTTPException: 404 if video not found
    """
    video = db.query(Video).filter(Video.id == video_id).first()

    if not video:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

    video.sign_id = request.sign_id
    db.commit()
    db.refresh(video)

    return _to_video_response(video)


@router.post("/{video_id}/suggestions", response_model=SuggestionsResponse, dependencies=[Depends(enforce_rate_limit)])
def get_label_suggestions(
    video_id: str,
    threshold: float = Query(default=0.75, ge=0.0, le=1.0),
    top_k: int = Query(default=5, ge=1, le=20),
    db: Session = Depends(get_db)
):
    """
    Get ML-based similar video suggestions for labeling.

    Pipeline:
    1. Load target video landmarks
    2. Find all unlabeled videos with landmarks
    3. Compute similarity scores
    4. Return top K videos above threshold

    Args:
        video_id: Reference video ID
        threshold: Minimum similarity (0.0-1.0, default: 0.75)
        top_k: Maximum suggestions (default: 5)

    Returns:
        SuggestionsResponse: Similar unlabeled videos with scores

    Raises:
        HTTPException: 404 if video not found, 400 if missing landmarks
    """
    target_video = db.query(Video).filter(Video.id == video_id).first()

    if not target_video:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

    if not target_video.landmarks_extracted or not target_video.landmarks_path:
        raise HTTPException(
            status_code=400,
            detail=f"Video {video_id} has no landmarks extracted"
        )

    # Find unlabeled candidates with landmarks
    candidates = db.query(Video).filter(
        Video.sign_id.is_(None),
        Video.landmarks_extracted,
        Video.landmarks_path.isnot(None),
        Video.id != video_id  # Exclude self
    ).all()

    if not candidates:
        return SuggestionsResponse(
            target_video_id=video_id,
            suggestions=[]
        )

    # Compute similarities
    candidate_paths = [Path(v.landmarks_path) for v in candidates]
    results = find_similar_videos(
        target_video_path=Path(target_video.landmarks_path),
        candidate_videos=candidate_paths,
        threshold=threshold,
        top_k=top_k
    )

    # Map paths back to video objects
    path_to_video = {v.landmarks_path: v for v in candidates}
    suggestions = []

    for path, score in results:
        video = path_to_video.get(str(path))
        if video:
            suggestions.append(SuggestionResponse(
                id=video.id,
                sign_id=video.sign_id,
                similarity_score=score,
                file_path=_public_media_path(video.id),
                thumbnail_path=None,
                resolution=video.resolution,
                landmarks_path=None,
                landmarks_extracted=video.landmarks_extracted,
                duration_ms=video.duration_ms,
                fps=video.fps,
                created_at=video.created_at,
            ))

    return SuggestionsResponse(
        target_video_id=video_id,
        suggestions=suggestions
    )


@router.patch("/bulk-label", response_model=dict, dependencies=[Depends(enforce_write_rate_limit)])
def bulk_label_videos(
    request: BulkLabelRequest,
    db: Session = Depends(get_db)
):
    """
    Label multiple videos with the same sign_id.

    Uses SQLAlchemy bulk update for efficiency.

    Args:
        request: Contains video_ids list and sign_id

    Returns:
        dict: Updated count

    Raises:
        HTTPException: 400 if no video_ids provided
    """
    if not request.video_ids:
        raise HTTPException(status_code=400, detail="video_ids list cannot be empty")

    # Bulk update
    stmt = (
        update(Video)
        .where(Video.id.in_(request.video_ids))
        .values(sign_id=request.sign_id)
    )

    result = db.execute(stmt)
    db.commit()

    return {"updated_count": result.rowcount}
