"""REST endpoints for media deletion and streaming."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.services.media_service import MediaService

router = APIRouter()
media_service = MediaService()


@router.delete("/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_video(video_id: str, db: Session = Depends(get_db)) -> None:
    """Delete one video by UUID."""
    deleted = media_service.delete_video(db, video_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")


@router.get("/{video_id}/stream")
def stream_video(video_id: str, db: Session = Depends(get_db)) -> FileResponse:
    """Return a streamable file response for video player."""
    path = media_service.get_video_path(db, video_id)
    if not path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
    return FileResponse(path=path, media_type="video/mp4")
