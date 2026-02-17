"""REST endpoints for media deletion and streaming."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.api.deps import enforce_rate_limit, enforce_write_rate_limit, get_db
from app.config import get_settings
from app.services.media_service import MediaService

router = APIRouter()
media_service = MediaService()


@router.delete("/{video_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(enforce_write_rate_limit)])
def delete_video(video_id: str, db: Session = Depends(get_db)) -> None:
    """Delete one video by UUID."""
    deleted = media_service.delete_video(db, video_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")


@router.get("/{video_id}/stream", dependencies=[Depends(enforce_rate_limit)])
def stream_video(video_id: str, db: Session = Depends(get_db)):
    """Stream video : redirect vers presigned URL S3 (prod) ou FileResponse local (dev)."""
    settings = get_settings()

    if settings.use_s3_storage:
        url = media_service.get_video_url(db, video_id)
        if not url:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
        # 307 Temporary Redirect : le client récupère la vidéo directement depuis MinIO
        return RedirectResponse(url=url, status_code=status.HTTP_307_TEMPORARY_REDIRECT)

    path = media_service.get_video_path(db, video_id)
    if not path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
    return FileResponse(path=path, media_type="video/mp4")
