"""Service layer for video upload, storage, and retrieval."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import numpy as np
import structlog
from fastapi import HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.ml.features import extract_landmarks_from_video
from app.models.sign import Sign
from app.models.video import Video
from app.schemas.video import Video as VideoSchema
from app.schemas.video import VideoCreateMetadata
from app.utils.video import compress_video, validate_video_upload

logger = structlog.get_logger(__name__)


class MediaService:
    """Handles video lifecycle for sign samples and references."""

    def add_video(
        self,
        db: Session,
        *,
        sign_id: str,
        upload: UploadFile,
        video_type: str,
        metadata: VideoCreateMetadata,
        settings: Settings,
    ) -> VideoSchema:
        """Validate and persist uploaded video, then register metadata in DB."""
        sign = db.scalar(select(Sign).where(Sign.id == sign_id))
        if not sign:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sign not found")

        filename = upload.filename or "upload.mp4"
        try:
            validate_video_upload(
                filename=filename,
                content_type=upload.content_type,
                duration_ms=metadata.duration_ms,
                max_video_seconds=settings.max_video_seconds,
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        max_bytes = settings.max_upload_mb * 1024 * 1024
        raw_bytes = upload.file.read()
        if not raw_bytes:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty")
        if len(raw_bytes) > max_bytes:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File too large")

        extension = Path(filename).suffix.lower() or ".mp4"
        folder = os.path.join(settings.video_dir, video_type)
        os.makedirs(folder, exist_ok=True)

        file_id = str(uuid.uuid4())
        raw_path = os.path.join(folder, f"{file_id}_raw{extension}")
        final_path = os.path.join(folder, f"{file_id}.mp4")
        with open(raw_path, "wb") as file_obj:
            file_obj.write(raw_bytes)

        try:
            compress_video(raw_path, final_path)
            os.remove(raw_path)
        except Exception:
            final_path = raw_path

        # Initialize video record
        video = Video(
            sign_id=sign_id,
            file_path=final_path,
            thumbnail_path=None,
            duration_ms=metadata.duration_ms,
            fps=metadata.fps,
            resolution=metadata.resolution,
            type=video_type,
            landmarks_extracted=False,
            landmarks_path=None,
        )
        db.add(video)

        sign.video_count += 1
        if video_type == "training":
            sign.training_sample_count += 1

        db.commit()
        db.refresh(video)

        # Extract landmarks asynchronously after initial commit
        try:
            logger.info("starting_landmark_extraction", video_id=str(video.id), file_path=final_path)
            result = extract_landmarks_from_video(
                video_path=final_path,
                include_face=False,  # Exclude face for performance
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            # Save landmarks to .npy file
            landmarks_filename = f"{file_id}_landmarks.npy"
            landmarks_path = os.path.join(folder, landmarks_filename)
            np.save(landmarks_path, result.landmarks)

            # Update video record with landmarks metadata
            video.landmarks_extracted = True
            video.landmarks_path = landmarks_path
            db.commit()

            logger.info(
                "landmark_extraction_complete",
                video_id=str(video.id),
                detection_rate=result.detection_rate,
                num_frames=result.num_frames,
                landmarks_shape=result.landmarks.shape,
            )

            if result.detection_rate < 0.8:
                logger.warning(
                    "low_detection_rate_warning",
                    video_id=str(video.id),
                    detection_rate=result.detection_rate,
                    message="Less than 80% of frames have landmarks detected",
                )

        except Exception as e:
            logger.error(
                "landmark_extraction_failed",
                video_id=str(video.id),
                error=str(e),
                exc_info=True,
            )
            # Don't fail the upload if landmark extraction fails
            # The video is still saved, landmarks can be extracted later

        db.refresh(video)
        return VideoSchema.model_validate(video)

    def list_sign_videos(self, db: Session, sign_id: str) -> list[VideoSchema]:
        """Return all videos for a sign sorted by creation timestamp."""
        videos = db.scalars(select(Video).where(Video.sign_id == sign_id).order_by(Video.created_at.desc())).all()
        return [VideoSchema.model_validate(video) for video in videos]

    def delete_video(self, db: Session, video_id: str) -> bool:
        """Delete a video record and local file if present."""
        video = db.get(Video, video_id)
        if not video:
            return False

        sign = db.get(Sign, video.sign_id)
        if sign:
            sign.video_count = max(0, sign.video_count - 1)
            if video.type == "training":
                sign.training_sample_count = max(0, sign.training_sample_count - 1)

        if os.path.exists(video.file_path):
            os.remove(video.file_path)

        db.delete(video)
        db.commit()
        return True

    def get_video_path(self, db: Session, video_id: str) -> str | None:
        """Return underlying file path for streaming."""
        video = db.get(Video, video_id)
        return video.file_path if video else None
