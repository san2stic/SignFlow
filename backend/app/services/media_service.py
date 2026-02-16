"""Service layer for video upload, storage, and retrieval."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import numpy as np
import structlog
from fastapi import HTTPException, UploadFile, status
from pydantic import ValidationError
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

    ALLOWED_VIDEO_TYPES = {"training", "reference", "example"}
    LEGACY_VIDEO_TYPE_ALIASES = {
        "train": "training",
        "training_video": "training",
        "ref": "reference",
        "reference_video": "reference",
        "sample": "example",
    }

    @staticmethod
    def _public_video_path(video_id: str) -> str:
        """Build safe API path for video playback."""
        return f"/api/v1/media/{video_id}/stream"

    def _normalize_video_type(self, raw_type: str | None) -> str:
        """Normalize legacy/unexpected DB values to valid API enum."""
        candidate = (raw_type or "").strip().lower()
        if not candidate:
            return "reference"

        candidate = self.LEGACY_VIDEO_TYPE_ALIASES.get(candidate, candidate)
        if candidate in self.ALLOWED_VIDEO_TYPES:
            return candidate
        return "reference"

    @staticmethod
    def _estimate_quality_score(landmarks: np.ndarray, detection_rate: float) -> tuple[float, bool]:
        """
        Estimate clip quality from landmarks and extraction stats.

        Returns:
            (quality_score, is_trainable_candidate)
        """
        if landmarks.ndim != 2 or landmarks.shape[0] == 0:
            return 0.0, False

        hands = landmarks[:, :126] if landmarks.shape[1] >= 126 else landmarks
        visible_ratio = float(np.mean(np.sum(np.abs(hands), axis=1) > 1e-4))
        if landmarks.shape[0] > 1:
            motion_energy = float(np.mean(np.abs(np.diff(hands, axis=0))))
        else:
            motion_energy = 0.0
        motion_component = float(np.clip(motion_energy / 0.02, 0.0, 1.0))

        quality = (
            0.60 * float(np.clip(detection_rate, 0.0, 1.0))
            + 0.25 * visible_ratio
            + 0.15 * motion_component
        )
        quality = float(np.clip(quality, 0.0, 1.0))
        is_candidate = detection_rate >= 0.8 and quality >= 0.55
        return quality, is_candidate

    def _to_schema(self, video: Video) -> VideoSchema:
        """Convert ORM object into public-safe schema."""
        normalized_type = self._normalize_video_type(video.type)
        if normalized_type != (video.type or "").strip().lower():
            logger.warning(
                "coercing_legacy_video_type",
                video_id=str(video.id),
                original_type=video.type,
                normalized_type=normalized_type,
            )

        return VideoSchema(
            id=video.id,
            sign_id=video.sign_id,
            file_path=self._public_video_path(video.id),
            thumbnail_path=None,
            duration_ms=video.duration_ms,
            fps=video.fps,
            resolution=video.resolution,
            type=normalized_type,
            landmarks_extracted=video.landmarks_extracted,
            landmarks_path=None,
            detection_rate=float(video.detection_rate or 0.0),
            quality_score=float(video.quality_score or 0.0),
            is_trainable=bool(video.is_trainable),
            landmark_feature_dim=int(video.landmark_feature_dim or 225),
            created_at=video.created_at,
        )

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

        extension = Path(filename).suffix.lower() or ".mp4"
        folder = os.path.join(settings.video_dir, video_type)
        os.makedirs(folder, exist_ok=True)

        file_id = str(uuid.uuid4())
        raw_path = os.path.join(folder, f"{file_id}_raw{extension}")
        final_path = os.path.join(folder, f"{file_id}.mp4")
        max_bytes = settings.max_upload_mb * 1024 * 1024
        written_bytes = 0
        chunk_size = 1024 * 1024

        try:
            with open(raw_path, "wb") as file_obj:
                while True:
                    chunk = upload.file.read(chunk_size)
                    if not chunk:
                        break
                    written_bytes += len(chunk)
                    if written_bytes > max_bytes:
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail="File too large",
                        )
                    file_obj.write(chunk)
        except HTTPException:
            if os.path.exists(raw_path):
                os.remove(raw_path)
            raise

        if written_bytes == 0:
            if os.path.exists(raw_path):
                os.remove(raw_path)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty")

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
            detection_rate=0.0,
            quality_score=0.0,
            is_trainable=False,
            landmark_feature_dim=225,
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
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7,
            )

            # Save landmarks to .npy file
            landmarks_filename = f"{file_id}_landmarks.npy"
            landmarks_path = os.path.join(folder, landmarks_filename)
            np.save(landmarks_path, result.landmarks)

            # Update video record with landmarks metadata
            video.landmarks_extracted = True
            video.landmarks_path = landmarks_path
            video.detection_rate = float(result.detection_rate)
            video.landmark_feature_dim = int(result.landmarks.shape[1])
            quality_score, candidate_trainable = self._estimate_quality_score(
                result.landmarks,
                detection_rate=result.detection_rate,
            )
            video.quality_score = quality_score
            video.is_trainable = bool(candidate_trainable)
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
        return self._to_schema(video)

    def list_sign_videos(self, db: Session, sign_id: str) -> list[VideoSchema]:
        """Return all videos for a sign sorted by creation timestamp."""
        videos = db.scalars(select(Video).where(Video.sign_id == sign_id).order_by(Video.created_at.desc())).all()
        payload: list[VideoSchema] = []
        for video in videos:
            try:
                payload.append(self._to_schema(video))
            except ValidationError as exc:
                # Do not fail the whole endpoint because of one malformed legacy row.
                logger.warning(
                    "skipping_invalid_video_row",
                    sign_id=sign_id,
                    video_id=str(video.id),
                    error=str(exc),
                )
        return payload

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
