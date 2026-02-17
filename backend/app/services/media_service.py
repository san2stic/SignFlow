"""Service layer for video upload, storage, and retrieval."""

from __future__ import annotations

import io
import os
import shutil
import tempfile
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
        file_id = str(uuid.uuid4())
        max_bytes = settings.max_upload_mb * 1024 * 1024

        if settings.use_s3_storage:
            return self._add_video_s3(
                db=db,
                sign=sign,
                sign_id=sign_id,
                upload=upload,
                video_type=video_type,
                metadata=metadata,
                settings=settings,
                file_id=file_id,
                extension=extension,
                max_bytes=max_bytes,
            )

        return self._add_video_local(
            db=db,
            sign=sign,
            sign_id=sign_id,
            upload=upload,
            video_type=video_type,
            metadata=metadata,
            settings=settings,
            file_id=file_id,
            extension=extension,
            max_bytes=max_bytes,
        )

    def _add_video_local(
        self,
        *,
        db: Session,
        sign,
        sign_id: str,
        upload: UploadFile,
        video_type: str,
        metadata: VideoCreateMetadata,
        settings: Settings,
        file_id: str,
        extension: str,
        max_bytes: int,
    ) -> VideoSchema:
        """Upload vidéo vers le filesystem local (mode développement)."""
        folder = os.path.join(settings.video_dir, video_type)
        os.makedirs(folder, exist_ok=True)

        raw_path = os.path.join(folder, f"{file_id}_raw{extension}")
        final_path = os.path.join(folder, f"{file_id}.mp4")
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

        try:
            logger.info("starting_landmark_extraction", video_id=str(video.id), file_path=final_path)
            result = extract_landmarks_from_video(
                video_path=final_path,
                include_face=False,
                include_face_expressions=True,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7,
            )

            landmarks_filename = f"{file_id}_landmarks.npy"
            landmarks_path = os.path.join(folder, landmarks_filename)
            np.save(landmarks_path, result.landmarks)

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

        db.refresh(video)
        return self._to_schema(video)

    def _add_video_s3(
        self,
        *,
        db: Session,
        sign,
        sign_id: str,
        upload: UploadFile,
        video_type: str,
        metadata: VideoCreateMetadata,
        settings: Settings,
        file_id: str,
        extension: str,
        max_bytes: int,
    ) -> VideoSchema:
        """Upload vidéo vers MinIO/S3 (mode production serveur)."""
        from app.storage.factory import get_storage

        storage = get_storage()
        tmp_dir = Path(tempfile.mkdtemp(prefix="signflow_upload_"))

        try:
            raw_path = tmp_dir / f"{file_id}_raw{extension}"
            final_path = tmp_dir / f"{file_id}.mp4"
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
                raise

            if written_bytes == 0:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty")

            try:
                compress_video(str(raw_path), str(final_path))
                raw_path.unlink(missing_ok=True)
            except Exception:
                final_path = raw_path

            # Clé S3 (stockée en DB à la place du chemin local)
            s3_key = f"videos/{video_type}/{file_id}.mp4"
            storage.upload_file(final_path, s3_key, settings.s3_bucket_videos)
            logger.info("s3_video_uploaded", key=s3_key, bucket=settings.s3_bucket_videos)

            video = Video(
                sign_id=sign_id,
                file_path=s3_key,  # Clé S3, pas un chemin local
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

            # Extraction landmarks sur fichier local temporaire
            try:
                logger.info("starting_landmark_extraction", video_id=str(video.id), s3_key=s3_key)
                result = extract_landmarks_from_video(
                    video_path=str(final_path),
                    include_face=False,
                    include_face_expressions=True,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7,
                )

                # Upload landmarks vers S3
                buf = io.BytesIO()
                np.save(buf, result.landmarks)
                lm_key = f"videos/{video_type}/{file_id}_landmarks.npy"
                storage.upload_bytes(buf.getvalue(), lm_key, settings.s3_bucket_videos, "application/octet-stream")

                video.landmarks_extracted = True
                video.landmarks_path = lm_key  # Clé S3
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

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

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
        """Delete a video record and associated storage (local or S3)."""
        from app.config import get_settings

        video = db.get(Video, video_id)
        if not video:
            return False

        sign = db.get(Sign, video.sign_id)
        if sign:
            sign.video_count = max(0, sign.video_count - 1)
            if video.type == "training":
                sign.training_sample_count = max(0, sign.training_sample_count - 1)

        settings = get_settings()
        if settings.use_s3_storage:
            from app.storage.factory import get_storage
            storage = get_storage()
            storage.delete_object(video.file_path, settings.s3_bucket_videos)
            if video.landmarks_path:
                storage.delete_object(video.landmarks_path, settings.s3_bucket_videos)
        else:
            if video.file_path and os.path.exists(video.file_path):
                os.remove(video.file_path)
            if video.landmarks_path and os.path.exists(video.landmarks_path):
                os.remove(video.landmarks_path)

        db.delete(video)
        db.commit()
        return True

    def get_video_path(self, db: Session, video_id: str) -> str | None:
        """Return local file path for streaming (mode dev uniquement).

        En mode S3, retourne None — utiliser get_video_url() à la place.
        """
        from app.config import get_settings
        settings = get_settings()
        if settings.use_s3_storage:
            return None
        video = db.get(Video, video_id)
        return video.file_path if video else None

    def get_video_url(self, db: Session, video_id: str) -> str | None:
        """Return a URL for video streaming.

        - Mode S3 : presigned URL MinIO (accès direct, expirée après s3_presigned_url_expiry secondes)
        - Mode local : URL API interne /api/v1/media/{video_id}/stream
        """
        from app.config import get_settings
        settings = get_settings()
        video = db.get(Video, video_id)
        if not video:
            return None
        if settings.use_s3_storage:
            from app.storage.factory import get_storage
            return get_storage().get_presigned_url(
                video.file_path,
                settings.s3_bucket_videos,
                expiry=settings.s3_presigned_url_expiry,
            )
        return self._public_video_path(video_id)
