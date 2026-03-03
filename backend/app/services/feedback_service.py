"""Service layer for user feedback / prediction corrections."""

from __future__ import annotations

import uuid as _uuid_module
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import structlog
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models.feedback import FeedbackCorrection
from app.schemas.feedback import FeedbackStats

logger = structlog.get_logger(__name__)


class FeedbackService:
    """Persist correction feedback and optionally trigger few-shot retraining."""

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def submit_correction(
        self,
        db: Session,
        *,
        predicted_sign: str,
        corrected_sign: str,
        confidence: Optional[float] = None,
        session_id: Optional[str] = None,
        landmarks_data: Optional[list] = None,
    ) -> FeedbackCorrection:
        """Store a user correction in the database.

        If ``landmarks_data`` is provided the raw landmark array is saved to
        disk and the resulting path is stored on the record.
        """
        landmarks_path: Optional[str] = None
        if landmarks_data:
            try:
                landmarks_path = self.save_landmarks(landmarks_data, sign_name=corrected_sign)
            except Exception as exc:  # noqa: BLE001
                logger.warning("feedback_landmarks_save_failed", error=str(exc))

        correction = FeedbackCorrection(
            predicted_sign=predicted_sign,
            corrected_sign=corrected_sign,
            confidence=confidence,
            landmarks_path=landmarks_path,
            session_id=session_id,
            status="pending",
        )
        db.add(correction)
        db.commit()
        db.refresh(correction)

        logger.info(
            "feedback_correction_stored",
            id=correction.id,
            predicted=predicted_sign,
            corrected=corrected_sign,
        )
        return correction

    def get_pending_count(self, db: Session, sign_name: str) -> int:
        """Return the number of pending corrections for a given corrected sign."""
        result = db.scalar(
            select(func.count(FeedbackCorrection.id)).where(
                FeedbackCorrection.corrected_sign == sign_name,
                FeedbackCorrection.status == "pending",
            )
        )
        return int(result or 0)

    def get_stats(self, db: Session) -> list[FeedbackStats]:
        """Return per-sign correction statistics (total + pending counts).

        Uses an explicit Python aggregation to stay DB-agnostic (SQLite / PostgreSQL).
        """
        all_corrections = db.execute(
            select(FeedbackCorrection.corrected_sign, FeedbackCorrection.status)
        ).all()

        name_counts: dict[str, dict[str, int]] = {}
        for sign, status in all_corrections:
            entry = name_counts.setdefault(sign, {"total": 0, "pending": 0})
            entry["total"] += 1
            if status == "pending":
                entry["pending"] += 1

        return [
            FeedbackStats(
                sign_name=sign,
                correction_count=counts["total"],
                pending_count=counts["pending"],
            )
            for sign, counts in sorted(name_counts.items())
        ]

    def check_and_trigger_training(
        self,
        db: Session,
        corrected_sign: str,
        threshold: Optional[int] = None,
    ) -> bool:
        """Trigger a few-shot training session when enough corrections are accumulated.

        Returns ``True`` if training was triggered, ``False`` otherwise.
        """
        settings = get_settings()

        if not settings.feedback_enabled:
            return False

        effective_threshold = threshold if threshold is not None else settings.feedback_training_trigger_count
        pending_count = self.get_pending_count(db, corrected_sign)

        if pending_count < effective_threshold:
            return False

        # Resolve Sign row for the corrected label.
        from app.models.sign import Sign  # local import to avoid circular

        sign_row = db.scalar(
            select(Sign).where(Sign.slug == corrected_sign)
        )
        if sign_row is None:
            # Try by name as fallback
            sign_row = db.scalar(
                select(Sign).where(Sign.name == corrected_sign)
            )

        if sign_row is None:
            logger.warning(
                "feedback_trigger_sign_not_found",
                corrected_sign=corrected_sign,
                pending_count=pending_count,
            )
            return False

        # Trigger few-shot training via TrainingService.
        try:
            from app.schemas.training import TrainingSessionCreate, TrainingConfig
            from app.services.training_service import training_service

            payload = TrainingSessionCreate(
                sign_id=_uuid_module.UUID(str(sign_row.id)),
                mode="few-shot",
                config=TrainingConfig(),
            )
            training_session = training_service.create_session(db, payload)

            # Mark corrections as trained.
            pending_corrections = db.scalars(
                select(FeedbackCorrection).where(
                    FeedbackCorrection.corrected_sign == corrected_sign,
                    FeedbackCorrection.status == "pending",
                )
            ).all()
            now = datetime.now(tz=timezone.utc)
            for corr in pending_corrections:
                corr.status = "trained"
                corr.trained_at = now
            db.commit()

            logger.info(
                "feedback_few_shot_training_triggered",
                corrected_sign=corrected_sign,
                training_session_id=str(training_session.id),
                corrections_marked=len(pending_corrections),
            )
            return True

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "feedback_trigger_training_failed",
                corrected_sign=corrected_sign,
                error=str(exc),
            )
            return False

    def save_landmarks(self, landmarks_data: list, sign_name: str) -> Optional[str]:
        """Save a list of landmark frames as a .npy file.

        Returns the path string on success, or ``None`` on failure.
        """
        settings = get_settings()
        feedback_dir = Path(settings.feedback_landmarks_dir)
        try:
            feedback_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error("feedback_landmarks_dir_create_failed", path=str(feedback_dir), error=str(exc))
            return None

        uid = _uuid_module.uuid4().hex[:12]
        safe_sign = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in sign_name)
        file_path = feedback_dir / f"{safe_sign}_{uid}.npy"

        try:
            arr = np.array(landmarks_data, dtype=np.float32)
            np.save(str(file_path), arr)
            logger.debug("feedback_landmarks_saved", path=str(file_path), shape=arr.shape)
            return str(file_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("feedback_landmarks_save_error", path=str(file_path), error=str(exc))
            return None


# Module-level singleton to mirror the pattern used by training_service.
feedback_service = FeedbackService()
