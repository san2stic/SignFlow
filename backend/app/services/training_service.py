"""Service layer for training session lifecycle with real PyTorch training."""

from __future__ import annotations

import threading
from multiprocessing import current_process
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import structlog
import torch
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import SessionLocal
from app.ml.augmentation import augment_dataset
from app.ml.dataset import LandmarkDataset, SignSample, load_landmarks_from_file
from app.ml.fewshot import prepare_few_shot_model
from app.ml.model import SignTransformer
from app.ml.prototypical import run_prototypical_fallback
from app.ml.trainer import SignTrainer, TrainingConfig as MLTrainingConfig, TrainingMetrics
from app.models.model_version import ModelVersion
from app.models.sign import Sign
from app.models.training import TrainingSession
from app.models.video import Video
from app.schemas.training import TrainingConfig, TrainingSessionCreate

logger = structlog.get_logger(__name__)


class TrainingService:
    """Creates and tracks training sessions; runs CPU-friendly training workers."""

    def __init__(self) -> None:
        self._stop_flags: dict[str, bool] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _resolve_deploy_threshold(config: dict | None) -> float:
        """Read deploy threshold from config with safe fallback."""
        try:
            return float((config or {}).get("min_deploy_accuracy", 0.85))
        except (TypeError, ValueError):
            return 0.85

    @staticmethod
    def _resolve_device() -> str:
        """Pick best available torch device for training."""
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _resolve_dataloader_workers(default_workers: int = 2) -> int:
        """Avoid DataLoader child processes when already inside a daemon worker process."""
        workers = max(0, int(default_workers))
        try:
            if current_process().daemon:
                return 0
        except Exception:  # noqa: BLE001
            pass
        return workers

    @staticmethod
    def _stratified_train_val_indices(
        labels: list[int],
        *,
        val_ratio: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create stratified train/val split to stabilize minority classes."""
        if not labels:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        by_label: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            by_label[int(label)].append(index)

        train_indices: list[int] = []
        val_indices: list[int] = []
        rng = np.random.default_rng()

        for class_indices in by_label.values():
            shuffled = np.array(class_indices, dtype=np.int64)
            rng.shuffle(shuffled)
            val_count = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 1
            val_count = min(val_count, max(1, len(shuffled) - 1)) if len(shuffled) > 1 else 1
            val_indices.extend(shuffled[:val_count].tolist())
            train_indices.extend(shuffled[val_count:].tolist())

        # Guard against degenerate splits.
        if not train_indices:
            train_indices, val_indices = val_indices[:-1], val_indices[-1:]
        if not val_indices:
            val_indices = train_indices[:1]
            train_indices = train_indices[1:] if len(train_indices) > 1 else train_indices

        return np.array(train_indices, dtype=np.int64), np.array(val_indices, dtype=np.int64)

    def _build_session_metrics(
        self,
        *,
        loss: float,
        accuracy: float,
        val_accuracy: float,
        deploy_threshold: float,
        current_epoch: int = 0,
        deployment_ready: bool = False,
        final_val_accuracy: float | None = None,
        recommended_next_action: str = "wait",
    ) -> dict:
        """Build normalized training metrics payload stored in JSON."""
        return {
            "loss": round(float(loss), 4),
            "accuracy": round(float(accuracy), 4),
            "val_accuracy": round(float(val_accuracy), 4),
            "current_epoch": int(current_epoch),
            "deployment_ready": bool(deployment_ready),
            "deploy_threshold": round(float(deploy_threshold), 4),
            "final_val_accuracy": round(float(final_val_accuracy), 4)
            if final_val_accuracy is not None
            else None,
            "recommended_next_action": recommended_next_action,
        }

    def _mark_failed_session(self, db: Session, session: TrainingSession, message: str) -> None:
        """Set failure status consistently for training sessions."""
        threshold = self._resolve_deploy_threshold(session.config)
        previous = session.metrics or {}
        final_val = previous.get("final_val_accuracy")
        try:
            final_val_accuracy = float(final_val) if final_val is not None else None
        except (TypeError, ValueError):
            final_val_accuracy = None
        session.metrics = self._build_session_metrics(
            loss=float(previous.get("loss", 1.0)),
            accuracy=float(previous.get("accuracy", 0.0)),
            val_accuracy=float(previous.get("val_accuracy", 0.0)),
            current_epoch=int(previous.get("current_epoch", 0)),
            deploy_threshold=threshold,
            deployment_ready=False,
            final_val_accuracy=final_val_accuracy,
            recommended_next_action="review_error",
        )
        session.status = "failed"
        session.error_message = message
        session.completed_at = datetime.now(timezone.utc)
        db.commit()

    def list_sessions(self, db: Session) -> list[TrainingSession]:
        """Return all training sessions ordered by creation date."""
        return db.scalars(select(TrainingSession).order_by(TrainingSession.created_at.desc())).all()

    def get_session(self, db: Session, session_id: str) -> TrainingSession | None:
        """Fetch a single training session by ID."""
        return db.get(TrainingSession, session_id)

    def get_session_model(self, db: Session, session: TrainingSession) -> ModelVersion | None:
        """Resolve model version produced by a completed session."""
        if not session.model_version_produced:
            return None
        return db.scalar(
            select(ModelVersion).where(ModelVersion.version == session.model_version_produced)
        )

    def create_session(self, db: Session, payload: TrainingSessionCreate) -> TrainingSession:
        """Insert a queued session row and dispatch background execution."""
        config = payload.config.model_dump()
        deploy_threshold = self._resolve_deploy_threshold(config)
        session = TrainingSession(
            sign_id=str(payload.sign_id) if payload.sign_id else None,
            mode=payload.mode,
            status="queued",
            progress=0.0,
            config=config,
            metrics=self._build_session_metrics(
                loss=1.0,
                accuracy=0.0,
                val_accuracy=0.0,
                deploy_threshold=deploy_threshold,
                deployment_ready=False,
                final_val_accuracy=None,
                recommended_next_action="wait",
            ),
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        self._dispatch_background_worker(session.id)
        return session

    def _dispatch_background_worker(self, session_id: str) -> None:
        """Dispatch session to Celery if enabled, otherwise use local thread."""
        settings = get_settings()

        if settings.training_use_celery:
            enqueued = self._enqueue_celery_job(session_id)
            if enqueued:
                return

        self._launch_background_worker(session_id)

    def _enqueue_celery_job(self, session_id: str) -> bool:
        """Try to queue training job on Celery; return False on dispatch failure."""
        try:
            from app.celery_app import celery_app

            celery_app.send_task(
                "app.tasks.training.run_training_session",
                args=[session_id],
                queue="training",
            )
            logger.info("training_session_enqueued_celery", session_id=session_id)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "training_celery_dispatch_failed",
                session_id=session_id,
                error=str(exc),
            )
            return False

    def stop_session(self, session_id: str) -> None:
        """Signal running worker to stop gracefully."""
        with self._lock:
            self._stop_flags[session_id] = True

    def _launch_background_worker(self, session_id: str) -> None:
        """Start detached worker thread for real PyTorch training."""
        thread = threading.Thread(target=self.run_training_session, args=(session_id,), daemon=True)
        thread.start()

    def run_training_session(self, session_id: str) -> None:
        """Execute real PyTorch training with landmark data."""
        db = SessionLocal()
        try:
            session = db.get(TrainingSession, session_id)
            if not session:
                logger.error("training_session_not_found", session_id=session_id)
                return

            logger.info("starting_training_worker", session_id=session_id, mode=session.mode)
            deploy_threshold = self._resolve_deploy_threshold(session.config)

            # === PHASE 1: PREPROCESSING ===
            session.status = "preprocessing"
            session.started_at = datetime.now(timezone.utc)
            session.progress = 5.0
            db.commit()

            try:
                # Load training data from database
                sign_id = session.sign_id
                mode = session.mode
                config = session.config
                active_model: ModelVersion | None = None
                target_count = 0

                if mode == "few-shot" and not sign_id:
                    raise ValueError("few-shot mode requires a sign_id")

                if mode == "few-shot":
                    active_model = db.scalar(
                        select(ModelVersion)
                        .where(ModelVersion.is_active.is_(True))
                        .order_by(ModelVersion.created_at.desc())
                    )

                # Always include full existing corpus (all video types) for stable class mapping.
                videos = db.scalars(
                    select(Video)
                    .where(Video.landmarks_extracted.is_(True))
                    .order_by(Video.created_at.asc())
                ).all()
                if not videos:
                    raise ValueError("No videos with extracted landmarks available")

                logger.info("loading_landmarks", num_videos=len(videos))

                sign_ids = {video.sign_id for video in videos}
                signs = db.scalars(select(Sign).where(Sign.id.in_(sign_ids))).all() if sign_ids else []
                slug_by_id = {item.id: item.slug for item in signs}

                if mode == "few-shot":
                    target_count = sum(1 for video in videos if video.sign_id == sign_id)
                    if target_count == 0:
                        raise ValueError("few-shot requested sign has no videos with extracted landmarks")

                # Load landmarks from files
                sequences: list[np.ndarray] = []
                labels: list[int] = []
                label_map: dict[str, int] = {}  # sign slug -> class index

                # Preserve active model class order for consistent logits -> labels mapping.
                if mode == "few-shot" and active_model and active_model.class_labels:
                    for existing_label in active_model.class_labels:
                        if existing_label and existing_label not in label_map:
                            label_map[existing_label] = len(label_map)

                for video in videos:
                    if not video.landmarks_path:
                        continue

                    try:
                        landmarks = load_landmarks_from_file(video.landmarks_path)
                        sequences.append(landmarks)

                        # Assign label
                        label_slug = slug_by_id.get(video.sign_id, f"unknown-{video.sign_id}")
                        if label_slug not in label_map:
                            label_map[label_slug] = len(label_map)
                        labels.append(label_map[label_slug])

                    except Exception as e:
                        logger.warning(
                            "failed_to_load_landmarks",
                            video_id=str(video.id),
                            error=str(e),
                        )

                if not sequences:
                    raise ValueError("No valid landmark sequences loaded")

                if len(sequences) < 2:
                    raise ValueError("Need at least 2 training samples for train/validation split")

                num_classes = len(label_map)
                class_labels = [item[0] for item in sorted(label_map.items(), key=lambda item: item[1])]
                logger.info("data_loaded", num_samples=len(sequences), num_classes=num_classes)

                # Apply data augmentation for all modes
                if config.get("augmentation", True):
                    logger.info("applying_data_augmentation", mode=mode)
                    sequences, labels = augment_dataset(
                        sequences,
                        labels,
                        num_augmentations_per_sample=5,
                        augmentation_probability=0.5,
                    )
                    logger.info("augmentation_complete", total_samples=len(sequences))

                # Create stratified train/val split.
                train_indices, val_indices = self._stratified_train_val_indices(labels, val_ratio=0.2)

                train_samples = [SignSample(sequences[i], labels[i]) for i in train_indices]
                val_samples = [SignSample(sequences[i], labels[i]) for i in val_indices]

                sequence_length = int(config.get("sequence_length", 64))
                sequence_length = max(8, min(256, sequence_length))

                # Create datasets with resampling + enriched features
                train_dataset = LandmarkDataset(
                    train_samples,
                    sequence_length=sequence_length,
                    apply_sliding_window=False,
                    use_enriched_features=True,
                )
                val_dataset = LandmarkDataset(
                    val_samples,
                    sequence_length=sequence_length,
                    apply_sliding_window=False,
                    use_enriched_features=True,
                )

                session.progress = 10.0
                db.commit()

            except Exception as e:
                logger.error("preprocessing_failed", error=str(e), exc_info=True)
                self._mark_failed_session(db, session, f"Preprocessing failed: {str(e)}")
                return

            # === PHASE 2: TRAINING ===
            session.status = "training"
            db.commit()

            try:
                device = self._resolve_device()

                # Create model
                from app.ml.feature_engineering import ENRICHED_FEATURE_DIM
                num_features = ENRICHED_FEATURE_DIM

                if mode == "few-shot":
                    active_checkpoint = (
                        active_model.file_path
                        if active_model and active_model.file_path and Path(active_model.file_path).exists()
                        else None
                    )
                    prepared = prepare_few_shot_model(
                        checkpoint_path=active_checkpoint,
                        num_features=num_features,
                        num_classes=num_classes,
                        d_model=128,
                        device=device,
                        freeze_until_layer=int(config.get("freeze_until_layer", 2)),
                    )
                    model = prepared.model
                    num_classes = model.num_classes
                else:
                    model = SignTransformer(
                        num_features=num_features,
                        num_classes=num_classes,
                    )

                # Training configuration
                ml_config = MLTrainingConfig(
                    num_epochs=int(config.get("epochs", 50)),
                    learning_rate=float(
                        config.get("learning_rate", 1e-4 if mode == "few-shot" else 3e-4)
                    ),
                    batch_size=32,
                    num_workers=self._resolve_dataloader_workers(int(config.get("num_workers", 2))),
                    device=device,
                    early_stopping_patience=int(config.get("early_stopping_patience", 15)),
                    early_stopping_min_delta=float(config.get("early_stopping_min_delta", 1e-4)),
                    weight_decay=float(config.get("weight_decay", 0.05)),
                    classifier_lr_multiplier=float(config.get("classifier_lr_multiplier", 2.0)),
                    label_smoothing=float(config.get("label_smoothing", 0.1)),
                    warmup_epochs=int(config.get("warmup_epochs", 3)),
                    use_focal_loss=(mode == "few-shot"),  # Focal loss for few-shot
                    use_mixup=bool(config.get("use_mixup", True)),
                    mixup_alpha=float(config.get("mixup_alpha", 0.3)),
                    use_ema=bool(config.get("use_ema", True)),
                    ema_decay=float(config.get("ema_decay", 0.995)),
                    gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
                    temporal_mask_prob=float(config.get("temporal_mask_prob", 0.15)),
                    temporal_mask_span_ratio=float(config.get("temporal_mask_span_ratio", 0.2)),
                    use_amp=bool(config.get("use_amp", True)),
                    amp_dtype=str(config.get("amp_dtype", "float16")),
                    use_swa=bool(config.get("use_swa", True)),
                    swa_start_ratio=float(config.get("swa_start_ratio", 0.75)),
                    swa_lr=(
                        float(config["swa_lr"])
                        if config.get("swa_lr") not in (None, "")
                        else None
                    ),
                )

                # Progress callback
                def progress_callback(metrics: TrainingMetrics) -> None:
                    # Update session with real metrics
                    progress = 10 + (metrics.epoch / ml_config.num_epochs) * 80
                    session.progress = progress
                    session.metrics = self._build_session_metrics(
                        loss=metrics.train_loss,
                        accuracy=metrics.train_accuracy,
                        val_accuracy=metrics.val_accuracy,
                        current_epoch=metrics.epoch,
                        deploy_threshold=deploy_threshold,
                        deployment_ready=False,
                        final_val_accuracy=None,
                        recommended_next_action="wait",
                    )
                    db.commit()

                # Stop signal
                def stop_signal() -> bool:
                    return self._should_stop(session_id)

                # Create trainer
                trainer = SignTrainer(
                    model=model,
                    config=ml_config,
                    progress_callback=progress_callback,
                    stop_signal=stop_signal,
                )

                use_prototypical = mode == "few-shot" and target_count < 5
                if use_prototypical:
                    logger.info(
                        "starting_prototypical_fallback",
                        target_count=target_count,
                    )
                    proto_metric, _prototypes = run_prototypical_fallback(
                        model=model,
                        train_dataset=train_dataset,
                        val_dataset=val_dataset,
                        device=device,
                        batch_size=ml_config.batch_size,
                    )
                    progress_callback(proto_metric)
                    trainer.metrics_history = [proto_metric]
                    trainer.best_val_loss = proto_metric.val_loss
                    metrics_history = trainer.metrics_history
                else:
                    # Train model
                    logger.info("starting_pytorch_training")
                    metrics_history = trainer.fit(train_dataset, val_dataset)

                if self._should_stop(session_id):
                    self._mark_failed_session(db, session, "Stopped by user")
                    return

                logger.info(
                    "training_complete",
                    num_epochs=len(metrics_history),
                    final_val_acc=metrics_history[-1].val_accuracy if metrics_history else 0.0,
                )

            except Exception as e:
                logger.error("training_failed", error=str(e), exc_info=True)
                self._mark_failed_session(db, session, f"Training failed: {str(e)}")
                return

            # === PHASE 3: VALIDATION & SAVE ===
            session.status = "validating"
            session.progress = 95.0
            db.commit()

            try:
                # Save model checkpoint
                settings = get_settings()
                models_dir = Path(settings.model_dir)
                models_dir.mkdir(parents=True, exist_ok=True)

                # Create version
                existing = db.scalars(select(ModelVersion).order_by(ModelVersion.created_at.asc())).all()
                version_idx = len(existing) + 1
                version = f"v{version_idx}"

                model_file = models_dir / f"model_{version}.pt"
                if len(class_labels) != num_classes:
                    logger.warning(
                        "class_label_count_mismatch",
                        class_labels=len(class_labels),
                        num_classes=num_classes,
                    )
                    if len(class_labels) < num_classes:
                        for index in range(len(class_labels), num_classes):
                            class_labels.append(f"class_{index}")
                    else:
                        class_labels = class_labels[:num_classes]
                trainer.save_model(model_file, class_labels=class_labels)

                logger.info("model_saved", path=str(model_file), version=version)

                # Create model version record
                final_accuracy = metrics_history[-1].val_accuracy if metrics_history else 0.0
                deployment_ready = final_accuracy >= deploy_threshold
                next_action = "deploy" if deployment_ready else "collect_more_examples"

                parent_version = existing[-1].version if existing else None
                model_version = ModelVersion(
                    version=version,
                    is_active=False,
                    num_classes=num_classes,
                    accuracy=final_accuracy,
                    class_labels=class_labels,
                    training_session_id=session.id,
                    file_path=str(model_file),
                    file_size_mb=round(model_file.stat().st_size / 1_048_576, 4),
                    parent_version=parent_version,
                )
                db.add(model_version)
                db.flush()

                session.status = "completed"
                session.progress = 100.0
                session.model_version_produced = version
                session.metrics = self._build_session_metrics(
                    loss=metrics_history[-1].train_loss if metrics_history else 0.0,
                    accuracy=metrics_history[-1].train_accuracy if metrics_history else 0.0,
                    val_accuracy=final_accuracy,
                    current_epoch=metrics_history[-1].epoch if metrics_history else 0,
                    deploy_threshold=deploy_threshold,
                    deployment_ready=deployment_ready,
                    final_val_accuracy=final_accuracy,
                    recommended_next_action=next_action,
                )
                session.completed_at = datetime.now(timezone.utc)
                db.commit()

                logger.info("training_session_complete", session_id=session_id, version=version)

            except Exception as e:
                logger.error("model_save_failed", error=str(e), exc_info=True)
                self._mark_failed_session(db, session, f"Model save failed: {str(e)}")
                return

        except Exception as e:
            logger.error("training_worker_error", error=str(e), exc_info=True)
            try:
                session = db.get(TrainingSession, session_id)
                if session:
                    self._mark_failed_session(db, session, f"Unexpected error: {str(e)}")
            except Exception:
                pass
        finally:
            db.close()

    def _should_stop(self, session_id: str) -> bool:
        """Read and consume stop signal for a session."""
        with self._lock:
            return self._stop_flags.pop(session_id, False)


training_service = TrainingService()


def normalize_training_config(config_dict: dict) -> TrainingConfig:
    """Normalize config payload before API serialization."""
    return TrainingConfig(**config_dict)
