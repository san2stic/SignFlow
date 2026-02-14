"""Service layer for training session lifecycle with real PyTorch training."""

from __future__ import annotations

import threading
from collections import defaultdict
from multiprocessing import current_process
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import structlog
import torch
from sqlalchemy import select
from sqlalchemy.orm import Session
from torch.utils.data import DataLoader

from app.config import get_settings
from app.database import SessionLocal
from app.ml.augmentation import augment_dataset
from app.ml.dataset import LandmarkDataset, SignSample, load_landmarks_from_file
from app.ml.fewshot import prepare_few_shot_model
from app.ml.model import SignTransformer
from app.ml.prototypical import run_prototypical_fallback
from app.ml.trainer import (
    SignTrainer,
    TrainingConfig as MLTrainingConfig,
    TrainingMetrics,
    load_model_checkpoint,
)
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
    def _has_ready_landmarks(video: Video) -> bool:
        """Check whether a video has extracted landmarks available on disk."""
        return bool(video.landmarks_extracted and video.landmarks_path)

    @staticmethod
    def _passes_detection_gate(video: Video, *, min_detection_rate: float) -> bool:
        """Check if clip quality is sufficient for training by detection-rate gate."""
        if not TrainingService._has_ready_landmarks(video):
            return False
        detection_rate = float(video.detection_rate or 0.0)
        return detection_rate >= float(min_detection_rate)

    @staticmethod
    def _validate_class_space(
        *,
        mode: str,
        num_classes: int,
        open_set_enabled: bool,
        generated_none_count: int,
    ) -> None:
        """Fail fast when supervision has no discriminative class boundary."""
        if num_classes >= 2:
            return

        if mode == "few-shot":
            raise ValueError(
                "Few-shot training requires at least 2 classes after preprocessing. "
                f"Got num_classes={num_classes}. "
                f"Open-set enabled={open_set_enabled}, generated_none={generated_none_count}. "
                "Add labeled videos for another sign, provide idle unlabeled clips for [NONE], "
                "or start from an existing multi-class checkpoint."
            )

        raise ValueError(
            "Training requires at least 2 classes after preprocessing. "
            f"Got num_classes={num_classes}. "
            f"Open-set enabled={open_set_enabled}, generated_none={generated_none_count}. "
            "Collect additional sign classes or [NONE] clips."
        )

    @staticmethod
    def _stratified_train_val_indices(
        labels: list[int],
        *,
        val_ratio: float = 0.2,
        min_val_samples_per_class: int = 1,
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

        min_per_class = max(1, int(min_val_samples_per_class))
        for class_indices in by_label.values():
            shuffled = np.array(class_indices, dtype=np.int64)
            rng.shuffle(shuffled)
            if len(shuffled) == 1:
                # Keep singleton classes in train split; they cannot be both train + val.
                train_indices.extend(shuffled.tolist())
                continue

            val_count = max(min_per_class, int(round(len(shuffled) * val_ratio)))
            val_count = min(val_count, max(1, len(shuffled) - 1))
            val_indices.extend(shuffled[:val_count].tolist())
            train_indices.extend(shuffled[val_count:].tolist())

        # Guard against degenerate splits.
        if not train_indices and val_indices:
            train_indices.append(val_indices.pop())

        if not val_indices and len(train_indices) > 1:
            # Prefer moving a sample from a class with >1 train samples to keep coverage.
            by_label_train: dict[int, list[int]] = defaultdict(list)
            for index in train_indices:
                by_label_train[int(labels[index])].append(index)

            moved_index: int | None = None
            for class_train_indices in by_label_train.values():
                if len(class_train_indices) > 1:
                    moved_index = class_train_indices[-1]
                    break

            if moved_index is None:
                moved_index = train_indices[-1]

            train_indices.remove(moved_index)
            val_indices.append(moved_index)

        if not val_indices and train_indices:
            # Final fallback for ultra-small datasets (e.g., one single sample).
            val_indices = train_indices[:1]

        return np.array(train_indices, dtype=np.int64), np.array(val_indices, dtype=np.int64)

    @staticmethod
    def _split_and_prepare_sequences(
        sequences: list[np.ndarray],
        labels: list[int],
        *,
        val_ratio: float = 0.2,
        min_val_samples_per_class: int = 1,
        apply_augmentation: bool = True,
        num_augmentations_per_sample: int = 5,
        augmentation_probability: float = 0.5,
        max_augmented_train_samples: int | None = None,
    ) -> tuple[list[np.ndarray], list[int], list[np.ndarray], list[int]]:
        """
        Split dataset first, then augment train split only.

        This avoids validation leakage where augmented variants of one clip end up
        in both train and validation, which can inflate validation accuracy.
        """
        if len(sequences) != len(labels):
            raise ValueError("sequences and labels must have the same length")
        if not sequences:
            raise ValueError("No sequences available for splitting")

        train_indices, val_indices = TrainingService._stratified_train_val_indices(
            labels,
            val_ratio=val_ratio,
            min_val_samples_per_class=min_val_samples_per_class,
        )

        train_sequences = [sequences[index] for index in train_indices.tolist()]
        train_labels = [labels[index] for index in train_indices.tolist()]
        val_sequences = [sequences[index] for index in val_indices.tolist()]
        val_labels = [labels[index] for index in val_indices.tolist()]

        if not train_sequences or not val_sequences:
            raise ValueError("Could not build a valid train/validation split")

        if apply_augmentation:
            train_sequences, train_labels = augment_dataset(
                train_sequences,
                train_labels,
                num_augmentations_per_sample=num_augmentations_per_sample,
                augmentation_probability=augmentation_probability,
            )
            if max_augmented_train_samples is not None and len(train_sequences) > max_augmented_train_samples:
                # Keep all originals and subsample augmented tails if needed.
                original_count = len(train_indices)
                max_total = max(original_count, int(max_augmented_train_samples))
                keep_augmented = max_total - original_count

                if keep_augmented <= 0:
                    train_sequences = train_sequences[:original_count]
                    train_labels = train_labels[:original_count]
                else:
                    augmented_start = original_count
                    augmented_indices = list(range(augmented_start, len(train_sequences)))
                    rng = np.random.default_rng()
                    rng.shuffle(augmented_indices)
                    kept_augmented_indices = augmented_indices[:keep_augmented]
                    selected = list(range(original_count)) + kept_augmented_indices
                    train_sequences = [train_sequences[index] for index in selected]
                    train_labels = [train_labels[index] for index in selected]

        return train_sequences, train_labels, val_sequences, val_labels

    @staticmethod
    def _resolve_augmentation_policy(
        *,
        mode: str,
        config: dict | None,
        train_size: int,
        target_class_samples: int | None = None,
    ) -> tuple[int, float, int]:
        """
        Resolve augmentation intensity using mode-aware defaults and sample budgets.

        Returns:
            (num_augmentations_per_sample, augmentation_probability, max_train_samples)
        """
        cfg = config or {}
        normalized_mode = str(mode)

        if normalized_mode == "few-shot":
            target = int(target_class_samples or 0)
            if target < 10:
                default_num_aug = 16
                default_probability = 0.70
            elif target <= 30:
                default_num_aug = 10
                default_probability = 0.60
            else:
                default_num_aug = 6
                default_probability = 0.50
            default_max_train_samples = 12000
        else:
            default_num_aug = 4
            default_probability = 0.45
            default_max_train_samples = 40000

        if cfg.get("num_augmentations_per_sample") is None:
            requested_num_aug = default_num_aug
        else:
            requested_num_aug = int(cfg["num_augmentations_per_sample"])
        requested_num_aug = max(0, min(128, requested_num_aug))

        if cfg.get("augmentation_probability") is None:
            requested_probability = default_probability
        else:
            requested_probability = float(cfg["augmentation_probability"])
        requested_probability = float(np.clip(requested_probability, 0.0, 1.0))

        requested_max_samples = int(cfg.get("max_augmented_train_samples", default_max_train_samples))
        requested_max_samples = max(int(train_size), requested_max_samples)

        if train_size <= 0 or requested_num_aug <= 0:
            return 0, requested_probability, requested_max_samples

        # Base set size + per-sample augmentations, bounded by max sample budget.
        max_extra = max(0, requested_max_samples - train_size)
        max_aug_per_sample = max_extra // train_size
        effective_num_aug = min(requested_num_aug, max_aug_per_sample)
        effective_num_aug = max(0, effective_num_aug)

        return effective_num_aug, requested_probability, requested_max_samples

    @staticmethod
    def _compute_motion_signal(sequence: np.ndarray) -> np.ndarray:
        """Compute frame-level hand motion magnitude for one sequence."""
        if sequence.ndim != 2 or sequence.shape[0] == 0:
            return np.array([], dtype=np.float32)
        hands = sequence[:, :126] if sequence.shape[1] >= 126 else sequence
        if hands.shape[0] < 2:
            return np.zeros((hands.shape[0],), dtype=np.float32)
        deltas = np.mean(np.abs(np.diff(hands, axis=0)), axis=1)
        first = np.array([float(deltas[0])], dtype=np.float32)
        return np.concatenate([first, deltas.astype(np.float32)], axis=0)

    @staticmethod
    def _extract_rest_segments(
        sequence: np.ndarray,
        *,
        motion_threshold: float = 0.0015,
        min_len: int = 12,
        max_segments: int = 3,
    ) -> list[np.ndarray]:
        """Extract low-motion contiguous segments to represent [NONE] windows."""
        motion = TrainingService._compute_motion_signal(sequence)
        if motion.size == 0:
            return []

        low_motion = motion < float(motion_threshold)
        segments: list[np.ndarray] = []
        start: int | None = None
        for index, value in enumerate(low_motion.tolist() + [False]):
            if value and start is None:
                start = index
                continue
            if not value and start is not None:
                end = index
                if end - start >= min_len:
                    candidate = sequence[start:end]
                    if candidate.size:
                        segments.append(candidate)
                start = None
                if len(segments) >= max_segments:
                    break
        return segments

    @staticmethod
    def _generate_open_set_sequences(
        *,
        labeled_sequences: list[np.ndarray],
        unlabeled_sequences: list[np.ndarray],
        max_count: int,
    ) -> list[np.ndarray]:
        """Build [NONE] sequences from rest windows and low-motion unlabeled clips."""
        if max_count <= 0:
            return []

        generated: list[np.ndarray] = []
        for sequence in labeled_sequences:
            generated.extend(TrainingService._extract_rest_segments(sequence))
            if len(generated) >= max_count:
                return generated[:max_count]

        # Retry with progressively relaxed thresholds for short or noisy clips.
        relaxed_profiles = [
            (0.0025, 10),
            (0.0040, 8),
            (0.0065, 6),
        ]
        for motion_threshold, min_len in relaxed_profiles:
            if len(generated) >= max_count:
                break
            for sequence in labeled_sequences:
                generated.extend(
                    TrainingService._extract_rest_segments(
                        sequence,
                        motion_threshold=motion_threshold,
                        min_len=min_len,
                        max_segments=2,
                    )
                )
                if len(generated) >= max_count:
                    return generated[:max_count]

        for sequence in unlabeled_sequences:
            motion = TrainingService._compute_motion_signal(sequence)
            if motion.size == 0:
                continue
            if float(np.mean(motion)) < 0.002:
                generated.append(sequence)
            if len(generated) >= max_count:
                break

        # Final fallback: synthesize low-motion windows from existing labeled clips.
        # This prevents cold-start few-shot runs from failing when users only upload one sign class.
        if len(generated) < max_count and labeled_sequences:
            fallback_target = min(max_count, max(2, min(8, len(labeled_sequences))))
            if len(generated) >= fallback_target:
                return generated[:max_count]
            rng = np.random.default_rng()
            target_len = max(8, min(24, int(np.median([seq.shape[0] for seq in labeled_sequences]))))
            sample_index = 0
            max_attempts = fallback_target * 4
            while len(generated) < fallback_target and sample_index < max_attempts:
                source = labeled_sequences[sample_index % len(labeled_sequences)]
                if source.ndim != 2 or source.shape[0] == 0:
                    sample_index += 1
                    continue
                frame_idx = int(np.argmin(TrainingService._compute_motion_signal(source)))
                anchor = source[frame_idx : frame_idx + 1].astype(np.float32, copy=True)
                synthetic = np.repeat(anchor, repeats=target_len, axis=0)
                if synthetic.shape[1] >= 126:
                    # Pull hands closer to neutral to emulate idle posture.
                    synthetic[:, :126] *= 0.2
                noise = rng.normal(loc=0.0, scale=4e-4, size=synthetic.shape).astype(np.float32)
                generated.append((synthetic + noise).astype(np.float32))
                sample_index += 1

        return generated[:max_count]

    @staticmethod
    def _collect_logits_and_labels(
        *,
        model: SignTransformer,
        dataset: LandmarkDataset,
        device: str,
        batch_size: int = 64,
        repeats: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Collect logits/labels from one dataset for calibration and evaluation."""
        loader = DataLoader(
            dataset,
            batch_size=max(1, int(batch_size)),
            shuffle=False,
            num_workers=0,
        )
        torch_device = torch.device(device)
        model = model.to(torch_device)
        model.eval()

        logits_chunks: list[np.ndarray] = []
        label_chunks: list[np.ndarray] = []
        repeat_count = max(1, int(repeats))
        with torch.no_grad():
            for _ in range(repeat_count):
                for landmarks, labels in loader:
                    logits = model(landmarks.to(torch_device))
                    logits_chunks.append(logits.detach().cpu().numpy())
                    label_chunks.append(labels.detach().cpu().numpy())

        if not logits_chunks:
            return (
                np.zeros((0, model.num_classes), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )
        return (
            np.concatenate(logits_chunks, axis=0).astype(np.float32),
            np.concatenate(label_chunks, axis=0).astype(np.int64),
        )

    @staticmethod
    def _fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
        """Fit temperature scaling on validation logits by minimizing NLL."""
        if logits.size == 0 or labels.size == 0:
            return 1.0

        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        log_temperature = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)
        criterion = torch.nn.CrossEntropyLoss()

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            temperature = torch.exp(log_temperature).clamp(0.5, 10.0)
            loss = criterion(logits_tensor / temperature, labels_tensor)
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
            value = float(torch.exp(log_temperature).detach().cpu().item())
            return float(np.clip(value, 0.5, 10.0))
        except Exception:  # noqa: BLE001
            return 1.0

    @staticmethod
    def _apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature-scaled softmax to logits."""
        temp = float(max(1e-3, temperature))
        scaled = logits / temp
        scaled = scaled - np.max(scaled, axis=1, keepdims=True)
        exp = np.exp(scaled)
        denom = np.sum(exp, axis=1, keepdims=True)
        denom = np.clip(denom, 1e-9, None)
        return exp / denom

    @staticmethod
    def _binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute binary F1 with safe zero handling."""
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        if tp == 0:
            return 0.0
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        if precision + recall <= 0:
            return 0.0
        return float((2.0 * precision * recall) / (precision + recall))

    @staticmethod
    def _learn_class_thresholds(
        probs: np.ndarray,
        labels: np.ndarray,
        class_labels: list[str],
    ) -> dict[str, float]:
        """Learn per-class confidence thresholds from validation predictions."""
        if probs.size == 0 or labels.size == 0:
            return {}

        thresholds: dict[str, float] = {}
        candidates = np.linspace(0.35, 0.95, 13)
        for class_index, class_label in enumerate(class_labels):
            y_true = (labels == class_index).astype(np.int32)
            if int(np.sum(y_true)) == 0:
                thresholds[class_label] = 0.75
                continue

            best_threshold = 0.7
            best_f1 = -1.0
            for candidate in candidates:
                y_pred = (probs[:, class_index] >= candidate).astype(np.int32)
                score = TrainingService._binary_f1(y_true, y_pred)
                if score > best_f1:
                    best_f1 = score
                    best_threshold = float(candidate)
            thresholds[class_label] = float(best_threshold)
        return thresholds

    @staticmethod
    def _predict_with_thresholds(
        probs: np.ndarray,
        *,
        class_labels: list[str],
        class_thresholds: dict[str, float],
        none_label: str = "[NONE]",
    ) -> np.ndarray:
        """Decode class indices with per-class confidence thresholds."""
        if probs.size == 0:
            return np.zeros((0,), dtype=np.int64)
        none_index = class_labels.index(none_label) if none_label in class_labels else -1
        predictions: list[int] = []
        for row in probs:
            top_index = int(np.argmax(row))
            label = class_labels[top_index] if top_index < len(class_labels) else f"class_{top_index}"
            threshold = max(0.5, float(class_thresholds.get(label, 0.7)))
            if float(row[top_index]) < threshold and none_index >= 0:
                predictions.append(none_index)
                continue
            if none_index >= 0 and top_index != none_index:
                none_threshold = max(0.5, float(class_thresholds.get(none_label, 0.7)))
                if float(row[none_index]) >= none_threshold and float(row[none_index]) > float(row[top_index]):
                    predictions.append(none_index)
                    continue
            predictions.append(top_index)
        return np.array(predictions, dtype=np.int64)

    @staticmethod
    def _f1_for_label(y_true: np.ndarray, y_pred: np.ndarray, label_index: int) -> float:
        """Compute one-vs-rest F1 for a single class index."""
        truth = (y_true == label_index).astype(np.int32)
        pred = (y_pred == label_index).astype(np.int32)
        return TrainingService._binary_f1(truth, pred)

    @staticmethod
    def _compute_eval_metrics(
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_labels: list[str],
        target_label: str | None,
        none_label: str = "[NONE]",
    ) -> dict[str, float | None]:
        """Compute macro F1, target sign F1, and open-set false positive rate."""
        if y_true.size == 0:
            return {
                "macro_f1": 0.0,
                "target_sign_f1": None,
                "open_set_fpr": None,
            }

        none_index = class_labels.index(none_label) if none_label in class_labels else -1
        labels_for_macro = sorted(set(int(v) for v in y_true.tolist()))
        if none_index >= 0:
            labels_for_macro = [label for label in labels_for_macro if label != none_index]
        if not labels_for_macro:
            labels_for_macro = sorted(set(int(v) for v in y_true.tolist()))

        f1_scores = [TrainingService._f1_for_label(y_true, y_pred, idx) for idx in labels_for_macro]
        macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

        target_sign_f1: float | None = None
        if target_label and target_label in class_labels:
            target_index = class_labels.index(target_label)
            target_sign_f1 = TrainingService._f1_for_label(y_true, y_pred, target_index)

        open_set_fpr: float | None = None
        if none_index >= 0:
            none_mask = y_true == none_index
            if int(np.sum(none_mask)) > 0:
                false_positive = int(np.sum((y_pred != none_index) & none_mask))
                open_set_fpr = float(false_positive / int(np.sum(none_mask)))
            else:
                open_set_fpr = 0.0

        return {
            "macro_f1": macro_f1,
            "target_sign_f1": target_sign_f1,
            "open_set_fpr": open_set_fpr,
        }

    @staticmethod
    def _estimate_latency_p95_ms(
        *,
        model: SignTransformer,
        dataset: LandmarkDataset,
        device: str,
        max_samples: int = 200,
    ) -> float:
        """Estimate p95 per-sample inference latency on a dataset."""
        if len(dataset) == 0:
            return 0.0
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        torch_device = torch.device(device)
        model = model.to(torch_device)
        model.eval()

        timings: list[float] = []
        with torch.no_grad():
            for index, (landmarks, _labels) in enumerate(loader):
                if index >= max_samples:
                    break
                start = time.perf_counter()
                _ = model(landmarks.to(torch_device))
                duration_ms = (time.perf_counter() - start) * 1000.0
                timings.append(float(duration_ms))

        if not timings:
            return 0.0
        return float(np.percentile(np.array(timings, dtype=np.float32), 95))

    @staticmethod
    def _passes_deployment_gate(
        *,
        mode: str,
        config: dict,
        macro_f1: float,
        target_sign_f1: float | None,
        open_set_fpr: float | None,
        latency_p95_ms: float,
    ) -> bool:
        """Evaluate multi-metric deployment gate."""
        macro_gate = float(config.get("macro_f1_gate", 0.82))
        target_gate = float(config.get("target_sign_f1_gate", 0.85))
        open_set_gate = float(config.get("open_set_fpr_gate", 0.05))
        latency_gate = float(config.get("latency_p95_ms_gate", 120.0))

        if macro_f1 < macro_gate:
            return False
        if mode == "few-shot":
            if target_sign_f1 is None or target_sign_f1 < target_gate:
                return False
        if open_set_fpr is not None and open_set_fpr > open_set_gate:
            return False
        if latency_p95_ms > latency_gate:
            return False
        return True

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
        macro_f1: float = 0.0,
        target_sign_f1: float | None = None,
        open_set_fpr: float | None = None,
        latency_p95_ms: float | None = None,
        calibration_temperature: float | None = None,
        deployment_gate_passed: bool = False,
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
            "macro_f1": round(float(macro_f1), 4),
            "target_sign_f1": round(float(target_sign_f1), 4) if target_sign_f1 is not None else None,
            "open_set_fpr": round(float(open_set_fpr), 4) if open_set_fpr is not None else None,
            "latency_p95_ms": round(float(latency_p95_ms), 2) if latency_p95_ms is not None else None,
            "calibration_temperature": (
                round(float(calibration_temperature), 4)
                if calibration_temperature is not None
                else None
            ),
            "deployment_gate_passed": bool(deployment_gate_passed),
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
        config = payload.config.model_dump(exclude_unset=True)
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

                quality_min_detection_rate = float(config.get("quality_min_detection_rate", 0.8))
                open_set_enabled = bool(config.get("open_set_enabled", True))

                # Supervised classes: labeled clips with landmarks passing detection gate.
                # We intentionally gate by measured detection_rate rather than is_trainable,
                # so config-driven threshold changes immediately affect eligibility.
                all_labeled_videos = db.scalars(
                    select(Video)
                    .where(Video.sign_id.is_not(None))
                    .order_by(Video.created_at.asc())
                ).all()
                videos = [
                    video
                    for video in all_labeled_videos
                    if self._passes_detection_gate(video, min_detection_rate=quality_min_detection_rate)
                ]

                labeled_total = len(all_labeled_videos)
                labeled_with_landmarks = sum(1 for video in all_labeled_videos if self._has_ready_landmarks(video))
                labeled_passing_detection = len(videos)
                labeled_marked_trainable = sum(
                    1
                    for video in all_labeled_videos
                    if self._has_ready_landmarks(video) and bool(video.is_trainable)
                )

                target_videos = (
                    [video for video in all_labeled_videos if video.sign_id == sign_id]
                    if sign_id
                    else []
                )
                target_total = len(target_videos)
                target_with_landmarks = sum(1 for video in target_videos if self._has_ready_landmarks(video))
                target_passing_detection = sum(
                    1
                    for video in target_videos
                    if self._passes_detection_gate(video, min_detection_rate=quality_min_detection_rate)
                )

                logger.info(
                    "training_video_eligibility",
                    mode=mode,
                    sign_id=sign_id,
                    quality_min_detection_rate=quality_min_detection_rate,
                    labeled_total=labeled_total,
                    labeled_with_landmarks=labeled_with_landmarks,
                    labeled_passing_detection=labeled_passing_detection,
                    labeled_marked_trainable=labeled_marked_trainable,
                    target_total=target_total,
                    target_with_landmarks=target_with_landmarks,
                    target_passing_detection=target_passing_detection,
                )

                if not videos:
                    raise ValueError(
                        "No eligible labeled videos available after quality filtering "
                        f"(required detection_rate >= {quality_min_detection_rate:.2f}). "
                        f"Dataset: total={labeled_total}, with_landmarks={labeled_with_landmarks}, "
                        f"passing_detection={labeled_passing_detection}, marked_trainable={labeled_marked_trainable}."
                    )

                logger.info(
                    "loading_landmarks",
                    num_videos=len(videos),
                    quality_min_detection_rate=quality_min_detection_rate,
                    open_set_enabled=open_set_enabled,
                )

                sign_ids = {video.sign_id for video in videos if video.sign_id}
                signs = db.scalars(select(Sign).where(Sign.id.in_(sign_ids))).all() if sign_ids else []
                slug_by_id = {item.id: item.slug for item in signs}

                if mode == "few-shot":
                    target_count = sum(1 for video in videos if video.sign_id == sign_id)
                    if target_count == 0:
                        raise ValueError(
                            "few-shot requested sign has no eligible videos after quality filtering "
                            f"(required detection_rate >= {quality_min_detection_rate:.2f}). "
                            f"Target sign: total={target_total}, with_landmarks={target_with_landmarks}, "
                            f"passing_detection={target_passing_detection}."
                        )

                # Load supervised landmarks
                sequences: list[np.ndarray] = []
                labels: list[int] = []
                label_map: dict[str, int] = {}  # sign slug -> class index
                none_label = "[NONE]"
                unlabeled_sequences: list[np.ndarray] = []

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

                if open_set_enabled:
                    unlabeled_videos = db.scalars(
                        select(Video)
                        .where(
                            Video.sign_id.is_(None),
                            Video.landmarks_extracted.is_(True),
                            Video.landmarks_path.is_not(None),
                        )
                        .order_by(Video.created_at.asc())
                    ).all()
                    for video in unlabeled_videos:
                        if not video.landmarks_path:
                            continue
                        try:
                            unlabeled_sequences.append(load_landmarks_from_file(video.landmarks_path))
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "failed_to_load_unlabeled_landmarks",
                                video_id=str(video.id),
                                error=str(exc),
                            )

                if not sequences:
                    raise ValueError("No valid landmark sequences loaded")

                if len(sequences) < 2:
                    raise ValueError("Need at least 2 training samples for train/validation split")

                none_sequences: list[np.ndarray] = []
                if open_set_enabled:
                    if mode == "few-shot":
                        desired_none_count = max(2, min(64, len(sequences) // 6))
                        if target_count > 0:
                            desired_none_count = min(desired_none_count, max(2, target_count // 2))
                    else:
                        desired_none_count = max(8, min(256, len(sequences) // 4))
                    none_sequences = self._generate_open_set_sequences(
                        labeled_sequences=sequences,
                        unlabeled_sequences=unlabeled_sequences,
                        max_count=desired_none_count,
                    )
                    if none_sequences:
                        if none_label not in label_map:
                            label_map[none_label] = len(label_map)
                        none_index = label_map[none_label]
                        for sample in none_sequences:
                            sequences.append(sample)
                            labels.append(none_index)
                    logger.info(
                        "open_set_none_generated",
                        requested=desired_none_count,
                        generated=len(none_sequences),
                    )
                elif none_label in label_map:
                    # Explicitly remove NONE class when open-set training is disabled.
                    filtered = {name: idx for name, idx in label_map.items() if name != none_label}
                    idx_to_name = {idx: name for name, idx in filtered.items()}
                    reordered_names = [item[0] for item in sorted(filtered.items(), key=lambda item: item[1])]
                    label_map = {name: index for index, name in enumerate(reordered_names)}
                    labels = [
                        label_map[idx_to_name[idx]]
                        for idx in labels
                        if idx in idx_to_name
                    ]

                num_classes = len(label_map)
                class_labels = [item[0] for item in sorted(label_map.items(), key=lambda item: item[1])]
                self._validate_class_space(
                    mode=mode,
                    num_classes=num_classes,
                    open_set_enabled=open_set_enabled,
                    generated_none_count=len(none_sequences),
                )
                logger.info("data_loaded", num_samples=len(sequences), num_classes=num_classes)

                requested_train_size = int(round(len(sequences) * 0.8))
                num_aug_per_sample, aug_probability, max_aug_train_samples = (
                    self._resolve_augmentation_policy(
                        mode=mode,
                        config=config,
                        train_size=max(1, requested_train_size),
                        target_class_samples=target_count,
                    )
                )

                # Split first, augment train only to avoid train/val leakage.
                split_val_ratio = float(config.get("val_ratio", 0.25 if mode == "few-shot" else 0.2))
                split_val_ratio = float(np.clip(split_val_ratio, 0.05, 0.5))
                min_val_per_class = int(config.get("min_val_samples_per_class", 2 if mode == "few-shot" else 1))
                min_val_per_class = max(1, min(8, min_val_per_class))
                train_sequences, train_labels, val_sequences, val_labels = self._split_and_prepare_sequences(
                    sequences,
                    labels,
                    val_ratio=split_val_ratio,
                    min_val_samples_per_class=min_val_per_class,
                    apply_augmentation=bool(config.get("augmentation", True)),
                    num_augmentations_per_sample=num_aug_per_sample,
                    augmentation_probability=aug_probability,
                    max_augmented_train_samples=max_aug_train_samples,
                )
                min_val_samples_required = int(
                    config.get("min_validation_samples", 3 if mode == "few-shot" else 8)
                )
                min_val_samples_required = max(1, min_val_samples_required)
                if len(val_sequences) < min_val_samples_required:
                    raise ValueError(
                        "Validation split too small for reliable metrics "
                        f"(val_samples={len(val_sequences)} < required={min_val_samples_required}). "
                        "Add more clips before training."
                    )
                logger.info(
                    "dataset_split_ready",
                    train_samples=len(train_sequences),
                    val_samples=len(val_sequences),
                    val_ratio=split_val_ratio,
                    min_val_samples_per_class=min_val_per_class,
                    augmentation=bool(config.get("augmentation", True)),
                    augmentations_per_sample=num_aug_per_sample,
                    augmentation_probability=aug_probability,
                    max_augmented_train_samples=max_aug_train_samples,
                )

                train_samples = [
                    SignSample(sequence, label) for sequence, label in zip(train_sequences, train_labels)
                ]
                val_samples = [
                    SignSample(sequence, label) for sequence, label in zip(val_sequences, val_labels)
                ]

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
                        d_model=192,
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
                use_class_weights = bool(config.get("use_class_weights", mode != "few-shot"))
                use_weighted_sampler = bool(config.get("use_weighted_sampler", mode == "few-shot"))
                use_focal_loss = bool(config.get("use_focal_loss", mode == "few-shot"))
                if mode == "few-shot":
                    # Avoid stacking class weights with sampler+focal simultaneously.
                    use_class_weights = False
                    use_weighted_sampler = True
                    use_focal_loss = True
                else:
                    use_class_weights = True
                    use_weighted_sampler = False
                    use_focal_loss = False

                teacher_model: SignTransformer | None = None
                use_distillation = bool(config.get("use_distillation", False))
                teacher_model_path = str(config.get("teacher_model_path", "")).strip()
                if use_distillation:
                    try:
                        if teacher_model_path and Path(teacher_model_path).exists():
                            teacher_model = load_model_checkpoint(teacher_model_path, device=device)
                        elif mode == "few-shot" and active_model and active_model.file_path and Path(active_model.file_path).exists():
                            teacher_model = load_model_checkpoint(active_model.file_path, device=device)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("teacher_model_load_failed", error=str(exc))
                        teacher_model = None
                        use_distillation = False
                if use_distillation and teacher_model is not None:
                    teacher_num_classes = int(getattr(teacher_model, "num_classes", 0))
                    if teacher_num_classes != int(num_classes):
                        logger.warning(
                            "distillation_disabled_class_mismatch",
                            teacher_num_classes=teacher_num_classes,
                            student_num_classes=int(num_classes),
                        )
                        teacher_model = None
                        use_distillation = False

                ml_config = MLTrainingConfig(
                    num_epochs=int(config.get("epochs", 50)),
                    learning_rate=float(
                        config.get("learning_rate", 1e-4 if mode == "few-shot" else 3e-4)
                    ),
                    batch_size=32,
                    num_workers=self._resolve_dataloader_workers(int(config.get("num_workers", 2))),
                    device=device,
                    sequence_length=sequence_length,
                    early_stopping_patience=int(config.get("early_stopping_patience", 15)),
                    early_stopping_min_delta=float(config.get("early_stopping_min_delta", 1e-4)),
                    weight_decay=float(config.get("weight_decay", 0.05)),
                    classifier_lr_multiplier=float(config.get("classifier_lr_multiplier", 2.0)),
                    label_smoothing=float(config.get("label_smoothing", 0.1)),
                    warmup_epochs=int(config.get("warmup_epochs", 3)),
                    use_focal_loss=use_focal_loss,
                    use_class_weights=use_class_weights,
                    use_weighted_sampler=use_weighted_sampler,
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
                    use_distillation=use_distillation,
                    distillation_alpha=float(config.get("distillation_alpha", 0.25)),
                    distillation_temperature=float(config.get("distillation_temperature", 2.0)),
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
                    teacher_model=teacher_model,
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

                eval_repeats = int(config.get("eval_repeats", 1))
                logits, eval_labels = self._collect_logits_and_labels(
                    model=model,
                    dataset=val_dataset,
                    device=device,
                    batch_size=ml_config.batch_size,
                    repeats=eval_repeats,
                )
                calibration_enabled = bool(config.get("calibration_enabled", True))
                calibration_temperature = (
                    self._fit_temperature(logits, eval_labels)
                    if calibration_enabled
                    else 1.0
                )
                probs = self._apply_temperature(logits, calibration_temperature)
                class_thresholds = self._learn_class_thresholds(probs, eval_labels, class_labels)
                decoded = self._predict_with_thresholds(
                    probs,
                    class_labels=class_labels,
                    class_thresholds=class_thresholds,
                )

                target_sign_slug = slug_by_id.get(sign_id) if sign_id else None
                eval_metrics = self._compute_eval_metrics(
                    y_true=eval_labels,
                    y_pred=decoded,
                    class_labels=class_labels,
                    target_label=target_sign_slug,
                )
                macro_f1 = float(eval_metrics.get("macro_f1") or 0.0)
                target_sign_f1 = (
                    float(eval_metrics["target_sign_f1"])
                    if eval_metrics.get("target_sign_f1") is not None
                    else None
                )
                open_set_fpr = (
                    float(eval_metrics["open_set_fpr"])
                    if eval_metrics.get("open_set_fpr") is not None
                    else None
                )
                latency_p95_ms = self._estimate_latency_p95_ms(
                    model=model,
                    dataset=val_dataset,
                    device=device,
                )
                deployment_ready = self._passes_deployment_gate(
                    mode=mode,
                    config=config,
                    macro_f1=macro_f1,
                    target_sign_f1=target_sign_f1,
                    open_set_fpr=open_set_fpr,
                    latency_p95_ms=latency_p95_ms,
                )

                model_eval_report = {
                    "macro_f1": round(macro_f1, 4),
                    "target_sign_f1": round(float(target_sign_f1), 4)
                    if target_sign_f1 is not None
                    else None,
                    "open_set_fpr": round(float(open_set_fpr), 4)
                    if open_set_fpr is not None
                    else None,
                    "latency_p95_ms": round(float(latency_p95_ms), 2),
                    "deployment_gate_passed": bool(deployment_ready),
                }

                runtime_metadata = {
                    "class_thresholds": class_thresholds,
                    "calibration_temperature": calibration_temperature,
                    "eval_report": model_eval_report,
                }
                trainer.save_model(
                    model_file,
                    class_labels=class_labels,
                    metadata=runtime_metadata,
                )

                logger.info("model_saved", path=str(model_file), version=version)

                # Create model version record
                final_accuracy = metrics_history[-1].val_accuracy if metrics_history else 0.0
                next_action = "deploy" if deployment_ready else "collect_more_examples"

                parent_version = existing[-1].version if existing else None
                model_version = ModelVersion(
                    version=version,
                    is_active=False,
                    num_classes=num_classes,
                    accuracy=final_accuracy,
                    class_labels=class_labels,
                    artifact_metadata={
                        "class_thresholds": class_thresholds,
                        "calibration": {"temperature": calibration_temperature},
                        "eval_report": model_eval_report,
                    },
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
                    macro_f1=macro_f1,
                    target_sign_f1=target_sign_f1,
                    open_set_fpr=open_set_fpr,
                    latency_p95_ms=latency_p95_ms,
                    calibration_temperature=calibration_temperature,
                    deployment_gate_passed=deployment_ready,
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
