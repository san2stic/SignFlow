"""Training loop for sign language recognition models."""

from __future__ import annotations

import contextlib
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import structlog
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from app.ml.curriculum import CurriculumSampler
from app.ml.distillation import DistillationTrainer
from app.ml.dataset import LandmarkDataset
from app.ml.feature_alignment import align_torch_features, resolve_model_feature_dim
from app.ml.model import SignTransformer
from app.ml.tracking import MLFlowTracker, create_default_tracker
from app.ml.gpu_manager import GPUManager, GPUConfig, get_gpu_manager

logger = structlog.get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""

    num_epochs: int = 50
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_workers: int = 2
    device: str = "cpu"  # "cpu" or "cuda" or "mps"
    sequence_length: int = 64
    early_stopping_patience: int = 15       # was 10
    early_stopping_min_delta: float = 1e-4
    gradient_clip_max_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    temporal_mask_prob: float = 0.15
    temporal_mask_span_ratio: float = 0.2

    weight_decay: float = 0.05              # was 0.01
    classifier_lr_multiplier: float = 2.0
    label_smoothing: float = 0.1            # was 0.05

    use_focal_loss: bool = False  # Useful for few-shot learning
    focal_loss_gamma: float = 2.0
    focal_loss_alpha: float = 0.25

    warmup_epochs: int = 3                  # was 5
    use_class_weights: bool = True
    use_weighted_sampler: bool = True

    use_mixup: bool = True                  # was False
    mixup_alpha: float = 0.3               # was 0.2

    use_ema: bool = True
    ema_decay: float = 0.995

    use_amp: bool = True
    amp_dtype: str = "float16"  # "float16" | "bfloat16"
    use_swa: bool = True
    swa_start_ratio: float = 0.75
    swa_lr: float | None = None

    use_distillation: bool = False
    distillation_alpha: float = 0.5
    distillation_temperature: float = 4.0
    use_curriculum: bool = False
    curriculum_strategy: str = "length"  # "length" | "confidence"
    curriculum_start_fraction: float = 0.4
    curriculum_warmup_epochs: int = 2
    curriculum_min_samples: int = 64
    curriculum_confidence_momentum: float = 0.8

    # Stable imbalance controls.
    class_weight_power: float = 0.5
    class_weight_min: float = 0.35
    class_weight_max: float = 4.0
    weighted_sampler_power: float = 0.75
    weighted_sampler_min: float = 0.25
    weighted_sampler_max: float = 5.0

    # Reproducibility
    seed: int = 42

    # MLflow tracking
    use_mlflow: bool = True
    mlflow_run_name: str | None = None
    mlflow_tags: dict[str, str] | None = None

    # [MLflow Sentinel] Minimum validation set size guard
    min_val_size_warn: int = 20
    min_val_size_error: int = 0  # 0 = no hard block


@dataclass
class TrainingMetrics:
    """Metrics for one epoch."""

    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    learning_rate: float
    duration_sec: float


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in few-shot learning."""

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Model output logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size]

        Returns:
            Scalar loss value
        """
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        num_classes = logits.size(1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.label_smoothing / max(1, num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        ce = -(true_dist * log_probs).sum(dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0)
        focal_factor = (1 - pt) ** self.gamma
        alpha_factor = torch.full_like(pt, self.alpha)

        if self.class_weights is not None:
            sample_weights = self.class_weights.to(logits.device)[targets]
        else:
            sample_weights = torch.ones_like(pt)

        loss = alpha_factor * focal_factor * ce * sample_weights
        return loss.mean()


class SignTrainer:
    """PyTorch trainer for sign language recognition models."""

    def __init__(
        self,
        model: SignTransformer,
        config: TrainingConfig,
        progress_callback: Callable[[TrainingMetrics], None] | None = None,
        stop_signal: Callable[[], bool] | None = None,
        teacher_model: SignTransformer | None = None,
    ):
        """
        Initialize trainer.

        Args:
            model: SignTransformer model to train
            config: Training configuration
            progress_callback: Optional callback called after each epoch with metrics
            stop_signal: Optional callable that returns True if training should stop
        """
        self.model = model
        self.config = config
        self.progress_callback = progress_callback
        self.stop_signal = stop_signal
        self.teacher_model = teacher_model
        self.distillation_helper = DistillationTrainer(
            temperature=float(self.config.distillation_temperature),
            alpha=float(self.config.distillation_alpha),
        )

        # Setup GPU manager and device
        if config.device == "auto":
            self.gpu_manager = get_gpu_manager()
            self.device = self.gpu_manager.get_device()
        else:
            self.device = torch.device(config.device)
            gpu_config = GPUConfig(
                device=config.device,
                enable_amp=config.use_amp,
                amp_dtype=config.amp_dtype,
            )
            self.gpu_manager = GPUManager(gpu_config)
            self.gpu_manager.setup_device()

        # Move model to device
        self.model.to(self.device)
        if self.teacher_model is not None:
            self.teacher_model.to(self.device)
            self.teacher_model.eval()
            for parameter in self.teacher_model.parameters():
                parameter.requires_grad = False
        self.accumulation_steps = max(1, int(self.config.gradient_accumulation_steps))

        # Configure AMP based on device capabilities
        self.autocast_enabled = self.gpu_manager.supports_amp()
        if self.autocast_enabled:
            amp_dtype = self.gpu_manager.get_amp_dtype()
            self.autocast_dtype = amp_dtype if amp_dtype else torch.float16
        else:
            self.autocast_dtype = torch.float16
        # Setup gradient scaler for AMP
        if self.device.type == "cuda" and self.autocast_enabled:
            try:
                self.grad_scaler = torch.amp.GradScaler("cuda", enabled=True)
            except (AttributeError, TypeError):
                self.grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            # MPS and CPU don't use gradient scaler
            self.grad_scaler = None

        # Loss function (initialized in fit once class distribution is known)
        self.criterion: nn.Module = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)

        # Optimizer with differential learning rates (faster adaptation for classifier)
        classifier_params = list(self.model.classifier.parameters())
        classifier_param_ids = {id(parameter) for parameter in classifier_params}
        backbone_params = [
            parameter
            for parameter in self.model.parameters()
            if id(parameter) not in classifier_param_ids and parameter.requires_grad
        ]
        optimizer_groups = []
        if backbone_params:
            optimizer_groups.append(
                {
                    "params": backbone_params,
                    "lr": self.config.learning_rate,
                    "weight_decay": self.config.weight_decay,
                }
            )
        if classifier_params:
            optimizer_groups.append(
                {
                    "params": classifier_params,
                    "lr": self.config.learning_rate * self.config.classifier_lr_multiplier,
                    "weight_decay": self.config.weight_decay,
                }
            )
        self.optimizer = AdamW(optimizer_groups if optimizer_groups else self.model.parameters())

        # Scheduler with warmup + cosine decay.
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=self._build_lr_lambda(
                num_epochs=max(1, self.config.num_epochs),
                warmup_epochs=max(0, self.config.warmup_epochs),
            ),
        )

        # Training state
        self.best_val_loss = float("inf")
        self.best_model_state: dict[str, torch.Tensor] | None = None
        self.epochs_without_improvement = 0
        self._optimizer_step_count = 0
        self.metrics_history: list[TrainingMetrics] = []
        self._mlflow_tracker: MLFlowTracker | None = None
        self.mlflow_run_id: str | None = None

        # EMA state (helps stabilization for few-shot and noisy samples).
        self.ema_state: dict[str, torch.Tensor] | None = None
        if self.config.use_ema:
            self.ema_state = {
                name: parameter.detach().clone()
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad
            }

        # Stochastic Weight Averaging for stronger final generalization.
        self.swa_model: AveragedModel | None = None
        self._swa_started = False
        self._swa_start_epoch = max(1, int(math.ceil(self.config.num_epochs * self.config.swa_start_ratio)))
        if self.config.use_swa:
            self.swa_model = AveragedModel(self.model)

    def _autocast_context(self):
        """Return autocast context when CUDA AMP is enabled."""
        if not self.autocast_enabled:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)

    @staticmethod
    def _build_lr_lambda(num_epochs: int, warmup_epochs: int) -> Callable[[int], float]:
        """Create warmup + cosine scheduler lambda."""
        warmup_epochs = min(warmup_epochs, max(0, num_epochs - 1))

        def lr_lambda(epoch_idx: int) -> float:
            step = epoch_idx + 1
            if warmup_epochs > 0 and step <= warmup_epochs:
                return step / float(warmup_epochs)

            cosine_steps = max(1, num_epochs - warmup_epochs)
            progress = (step - warmup_epochs) / float(cosine_steps)
            progress = min(max(progress, 0.0), 1.0)
            min_ratio = 0.05
            return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

        return lr_lambda

    @staticmethod
    def _dataset_labels(dataset: Dataset) -> list[int]:
        """Extract labels from LandmarkDataset or a torch Subset wrapping it."""
        if isinstance(dataset, LandmarkDataset):
            return [int(label) for _sequence, label in dataset.processed_samples]

        if isinstance(dataset, Subset):
            base = dataset.dataset
            if isinstance(base, LandmarkDataset):
                return [
                    int(base.processed_samples[int(index)][1])
                    for index in dataset.indices
                ]

        labels: list[int] = []
        for sample_index in range(len(dataset)):
            _sequence, label = dataset[sample_index]
            labels.append(int(label))
        return labels

    @classmethod
    def _class_distribution(cls, dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
        """Return class ids and counts from any compatible torch dataset."""
        labels = cls._dataset_labels(dataset)
        if not labels:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        return np.unique(np.array(labels, dtype=np.int64), return_counts=True)

    @staticmethod
    def _stable_inverse_frequency_weights(
        counts: np.ndarray,
        *,
        power: float,
        min_weight: float,
        max_weight: float,
    ) -> np.ndarray:
        """Compute numerically stable inverse-frequency style weights with clipping."""
        if counts.size == 0:
            return np.array([], dtype=np.float32)
        base = np.maximum(counts.astype(np.float32), 1.0)
        normalized = base / max(float(np.mean(base)), 1e-6)
        inv = 1.0 / np.power(normalized, max(0.0, float(power)))
        inv = inv / max(float(np.mean(inv)), 1e-6)
        inv = np.clip(inv, float(min_weight), float(max_weight)).astype(np.float32)
        return inv

    def _configure_criterion(self, train_dataset: Dataset) -> None:
        """Configure criterion using class weights from train set if requested."""
        class_weights_tensor: torch.Tensor | None = None
        class_ids, counts = self._class_distribution(train_dataset)
        if self.config.use_class_weights and len(class_ids) > 0:
            num_classes = self.model.num_classes
            weights = np.ones((num_classes,), dtype=np.float32)
            inv = self._stable_inverse_frequency_weights(
                counts,
                power=self.config.class_weight_power,
                min_weight=self.config.class_weight_min,
                max_weight=self.config.class_weight_max,
            )
            for class_id, value in zip(class_ids.tolist(), inv.tolist()):
                if 0 <= class_id < num_classes:
                    weights[class_id] = float(value)
            class_weights_tensor = torch.from_numpy(weights).to(self.device)

        if self.config.use_focal_loss:
            self.criterion = FocalLoss(
                alpha=self.config.focal_loss_alpha,
                gamma=self.config.focal_loss_gamma,
                class_weights=class_weights_tensor,
                label_smoothing=self.config.label_smoothing,
            )
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights_tensor,
                label_smoothing=self.config.label_smoothing,
            )

    def _build_train_loader(self, train_dataset: Dataset) -> DataLoader:
        """Build training dataloader with optional class-balanced sampler."""
        sampler = None
        shuffle = True

        if self.config.use_weighted_sampler and len(train_dataset) > 0:
            class_ids, counts = self._class_distribution(train_dataset)
            if len(class_ids) > 0:
                inv = self._stable_inverse_frequency_weights(
                    counts,
                    power=self.config.weighted_sampler_power,
                    min_weight=self.config.weighted_sampler_min,
                    max_weight=self.config.weighted_sampler_max,
                )
                class_to_weight = {
                    int(class_id): float(weight)
                    for class_id, weight in zip(class_ids.tolist(), inv.tolist())
                }
                labels = self._dataset_labels(train_dataset)
                sample_weights = [class_to_weight.get(int(label), 1.0) for label in labels]
                sampler = WeightedRandomSampler(
                    weights=torch.tensor(sample_weights, dtype=torch.double),
                    num_samples=len(sample_weights),
                    replacement=True,
                )
                shuffle = False

        return DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.config.num_workers > 0,
        )

    def _mixup_batch(
        self,
        landmarks: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply mixup on batch."""
        alpha = max(self.config.mixup_alpha, 1e-6)
        lam = float(np.random.beta(alpha, alpha))
        permutation = torch.randperm(landmarks.size(0), device=landmarks.device)

        mixed_landmarks = lam * landmarks + (1.0 - lam) * landmarks[permutation]
        labels_a = labels
        labels_b = labels[permutation]
        return mixed_landmarks, labels_a, labels_b, lam

    def _compute_mixup_loss(
        self,
        logits: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """Compute criterion under mixup target interpolation."""
        return (lam * self.criterion(logits, labels_a)) + ((1.0 - lam) * self.criterion(logits, labels_b))

    def _compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL distillation loss from teacher to student."""
        return self.distillation_helper.distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
        )

    def _update_ema(self) -> None:
        """Update exponential moving average of trainable parameters."""
        if self.ema_state is None:
            return
        decay = self.config.ema_decay
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if not parameter.requires_grad:
                    continue
                if name not in self.ema_state:
                    self.ema_state[name] = parameter.detach().clone()
                    continue
                self.ema_state[name].mul_(decay).add_(parameter.detach(), alpha=(1.0 - decay))

    def _apply_temporal_mask(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Mask contiguous temporal spans to regularize against tracking dropouts."""
        if (
            self.config.temporal_mask_prob <= 0.0
            or self.config.temporal_mask_span_ratio <= 0.0
            or landmarks.ndim != 3
            or landmarks.size(1) < 4
        ):
            return landmarks

        masked = landmarks.clone()
        batch_size, seq_len, _num_features = masked.shape
        max_span = max(1, int(seq_len * self.config.temporal_mask_span_ratio))
        mask_prob = float(np.clip(self.config.temporal_mask_prob, 0.0, 1.0))

        for batch_idx in range(batch_size):
            if np.random.random() > mask_prob:
                continue
            span = int(np.random.randint(1, max_span + 1))
            start = int(np.random.randint(0, max(1, seq_len - span + 1)))
            masked[batch_idx, start : start + span] = 0.0

        return masked

    def _optimizer_step(self) -> None:
        """Apply optimizer step with optional AMP scaler."""
        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(self.optimizer)

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.gradient_clip_max_norm,
        )

        if self.grad_scaler is not None:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()

        self._update_ema()
        self._optimizer_step_count += 1
        self.optimizer.zero_grad(set_to_none=True)

    @staticmethod
    def _align_landmarks_for_model(landmarks: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """Adapt batch feature width to match model.num_features."""
        target_dim = resolve_model_feature_dim(model)
        return align_torch_features(landmarks, target_dim)

    def _validate_with_model(
        self,
        model: nn.Module,
        dataloader: DataLoader,
    ) -> tuple[float, float]:
        """Run validation metrics on a provided model without EMA swapping."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        steps = 0

        with torch.no_grad():
            for landmarks, labels in dataloader:
                landmarks = landmarks.to(self.device)
                labels = labels.to(self.device)
                landmarks = self._align_landmarks_for_model(landmarks, model)

                with self._autocast_context():
                    logits = model(landmarks)
                    loss = self.criterion(logits, labels)

                total_loss += float(loss.item())
                predictions = torch.argmax(logits, dim=1)
                correct += int((predictions == labels).sum().item())
                total += int(labels.size(0))
                steps += 1

        avg_loss = total_loss / steps if steps > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    def _maybe_update_swa(self, epoch: int) -> None:
        """Accumulate SWA weights once the configured epoch ratio is reached."""
        if self.swa_model is None:
            return
        if epoch < self._swa_start_epoch:
            return
        self.swa_model.update_parameters(self.model)
        self._swa_started = True

    def _maybe_select_swa_model(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Evaluate SWA model and keep it if it outperforms the current best checkpoint."""
        if self.swa_model is None or not self._swa_started:
            return

        try:
            sample_dim = None
            train_dataset = getattr(train_loader, "dataset", None)
            if train_dataset is not None and len(train_dataset) > 0:
                try:
                    sample_tensor, _sample_label = train_dataset[0]
                    sample_dim = int(sample_tensor.shape[-1])
                except Exception:  # noqa: BLE001
                    sample_dim = None
            swa_dim = resolve_model_feature_dim(self.swa_model)
            if sample_dim is not None and swa_dim is not None and sample_dim != swa_dim:
                logger.info(
                    "swa_batch_norm_update_skipped_feature_mismatch",
                    sample_dim=sample_dim,
                    model_dim=swa_dim,
                )
            else:
                update_bn(train_loader, self.swa_model, device=self.device)
        except Exception as exc:  # noqa: BLE001
            logger.warning("swa_batch_norm_update_failed", error=str(exc))

        swa_val_loss, swa_val_accuracy = self._validate_with_model(self.swa_model, val_loader)
        if swa_val_loss < (self.best_val_loss - self.config.early_stopping_min_delta):
            self.best_val_loss = swa_val_loss
            self.best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in self.swa_model.module.state_dict().items()
            }
            logger.info(
                "swa_selected_as_best_model",
                swa_val_loss=swa_val_loss,
                swa_val_acc=swa_val_accuracy,
            )

    @contextlib.contextmanager
    def _swap_to_ema_weights(self):
        """Temporarily swap model weights to EMA values during evaluation."""
        if self.ema_state is None:
            yield
            return

        backup = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and name in self.ema_state
        }

        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad and name in self.ema_state:
                    parameter.copy_(self.ema_state[name])

        try:
            yield
        finally:
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    if parameter.requires_grad and name in backup:
                        parameter.copy_(backup[name])

    def train_epoch(self, dataloader: DataLoader) -> tuple[float, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training DataLoader

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0.0
        total = 0
        steps = 0
        pending_grad_batches = 0
        total_batches = len(dataloader)

        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, (landmarks, labels) in enumerate(dataloader):
            # Move to device
            landmarks = landmarks.to(self.device)  # [batch, seq_len, features]
            labels = labels.to(self.device)  # [batch]
            landmarks = self._apply_temporal_mask(landmarks)
            landmarks = self._align_landmarks_for_model(landmarks, self.model)

            # Forward pass
            with self._autocast_context():
                distillation_inputs = landmarks
                if self.config.use_mixup and labels.size(0) >= 2:
                    mixed_landmarks, labels_a, labels_b, lam = self._mixup_batch(landmarks, labels)
                    logits = self.model(mixed_landmarks)
                    distillation_inputs = mixed_landmarks
                    loss = self._compute_mixup_loss(logits, labels_a, labels_b, lam)
                    predictions = torch.argmax(logits, dim=1)
                    # Soft accuracy estimate under mixed labels.
                    batch_correct = lam * (predictions == labels_a).sum().item() + (1.0 - lam) * (
                        predictions == labels_b
                    ).sum().item()
                    correct += float(batch_correct)
                else:
                    logits = self.model(landmarks)  # [batch, num_classes]
                    loss = self.criterion(logits, labels)
                    predictions = torch.argmax(logits, dim=1)
                    correct += float((predictions == labels).sum().item())

                if self.config.use_distillation and self.teacher_model is not None:
                    with torch.no_grad():
                        teacher_inputs = self._align_landmarks_for_model(
                            distillation_inputs,
                            self.teacher_model,
                        )
                        teacher_logits = self.teacher_model(teacher_inputs)
                    loss = self.distillation_helper.combined_loss(
                        hard_loss=loss,
                        student_logits=logits,
                        teacher_logits=teacher_logits,
                    )

            loss_value = float(loss.item())
            scaled_loss = loss / self.accumulation_steps

            # Backward pass
            if self.grad_scaler is not None:
                self.grad_scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            pending_grad_batches += 1

            should_step = pending_grad_batches >= self.accumulation_steps or (
                batch_idx + 1
            ) == total_batches
            if should_step:
                self._optimizer_step()
                pending_grad_batches = 0
                # GPU memory management
                self.gpu_manager.on_batch_end()

            # Metrics
            total_loss += loss_value
            total += int(labels.size(0))
            steps += 1

            # Check stop signal
            if self.stop_signal and self.stop_signal():
                logger.info("training_interrupted_by_stop_signal")
                break

        # Ensure no gradients remain pending if the loop exits early.
        if pending_grad_batches > 0:
            self._optimizer_step()

        avg_loss = total_loss / steps if steps > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def validate(self, dataloader: DataLoader) -> tuple[float, float]:
        """
        Validate model on validation set.

        Args:
            dataloader: Validation DataLoader

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        steps = 0

        with self._swap_to_ema_weights(), torch.no_grad():
            for landmarks, labels in dataloader:
                # Move to device
                landmarks = landmarks.to(self.device)
                labels = labels.to(self.device)
                landmarks = self._align_landmarks_for_model(landmarks, self.model)

                # Forward pass
                with self._autocast_context():
                    logits = self.model(landmarks)
                    # Compute loss
                    loss = self.criterion(logits, labels)

                # Metrics
                total_loss += float(loss.item())
                predictions = torch.argmax(logits, dim=1)
                correct += int((predictions == labels).sum().item())
                total += int(labels.size(0))
                steps += 1

        avg_loss = total_loss / steps if steps > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def _compute_eval_report(self, val_loader: DataLoader) -> dict:
        """[MLflow Sentinel] Compute F1, precision, recall and confusion matrix."""
        all_preds: list[int] = []
        all_labels: list[int] = []

        self.model.eval()  # PyTorch inference mode
        with self._swap_to_ema_weights(), torch.no_grad():
            for landmarks, labels in val_loader:
                landmarks = landmarks.to(self.device)
                labels = labels.to(self.device)
                landmarks = self._align_landmarks_for_model(landmarks, self.model)
                with self._autocast_context():
                    logits = self.model(landmarks)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        if not all_labels:
            return {}

        num_classes = self.model.num_classes
        tp = [0] * num_classes
        fp = [0] * num_classes
        fn = [0] * num_classes
        confusion = [[0] * num_classes for _ in range(num_classes)]

        for true, pred in zip(all_labels, all_preds):
            confusion[true][pred] += 1
            if pred == true:
                tp[pred] += 1
            else:
                fp[pred] += 1
                fn[true] += 1

        precision_per_class = []
        recall_per_class = []
        f1_per_class = []
        support_per_class = []

        for c in range(num_classes):
            p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
            r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            support = sum(1 for lbl in all_labels if lbl == c)
            precision_per_class.append(round(p, 4))
            recall_per_class.append(round(r, 4))
            f1_per_class.append(round(f1, 4))
            support_per_class.append(support)

        total_support = sum(support_per_class)
        f1_macro = sum(f1_per_class) / num_classes if num_classes > 0 else 0.0
        precision_macro = sum(precision_per_class) / num_classes if num_classes > 0 else 0.0
        recall_macro = sum(recall_per_class) / num_classes if num_classes > 0 else 0.0
        f1_weighted = (
            sum(f * s for f, s in zip(f1_per_class, support_per_class)) / total_support
            if total_support > 0 else 0.0
        )

        report = {
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "precision_macro": round(precision_macro, 4),
            "recall_macro": round(recall_macro, 4),
            "per_class": {
                str(c): {
                    "precision": precision_per_class[c],
                    "recall": recall_per_class[c],
                    "f1": f1_per_class[c],
                    "support": support_per_class[c],
                }
                for c in range(num_classes)
            },
            "confusion_matrix": confusion,
            "val_size": len(all_labels),
        }

        logger.info(
            "evaluation_report",
            f1_macro=report["f1_macro"],
            f1_weighted=report["f1_weighted"],
            precision_macro=report["precision_macro"],
            recall_macro=report["recall_macro"],
            val_size=report["val_size"],
        )

        return report

    def _estimate_curriculum_confidence(self, dataset: LandmarkDataset) -> dict[int, float]:
        """Estimate per-sample true-label confidence for confidence-based curriculum."""
        if len(dataset) == 0:
            return {}

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.config.num_workers > 0,
        )
        confidence_by_index: dict[int, float] = {}
        offset = 0

        self.model.eval()
        with self._swap_to_ema_weights(), torch.no_grad():
            for landmarks, labels in loader:
                landmarks = landmarks.to(self.device)
                labels = labels.to(self.device).long()
                landmarks = self._align_landmarks_for_model(landmarks, self.model)
                with self._autocast_context():
                    logits = self.model(landmarks)
                    probs = torch.softmax(logits, dim=1)
                batch_confidence = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
                values = batch_confidence.detach().cpu().numpy()
                for value in values.tolist():
                    confidence_by_index[offset] = float(np.clip(value, 0.0, 1.0))
                    offset += 1

        return confidence_by_index

    def fit(
        self,
        train_dataset: LandmarkDataset,
        val_dataset: LandmarkDataset,
    ) -> list[TrainingMetrics]:
        """
        Train model on train and validation datasets.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset

        Returns:
            List of training metrics for each epoch
        """
        # [MLflow Sentinel] Fix reproducibility — seed all RNGs
        _seed = self.config.seed
        random.seed(_seed)
        np.random.seed(_seed)
        torch.manual_seed(_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_seed)

        # [MLflow Sentinel] Validate minimum val_size
        val_size = len(val_dataset)
        if self.config.min_val_size_error > 0 and val_size < self.config.min_val_size_error:
            raise ValueError(
                f"Validation set too small: {val_size} < {self.config.min_val_size_error}. "
                "Set config.min_val_size_error=0 to disable this check."
            )
        if val_size < self.config.min_val_size_warn:
            logger.warning(
                "small_validation_set",
                val_size=val_size,
                recommended_min=self.config.min_val_size_warn,
                msg=f"val_size={val_size} is too small for reliable evaluation. "
                    f"Metrics may have high variance. Recommended: >= {self.config.min_val_size_warn}.",
            )

        # [MLflow Sentinel] Warn if model is over-parameterized for dataset size
        train_size = len(train_dataset)
        num_params = sum(p.numel() for p in self.model.parameters())
        params_per_sample = num_params / max(train_size, 1)
        if params_per_sample > 100:
            logger.warning(
                "over_parameterized_model",
                num_params=num_params,
                train_size=train_size,
                params_per_sample=round(params_per_sample, 1),
                msg=f"Model has {num_params:,} params for {train_size} samples "
                    f"(ratio {params_per_sample:.0f}:1). Consider using LIGHTWEIGHT_CONFIG "
                    f"from model_configs.py for better generalization.",
            )

        logger.info(
            "starting_training",
            num_epochs=self.config.num_epochs,
            train_size=len(train_dataset),
            val_size=val_size,
            batch_size=self.config.batch_size,
            seed=_seed,
        )

        # Initialize MLflow tracker
        mlflow_tracker = create_default_tracker(enabled=self.config.use_mlflow)
        self._mlflow_tracker = mlflow_tracker

        self._configure_criterion(train_dataset)

        # Create dataloaders
        full_train_loader = self._build_train_loader(train_dataset)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.config.num_workers > 0,
        )
        curriculum_sampler: CurriculumSampler | None = None
        if self.config.use_curriculum and len(train_dataset) > 0:
            curriculum_sampler = CurriculumSampler(
                strategy=self.config.curriculum_strategy,
                start_fraction=self.config.curriculum_start_fraction,
                warmup_epochs=self.config.curriculum_warmup_epochs,
                min_samples=min(len(train_dataset), self.config.curriculum_min_samples),
                confidence_momentum=self.config.curriculum_confidence_momentum,
            )
            logger.info(
                "curriculum_enabled",
                strategy=self.config.curriculum_strategy,
                start_fraction=self.config.curriculum_start_fraction,
                warmup_epochs=self.config.curriculum_warmup_epochs,
                min_samples=min(len(train_dataset), self.config.curriculum_min_samples),
            )

        # Start MLflow run and log parameters
        with mlflow_tracker.start_run(
            run_name=self.config.mlflow_run_name,
            tags=self.config.mlflow_tags or {}
        ):
            self.mlflow_run_id = mlflow_tracker.run_id
            # Log hyperparameters
            mlflow_tracker.log_params({
                "num_epochs": self.config.num_epochs,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "device": str(self.device),
                "sequence_length": self.config.sequence_length,
                "weight_decay": self.config.weight_decay,
                "label_smoothing": self.config.label_smoothing,
                "warmup_epochs": self.config.warmup_epochs,
                "use_mixup": self.config.use_mixup,
                "mixup_alpha": self.config.mixup_alpha,
                "use_ema": self.config.use_ema,
                "ema_decay": self.config.ema_decay,
                "use_amp": self.config.use_amp,
                "use_swa": self.config.use_swa,
                "use_focal_loss": self.config.use_focal_loss,
                "use_distillation": self.config.use_distillation,
                "distillation_alpha": self.config.distillation_alpha,
                "distillation_temperature": self.config.distillation_temperature,
                "use_curriculum": self.config.use_curriculum,
                "curriculum_strategy": self.config.curriculum_strategy,
                "curriculum_start_fraction": self.config.curriculum_start_fraction,
                "curriculum_warmup_epochs": self.config.curriculum_warmup_epochs,
                "curriculum_min_samples": self.config.curriculum_min_samples,
                "d_model": self.model.d_model,
                "num_layers": self.model.num_layers,
                "nhead": self.model.nhead,
                "num_classes": self.model.num_classes,
                "num_features": self.model.num_features,
                "train_size": len(train_dataset),
                "val_size": val_size,
                "seed": _seed,
            })

            # Training loop
            for epoch in range(self.config.num_epochs):
                epoch_start = time.time()
                curriculum_fraction = 1.0
                curriculum_selected = len(train_dataset)
                epoch_train_loader = full_train_loader

                # Check stop signal before starting epoch
                if self.stop_signal and self.stop_signal():
                    logger.info("training_stopped_before_epoch", epoch=epoch)
                    break

                if curriculum_sampler is not None:
                    if curriculum_sampler.strategy == "confidence":
                        confidence_by_index = self._estimate_curriculum_confidence(train_dataset)
                        curriculum_sampler.update_confidence(confidence_by_index)

                    selected_indices, snapshot = curriculum_sampler.select_indices(
                        train_dataset,
                        epoch=epoch,
                        total_epochs=self.config.num_epochs,
                    )
                    curriculum_fraction = float(snapshot.fraction)
                    curriculum_selected = int(snapshot.selected_samples)
                    if curriculum_selected < len(train_dataset):
                        epoch_train_loader = self._build_train_loader(
                            Subset(train_dataset, selected_indices)
                        )
                    logger.debug(
                        "curriculum_epoch_subset_selected",
                        epoch=epoch + 1,
                        strategy=snapshot.strategy,
                        fraction=curriculum_fraction,
                        selected_samples=curriculum_selected,
                        total_samples=int(snapshot.total_samples),
                    )

                # Train
                optimizer_steps_before_epoch = self._optimizer_step_count
                train_loss, train_accuracy = self.train_epoch(epoch_train_loader)
                optimizer_steps_in_epoch = self._optimizer_step_count - optimizer_steps_before_epoch

                # Validate
                val_loss, val_accuracy = self.validate(val_loader)
                self._maybe_update_swa(epoch + 1)

                # Scheduler step
                if optimizer_steps_in_epoch > 0:
                    self.scheduler.step()
                else:
                    logger.warning(
                        "scheduler_step_skipped_no_optimizer_updates",
                        epoch=epoch + 1,
                    )
                current_lr = float(self.optimizer.param_groups[0]["lr"])

                # Metrics
                epoch_duration = time.time() - epoch_start
                metrics = TrainingMetrics(
                    epoch=epoch + 1,
                    train_loss=train_loss,
                    train_accuracy=train_accuracy,
                    val_loss=val_loss,
                    val_accuracy=val_accuracy,
                    learning_rate=current_lr,
                    duration_sec=epoch_duration,
                )
                self.metrics_history.append(metrics)

                # Logging
                logger.info(
                    "epoch_complete",
                    epoch=epoch + 1,
                    train_loss=train_loss,
                    train_acc=train_accuracy,
                    val_loss=val_loss,
                    val_acc=val_accuracy,
                    lr=current_lr,
                    curriculum_fraction=curriculum_fraction,
                    curriculum_selected=curriculum_selected,
                )

                # Log metrics to MLflow
                mlflow_tracker.log_metrics({
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "learning_rate": current_lr,
                    "epoch_duration_sec": epoch_duration,
                    "curriculum_fraction": curriculum_fraction,
                    "curriculum_selected_samples": float(curriculum_selected),
                }, step=epoch + 1)

                # Progress callback
                if self.progress_callback:
                    self.progress_callback(metrics)

                # Early stopping check
                improved = val_loss < (self.best_val_loss - self.config.early_stopping_min_delta)
                if improved:
                    self.best_val_loss = val_loss
                    # Store the EMA-smoothed weights when enabled.
                    with self._swap_to_ema_weights():
                        self.best_model_state = {
                            key: value.detach().cpu().clone()
                            for key, value in self.model.state_dict().items()
                        }
                    self.epochs_without_improvement = 0
                    logger.debug("new_best_model", val_loss=val_loss)
                else:
                    self.epochs_without_improvement += 1
                    logger.debug(
                        "no_improvement",
                        epochs_without_improvement=self.epochs_without_improvement,
                    )

                    if self.epochs_without_improvement >= self.config.early_stopping_patience:
                        logger.info(
                            "early_stopping_triggered",
                            patience=self.config.early_stopping_patience,
                            best_val_loss=self.best_val_loss,
                        )
                        break

            self._maybe_select_swa_model(full_train_loader, val_loader)

            # Restore best model
            if self.best_model_state is not None:
                self.model.load_state_dict(self.best_model_state)
                logger.info("restored_best_model", val_loss=self.best_val_loss)

            # [MLflow Sentinel] Compute detailed evaluation metrics
            eval_report = self._compute_eval_report(val_loader)

            # Log final metrics summary to MLflow
            if self.metrics_history:
                final_metrics = self.metrics_history[-1]
                final_mlflow_metrics: dict[str, float] = {
                    "final_train_loss": final_metrics.train_loss,
                    "final_train_accuracy": final_metrics.train_accuracy,
                    "final_val_loss": final_metrics.val_loss,
                    "final_val_accuracy": final_metrics.val_accuracy,
                    "best_val_loss": self.best_val_loss,
                    "total_epochs": len(self.metrics_history),
                }
                # [MLflow Sentinel] Add F1/precision/recall if available
                if eval_report:
                    final_mlflow_metrics.update({
                        "val_f1_macro": eval_report["f1_macro"],
                        "val_f1_weighted": eval_report["f1_weighted"],
                        "val_precision_macro": eval_report["precision_macro"],
                        "val_recall_macro": eval_report["recall_macro"],
                    })
                mlflow_tracker.log_metrics(final_mlflow_metrics)

            # [MLflow Sentinel] Log artifacts: classification report + best model
            if eval_report:
                mlflow_tracker.log_dict(eval_report, "classification_report.json")
            # Log best model checkpoint and register in Model Registry
            if self.best_model_state is not None:
                mlflow_tracker.log_model(
                    self.model,
                    artifact_path="best_model",
                    registered_model_name="signflow-model"  # Register in Model Registry
                )

        return self.metrics_history

    def save_model(
        self,
        save_path: str | Path,
        class_labels: list[str] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """
        Save trained model to file.

        Args:
            save_path: Path to save model checkpoint
            class_labels: Ordered class labels matching output logits
            metadata: Optional runtime metadata for inference calibration/gating
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "num_classes": self.model.num_classes,
            "num_features": self.model.num_features,
            "d_model": self.model.d_model,
            "nhead": self.model.nhead,
            "num_layers": self.model.num_layers,
            "dim_feedforward": getattr(self.model, "dim_feedforward", 768),
            "dropout": getattr(self.model, "dropout", 0.2),
            "feature_dropout": getattr(self.model, "feature_dropout", 0.15),
            "pooling_dropout": getattr(self.model, "pooling_dropout_value", 0.2),
            "use_cls_token": getattr(self.model, "use_cls_token", True),
            "token_dropout": getattr(self.model, "token_dropout", 0.0),
            "temporal_smoothing": getattr(self.model, "temporal_smoothing", 0.0),
            "use_multiscale_stem": getattr(self.model, "use_multiscale_stem", False),
            "use_cosine_head": getattr(self.model, "use_cosine_head", False),
            "relative_bias_max_distance": getattr(self.model, "relative_bias_max_distance", 64),
            "cosine_head_weight": getattr(self.model, "cosine_head_weight", 0.35),
            "class_labels": class_labels or [],
            "config": {
                "num_epochs": self.config.num_epochs,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "sequence_length": self.config.sequence_length,
                "weight_decay": self.config.weight_decay,
                "label_smoothing": self.config.label_smoothing,
                "warmup_epochs": self.config.warmup_epochs,
                "use_mixup": self.config.use_mixup,
                "mixup_alpha": self.config.mixup_alpha,
                "use_class_weights": self.config.use_class_weights,
                "use_weighted_sampler": self.config.use_weighted_sampler,
                "use_ema": self.config.use_ema,
                "ema_decay": self.config.ema_decay,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "temporal_mask_prob": self.config.temporal_mask_prob,
                "temporal_mask_span_ratio": self.config.temporal_mask_span_ratio,
                "use_amp": self.config.use_amp,
                "amp_dtype": self.config.amp_dtype,
                "use_swa": self.config.use_swa,
                "swa_start_ratio": self.config.swa_start_ratio,
                "swa_lr": self.config.swa_lr,
                "use_distillation": self.config.use_distillation,
                "distillation_alpha": self.config.distillation_alpha,
                "distillation_temperature": self.config.distillation_temperature,
                "use_curriculum": self.config.use_curriculum,
                "curriculum_strategy": self.config.curriculum_strategy,
                "curriculum_start_fraction": self.config.curriculum_start_fraction,
                "curriculum_warmup_epochs": self.config.curriculum_warmup_epochs,
                "curriculum_min_samples": self.config.curriculum_min_samples,
                "curriculum_confidence_momentum": self.config.curriculum_confidence_momentum,
                "class_weight_power": self.config.class_weight_power,
                "class_weight_min": self.config.class_weight_min,
                "class_weight_max": self.config.class_weight_max,
                "weighted_sampler_power": self.config.weighted_sampler_power,
                "weighted_sampler_min": self.config.weighted_sampler_min,
                "weighted_sampler_max": self.config.weighted_sampler_max,
            },
            "metrics_history": [
                {
                    "epoch": m.epoch,
                    "train_loss": m.train_loss,
                    "train_accuracy": m.train_accuracy,
                    "val_loss": m.val_loss,
                    "val_accuracy": m.val_accuracy,
                }
                for m in self.metrics_history
            ],
            "best_val_loss": self.best_val_loss,
            "metadata": metadata or {},
        }

        torch.save(checkpoint, save_path)
        logger.info("model_saved", path=str(save_path))

        if not self.config.use_mlflow:
            return

        tracker = self._mlflow_tracker or create_default_tracker(enabled=True)
        resumed_run_id: str | None = None

        if self.mlflow_run_id:
            resumed_run_id = tracker.start(run_id=self.mlflow_run_id)
        if resumed_run_id is None:
            resumed_run_id = tracker.start(
                run_name=self.config.mlflow_run_name,
                tags=self.config.mlflow_tags or {},
            )
        self.mlflow_run_id = resumed_run_id or self.mlflow_run_id

        if resumed_run_id is None:
            logger.warning("mlflow_model_artifact_skipped", reason="run_start_failed")
            return

        tracker.log_artifact(save_path, artifact_path="model")
        tracker.log_dict(checkpoint["config"], filename="model/training_config.json")
        if metadata:
            tracker.log_dict(metadata, filename="model/metadata.json")
        tracker.end()


def load_model_checkpoint(checkpoint_path: str | Path, device: str = "cpu") -> SignTransformer:
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded SignTransformer model

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        ValueError: If checkpoint is invalid
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

        # Create model with saved architecture.
        model = SignTransformer(
            num_features=checkpoint.get("num_features", 225),
            num_classes=checkpoint["num_classes"],
            d_model=checkpoint.get("d_model", 192),
            nhead=checkpoint.get("nhead", 6),
            num_layers=checkpoint.get("num_layers", 4),
            dim_feedforward=checkpoint.get("dim_feedforward", 768),
            dropout=checkpoint.get("dropout", 0.2),
            feature_dropout=checkpoint.get("feature_dropout", 0.15),
            pooling_dropout=checkpoint.get("pooling_dropout", 0.2),
            use_cls_token=checkpoint.get("use_cls_token", True),
            token_dropout=checkpoint.get("token_dropout", 0.0),
            temporal_smoothing=checkpoint.get("temporal_smoothing", 0.0),
            use_multiscale_stem=checkpoint.get("use_multiscale_stem", False),
            use_cosine_head=checkpoint.get("use_cosine_head", False),
            relative_bias_max_distance=checkpoint.get("relative_bias_max_distance", 64),
            cosine_head_weight=checkpoint.get("cosine_head_weight", 0.35),
        )

        # Load weights (strict first, fallback for backward compatibility).
        try:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except RuntimeError as strict_error:
            logger.warning(
                "checkpoint_strict_load_failed_falling_back",
                path=str(checkpoint_path),
                error=str(strict_error),
            )
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        model.set_inference_mode()

        logger.info(
            "model_loaded",
            path=str(checkpoint_path),
            num_classes=model.num_classes,
        )

        return model

    except Exception as e:
        logger.error("failed_to_load_checkpoint", path=str(checkpoint_path), error=str(e))
        raise ValueError(f"Failed to load checkpoint: {e}") from e


def train_base_model() -> None:
    """Placeholder entrypoint for full retrain mode."""
    return None
