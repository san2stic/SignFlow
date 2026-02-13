"""Training loop for sign language recognition models."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import structlog
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from app.ml.dataset import LandmarkDataset
from app.ml.model import SignTransformer

logger = structlog.get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""

    num_epochs: int = 50
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_workers: int = 2
    device: str = "cpu"  # "cpu" or "cuda"
    early_stopping_patience: int = 10
    gradient_clip_max_norm: float = 1.0
    use_focal_loss: bool = False  # For few-shot learning
    focal_loss_gamma: float = 2.0
    focal_loss_alpha: float = 0.25


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

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Model output logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size]

        Returns:
            Scalar loss value
        """
        ce_loss = self.ce_loss(logits, targets)
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class SignTrainer:
    """PyTorch trainer for sign language recognition models."""

    def __init__(
        self,
        model: SignTransformer,
        config: TrainingConfig,
        progress_callback: Callable[[TrainingMetrics], None] | None = None,
        stop_signal: Callable[[], bool] | None = None,
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

        # Move model to device
        self.device = torch.device(config.device)
        self.model.to(self.device)

        # Loss function
        if config.use_focal_loss:
            self.criterion = FocalLoss(
                alpha=config.focal_loss_alpha, gamma=config.focal_loss_gamma
            )
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=config.learning_rate * 0.01,
        )

        # Training state
        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.epochs_without_improvement = 0
        self.metrics_history: list[TrainingMetrics] = []

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
        correct = 0
        total = 0

        for batch_idx, (landmarks, labels) in enumerate(dataloader):
            # Move to device
            landmarks = landmarks.to(self.device)  # [batch, seq_len, features]
            labels = labels.to(self.device)  # [batch]

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(landmarks)  # [batch, num_classes]

            # Compute loss
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.gradient_clip_max_norm,
            )

            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Check stop signal
            if self.stop_signal and self.stop_signal():
                logger.info("training_interrupted_by_stop_signal")
                break

        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
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

        with torch.no_grad():
            for landmarks, labels in dataloader:
                # Move to device
                landmarks = landmarks.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                logits = self.model(landmarks)

                # Compute loss
                loss = self.criterion(logits, labels)

                # Metrics
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

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
        logger.info(
            "starting_training",
            num_epochs=self.config.num_epochs,
            train_size=len(train_dataset),
            val_size=len(val_dataset),
            batch_size=self.config.batch_size,
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
        )

        # Training loop
        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()

            # Check stop signal before starting epoch
            if self.stop_signal and self.stop_signal():
                logger.info("training_stopped_before_epoch", epoch=epoch)
                break

            # Train
            train_loss, train_accuracy = self.train_epoch(train_loader)

            # Validate
            val_loss, val_accuracy = self.validate(val_loader)

            # Scheduler step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

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
            )

            # Progress callback
            if self.progress_callback:
                self.progress_callback(metrics)

            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
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

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("restored_best_model", val_loss=self.best_val_loss)

        return self.metrics_history

    def save_model(self, save_path: str | Path, class_labels: list[str] | None = None) -> None:
        """
        Save trained model to file.

        Args:
            save_path: Path to save model checkpoint
            class_labels: Ordered class labels matching output logits
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "num_classes": self.model.num_classes,
            "num_features": self.model.num_features,
            "d_model": self.model.d_model,
            "class_labels": class_labels or [],
            "config": {
                "num_epochs": self.config.num_epochs,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
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
        }

        torch.save(checkpoint, save_path)
        logger.info("model_saved", path=str(save_path))


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
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Create model with saved architecture
        model = SignTransformer(
            num_features=checkpoint.get("num_features", 225),
            num_classes=checkpoint["num_classes"],
            d_model=checkpoint.get("d_model", 256),
        )

        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
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
