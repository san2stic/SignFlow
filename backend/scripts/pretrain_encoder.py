#!/usr/bin/env python3
"""Pretrain spatial-temporal encoder with self-supervised masked landmark modeling.

This script performs self-supervised pretraining on unlabeled video data before
fine-tuning on labeled sign language datasets.

Usage:
    python scripts/pretrain_encoder.py --data-dir data/videos/unlabeled --epochs 50

The pretrained encoder can then be loaded and fine-tuned on labeled data:
    python scripts/train.py --pretrained-encoder data/models/pretrained_encoder.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import structlog
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Import ML modules
from app.ml.pretraining import MaskedLandmarkConfig, MaskedLandmarkModel
from app.ml.spatial_encoder import SpatialTemporalTransformer
from app.ml.tracking import MLFlowTracker

logger = structlog.get_logger(__name__)


class UnlabeledLandmarkDataset(Dataset):
    """
    Dataset for unlabeled landmark sequences.

    Loads raw landmark sequences from video files without requiring labels.
    """

    def __init__(self, data_dir: Path, max_samples: int | None = None):
        """
        Args:
            data_dir: Directory containing unlabeled video files
            max_samples: Optional limit on number of samples to load
        """
        self.data_dir = Path(data_dir)
        self.samples = self._load_samples()

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        logger.info("unlabeled_dataset_loaded", num_samples=len(self.samples))

    def _load_samples(self) -> list[Path]:
        """Load list of video file paths."""
        video_extensions = [".mp4", ".avi", ".mov", ".npy", ".pt"]
        samples = []

        for ext in video_extensions:
            samples.extend(self.data_dir.glob(f"**/*{ext}"))

        return sorted(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Load landmark sequence from file.

        Returns:
            [seq_len, num_landmarks * landmark_dim] tensor
        """
        file_path = self.samples[idx]

        # Load based on file extension
        if file_path.suffix == ".npy":
            import numpy as np

            landmarks = np.load(file_path)
            landmarks = torch.from_numpy(landmarks).float()
        elif file_path.suffix == ".pt":
            landmarks = torch.load(file_path)
        else:
            # For video files, extract landmarks using MediaPipe
            landmarks = self._extract_landmarks_from_video(file_path)

        return landmarks

    def _extract_landmarks_from_video(self, video_path: Path) -> torch.Tensor:
        """
        Extract landmarks from video file using MediaPipe.

        Args:
            video_path: Path to video file

        Returns:
            [seq_len, num_landmarks * landmark_dim] tensor
        """
        # Import feature extraction
        from app.ml.features import extract_landmarks_from_video

        try:
            landmarks = extract_landmarks_from_video(str(video_path))
            return torch.from_numpy(landmarks).float()
        except Exception as e:
            logger.error("landmark_extraction_failed", path=str(video_path), error=str(e))
            # Return zero tensor as fallback
            return torch.zeros(64, 225)  # 64 frames, 225 features (75 landmarks * 3)


def collate_variable_length(batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Collate variable-length sequences into batched tensor with padding.

    Args:
        batch: List of [seq_len, features] tensors

    Returns:
        {
            "landmarks": [batch, max_seq_len, features] padded tensor,
            "lengths": [batch] actual sequence lengths
        }
    """
    # Find max sequence length in batch
    lengths = torch.tensor([x.size(0) for x in batch])
    max_len = lengths.max().item()
    feature_dim = batch[0].size(-1)

    # Pad sequences
    padded_batch = torch.zeros(len(batch), max_len, feature_dim)
    for i, seq in enumerate(batch):
        seq_len = seq.size(0)
        padded_batch[i, :seq_len] = seq

    return {"landmarks": padded_batch, "lengths": lengths}


def train_pretrain_step(
    model: MaskedLandmarkModel,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """
    Single pretraining step.

    Args:
        model: Masked landmark model
        batch: Batch of landmarks
        optimizer: Optimizer
        device: Device to use

    Returns:
        Loss dict with scalar values
    """
    model.train()
    optimizer.zero_grad()

    # Move to device
    landmarks = batch["landmarks"].to(device)

    # Forward pass with masking
    outputs = model(landmarks, return_loss=True)

    # Backward pass
    loss = outputs["loss"]
    loss.backward()
    optimizer.step()

    # Return scalar losses
    return {
        "loss": loss.item(),
        "reconstruction_loss": outputs["reconstruction_loss"].item(),
        "temporal_smoothness_loss": outputs["temporal_smoothness_loss"].item(),
    }


def pretrain_epoch(
    model: MaskedLandmarkModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """
    Run one pretraining epoch.

    Args:
        model: Model to train
        dataloader: Data loader
        optimizer: Optimizer
        device: Device
        epoch: Current epoch number

    Returns:
        Average losses for the epoch
    """
    model.train()
    total_losses = {"loss": 0.0, "reconstruction_loss": 0.0, "temporal_smoothness_loss": 0.0}
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        losses = train_pretrain_step(model, batch, optimizer, device)

        # Accumulate losses
        for key, value in losses.items():
            total_losses[key] += value

        num_batches += 1

        # Update progress bar
        pbar.set_postfix(loss=f"{losses['loss']:.4f}")

    # Average losses
    avg_losses = {key: value / num_batches for key, value in total_losses.items()}

    return avg_losses


def main():
    parser = argparse.ArgumentParser(description="Pretrain encoder with masked landmark modeling")
    parser.add_argument("--data-dir", type=Path, required=True, help="Unlabeled data directory")
    parser.add_argument("--output-dir", type=Path, default=Path("data/models"), help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of pretraining epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/mps/cpu/auto)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--max-samples", type=int, default=None, help="Max training samples")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow tracking")
    parser.add_argument("--run-name", type=str, default="pretrain", help="MLflow run name")

    args = parser.parse_args()

    # Setup device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    logger.info("pretraining_setup", device=str(device), epochs=args.epochs)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = UnlabeledLandmarkDataset(args.data_dir, max_samples=args.max_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_variable_length,
    )

    # Create encoder
    encoder = SpatialTemporalTransformer(
        num_landmarks=75,
        landmark_dim=3,
        num_classes=100,  # Dummy, not used during pretraining
        spatial_hidden_dim=128,
        spatial_output_dim=256,
        num_gcn_layers=2,
        d_model=384,
        nhead=8,
        num_transformer_layers=6,
        dim_feedforward=1536,
        dropout=0.3,
    ).to(device)

    # Wrap with masked landmark model
    config = MaskedLandmarkConfig(
        mask_prob=0.15,
        num_landmarks=75,
        landmark_dim=3,
    )
    model = MaskedLandmarkModel(encoder, config, encoder_output_dim=384).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # MLflow tracking
    tracker = None
    if args.mlflow:
        tracker = MLFlowTracker(experiment_name="signflow-pretraining")

    # Training loop
    if tracker:
        with tracker.start_run(run_name=args.run_name):
            # Log config
            tracker.log_params(
                {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "mask_prob": config.mask_prob,
                    "device": str(device),
                }
            )

            for epoch in range(1, args.epochs + 1):
                avg_losses = pretrain_epoch(model, dataloader, optimizer, device, epoch)

                # Log metrics
                tracker.log_metrics(avg_losses, step=epoch)

                # Log learning rate
                tracker.log_metrics({"lr": optimizer.param_groups[0]["lr"]}, step=epoch)

                logger.info("epoch_complete", epoch=epoch, **avg_losses)

                # Step scheduler
                scheduler.step()

                # Save checkpoint every 10 epochs
                if epoch % 10 == 0:
                    checkpoint_path = args.output_dir / f"pretrained_encoder_epoch{epoch}.pt"
                    torch.save(encoder.state_dict(), checkpoint_path)
                    logger.info("checkpoint_saved", path=str(checkpoint_path))

            # Save final encoder
            final_path = args.output_dir / "pretrained_encoder.pt"
            torch.save(encoder.state_dict(), final_path)
            tracker.log_artifact(final_path)
            logger.info("final_encoder_saved", path=str(final_path))

    else:
        # Train without MLflow
        for epoch in range(1, args.epochs + 1):
            avg_losses = pretrain_epoch(model, dataloader, optimizer, device, epoch)
            logger.info("epoch_complete", epoch=epoch, **avg_losses)
            scheduler.step()

            if epoch % 10 == 0:
                checkpoint_path = args.output_dir / f"pretrained_encoder_epoch{epoch}.pt"
                torch.save(encoder.state_dict(), checkpoint_path)
                logger.info("checkpoint_saved", path=str(checkpoint_path))

        final_path = args.output_dir / "pretrained_encoder.pt"
        torch.save(encoder.state_dict(), final_path)
        logger.info("final_encoder_saved", path=str(final_path))


if __name__ == "__main__":
    main()
