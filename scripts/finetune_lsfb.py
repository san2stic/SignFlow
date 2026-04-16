#!/usr/bin/env python3
"""Fine-tune a pre-trained SignTransformer on the LSFB-ISOL dataset.

This script implements a two-stage transfer learning pipeline:
  1. Load a pre-trained checkpoint (WLASL or any SignTransformer checkpoint).
  2. Expand the classifier head for LSFB-ISOL classes.
  3. Freeze early transformer layers and fine-tune on LSFB-ISOL landmarks.

The script reads converted landmarks directly from disk (produced by
``scripts/convert_lsfb.py``) and does not require the DB to be seeded.
However, it does save the trained model to the standard ``data/models/``
directory and can optionally register it via the API.

Usage:
    # From scratch (no pre-trained checkpoint):
    python scripts/finetune_lsfb.py --lsfb-dir backend/data/datasets/lsfb_isol

    # Fine-tune from a WLASL pre-trained checkpoint:
    python scripts/finetune_lsfb.py \\
        --lsfb-dir backend/data/datasets/lsfb_isol \\
        --checkpoint backend/data/models/wlasl_base.pt \\
        --epochs 80 --lr 1e-4

    # Quick test on a small subset:
    python scripts/finetune_lsfb.py --lsfb-dir ... --max-signs 50 --epochs 10
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.dataset import LandmarkDataset, SignSample  # noqa: E402
from app.ml.fewshot import (  # noqa: E402
    expand_classifier_head,
    freeze_transformer_encoder_layers,
    prepare_few_shot_model,
)
from app.ml.lsfb_adapter import compute_detection_rate, load_split_json  # noqa: E402
from app.ml.trainer import SignTrainer, TrainingConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune SignTransformer on LSFB-ISOL dataset.",
    )
    # Dataset
    parser.add_argument(
        "--lsfb-dir",
        default=str(REPO_ROOT / "backend" / "data" / "datasets" / "lsfb_isol"),
        help="Root of the LSFB-ISOL dataset with converted/ subdirectory.",
    )
    parser.add_argument(
        "--converted-dir",
        default=None,
        help="Directory with converted .npy files (default: <lsfb-dir>/converted).",
    )
    parser.add_argument(
        "--max-signs",
        type=int,
        default=None,
        help="Limit to the N most frequent signs (for quick experimentation).",
    )
    parser.add_argument(
        "--min-clips-per-sign",
        type=int,
        default=10,
        help="Minimum training clips for a sign to be included (default: 10).",
    )
    parser.add_argument(
        "--min-detection-rate",
        type=float,
        default=0.5,
        help="Minimum landmark detection rate (default: 0.5).",
    )

    # Model
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to pre-trained .pt checkpoint for transfer learning.",
    )
    parser.add_argument(
        "--freeze-layers",
        type=int,
        default=2,
        help="Freeze the first N transformer encoder layers (default: 2).",
    )
    parser.add_argument(
        "--freeze-embedding",
        action="store_true",
        default=True,
        help="Freeze input embedding/normalization layers (default: true).",
    )
    parser.add_argument(
        "--no-freeze-embedding",
        dest="freeze_embedding",
        action="store_false",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, mps.")
    parser.add_argument("--classifier-lr-mult", type=float, default=3.0, help="LR multiplier for classifier head.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio if no test split file.")
    parser.add_argument("--sequence-length", type=int, default=64, help="Temporal sequence length.")

    # Output
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "backend" / "data" / "models" / "lsfb_isol_finetuned.pt"),
        help="Path to save the fine-tuned model checkpoint.",
    )
    parser.add_argument(
        "--augmentations",
        type=int,
        default=4,
        help="Number of augmented copies per training sample (default: 4).",
    )

    return parser.parse_args()


def load_lsfb_samples(
    instances_csv: Path,
    converted_dir: Path,
    split_ids: set[str] | None,
    gloss_to_label: dict[str, int],
    min_detection_rate: float,
) -> list[SignSample]:
    """Load converted landmarks into SignSample list.

    Args:
        instances_csv: Path to instances.csv.
        converted_dir: Directory with {instance_id}_landmarks.npy files.
        split_ids: Optional set of instance IDs to include.
        gloss_to_label: Mapping from sign gloss to integer label.
        min_detection_rate: Minimum detection rate threshold.

    Returns:
        List of SignSample ready for LandmarkDataset.
    """
    samples: list[SignSample] = []
    skipped = 0

    with open(instances_csv, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            instance_id = row.get("id", "").strip()
            gloss = row.get("sign", "").strip()

            if not instance_id or not gloss:
                continue

            if split_ids is not None and instance_id not in split_ids:
                continue

            if gloss not in gloss_to_label:
                continue

            landmarks_path = converted_dir / f"{instance_id}_landmarks.npy"
            if not landmarks_path.exists():
                skipped += 1
                continue

            try:
                landmarks = np.load(landmarks_path)
            except Exception:
                skipped += 1
                continue

            if landmarks.ndim != 2 or landmarks.shape[1] < 225:
                skipped += 1
                continue

            det_rate = compute_detection_rate(landmarks)
            if det_rate < min_detection_rate:
                skipped += 1
                continue

            samples.append(SignSample(landmarks=landmarks, label=gloss_to_label[gloss]))

    if skipped > 0:
        print(f"  Skipped {skipped} instances (missing, corrupt, or low quality)")

    return samples


def build_gloss_to_label(
    instances_csv: Path,
    split_ids: set[str] | None,
    converted_dir: Path,
    max_signs: int | None,
    min_clips: int,
) -> tuple[dict[str, int], list[str]]:
    """Build gloss→label mapping from instances.csv, filtering by frequency.

    Returns:
        Tuple of (gloss_to_label dict, ordered class_labels list).
    """
    # Count occurrences per sign in the target split
    gloss_counts: dict[str, int] = {}
    with open(instances_csv, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            instance_id = row.get("id", "").strip()
            gloss = row.get("sign", "").strip()
            if not instance_id or not gloss:
                continue
            if split_ids is not None and instance_id not in split_ids:
                continue
            # Only count if converted file exists
            if (converted_dir / f"{instance_id}_landmarks.npy").exists():
                gloss_counts[gloss] = gloss_counts.get(gloss, 0) + 1

    # Filter by minimum clip count
    eligible = {g: c for g, c in gloss_counts.items() if c >= min_clips}

    # Sort by frequency (descending) for deterministic ordering
    sorted_glosses = sorted(eligible.keys(), key=lambda g: (-eligible[g], g))

    if max_signs is not None:
        sorted_glosses = sorted_glosses[:max_signs]

    gloss_to_label = {g: i for i, g in enumerate(sorted_glosses)}
    return gloss_to_label, sorted_glosses


def main() -> int:
    args = parse_args()

    lsfb_dir = Path(args.lsfb_dir).expanduser().resolve()
    converted_dir = Path(args.converted_dir) if args.converted_dir else lsfb_dir / "converted"
    converted_dir = converted_dir.expanduser().resolve()

    instances_csv = lsfb_dir / "instances.csv"
    if not instances_csv.exists():
        print(f"ERROR: instances.csv not found at {instances_csv}", file=sys.stderr)
        return 1
    if not converted_dir.is_dir():
        print(f"ERROR: Converted directory not found: {converted_dir}", file=sys.stderr)
        return 1

    # ----- Load splits -----
    train_split_path = lsfb_dir / "metadata" / "splits" / "train.json"
    test_split_path = lsfb_dir / "metadata" / "splits" / "test.json"

    train_ids: set[str] | None = None
    test_ids: set[str] | None = None

    if train_split_path.exists() and test_split_path.exists():
        train_ids = load_split_json(train_split_path)
        test_ids = load_split_json(test_split_path)
        print(f"Using LSFB official splits: {len(train_ids)} train, {len(test_ids)} test")
    else:
        print("WARNING: No official splits found, will use random 80/20 split")

    # ----- Build class mapping -----
    # Build from train split (or all if no splits)
    print("\nBuilding class mapping...")
    gloss_to_label, class_labels = build_gloss_to_label(
        instances_csv=instances_csv,
        split_ids=train_ids,
        converted_dir=converted_dir,
        max_signs=args.max_signs,
        min_clips=args.min_clips_per_sign,
    )

    num_classes = len(class_labels)
    print(f"  Classes: {num_classes} signs (min {args.min_clips_per_sign} clips each)")

    if num_classes == 0:
        print("ERROR: No eligible signs found. Check conversion / min-clips.", file=sys.stderr)
        return 1

    # ----- Load training data -----
    print("\nLoading training samples...")
    train_samples = load_lsfb_samples(
        instances_csv=instances_csv,
        converted_dir=converted_dir,
        split_ids=train_ids,
        gloss_to_label=gloss_to_label,
        min_detection_rate=args.min_detection_rate,
    )
    print(f"  Train samples: {len(train_samples)}")

    # ----- Load validation data -----
    if test_ids is not None:
        print("Loading validation samples (test split)...")
        val_samples = load_lsfb_samples(
            instances_csv=instances_csv,
            converted_dir=converted_dir,
            split_ids=test_ids,
            gloss_to_label=gloss_to_label,
            min_detection_rate=args.min_detection_rate,
        )
    else:
        # Random split
        from sklearn.model_selection import train_test_split
        train_samples, val_samples = train_test_split(
            train_samples, test_size=args.val_split, random_state=42,
            stratify=[s.label for s in train_samples],
        )

    print(f"  Val samples:   {len(val_samples)}")

    if len(train_samples) == 0:
        print("ERROR: No training samples loaded.", file=sys.stderr)
        return 1

    # ----- Apply augmentation to training data -----
    if args.augmentations > 0:
        from app.ml.augmentation import augment_dataset

        train_landmarks = [s.landmarks for s in train_samples]
        train_labels = [s.label for s in train_samples]

        aug_landmarks, aug_labels = augment_dataset(
            train_landmarks, train_labels,
            num_augmentations_per_sample=args.augmentations,
        )

        train_samples = [
            SignSample(landmarks=lm, label=lb)
            for lm, lb in zip(aug_landmarks, aug_labels)
        ]
        print(f"  After augmentation: {len(train_samples)} train samples")

    # ----- Create datasets -----
    train_dataset = LandmarkDataset(
        samples=train_samples,
        sequence_length=args.sequence_length,
        use_enriched_features=True,
    )
    val_dataset = LandmarkDataset(
        samples=val_samples,
        sequence_length=args.sequence_length,
        use_enriched_features=True,
    )

    print(f"\n  Train dataset size: {len(train_dataset)}")
    print(f"  Val dataset size:   {len(val_dataset)}")

    # ----- Prepare model -----
    print("\nPreparing model...")
    from app.ml.feature_engineering import ENRICHED_FEATURE_DIM

    preparation = prepare_few_shot_model(
        checkpoint_path=args.checkpoint,
        num_features=ENRICHED_FEATURE_DIM,
        num_classes=num_classes,
        device=args.device if args.device != "auto" else "cpu",
        freeze_until_layer=args.freeze_layers,
        freeze_embedding=args.freeze_embedding,
    )
    model = preparation.model

    if preparation.loaded_from_checkpoint:
        print(f"  Loaded from checkpoint: {args.checkpoint}")
        print(f"  Frozen layers: {preparation.frozen_layer_count}")
    else:
        print("  Training from scratch (no checkpoint)")

    # ----- Configure training -----
    use_amp = args.device not in ("mps",)  # AMP not supported on MPS
    config = TrainingConfig(
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        sequence_length=args.sequence_length,
        classifier_lr_multiplier=args.classifier_lr_mult,
        use_class_weights=True,
        use_weighted_sampler=True,
        use_focal_loss=True,
        use_ema=True,
        use_amp=use_amp,
        use_swa=True,
        early_stopping_patience=15,
        warmup_epochs=5,
        use_mixup=True,
        mixup_alpha=0.3,
        use_mlflow=False,
    )

    print("\n" + "=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"  Classes:         {num_classes}")
    print(f"  Train samples:   {len(train_dataset)}")
    print(f"  Val samples:     {len(val_dataset)}")
    print(f"  Epochs:          {config.num_epochs}")
    print(f"  LR:              {config.learning_rate}")
    print(f"  Batch size:      {config.batch_size}")
    print(f"  Device:          {config.device}")
    print(f"  Freeze layers:   {args.freeze_layers}")
    print(f"  Classifier LR:   {config.learning_rate * config.classifier_lr_multiplier}")
    print(f"  Checkpoint:      {args.checkpoint or 'none (from scratch)'}")
    print(f"  Output:          {args.output}")
    print("=" * 60)

    # ----- Train -----
    start = time.time()

    def progress_callback(metrics):
        if hasattr(metrics, "epoch"):
            epoch = getattr(metrics, "epoch", "?")
            loss = getattr(metrics, "loss", 0)
            acc = getattr(metrics, "accuracy", 0)
            val_acc = getattr(metrics, "val_accuracy", 0)
            print(f"  Epoch {epoch}: loss={loss:.4f} acc={acc:.3f} val_acc={val_acc:.3f}")

    trainer = SignTrainer(model=model, config=config, progress_callback=progress_callback)

    print("\nStarting training...")
    metrics_history = trainer.fit(train_dataset, val_dataset)

    elapsed = time.time() - start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    # ----- Save model -----
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trainer.save_model(
        save_path=str(output_path),
        class_labels=class_labels,
        metadata={
            "feature_version": 1,
            "source_dataset": "lsfb-isol",
            "source_checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "freeze_layers": args.freeze_layers,
            "num_train_samples": len(train_samples),
            "num_val_samples": len(val_samples),
            "augmentations_per_sample": args.augmentations,
            "licence": "CC BY-NC-SA 4.0",
            "citation": (
                "Fink, J. et al. (2021). LSFB-CONT and LSFB-ISOL: "
                "Two New Datasets for Vision-Based Sign Language Recognition. IJCNN 2021."
            ),
        },
    )

    # ----- Report -----
    print()
    print("=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"  Duration:         {minutes}m {seconds}s")
    print(f"  Model saved:      {output_path}")
    print(f"  Classes:          {num_classes}")
    print(f"  Class labels:     {class_labels[:10]}{'...' if num_classes > 10 else ''}")

    if metrics_history:
        last = metrics_history[-1]
        best_val_acc = max(getattr(m, "val_accuracy", 0) for m in metrics_history)
        print(f"  Final train loss: {getattr(last, 'loss', 'N/A')}")
        print(f"  Final val acc:    {getattr(last, 'val_accuracy', 'N/A')}")
        print(f"  Best val acc:     {best_val_acc:.4f}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
