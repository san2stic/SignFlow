#!/usr/bin/env python3
"""Test MLflow Model Registry integration by training a tiny model."""

import os
import sys
from pathlib import Path

# Set environment variables FIRST, before any imports
models_dir = Path(__file__).parent.parent / "data" / "models"
mlruns_dir = models_dir / "mlruns"
mlruns_dir.mkdir(parents=True, exist_ok=True)

os.environ["MODEL_DIR"] = str(models_dir.resolve())
os.environ["MLFLOW_TRACKING_URI"] = f"file://{mlruns_dir.resolve()}"

# Add backend to path AFTER env vars are set
sys.path.insert(0, str(Path(__file__).parent.parent))

# Now import everything
import numpy as np
import torch
from app.ml.model import SignTransformer
from app.ml.trainer import SignTrainer, TrainingConfig
from app.ml.dataset import LandmarkDataset, SignSample


def create_dummy_data(num_samples: int = 20, num_classes: int = 5):
    """Create dummy landmark data for testing."""
    data = []
    for i in range(num_samples):
        # Create random sequence of 20-40 frames
        seq_len = np.random.randint(20, 40)
        landmarks = np.random.randn(seq_len, 225).astype(np.float32)
        label = i % num_classes
        # Create SignSample object
        sample = SignSample(landmarks=landmarks, label=label)
        data.append(sample)
    return data


def main():
    print("🧪 Testing MLflow Model Registry Integration\n")

    # 1. Create dummy dataset
    print("📊 Creating dummy dataset...")
    train_data = create_dummy_data(num_samples=20, num_classes=5)
    val_data = create_dummy_data(num_samples=10, num_classes=5)

    train_dataset = LandmarkDataset(
        train_data,
        sequence_length=64,
        apply_sliding_window=False,
        use_enriched_features=False  # Use raw features for speed
    )
    val_dataset = LandmarkDataset(
        val_data,
        sequence_length=64,
        apply_sliding_window=False,
        use_enriched_features=False
    )

    # 2. Create tiny model
    print("🏗️  Creating tiny model (d_model=32, 1 layer)...")
    model = SignTransformer(
        num_features=225,  # Raw landmark features
        num_classes=5,
        d_model=32,
        nhead=2,  # Must divide d_model
        num_layers=1,
        dim_feedforward=64,
        dropout=0.1,
    )

    # 3. Configure training with MLflow
    print("⚙️  Configuring training with MLflow enabled...")

    config = TrainingConfig(
        num_epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        use_mlflow=True,
    )

    # 4. Train
    print("🚀 Training model (2 epochs)...\n")
    trainer = SignTrainer(
        model=model,
        config=config,
    )

    history = trainer.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset
    )

    # 5. Check results
    print("\n✅ Training complete!")
    if isinstance(history, dict):
        print(f"Final train loss: {history.get('train_loss', [0])[-1]:.4f}")
        print(f"Final val loss: {history.get('val_loss', [0])[-1]:.4f}")
    else:
        print(f"History type: {type(history)}")

    print("\n📦 Model should now be registered in MLflow Model Registry:")
    print("   - Model name: signflow-model")
    print("   - Check at: http://localhost:5000/#/models/signflow-model")

    print("\n💡 Refresh the MLflow UI 'Models' tab to see the registered model.")


if __name__ == "__main__":
    main()
