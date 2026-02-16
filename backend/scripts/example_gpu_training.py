#!/usr/bin/env python3
"""Example script demonstrating GPU-accelerated training."""

import sys
from pathlib import Path

# Add backend to path
backend_root = Path(__file__).parent.parent
sys.path.insert(0, str(backend_root))

import numpy as np
import torch

from app.ml.gpu_manager import GPUConfig, get_gpu_manager
from app.ml.gpu_utils import get_device_info, profile_model
from app.ml.model import SignTransformer
from app.ml.trainer import SignTrainer, TrainingConfig
from app.ml.dataset import LandmarkDataset


def print_header(title: str) -> None:
    """Print section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def demo_device_detection() -> None:
    """Demonstrate automatic device detection."""
    print_header("1. Device Detection")

    # Get device info
    info = get_device_info()
    print("Device Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Get GPU manager
    manager = get_gpu_manager()
    device = manager.get_device()
    print(f"\n✓ Best device selected: {device}")

    if manager.supports_amp():
        print(f"✓ AMP supported: {manager.get_amp_dtype()}")
    else:
        print("⚠ AMP not supported on this device")


def demo_model_creation() -> None:
    """Demonstrate creating and moving model to GPU."""
    print_header("2. Model Creation and GPU Transfer")

    # Create model
    model = SignTransformer(
        num_features=469,
        num_classes=50,
        d_model=256,
        nhead=8,
        num_layers=4,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created:")
    print(f"  Parameters: {num_params:,}")
    print(f"  d_model: 256")
    print(f"  Layers: 4")

    # Move to GPU
    device = get_gpu_manager().get_device()
    model = model.to(device)
    print(f"\n✓ Model moved to: {device}")

    # Test forward pass
    dummy_input = torch.randn(4, 64, 469, device=device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"\n✓ Forward pass successful")
    print(f"  Input shape: {list(dummy_input.shape)}")
    print(f"  Output shape: {list(output.shape)}")


def demo_training_configuration() -> None:
    """Demonstrate training configuration with GPU."""
    print_header("3. Training Configuration")

    # Create training config with GPU optimizations
    config = TrainingConfig(
        device="auto",              # Auto-detect best device
        batch_size=32,
        num_epochs=5,
        learning_rate=1e-4,
        use_amp=True,               # Enable AMP
        amp_dtype="float16",
        use_ema=True,
        use_mixup=True,
        use_class_weights=True,
    )

    print("Training Configuration:")
    print(f"  Device: {config.device}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  AMP: {config.use_amp} ({config.amp_dtype})")
    print(f"  EMA: {config.use_ema}")
    print(f"  Mixup: {config.use_mixup}")

    print("\n✓ Configuration optimized for GPU training")


def create_dummy_dataset(num_samples: int, num_classes: int) -> LandmarkDataset:
    """Create a dummy dataset for demonstration."""
    samples = []
    for _ in range(num_samples):
        # Random sequence of landmarks
        sequence = np.random.randn(np.random.randint(30, 100), 225).astype(np.float32)
        label = np.random.randint(0, num_classes)
        samples.append((sequence, label))

    return LandmarkDataset(
        samples=samples,
        labels=[f"sign_{i}" for i in range(num_classes)],
        seq_len=64,
        augment=True,
    )


def demo_training() -> None:
    """Demonstrate GPU-accelerated training."""
    print_header("4. GPU-Accelerated Training")

    # Create dummy datasets
    print("Creating dummy datasets...")
    train_dataset = create_dummy_dataset(num_samples=500, num_classes=10)
    val_dataset = create_dummy_dataset(num_samples=100, num_classes=10)
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")

    # Create model
    model = SignTransformer(
        num_features=469,
        num_classes=10,
        d_model=128,
        nhead=4,
        num_layers=2,
    )

    # Create trainer with GPU config
    config = TrainingConfig(
        device="auto",
        batch_size=16,
        num_epochs=3,
        learning_rate=1e-4,
        use_amp=True,
        early_stopping_patience=5,
    )

    trainer = SignTrainer(model, config)
    print(f"\n✓ Trainer created with device: {trainer.device}")

    # Show memory stats before training
    manager = get_gpu_manager()
    if trainer.device.type != "cpu":
        stats = manager.get_memory_stats()
        print("\nInitial GPU Memory:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    # Train
    print("\nStarting training...")
    metrics_history = trainer.fit(train_dataset, val_dataset)

    # Show results
    print("\n✓ Training complete!")
    print("\nFinal Metrics:")
    final_metrics = metrics_history[-1]
    print(f"  Epoch: {final_metrics.epoch}")
    print(f"  Train Loss: {final_metrics.train_loss:.4f}")
    print(f"  Train Accuracy: {final_metrics.train_accuracy:.4f}")
    print(f"  Val Loss: {final_metrics.val_loss:.4f}")
    print(f"  Val Accuracy: {final_metrics.val_accuracy:.4f}")
    print(f"  Duration: {final_metrics.duration_sec:.2f}s")

    # Show final memory stats
    if trainer.device.type != "cpu":
        stats = manager.get_memory_stats()
        print("\nFinal GPU Memory:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


def demo_model_profiling() -> None:
    """Demonstrate model profiling."""
    print_header("5. Model Profiling")

    model = SignTransformer(
        num_features=469,
        num_classes=50,
        d_model=256,
        nhead=8,
        num_layers=4,
    )

    print("Profiling model with different batch sizes...\n")

    for batch_size in [1, 8, 16, 32]:
        results = profile_model(
            model=model,
            input_shape=(64, 469),
            batch_size=batch_size,
            num_iterations=50,
            warmup_iterations=10,
        )

        print(f"Batch Size = {batch_size}:")
        print(f"  Mean time: {results['mean_time_ms']:.2f} ms")
        print(f"  Throughput: {results['throughput_samples_per_sec']:.1f} samples/sec")
        print()


def demo_memory_optimization() -> None:
    """Demonstrate memory optimization techniques."""
    print_header("6. Memory Optimization")

    manager = get_gpu_manager()
    device = manager.get_device()

    if device.type == "cpu":
        print("⚠ Running on CPU, skipping memory optimization demo")
        return

    print("Testing memory management...")

    # Allocate large tensors
    print("\n1. Allocating tensors...")
    tensors = []
    for i in range(5):
        tensor = torch.randn(1000, 1000, device=device)
        tensors.append(tensor)

    stats = manager.get_memory_stats()
    print(f"  Allocated: {stats['allocated_mb']:.2f} MB")

    # Clear cache
    print("\n2. Clearing cache...")
    tensors.clear()
    manager.empty_cache()

    stats = manager.get_memory_stats()
    print(f"  Allocated after clear: {stats['allocated_mb']:.2f} MB")

    print("\n✓ Memory optimization successful")


def main() -> None:
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("  GPU SUPPORT DEMONSTRATION")
    print("  SignFlow - Real-time Sign Language Translation")
    print("=" * 70)

    demos = [
        demo_device_detection,
        demo_model_creation,
        demo_training_configuration,
        demo_training,
        demo_model_profiling,
        demo_memory_optimization,
    ]

    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"\n✗ Demo failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("  DEMONSTRATION COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
