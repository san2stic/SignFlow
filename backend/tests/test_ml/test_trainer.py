"""Trainer-level tests for advanced optimization features."""

from __future__ import annotations

import numpy as np
import torch

from app.ml.dataset import LandmarkDataset, SignSample
from app.ml.feature_engineering import ENRICHED_FEATURE_DIM
from app.ml.model import SignTransformer
from app.ml.trainer import SignTrainer, TrainingConfig


def test_temporal_mask_regularization_preserves_shape_and_masks_tokens() -> None:
    """Temporal masking should keep tensor shape and drop at least one span."""
    model = SignTransformer(num_features=ENRICHED_FEATURE_DIM, num_classes=3)
    config = TrainingConfig(
        num_epochs=1,
        batch_size=2,
        num_workers=0,
        device="cpu",
        temporal_mask_prob=1.0,
        temporal_mask_span_ratio=0.3,
    )
    trainer = SignTrainer(model=model, config=config)
    trainer.model.train()

    landmarks = torch.ones((2, 24, ENRICHED_FEATURE_DIM), dtype=torch.float32)
    masked = trainer._apply_temporal_mask(landmarks)

    assert masked.shape == landmarks.shape
    assert torch.any(masked == 0.0)


def test_trainer_cpu_disables_amp_even_if_requested() -> None:
    """AMP should only activate on CUDA devices."""
    model = SignTransformer(num_features=ENRICHED_FEATURE_DIM, num_classes=2)
    config = TrainingConfig(
        num_epochs=1,
        batch_size=2,
        num_workers=0,
        device="cpu",
        use_amp=True,
    )

    trainer = SignTrainer(model=model, config=config)
    assert trainer.autocast_enabled is False


def test_stable_inverse_frequency_weights_are_clipped_and_normalized() -> None:
    """Stable class-imbalance weights should remain bounded and finite."""
    counts = torch.tensor([50, 5, 1], dtype=torch.float32).numpy()
    weights = SignTrainer._stable_inverse_frequency_weights(
        counts,
        power=0.75,
        min_weight=0.3,
        max_weight=3.0,
    )

    assert weights.shape == (3,)
    assert float(weights.min()) >= 0.3
    assert float(weights.max()) <= 3.0
    assert weights[0] <= weights[1] <= weights[2]


def test_trainer_curriculum_progressively_expands_train_subset() -> None:
    """Trainer should increase curriculum subset size across epochs."""
    rng = np.random.default_rng(7)
    samples: list[SignSample] = []
    for label in [0, 1]:
        for _ in range(4):
            landmarks = rng.standard_normal((48, 225)).astype(np.float32) * (label + 1)
            samples.append(SignSample(landmarks=landmarks, label=label))

    train_dataset = LandmarkDataset(samples[:6], sequence_length=32)
    val_dataset = LandmarkDataset(samples[6:], sequence_length=32)

    model = SignTransformer(num_features=ENRICHED_FEATURE_DIM, num_classes=2)
    config = TrainingConfig(
        num_epochs=3,
        batch_size=2,
        num_workers=0,
        device="cpu",
        use_mlflow=False,
        use_curriculum=True,
        curriculum_strategy="length",
        curriculum_start_fraction=0.5,
        curriculum_warmup_epochs=0,
        curriculum_min_samples=1,
    )
    trainer = SignTrainer(model=model, config=config)

    seen_sizes: list[int] = []

    def fake_train_epoch(loader):  # type: ignore[no-untyped-def]
        seen_sizes.append(len(loader.dataset))
        return 0.9, 0.5

    trainer.train_epoch = fake_train_epoch  # type: ignore[method-assign]
    trainer.validate = lambda _loader: (0.8, 0.5)  # type: ignore[method-assign]

    metrics = trainer.fit(train_dataset, val_dataset)

    assert len(metrics) == 3
    assert seen_sizes[0] < seen_sizes[-1]
    assert seen_sizes[-1] == len(train_dataset)


def test_scheduler_step_skipped_when_no_optimizer_updates() -> None:
    """Scheduler should not step when an epoch performs no optimizer update."""
    rng = np.random.default_rng(11)
    samples = [
        SignSample(landmarks=rng.standard_normal((32, 225)).astype(np.float32), label=0),
        SignSample(landmarks=rng.standard_normal((32, 225)).astype(np.float32), label=1),
    ]
    train_dataset = LandmarkDataset(samples, sequence_length=32)
    val_dataset = LandmarkDataset(samples, sequence_length=32)

    model = SignTransformer(num_features=ENRICHED_FEATURE_DIM, num_classes=2)
    config = TrainingConfig(
        num_epochs=1,
        batch_size=2,
        num_workers=0,
        device="cpu",
        use_mlflow=False,
    )
    trainer = SignTrainer(model=model, config=config)

    scheduler_step_calls = 0

    def fake_scheduler_step() -> None:
        nonlocal scheduler_step_calls
        scheduler_step_calls += 1

    trainer.scheduler.step = fake_scheduler_step  # type: ignore[assignment]
    trainer.train_epoch = lambda _loader: (0.7, 0.5)  # type: ignore[method-assign]
    trainer.validate = lambda _loader: (0.6, 0.5)  # type: ignore[method-assign]

    _ = trainer.fit(train_dataset, val_dataset)

    assert scheduler_step_calls == 0


def test_save_model_logs_checkpoint_artifact_to_mlflow_run(tmp_path) -> None:
    """Checkpoint save should also push artifact + metadata into MLflow."""
    model = SignTransformer(num_features=ENRICHED_FEATURE_DIM, num_classes=2)
    config = TrainingConfig(
        num_epochs=1,
        batch_size=2,
        num_workers=0,
        device="cpu",
        use_mlflow=True,
    )
    trainer = SignTrainer(model=model, config=config)

    calls: list[tuple[str, object]] = []

    class _FakeTracker:
        def start(
            self,
            run_name: str | None = None,
            tags: dict[str, str] | None = None,
            run_id: str | None = None,
        ) -> str:
            calls.append(("start", run_id or run_name or "generated"))
            return run_id or "generated-run-id"

        def log_artifact(self, local_path, artifact_path=None) -> None:  # type: ignore[no-untyped-def]
            calls.append(("artifact", str(local_path), artifact_path))

        def log_dict(self, dictionary: dict, filename: str) -> None:
            calls.append(("dict", filename, bool(dictionary)))

        def end(self) -> None:
            calls.append(("end", True))

    trainer._mlflow_tracker = _FakeTracker()  # type: ignore[assignment]
    trainer.mlflow_run_id = "run-abc-123"

    checkpoint_path = tmp_path / "model.pt"
    trainer.save_model(
        checkpoint_path,
        class_labels=["hello", "bye"],
        metadata={"threshold": 0.8},
    )

    assert checkpoint_path.exists()
    assert ("start", "run-abc-123") in calls
    assert ("artifact", str(checkpoint_path), "model") in calls
    assert ("dict", "model/training_config.json", True) in calls
    assert ("dict", "model/metadata.json", True) in calls
    assert ("end", True) in calls
