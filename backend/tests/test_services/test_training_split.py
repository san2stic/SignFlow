"""Training split safeguards (anti data leakage for small datasets)."""

from __future__ import annotations

import numpy as np

import app.services.training_service as training_service_module
from app.services.training_service import TrainingService


def _make_sequences(num_samples: int) -> list[np.ndarray]:
    """Create deterministic synthetic samples with traceable identifiers."""
    return [
        np.full((12, 225), fill_value=float(index), dtype=np.float32)
        for index in range(num_samples)
    ]


def test_stratified_split_keeps_singleton_class_in_train() -> None:
    """A class with one sample should stay in train split (no impossible val holdout)."""
    labels = [0, 0, 1]  # class "1" is singleton

    train_indices, val_indices = TrainingService._stratified_train_val_indices(
        labels,
        val_ratio=0.2,
    )

    assert 2 in train_indices.tolist()
    assert len(val_indices) >= 1


def test_split_and_prepare_sequences_augments_train_only(monkeypatch) -> None:
    """Augmentation must only run on train split to avoid validation leakage."""
    sequences = _make_sequences(6)
    labels = [0, 0, 0, 1, 1, 1]

    observed_train_markers: list[float] = []

    def fake_augment(
        train_sequences: list[np.ndarray],
        train_labels: list[int],
        num_augmentations_per_sample: int,
        augmentation_probability: float,
    ) -> tuple[list[np.ndarray], list[int]]:
        del num_augmentations_per_sample, augmentation_probability
        observed_train_markers.extend(float(sequence[0, 0]) for sequence in train_sequences)
        duplicated_sequences = train_sequences + [sequence.copy() for sequence in train_sequences]
        duplicated_labels = train_labels + train_labels
        return duplicated_sequences, duplicated_labels

    monkeypatch.setattr(training_service_module, "augment_dataset", fake_augment)

    train_sequences, train_labels, val_sequences, _val_labels = TrainingService._split_and_prepare_sequences(
        sequences,
        labels,
        val_ratio=0.2,
        apply_augmentation=True,
        num_augmentations_per_sample=5,
        augmentation_probability=0.5,
    )

    val_markers = {float(sequence[0, 0]) for sequence in val_sequences}
    assert observed_train_markers, "augmentation should receive train split samples"
    assert val_markers.isdisjoint(set(observed_train_markers))
    assert len(train_sequences) == len(observed_train_markers) * 2
    assert len(train_labels) == len(train_sequences)
    assert len(val_sequences) == 2


def test_split_without_augmentation_preserves_total_sample_count() -> None:
    """When augmentation is disabled, split should preserve original sample count."""
    sequences = _make_sequences(8)
    labels = [0, 0, 0, 0, 1, 1, 1, 1]

    train_sequences, train_labels, val_sequences, val_labels = TrainingService._split_and_prepare_sequences(
        sequences,
        labels,
        val_ratio=0.25,
        apply_augmentation=False,
    )

    assert len(train_sequences) + len(val_sequences) == len(sequences)
    assert len(train_labels) + len(val_labels) == len(labels)

