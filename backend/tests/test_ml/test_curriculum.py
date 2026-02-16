"""Tests for curriculum learning sampler."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from app.ml.curriculum import CurriculumSampler


class _ToyDataset(Dataset):
    def __init__(self, lengths: list[int]) -> None:
        self.lengths = lengths

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        seq_len = 16
        features = 12
        active_len = self.lengths[index]
        sample = torch.zeros((seq_len, features), dtype=torch.float32)
        sample[:active_len] = 1.0
        return sample, 0


def test_curriculum_sampler_length_progressively_expands_subset() -> None:
    """Length strategy should start with easier samples and expand over epochs."""
    dataset = _ToyDataset([2, 4, 6, 8, 10, 12])
    sampler = CurriculumSampler(
        strategy="length",
        start_fraction=0.5,
        warmup_epochs=0,
        min_samples=1,
    )

    early_indices, early_snapshot = sampler.select_indices(dataset, epoch=0, total_epochs=4)
    late_indices, late_snapshot = sampler.select_indices(dataset, epoch=3, total_epochs=4)

    assert early_snapshot.selected_samples < late_snapshot.selected_samples
    assert late_snapshot.selected_samples == len(dataset)
    # First curriculum subset should prioritize shortest active sequences.
    assert set(early_indices).issubset({0, 1, 2, 3})


def test_curriculum_sampler_confidence_prioritizes_high_confidence_samples() -> None:
    """Confidence strategy should rank known high-confidence samples as easier."""
    dataset = _ToyDataset([8, 8, 8])
    sampler = CurriculumSampler(
        strategy="confidence",
        start_fraction=0.67,
        warmup_epochs=1,
        min_samples=1,
    )
    sampler.update_confidence({0: 0.9, 1: 0.1, 2: 0.8})

    indices, snapshot = sampler.select_indices(dataset, epoch=0, total_epochs=3)

    assert snapshot.selected_samples == 2
    assert 0 in indices
    assert 2 in indices
