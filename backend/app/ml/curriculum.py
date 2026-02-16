"""Curriculum-learning utilities for progressive sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


@dataclass
class CurriculumSnapshot:
    """Serializable curriculum state for one epoch."""

    strategy: str
    fraction: float
    selected_samples: int
    total_samples: int


class CurriculumSampler:
    """
    Progressive easy-to-hard sampler.

    Supported strategies:
    - ``length``: short/easier active sequences first, then longer ones.
    - ``confidence``: high-confidence samples first, then uncertain ones.
    """

    def __init__(
        self,
        *,
        strategy: str = "length",
        start_fraction: float = 0.4,
        warmup_epochs: int = 2,
        min_samples: int = 64,
        confidence_momentum: float = 0.8,
    ) -> None:
        self.strategy = strategy if strategy in {"length", "confidence"} else "length"
        self.start_fraction = float(np.clip(start_fraction, 0.1, 1.0))
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.min_samples = max(1, int(min_samples))
        self.confidence_momentum = float(np.clip(confidence_momentum, 0.0, 0.999))
        self._confidence_by_index: dict[int, float] = {}

    def update_confidence(self, confidence_by_index: dict[int, float]) -> None:
        """Update per-sample confidence with EMA smoothing."""
        for index, value in confidence_by_index.items():
            idx = int(index)
            new_value = float(np.clip(value, 0.0, 1.0))
            old_value = self._confidence_by_index.get(idx)
            if old_value is None:
                self._confidence_by_index[idx] = new_value
            else:
                momentum = self.confidence_momentum
                self._confidence_by_index[idx] = (momentum * old_value) + ((1.0 - momentum) * new_value)

    def select_indices(
        self,
        dataset: Dataset,
        *,
        epoch: int,
        total_epochs: int,
    ) -> tuple[list[int], CurriculumSnapshot]:
        """Return local dataset indices selected for the current epoch."""
        total = len(dataset)
        if total <= 0:
            snapshot = CurriculumSnapshot(
                strategy=self.strategy,
                fraction=1.0,
                selected_samples=0,
                total_samples=0,
            )
            return [], snapshot

        fraction = self._fraction_for_epoch(epoch=epoch, total_epochs=total_epochs)
        target_count = min(total, max(self.min_samples, int(round(total * fraction))))
        if target_count >= total:
            snapshot = CurriculumSnapshot(
                strategy=self.strategy,
                fraction=fraction,
                selected_samples=total,
                total_samples=total,
            )
            return list(range(total)), snapshot

        difficulties = self._difficulty_scores(dataset)
        ordered = np.argsort(difficulties, kind="stable")
        selected = ordered[:target_count].tolist()
        snapshot = CurriculumSnapshot(
            strategy=self.strategy,
            fraction=fraction,
            selected_samples=target_count,
            total_samples=total,
        )
        return selected, snapshot

    def _fraction_for_epoch(self, *, epoch: int, total_epochs: int) -> float:
        """Compute progressive sample ratio for an epoch."""
        total_epochs = max(1, int(total_epochs))
        epoch_idx = max(0, int(epoch))
        current_epoch = epoch_idx + 1

        if current_epoch <= self.warmup_epochs:
            return self.start_fraction

        progress_epochs = max(1, total_epochs - self.warmup_epochs)
        progress = (current_epoch - self.warmup_epochs) / float(progress_epochs)
        fraction = self.start_fraction + ((1.0 - self.start_fraction) * progress)
        return float(np.clip(fraction, self.start_fraction, 1.0))

    def _difficulty_scores(self, dataset: Dataset) -> np.ndarray:
        """Compute difficulty score for each local dataset index."""
        total = len(dataset)
        if total == 0:
            return np.zeros((0,), dtype=np.float32)

        if self.strategy == "confidence":
            confidence_scores = np.zeros((total,), dtype=np.float32)
            has_signal = False
            for local_idx in range(total):
                global_idx = self._global_index(dataset, local_idx)
                confidence = self._confidence_by_index.get(global_idx, 0.5)
                confidence_scores[local_idx] = 1.0 - float(np.clip(confidence, 0.0, 1.0))
                has_signal = has_signal or (global_idx in self._confidence_by_index)
            if has_signal:
                return confidence_scores

        lengths = np.zeros((total,), dtype=np.float32)
        for local_idx in range(total):
            sample, _label = dataset[local_idx]
            lengths[local_idx] = self._effective_sequence_length(sample)
        return lengths

    @staticmethod
    def _effective_sequence_length(sample: torch.Tensor | np.ndarray) -> float:
        """Estimate active sequence length (higher means more difficult)."""
        if isinstance(sample, torch.Tensor):
            array = sample.detach().cpu().numpy()
        else:
            array = np.asarray(sample)
        if array.ndim != 2 or array.shape[0] == 0:
            return 1.0
        active = np.abs(array).sum(axis=1) > 1e-6
        active_count = int(active.sum())
        return float(max(active_count, 1))

    @staticmethod
    def _global_index(dataset: Dataset, local_idx: int) -> int:
        """Map local index to underlying base-dataset index."""
        if isinstance(dataset, Subset):
            return int(dataset.indices[local_idx])
        return int(local_idx)
