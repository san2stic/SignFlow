"""Test-time augmentation helpers for inference ensembling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.ml.dataset import temporal_resample


@dataclass
class TTAConfig:
    """Configuration for inference-time landmark augmentations."""

    num_views: int = 1
    enable_mirror: bool = True
    enable_temporal_jitter: bool = True
    enable_spatial_noise: bool = True
    temporal_jitter_ratio: float = 0.05
    spatial_noise_std: float = 0.005


class TTAGenerator:
    """Generate deterministic-ish sets of augmented windows for ensembling."""

    def __init__(self, config: TTAConfig) -> None:
        self.config = config

    def generate(self, window: np.ndarray) -> list[np.ndarray]:
        """Generate a fixed number of views, including original input."""
        base = window.astype(np.float32, copy=False)
        views = [base]
        if self.config.num_views <= 1 or window.ndim != 2 or window.shape[0] < 8:
            return views

        generators = []
        if self.config.enable_mirror:
            generators.append(self._mirror_view)
        if self.config.enable_temporal_jitter:
            generators.append(self._temporal_jitter_view)
        if self.config.enable_spatial_noise:
            generators.append(self._spatial_noise_view)

        if not generators:
            return [base for _ in range(self.config.num_views)]

        turn = 0
        while len(views) < self.config.num_views:
            transform = generators[turn % len(generators)]
            variant = transform(base)
            views.append(variant.astype(np.float32, copy=False))
            turn += 1
        return views[: self.config.num_views]

    def _mirror_view(self, window: np.ndarray) -> np.ndarray:
        """
        Mirror left/right body landmarks around vertical axis.

        Layout assumption for first 225 dims:
        - left hand: 63
        - right hand: 63
        - pose: 99
        """
        mirrored = window.copy()
        if mirrored.shape[1] < 225:
            return mirrored

        left = mirrored[:, 0:63].copy()
        right = mirrored[:, 63:126].copy()
        pose = mirrored[:, 126:225].copy()

        # Swap hands and invert X coordinate for every xyz triplet.
        left[:, 0::3] *= -1.0
        right[:, 0::3] *= -1.0
        mirrored[:, 0:63] = right
        mirrored[:, 63:126] = left

        # Mirror body pose X coordinates.
        pose[:, 0::3] *= -1.0
        mirrored[:, 126:225] = pose
        return mirrored

    def _temporal_jitter_view(self, window: np.ndarray) -> np.ndarray:
        """Apply mild speed perturbation then resample back to original length."""
        if window.shape[0] < 8:
            return window

        ratio = float(np.clip(self.config.temporal_jitter_ratio, 0.0, 0.3))
        speed = float(np.random.uniform(1.0 - ratio, 1.0 + ratio))
        target = max(4, int(round(window.shape[0] / max(speed, 1e-3))))
        jittered = temporal_resample(window, target_len=target)
        restored = temporal_resample(jittered, target_len=window.shape[0])
        return restored

    def _spatial_noise_view(self, window: np.ndarray) -> np.ndarray:
        """Add Gaussian spatial perturbation on raw landmark coordinates."""
        noisy = window.copy()
        if noisy.shape[1] <= 0:
            return noisy

        std = float(max(0.0, self.config.spatial_noise_std))
        if std <= 0.0:
            return noisy

        coord_dims = min(225, noisy.shape[1])
        noise = np.random.normal(loc=0.0, scale=std, size=noisy[:, :coord_dims].shape)
        noisy[:, :coord_dims] = noisy[:, :coord_dims] + noise.astype(np.float32)
        return noisy
