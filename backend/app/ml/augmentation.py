"""Data augmentation utilities for landmark-based training."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import interpolate


def mirror_horizontal(sequence: np.ndarray) -> np.ndarray:
    """Mirror x-axis to simulate left/right handed variants."""
    mirrored = sequence.copy()
    mirrored[..., 0::3] *= -1
    return mirrored


def temporal_jitter(sequence: np.ndarray, max_shift: int = 5) -> np.ndarray:
    """Shift sequence in time while preserving length."""
    if len(sequence) == 0:
        return sequence
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(sequence, shift=shift, axis=0)


def gaussian_noise(sequence: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    """Inject gaussian noise into landmarks for robustness."""
    return sequence + np.random.normal(0, sigma, size=sequence.shape)


def speed_variation(sequence: np.ndarray, speed_factor: float | None = None) -> np.ndarray:
    """
    Change playback speed by interpolation (0.8x to 1.2x).

    Args:
        sequence: Input sequence of shape [num_frames, num_features]
        speed_factor: Speed multiplier. If None, randomly sample from [0.8, 1.2]

    Returns:
        Resampled sequence with same length but different temporal resolution
    """
    if len(sequence) == 0:
        return sequence

    if speed_factor is None:
        speed_factor = np.random.uniform(0.8, 1.2)

    num_frames, num_features = sequence.shape

    # Original time indices
    original_indices = np.arange(num_frames)

    # New time indices after speed change
    new_num_frames = int(num_frames / speed_factor)
    new_indices = np.linspace(0, num_frames - 1, new_num_frames)

    # Interpolate each feature dimension
    augmented = np.zeros((new_num_frames, num_features), dtype=sequence.dtype)

    for feature_idx in range(num_features):
        interpolator = interpolate.interp1d(
            original_indices,
            sequence[:, feature_idx],
            kind="linear",
            fill_value="extrapolate",
        )
        augmented[:, feature_idx] = interpolator(new_indices)

    # Resample back to original length to maintain fixed sequence size
    if new_num_frames != num_frames:
        final_indices = np.linspace(0, new_num_frames - 1, num_frames)
        final_augmented = np.zeros((num_frames, num_features), dtype=sequence.dtype)

        for feature_idx in range(num_features):
            interpolator = interpolate.interp1d(
                np.arange(new_num_frames),
                augmented[:, feature_idx],
                kind="linear",
                fill_value="extrapolate",
            )
            final_augmented[:, feature_idx] = interpolator(final_indices)

        return final_augmented

    return augmented


def swap_hands(sequence: np.ndarray) -> np.ndarray:
    """
    Swap left and right hand landmarks.

    Assumes sequence structure: [left_hand (63), right_hand (63), pose (99), ...]
    where each hand has 21 landmarks × 3 coordinates = 63 values.

    Args:
        sequence: Input sequence of shape [num_frames, num_features]

    Returns:
        Sequence with left and right hands swapped
    """
    if sequence.shape[1] < 126:  # Need at least hands (2 × 63)
        return sequence

    swapped = sequence.copy()

    # Swap hand landmarks (first 126 features: 63 left + 63 right)
    left_hand = swapped[:, :63].copy()
    right_hand = swapped[:, 63:126].copy()

    swapped[:, :63] = right_hand
    swapped[:, 63:126] = left_hand

    # Also flip x-coordinates after swapping
    swapped[:, 0:63:3] *= -1  # Left hand (now right) x-coords
    swapped[:, 63:126:3] *= -1  # Right hand (now left) x-coords

    return swapped


def apply_augmentations(
    sequence: np.ndarray,
    augmentations: list[Callable[[np.ndarray], np.ndarray]] | None = None,
    probability: float = 0.5,
) -> np.ndarray:
    """
    Apply a list of augmentations with given probability.

    Args:
        sequence: Input sequence of shape [num_frames, num_features]
        augmentations: List of augmentation functions. If None, use default set.
        probability: Probability of applying each augmentation

    Returns:
        Augmented sequence
    """
    if augmentations is None:
        # Default augmentation pipeline
        augmentations = [
            mirror_horizontal,
            lambda seq: temporal_jitter(seq, max_shift=5),
            lambda seq: gaussian_noise(seq, sigma=0.01),
            lambda seq: speed_variation(seq, speed_factor=None),
        ]

    augmented = sequence.copy()

    for aug_fn in augmentations:
        if np.random.random() < probability:
            augmented = aug_fn(augmented)

    return augmented


def augment_dataset(
    sequences: list[np.ndarray],
    labels: list[int],
    num_augmentations_per_sample: int = 3,
    augmentation_probability: float = 0.5,
) -> tuple[list[np.ndarray], list[int]]:
    """
    Create augmented versions of a dataset.

    Args:
        sequences: List of landmark sequences
        labels: Corresponding class labels
        num_augmentations_per_sample: How many augmented versions to create per sample
        augmentation_probability: Probability of applying each augmentation

    Returns:
        Tuple of (augmented_sequences, augmented_labels) including originals
    """
    augmented_sequences = sequences.copy()
    augmented_labels = labels.copy()

    for sequence, label in zip(sequences, labels):
        for _ in range(num_augmentations_per_sample):
            augmented = apply_augmentations(
                sequence,
                augmentations=None,
                probability=augmentation_probability,
            )
            augmented_sequences.append(augmented)
            augmented_labels.append(label)

    return augmented_sequences, augmented_labels
