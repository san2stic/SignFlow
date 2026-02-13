# ML Model Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix overfitting (0.0 loss from epoch 1) and add compound sign support by enriching features, adapting temporal handling, reducing model size, and adding inference state machine.

**Architecture:** Replace fixed 30-frame sliding window with full-video temporal resampling to 64 tokens. Enrich raw 225 coordinates with velocities, distances, and angles (~469 features). Shrink transformer (128-dim, 2 layers) to match dataset size. Replace per-frame inference with sign-level state machine.

**Tech Stack:** PyTorch, NumPy, SciPy (interpolation)

---

## Task 1: Feature Engineering Module

**Files:**
- Create: `backend/app/ml/feature_engineering.py`
- Test: `backend/tests/test_ml/test_feature_engineering.py`

**Step 1: Write failing tests**

```python
"""Tests for enriched feature computation."""
from __future__ import annotations

import numpy as np
import pytest

from app.ml.feature_engineering import (
    compute_enriched_features,
    compute_velocities,
    compute_hand_distances,
    compute_joint_angles,
    compute_hand_shape_features,
    ENRICHED_FEATURE_DIM,
)


def _make_sequence(num_frames: int = 10, num_features: int = 225) -> np.ndarray:
    """Create a synthetic landmark sequence."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((num_frames, num_features)).astype(np.float32)


def test_compute_velocities_shape():
    seq = _make_sequence(10, 225)
    vel = compute_velocities(seq)
    assert vel.shape == (10, 225)


def test_compute_velocities_first_frame_is_zero():
    seq = _make_sequence(10, 225)
    vel = compute_velocities(seq)
    assert np.allclose(vel[0], 0.0)


def test_compute_velocities_values():
    seq = np.array([[1.0, 2.0], [4.0, 6.0], [5.0, 5.0]], dtype=np.float32)
    vel = compute_velocities(seq)
    assert np.allclose(vel[1], [3.0, 4.0])
    assert np.allclose(vel[2], [1.0, -1.0])


def test_compute_hand_distances_shape():
    seq = _make_sequence(10, 225)
    dist = compute_hand_distances(seq)
    assert dist.shape == (10, 5)


def test_compute_joint_angles_shape():
    seq = _make_sequence(10, 225)
    angles = compute_joint_angles(seq)
    assert angles.shape == (10, 4)


def test_compute_hand_shape_features_shape():
    seq = _make_sequence(10, 225)
    shape_feats = compute_hand_shape_features(seq)
    assert shape_feats.shape == (10, 10)


def test_compute_enriched_features_output_dim():
    seq = _make_sequence(10, 225)
    enriched = compute_enriched_features(seq)
    assert enriched.shape == (10, ENRICHED_FEATURE_DIM)


def test_compute_enriched_features_no_nans():
    seq = _make_sequence(10, 225)
    seq[3, :] = 0.0  # simulate missing frame
    enriched = compute_enriched_features(seq)
    assert not np.any(np.isnan(enriched))


def test_compute_enriched_features_single_frame():
    seq = _make_sequence(1, 225)
    enriched = compute_enriched_features(seq)
    assert enriched.shape[0] == 1
    assert enriched.shape[1] == ENRICHED_FEATURE_DIM
```

**Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_ml/test_feature_engineering.py -v`
Expected: FAIL — module not found

**Step 3: Implement feature engineering module**

```python
"""Enriched feature computation for sign language landmark sequences.

Transforms raw 225-dimensional landmark coordinates into ~469 features
that capture motion, hand shape, and spatial relationships.

Feature layout of input (225 dims):
  - [0:63]   left hand  (21 points * 3 coords)
  - [63:126] right hand (21 points * 3 coords)
  - [126:225] pose      (33 points * 3 coords)
"""
from __future__ import annotations

import numpy as np

# Landmark index constants
_LEFT_HAND_START = 0
_LEFT_HAND_END = 63
_RIGHT_HAND_START = 63
_RIGHT_HAND_END = 126
_POSE_START = 126
_POSE_END = 225

# Pose landmark indices (within pose block, each point = 3 values)
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_ELBOW = 13
_RIGHT_ELBOW = 14
_LEFT_WRIST = 15
_RIGHT_WRIST = 16

# Hand landmark indices (within each hand block, each point = 3 values)
_WRIST = 0
_THUMB_TIP = 4
_INDEX_TIP = 8
_MIDDLE_TIP = 12
_RING_TIP = 16
_PINKY_TIP = 20

# Enriched feature dimension
ENRICHED_FEATURE_DIM = 225 + 225 + 5 + 4 + 10  # 469


def _pose_point(seq: np.ndarray, point_idx: int) -> np.ndarray:
    """Extract a pose point [num_frames, 3] from raw sequence."""
    start = _POSE_START + point_idx * 3
    return seq[:, start:start + 3]


def _hand_point(seq: np.ndarray, hand_start: int, point_idx: int) -> np.ndarray:
    """Extract a hand point [num_frames, 3] from raw sequence."""
    start = hand_start + point_idx * 3
    return seq[:, start:start + 3]


def _safe_norm(vectors: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute norm with epsilon to avoid division by zero."""
    return np.linalg.norm(vectors, axis=axis).clip(min=1e-8)


def compute_velocities(seq: np.ndarray) -> np.ndarray:
    """Frame-to-frame differences. First frame is zero."""
    vel = np.zeros_like(seq)
    if seq.shape[0] > 1:
        vel[1:] = seq[1:] - seq[:-1]
    return vel


def compute_hand_distances(seq: np.ndarray) -> np.ndarray:
    """5 inter-hand distances: wrist-wrist + 4 fingertip pairs."""
    num_frames = seq.shape[0]
    distances = np.zeros((num_frames, 5), dtype=np.float32)

    l_wrist = _hand_point(seq, _LEFT_HAND_START, _WRIST)
    r_wrist = _hand_point(seq, _RIGHT_HAND_START, _WRIST)
    distances[:, 0] = _safe_norm(l_wrist - r_wrist)

    for i, tip_idx in enumerate([_THUMB_TIP, _INDEX_TIP, _MIDDLE_TIP, _PINKY_TIP]):
        l_tip = _hand_point(seq, _LEFT_HAND_START, tip_idx)
        r_tip = _hand_point(seq, _RIGHT_HAND_START, tip_idx)
        distances[:, i + 1] = _safe_norm(l_tip - r_tip)

    return distances


def compute_joint_angles(seq: np.ndarray) -> np.ndarray:
    """4 arm joint angles: left/right elbow flexion, left/right wrist flexion."""
    num_frames = seq.shape[0]
    angles = np.zeros((num_frames, 4), dtype=np.float32)

    l_shoulder = _pose_point(seq, _LEFT_SHOULDER)
    r_shoulder = _pose_point(seq, _RIGHT_SHOULDER)
    l_elbow = _pose_point(seq, _LEFT_ELBOW)
    r_elbow = _pose_point(seq, _RIGHT_ELBOW)
    l_wrist = _pose_point(seq, _LEFT_WRIST)
    r_wrist = _pose_point(seq, _RIGHT_WRIST)

    def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Angle at vertex b formed by segments a-b and c-b."""
        ba = a - b
        bc = c - b
        cos = np.sum(ba * bc, axis=-1) / (_safe_norm(ba) * _safe_norm(bc))
        return np.arccos(np.clip(cos, -1.0, 1.0))

    angles[:, 0] = _angle(l_shoulder, l_elbow, l_wrist)
    angles[:, 1] = _angle(r_shoulder, r_elbow, r_wrist)

    # Wrist angles: use hand wrist landmark relative to pose elbow/wrist
    l_hand_wrist = _hand_point(seq, _LEFT_HAND_START, _WRIST)
    r_hand_wrist = _hand_point(seq, _RIGHT_HAND_START, _WRIST)
    l_hand_middle = _hand_point(seq, _LEFT_HAND_START, _MIDDLE_TIP)
    r_hand_middle = _hand_point(seq, _RIGHT_HAND_START, _MIDDLE_TIP)

    angles[:, 2] = _angle(l_elbow, l_hand_wrist, l_hand_middle)
    angles[:, 3] = _angle(r_elbow, r_hand_wrist, r_hand_middle)

    return np.nan_to_num(angles, nan=0.0).astype(np.float32)


def compute_hand_shape_features(seq: np.ndarray) -> np.ndarray:
    """10 hand shape features: 5 per hand (openness, 4 finger extensions)."""
    num_frames = seq.shape[0]
    features = np.zeros((num_frames, 10), dtype=np.float32)

    for hand_idx, hand_start in enumerate([_LEFT_HAND_START, _RIGHT_HAND_START]):
        wrist = _hand_point(seq, hand_start, _WRIST)
        tips = [_hand_point(seq, hand_start, t) for t in [_INDEX_TIP, _MIDDLE_TIP, _RING_TIP, _PINKY_TIP]]

        # Openness: average distance from wrist to all fingertips
        tip_dists = np.stack([_safe_norm(tip - wrist) for tip in tips], axis=-1)
        features[:, hand_idx * 5] = tip_dists.mean(axis=-1)

        # Individual finger extensions
        for i, tip in enumerate(tips):
            features[:, hand_idx * 5 + 1 + i] = _safe_norm(tip - wrist)

    return features


def compute_enriched_features(seq: np.ndarray) -> np.ndarray:
    """
    Compute enriched features from raw 225-dim landmark sequence.

    Input:  [num_frames, 225]
    Output: [num_frames, ENRICHED_FEATURE_DIM]  (~469)
    """
    velocities = compute_velocities(seq)
    hand_distances = compute_hand_distances(seq)
    joint_angles = compute_joint_angles(seq)
    hand_shapes = compute_hand_shape_features(seq)

    enriched = np.concatenate([
        seq,             # 225: raw coordinates
        velocities,      # 225: frame-to-frame velocities
        hand_distances,  # 5: inter-hand distances
        joint_angles,    # 4: arm/wrist angles
        hand_shapes,     # 10: hand shape features
    ], axis=1)

    return np.nan_to_num(enriched, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
```

**Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_ml/test_feature_engineering.py -v`
Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add backend/app/ml/feature_engineering.py backend/tests/test_ml/test_feature_engineering.py
git commit -m "feat(ml): add enriched feature engineering module

Computes velocities, inter-hand distances, joint angles, and hand shape
features from raw 225-dim landmarks. Output: 469-dim feature vectors."
```

---

## Task 2: Temporal Resampling

**Files:**
- Modify: `backend/app/ml/dataset.py`
- Test: `backend/tests/test_ml/test_temporal_resample.py`

**Step 1: Write failing tests**

```python
"""Tests for temporal resampling in dataset."""
from __future__ import annotations

import numpy as np
import pytest

from app.ml.dataset import temporal_resample


def test_resample_shorter_sequence_pads():
    """Sequence shorter than target should be interpolated up."""
    seq = np.arange(20).reshape(4, 5).astype(np.float32)
    result = temporal_resample(seq, target_len=8)
    assert result.shape == (8, 5)


def test_resample_longer_sequence_downsamples():
    """Sequence longer than target should be interpolated down."""
    seq = np.arange(100).reshape(20, 5).astype(np.float32)
    result = temporal_resample(seq, target_len=8)
    assert result.shape == (8, 5)


def test_resample_exact_length_unchanged():
    """Sequence exactly at target length should be unchanged."""
    seq = np.arange(40).reshape(8, 5).astype(np.float32)
    result = temporal_resample(seq, target_len=8)
    assert result.shape == (8, 5)
    assert np.allclose(result, seq)


def test_resample_preserves_endpoints():
    """First and last frames should be preserved after resampling."""
    seq = np.random.default_rng(42).standard_normal((30, 225)).astype(np.float32)
    result = temporal_resample(seq, target_len=64)
    assert np.allclose(result[0], seq[0], atol=1e-5)
    assert np.allclose(result[-1], seq[-1], atol=1e-5)


def test_resample_single_frame():
    """Single frame should be repeated to fill target length."""
    seq = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    result = temporal_resample(seq, target_len=8)
    assert result.shape == (8, 3)
    # All frames should be identical to the single input frame
    for i in range(8):
        assert np.allclose(result[i], seq[0])
```

**Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_ml/test_temporal_resample.py -v`
Expected: FAIL — `temporal_resample` not found

**Step 3: Add temporal_resample to dataset.py**

Add to `backend/app/ml/dataset.py` after imports:

```python
from scipy import interpolate as scipy_interpolate


def temporal_resample(sequence: np.ndarray, target_len: int = 64) -> np.ndarray:
    """
    Resample a variable-length sequence to a fixed number of frames.

    Uses linear interpolation to uniformly sample the temporal axis.
    Preserves first and last frames exactly.

    Args:
        sequence: Array of shape [num_frames, num_features]
        target_len: Desired output length (default: 64)

    Returns:
        Array of shape [target_len, num_features]
    """
    num_frames = sequence.shape[0]

    if num_frames == target_len:
        return sequence.copy()

    if num_frames == 1:
        return np.tile(sequence, (target_len, 1))

    original_indices = np.linspace(0, 1, num_frames)
    target_indices = np.linspace(0, 1, target_len)

    resampled = np.zeros((target_len, sequence.shape[1]), dtype=sequence.dtype)
    for feat_idx in range(sequence.shape[1]):
        interp_fn = scipy_interpolate.interp1d(
            original_indices, sequence[:, feat_idx], kind="linear"
        )
        resampled[:, feat_idx] = interp_fn(target_indices)

    return resampled
```

**Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_ml/test_temporal_resample.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add backend/app/ml/dataset.py backend/tests/test_ml/test_temporal_resample.py
git commit -m "feat(ml): add temporal_resample for variable-length sequences

Linear interpolation to fixed 64 tokens preserving endpoints.
Replaces fixed sliding window for training."
```

---

## Task 3: Update LandmarkDataset to use resampling + enriched features

**Files:**
- Modify: `backend/app/ml/dataset.py`
- Test: `backend/tests/test_ml/test_dataset_v2.py`

**Step 1: Write failing tests**

```python
"""Tests for updated LandmarkDataset with resampling and enriched features."""
from __future__ import annotations

import numpy as np
import torch

from app.ml.dataset import LandmarkDataset, SignSample
from app.ml.feature_engineering import ENRICHED_FEATURE_DIM


def _make_sample(num_frames: int, label: int = 0) -> SignSample:
    rng = np.random.default_rng(42 + label)
    landmarks = rng.standard_normal((num_frames, 225)).astype(np.float32)
    return SignSample(landmarks=landmarks, label=label)


def test_dataset_resample_mode_output_shape():
    """Each sample should be [target_len, ENRICHED_FEATURE_DIM]."""
    samples = [_make_sample(50, 0), _make_sample(100, 1)]
    ds = LandmarkDataset(samples, sequence_length=64, use_enriched_features=True, apply_sliding_window=False)
    tensor, label = ds[0]
    assert tensor.shape == (64, ENRICHED_FEATURE_DIM)


def test_dataset_resample_mode_short_video():
    """Short video should be resampled up to target_len."""
    samples = [_make_sample(10, 0)]
    ds = LandmarkDataset(samples, sequence_length=64, use_enriched_features=True, apply_sliding_window=False)
    tensor, _ = ds[0]
    assert tensor.shape == (64, ENRICHED_FEATURE_DIM)


def test_dataset_resample_mode_long_video():
    """Long video should be resampled down to target_len."""
    samples = [_make_sample(200, 0)]
    ds = LandmarkDataset(samples, sequence_length=64, use_enriched_features=True, apply_sliding_window=False)
    tensor, _ = ds[0]
    assert tensor.shape == (64, ENRICHED_FEATURE_DIM)


def test_dataset_one_sample_per_video():
    """Without sliding window, each video produces exactly 1 sample."""
    samples = [_make_sample(50, 0), _make_sample(100, 1), _make_sample(200, 2)]
    ds = LandmarkDataset(samples, sequence_length=64, use_enriched_features=True, apply_sliding_window=False)
    assert len(ds) == 3


def test_dataset_backward_compat_sliding_window():
    """Old sliding window mode still works with 225 features."""
    samples = [_make_sample(50, 0)]
    ds = LandmarkDataset(samples, sequence_length=30, stride=10, apply_sliding_window=True, use_enriched_features=False)
    tensor, _ = ds[0]
    assert tensor.shape[1] == 225
```

**Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_ml/test_dataset_v2.py -v`
Expected: FAIL — `use_enriched_features` not recognized

**Step 3: Update LandmarkDataset**

Modify `backend/app/ml/dataset.py` — add `use_enriched_features` parameter to `__init__`:

```python
from app.ml.feature_engineering import compute_enriched_features

class LandmarkDataset(Dataset):
    def __init__(
        self,
        samples: list[SignSample],
        sequence_length: int = 64,
        stride: int = 10,
        apply_sliding_window: bool = False,
        use_enriched_features: bool = True,
    ) -> None:
        self.sequence_length = sequence_length
        self.stride = stride
        self.apply_sliding_window = apply_sliding_window
        self.use_enriched_features = use_enriched_features

        self.processed_samples: list[tuple[torch.Tensor, int]] = []

        for sample in samples:
            if apply_sliding_window:
                sequences = self._create_sliding_windows(sample.landmarks, sample.label)
                self.processed_samples.extend(sequences)
            else:
                resampled = temporal_resample(sample.landmarks, target_len=sequence_length)
                if use_enriched_features:
                    resampled = compute_enriched_features(resampled)
                tensor = torch.from_numpy(resampled).float()
                self.processed_samples.append((tensor, sample.label))

        logger.debug(
            "dataset_initialized",
            num_samples=len(samples),
            num_sequences=len(self.processed_samples),
            sequence_length=sequence_length,
            use_enriched_features=use_enriched_features,
        )
```

Keep `_create_sliding_windows` and `_pad_or_truncate` unchanged for backward compatibility.

**Step 4: Run all dataset tests**

Run: `cd backend && python -m pytest tests/test_ml/test_dataset_v2.py tests/test_ml/test_temporal_resample.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add backend/app/ml/dataset.py backend/tests/test_ml/test_dataset_v2.py
git commit -m "feat(ml): integrate temporal resampling and enriched features in LandmarkDataset

New default: resample to 64 frames + compute 469 enriched features.
Old sliding window mode preserved for backward compatibility."
```

---

## Task 4: Update model defaults

**Files:**
- Modify: `backend/app/ml/model.py`
- Modify: `backend/tests/test_ml/test_model.py`

**Step 1: Write failing tests**

Add to `backend/tests/test_ml/test_model.py`:

```python
from app.ml.feature_engineering import ENRICHED_FEATURE_DIM


def test_model_reduced_defaults():
    """New defaults should create a smaller model."""
    model = SignTransformer(num_classes=5)
    assert model.d_model == 128
    assert model.nhead == 4
    assert model.num_layers == 2
    assert model.num_features == ENRICHED_FEATURE_DIM


def test_model_forward_with_enriched_features():
    """Model should handle enriched feature dimension."""
    model = SignTransformer(num_classes=5)
    x = torch.randn(2, 64, ENRICHED_FEATURE_DIM)
    logits = model(x)
    assert logits.shape == (2, 5)


def test_model_old_config_still_works():
    """Explicit old-style config should still work."""
    model = SignTransformer(
        num_features=225, num_classes=3, d_model=256, nhead=8, num_layers=4
    )
    x = torch.randn(1, 30, 225)
    logits = model(x)
    assert logits.shape == (1, 3)
```

**Step 2: Run to verify failure**

Run: `cd backend && python -m pytest tests/test_ml/test_model.py -v`
Expected: FAIL on `test_model_reduced_defaults` (d_model is 256, not 128)

**Step 3: Update model defaults**

Modify `backend/app/ml/model.py` `SignTransformer.__init__`:

```python
from app.ml.feature_engineering import ENRICHED_FEATURE_DIM

class SignTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_features: int = ENRICHED_FEATURE_DIM,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
        feature_dropout: float = 0.2,
        pooling_dropout: float = 0.3,
        use_cls_token: bool = True,
    ) -> None:
```

Only change default values — no structural changes.

**Step 4: Run all model tests**

Run: `cd backend && python -m pytest tests/test_ml/test_model.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add backend/app/ml/model.py backend/tests/test_ml/test_model.py
git commit -m "feat(ml): reduce model defaults for better generalization

d_model=128, num_layers=2, nhead=4, dropout=0.3.
~150k params instead of ~600k. Old configs still loadable."
```

---

## Task 5: Update trainer defaults and augmentation

**Files:**
- Modify: `backend/app/ml/trainer.py`
- Modify: `backend/app/ml/augmentation.py`
- Test: `backend/tests/test_ml/test_augmentation_v2.py`

**Step 1: Write failing test for temporal_crop**

```python
"""Tests for new augmentation: temporal_crop."""
from __future__ import annotations

import numpy as np

from app.ml.augmentation import temporal_crop


def test_temporal_crop_preserves_features():
    """Output should have same num_features as input."""
    seq = np.random.default_rng(42).standard_normal((64, 469)).astype(np.float32)
    cropped = temporal_crop(seq, min_ratio=0.7)
    assert cropped.shape == (64, 469)


def test_temporal_crop_is_different():
    """Cropped sequence should differ from original (stochastic)."""
    seq = np.random.default_rng(42).standard_normal((64, 469)).astype(np.float32)
    np.random.seed(123)
    cropped = temporal_crop(seq, min_ratio=0.7)
    assert not np.allclose(cropped, seq)


def test_temporal_crop_short_sequence():
    """Very short sequence should be returned as-is."""
    seq = np.random.default_rng(42).standard_normal((3, 10)).astype(np.float32)
    cropped = temporal_crop(seq, min_ratio=0.7)
    assert cropped.shape == (3, 10)
```

**Step 2: Run to verify failure**

Run: `cd backend && python -m pytest tests/test_ml/test_augmentation_v2.py -v`
Expected: FAIL — `temporal_crop` signature mismatch or not found

**Step 3: Add temporal_crop and update augmentation.py**

Add to `backend/app/ml/augmentation.py`:

```python
def temporal_crop(sequence: np.ndarray, min_ratio: float = 0.7) -> np.ndarray:
    """
    Take a random temporal sub-segment and resample to original length.

    Simulates variations in sign start/end timing.

    Args:
        sequence: Input [num_frames, num_features]
        min_ratio: Minimum ratio of frames to keep (0.7 = 70%)

    Returns:
        Resampled sequence with same shape as input
    """
    num_frames = sequence.shape[0]
    if num_frames < 4:
        return sequence

    ratio = np.random.uniform(min_ratio, 1.0)
    crop_len = max(2, int(num_frames * ratio))
    max_start = num_frames - crop_len
    start = np.random.randint(0, max(1, max_start + 1))
    cropped = sequence[start:start + crop_len]

    if cropped.shape[0] == num_frames:
        return cropped

    # Resample back to original length
    from app.ml.dataset import temporal_resample
    return temporal_resample(cropped, target_len=num_frames)
```

Update the default augmentation list in `apply_augmentations`:

```python
    if augmentations is None:
        augmentations = [
            mirror_horizontal,
            swap_hands,
            lambda seq: temporal_jitter(seq, max_shift=5),
            lambda seq: gaussian_noise(seq, sigma=0.01),
            lambda seq: speed_variation(seq, speed_factor=None),
            lambda seq: random_frame_dropout(seq, drop_ratio=0.1),
            lambda seq: temporal_cutout(seq, max_ratio=0.2),
            lambda seq: temporal_crop(seq, min_ratio=0.7),
        ]
```

**Step 4: Update TrainingConfig defaults in trainer.py**

In `backend/app/ml/trainer.py` `TrainingConfig`:

```python
@dataclass
class TrainingConfig:
    num_epochs: int = 50
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_workers: int = 2
    device: str = "cpu"
    early_stopping_patience: int = 15       # was 10
    early_stopping_min_delta: float = 1e-4
    gradient_clip_max_norm: float = 1.0

    weight_decay: float = 0.05              # was 0.01
    classifier_lr_multiplier: float = 2.0
    label_smoothing: float = 0.1            # was 0.05

    use_focal_loss: bool = False
    focal_loss_gamma: float = 2.0
    focal_loss_alpha: float = 0.25

    warmup_epochs: int = 3                  # was 5
    use_class_weights: bool = True
    use_weighted_sampler: bool = True

    use_mixup: bool = True                  # was False for few-shot
    mixup_alpha: float = 0.3               # was 0.2

    use_ema: bool = True
    ema_decay: float = 0.995
```

**Step 5: Run all tests**

Run: `cd backend && python -m pytest tests/test_ml/ -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add backend/app/ml/augmentation.py backend/app/ml/trainer.py backend/tests/test_ml/test_augmentation_v2.py
git commit -m "feat(ml): add temporal_crop augmentation and stronger regularization

temporal_crop simulates sign timing variations. Training defaults
updated: weight_decay=0.05, label_smoothing=0.1, mixup_alpha=0.3,
patience=15, warmup=3."
```

---

## Task 6: Inference Pipeline State Machine

**Files:**
- Modify: `backend/app/ml/pipeline.py`
- Test: `backend/tests/test_ml/test_pipeline_state_machine.py`

**Step 1: Write failing tests**

```python
"""Tests for inference pipeline state machine."""
from __future__ import annotations

import numpy as np

from app.ml.pipeline import SignFlowInferencePipeline, InferenceState


def _hand_payload(value: float = 0.5) -> dict:
    """Create a payload with visible hand landmarks."""
    return {
        "hands": {
            "left": [[value, value, 0.0]] * 21,
            "right": [[value + 0.1, value, 0.0]] * 21,
        },
        "pose": [[0.5, 0.5, 0.0]] * 33,
    }


def _idle_payload() -> dict:
    """Create a payload with hands at rest (zero)."""
    return {
        "hands": {"left": [[0.0, 0.0, 0.0]] * 21, "right": [[0.0, 0.0, 0.0]] * 21},
        "pose": [[0.5, 0.5, 0.0]] * 33,
    }


def test_pipeline_starts_in_idle():
    pipeline = SignFlowInferencePipeline()
    assert pipeline.state == InferenceState.IDLE


def test_pipeline_transitions_to_recording_on_motion():
    pipeline = SignFlowInferencePipeline()
    # Send frames with hand movement
    for i in range(5):
        pipeline.process_frame(_hand_payload(0.1 * (i + 1)))
    assert pipeline.state == InferenceState.RECORDING


def test_pipeline_returns_to_idle_after_rest():
    pipeline = SignFlowInferencePipeline(
        rest_frames_threshold=3, min_recording_frames=2
    )
    # Start recording
    for i in range(5):
        pipeline.process_frame(_hand_payload(0.1 * (i + 1)))
    # Send idle frames to trigger end-of-sign
    for _ in range(5):
        pipeline.process_frame(_idle_payload())
    assert pipeline.state == InferenceState.IDLE


def test_pipeline_prediction_on_sign_complete():
    """Pipeline should produce non-NONE prediction when sign ends (if model loaded)."""
    pipeline = SignFlowInferencePipeline(
        rest_frames_threshold=3, min_recording_frames=2
    )
    # No model loaded, so prediction will be NONE, but state machine should still cycle
    for i in range(10):
        pipeline.process_frame(_hand_payload(0.1 * (i + 1)))
    for _ in range(5):
        result = pipeline.process_frame(_idle_payload())
    # Without model, prediction should be NONE
    assert result.prediction == "NONE"
```

**Step 2: Run to verify failure**

Run: `cd backend && python -m pytest tests/test_ml/test_pipeline_state_machine.py -v`
Expected: FAIL — `InferenceState` not found

**Step 3: Add state machine to pipeline.py**

Add enum and update `SignFlowInferencePipeline`:

```python
from enum import Enum

class InferenceState(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    INFERRING = "inferring"
```

Update `SignFlowInferencePipeline.__init__` — add state machine params:

```python
    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        seq_len: int = 64,
        confidence_threshold: float = 0.7,
        smoothing_window: int = 5,
        min_hand_visibility: float = 0.2,
        min_prediction_margin: float = 0.1,
        min_motion_energy: float = 0.003,
        device: str = "cpu",
        max_buffer_frames: int = 180,
        rest_frames_threshold: int = 10,
        motion_start_threshold: float = 0.005,
        min_recording_frames: int = 15,
    ) -> None:
        # ... existing init ...
        self.frame_buffer: deque[np.ndarray] = deque(maxlen=max_buffer_frames)
        self.max_buffer_frames = max_buffer_frames
        self.rest_frames_threshold = rest_frames_threshold
        self.motion_start_threshold = motion_start_threshold
        self.min_recording_frames = min_recording_frames
        self.state = InferenceState.IDLE
        self._rest_frame_count = 0
        self._recording_frame_count = 0
```

Update `process_frame` to implement state machine:

```python
    def process_frame(self, payload: dict) -> Prediction:
        frame = FrameLandmarks(
            left_hand=payload.get("hands", {}).get("left", []) or [],
            right_hand=payload.get("hands", {}).get("right", []) or [],
            pose=payload.get("pose", []) or [],
            face=payload.get("face", []) or [],
        )

        hand_visibility = self._compute_hand_visibility(frame)
        self.hand_visibility_history.append(hand_visibility)

        features = normalize_landmarks(frame, include_face=False)
        self.frame_buffer.append(features)
        self._current_motion_energy = self._compute_motion_energy()
        self.motion_history.append(self._current_motion_energy)

        # State machine
        if self.state == InferenceState.IDLE:
            if self._current_motion_energy > self.motion_start_threshold and hand_visibility > self.min_hand_visibility:
                self.state = InferenceState.RECORDING
                self._recording_frame_count = 1
                self._rest_frame_count = 0
            return self._idle_prediction()

        if self.state == InferenceState.RECORDING:
            self._recording_frame_count += 1

            is_resting = self._current_motion_energy < self.motion_start_threshold
            if is_resting:
                self._rest_frame_count += 1
            else:
                self._rest_frame_count = 0

            sign_ended = (
                (self._rest_frame_count >= self.rest_frames_threshold and self._recording_frame_count >= self.min_recording_frames)
                or len(self.frame_buffer) >= self.max_buffer_frames
            )

            if sign_ended:
                self.state = InferenceState.INFERRING
                return self._infer_complete_sign()

            return self._recording_prediction()

        # INFERRING state — runs once then returns to IDLE
        self.state = InferenceState.IDLE
        return self._idle_prediction()

    def _idle_prediction(self) -> Prediction:
        return Prediction(
            prediction="NONE", confidence=0.0, alternatives=[],
            sentence_buffer=" ".join(self.sentence_tokens),
            is_sentence_complete=False,
        )

    def _recording_prediction(self) -> Prediction:
        return Prediction(
            prediction="RECORDING", confidence=0.0, alternatives=[],
            sentence_buffer=" ".join(self.sentence_tokens),
            is_sentence_complete=False,
        )

    def _infer_complete_sign(self) -> Prediction:
        """Run inference on the complete recorded sign."""
        from app.ml.dataset import temporal_resample
        from app.ml.feature_engineering import compute_enriched_features

        buffer_array = np.stack(list(self.frame_buffer), axis=0)
        resampled = temporal_resample(buffer_array, target_len=self.seq_len)
        enriched = compute_enriched_features(resampled)

        predicted_label, predicted_confidence, alternatives = self._infer_window(enriched)

        # Reset for next sign
        self.frame_buffer.clear()
        self._rest_frame_count = 0
        self._recording_frame_count = 0
        self.state = InferenceState.IDLE

        if predicted_label != "NONE" and predicted_confidence >= self.confidence_threshold:
            if not self.sentence_tokens or self.sentence_tokens[-1] != predicted_label:
                self.sentence_tokens.append(predicted_label)

        sentence = " ".join(self.sentence_tokens)
        return Prediction(
            prediction=predicted_label,
            confidence=predicted_confidence,
            alternatives=alternatives,
            sentence_buffer=sentence,
            is_sentence_complete=False,
        )
```

**Step 4: Run all pipeline tests**

Run: `cd backend && python -m pytest tests/test_ml/test_pipeline_state_machine.py tests/test_ml/test_pipeline.py -v`
Expected: All PASS (old tests may need minor adjustments for new default seq_len)

**Step 5: Fix old pipeline tests if needed**

Update `tests/test_ml/test_pipeline.py` — pass explicit `seq_len` and adjust for state machine:

- `test_pipeline_warmup_prediction`: Now with state machine, IDLE returns NONE — still passes
- `test_pipeline_smoothing_prefers_temporal_consensus`: Still tests internal `_smooth()` — no change
- `test_pipeline_rejects_predictions_when_hands_not_visible`: Adjust for state machine behavior

**Step 6: Commit**

```bash
git add backend/app/ml/pipeline.py backend/tests/test_ml/test_pipeline_state_machine.py backend/tests/test_ml/test_pipeline.py
git commit -m "feat(ml): add inference state machine for compound sign detection

IDLE → RECORDING → INFERRING cycle. Detects sign boundaries via
motion energy. Buffer 180 frames, resample to 64 before inference."
```

---

## Task 7: Update schemas and training service

**Files:**
- Modify: `backend/app/schemas/training.py`
- Modify: `backend/app/services/training_service.py`
- Modify: `backend/app/ml/fewshot.py`

**Step 1: Update training schema**

In `backend/app/schemas/training.py`:

```python
class TrainingConfig(BaseModel):
    epochs: int = Field(default=50, ge=1, le=500)
    learning_rate: float = Field(default=1e-4, gt=0, le=1)
    augmentation: bool = True
    sequence_length: int = Field(default=64, ge=8, le=256)   # was 30, max 256
    early_stopping_patience: int = Field(default=15, ge=1, le=100)  # was 10
    early_stopping_min_delta: float = Field(default=1e-4, ge=0, le=1)
    weight_decay: float = Field(default=0.05, ge=0, le=1)    # was 0.01
    classifier_lr_multiplier: float = Field(default=2.0, ge=0.1, le=20)
    label_smoothing: float = Field(default=0.1, ge=0.0, le=0.4)  # was 0.05
    warmup_epochs: int = Field(default=3, ge=0, le=100)      # was 5
    use_mixup: bool = True
    mixup_alpha: float = Field(default=0.3, ge=0.0, le=1.0)  # was 0.2
    use_ema: bool = True
    ema_decay: float = Field(default=0.995, ge=0.8, le=0.9999)
    min_deploy_accuracy: float = Field(default=0.85, ge=0.0, le=1.0)
```

Remove `stride` field (no longer used in new mode).

**Step 2: Update training service**

In `backend/app/services/training_service.py`, update the preprocessing phase:

Replace lines 337-354 (dataset creation) with:

```python
                sequence_length = int(config.get("sequence_length", 64))
                sequence_length = max(8, min(256, sequence_length))

                # Apply augmentation for all modes (not just few-shot)
                if config.get("augmentation", True):
                    logger.info("applying_data_augmentation", mode=mode)
                    sequences, labels = augment_dataset(
                        sequences,
                        labels,
                        num_augmentations_per_sample=5,
                        augmentation_probability=0.5,
                    )
                    logger.info("augmentation_complete", total_samples=len(sequences))

                # Create stratified split after augmentation
                train_indices, val_indices = self._stratified_train_val_indices(labels, val_ratio=0.2)
                train_samples = [SignSample(sequences[i], labels[i]) for i in train_indices]
                val_samples = [SignSample(sequences[i], labels[i]) for i in val_indices]

                # Create datasets with resampling + enriched features
                train_dataset = LandmarkDataset(
                    train_samples,
                    sequence_length=sequence_length,
                    apply_sliding_window=False,
                    use_enriched_features=True,
                )
                val_dataset = LandmarkDataset(
                    val_samples,
                    sequence_length=sequence_length,
                    apply_sliding_window=False,
                    use_enriched_features=True,
                )
```

Update model creation (lines 372-394) to use `ENRICHED_FEATURE_DIM`:

```python
                from app.ml.feature_engineering import ENRICHED_FEATURE_DIM
                num_features = ENRICHED_FEATURE_DIM

                if mode == "few-shot":
                    prepared = prepare_few_shot_model(
                        checkpoint_path=active_checkpoint,
                        num_features=num_features,
                        num_classes=num_classes,
                        d_model=128,
                        device=device,
                        freeze_until_layer=1,  # Only 2 layers, freeze first 1
                    )
                    model = prepared.model
                    num_classes = model.num_classes
                else:
                    model = SignTransformer(
                        num_features=num_features,
                        num_classes=num_classes,
                    )
```

**Step 3: Update fewshot.py defaults**

In `backend/app/ml/fewshot.py`, update `prepare_few_shot_model` default:

```python
def prepare_few_shot_model(
    *,
    checkpoint_path: str | Path | None,
    num_features: int,
    num_classes: int,
    d_model: int = 128,          # was 256
    device: str = "cpu",
    freeze_until_layer: int = 1,  # was 3 (now only 2 layers)
    freeze_embedding: bool = True,
) -> FewShotPreparation:
```

**Step 4: Run full test suite**

Run: `cd backend && python -m pytest tests/ -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add backend/app/schemas/training.py backend/app/services/training_service.py backend/app/ml/fewshot.py
git commit -m "feat(ml): wire new model config through training pipeline

Schema: seq_len=64, no stride, stronger regularization defaults.
Service: resampling+enriched features, augmentation for all modes.
Fewshot: d_model=128, freeze_until_layer=1."
```

---

## Task 8: Final integration test and cleanup

**Files:**
- Test: `backend/tests/test_ml/test_integration.py`

**Step 1: Write integration test**

```python
"""End-to-end integration test for the improved ML pipeline."""
from __future__ import annotations

import numpy as np
import torch

from app.ml.dataset import LandmarkDataset, SignSample, temporal_resample
from app.ml.feature_engineering import compute_enriched_features, ENRICHED_FEATURE_DIM
from app.ml.model import SignTransformer
from app.ml.trainer import SignTrainer, TrainingConfig


def test_full_training_pipeline_no_overfit():
    """Train on synthetic data and verify model doesn't achieve 0.0 loss in 1 epoch."""
    rng = np.random.default_rng(42)
    num_classes = 3
    samples_per_class = 10

    samples = []
    for class_idx in range(num_classes):
        for _ in range(samples_per_class):
            # Each class has a distinct pattern with noise
            num_frames = rng.integers(30, 120)
            base = rng.standard_normal((1, 225)).astype(np.float32) * (class_idx + 1)
            noise = rng.standard_normal((num_frames, 225)).astype(np.float32) * 0.3
            landmarks = base + noise
            samples.append(SignSample(landmarks=landmarks, label=class_idx))

    rng.shuffle(samples)
    train_samples = samples[:24]
    val_samples = samples[24:]

    train_ds = LandmarkDataset(train_samples, sequence_length=64, apply_sliding_window=False, use_enriched_features=True)
    val_ds = LandmarkDataset(val_samples, sequence_length=64, apply_sliding_window=False, use_enriched_features=True)

    model = SignTransformer(num_classes=num_classes)

    config = TrainingConfig(
        num_epochs=3,
        learning_rate=1e-4,
        batch_size=8,
        num_workers=0,
        device="cpu",
        early_stopping_patience=15,
    )

    trainer = SignTrainer(model=model, config=config)
    metrics = trainer.fit(train_ds, val_ds)

    # Key assertion: model should NOT have 0.0 loss after epoch 1
    assert metrics[0].train_loss > 0.01, f"Train loss too low at epoch 1: {metrics[0].train_loss}"
    # Model should be learning (loss decreasing or accuracy improving)
    assert len(metrics) == 3


def test_temporal_resample_then_enriched_features():
    """Verify the data pipeline: raw → resample → enrich."""
    raw = np.random.default_rng(42).standard_normal((100, 225)).astype(np.float32)
    resampled = temporal_resample(raw, target_len=64)
    enriched = compute_enriched_features(resampled)
    assert enriched.shape == (64, ENRICHED_FEATURE_DIM)
    assert not np.any(np.isnan(enriched))
```

**Step 2: Run integration test**

Run: `cd backend && python -m pytest tests/test_ml/test_integration.py -v`
Expected: All PASS — and critically, train_loss > 0.01 at epoch 1

**Step 3: Run full test suite**

Run: `cd backend && python -m pytest tests/ -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add backend/tests/test_ml/test_integration.py
git commit -m "test(ml): add integration test verifying no overfitting

Synthetic 3-class dataset, 10 samples each. Asserts train_loss > 0.01
after epoch 1 — the original bug where loss was 0.0 is now prevented."
```

---

## Summary of Changes

| Task | What | Files |
|------|------|-------|
| 1 | Feature engineering (velocities, distances, angles) | `feature_engineering.py` (NEW) |
| 2 | Temporal resampling | `dataset.py` |
| 3 | Dataset integration (resample + enrich) | `dataset.py` |
| 4 | Reduced model defaults | `model.py` |
| 5 | Stronger regularization + temporal_crop | `trainer.py`, `augmentation.py` |
| 6 | Inference state machine | `pipeline.py` |
| 7 | Schema + service wiring | `training.py`, `training_service.py`, `fewshot.py` |
| 8 | Integration test | `test_integration.py` |
