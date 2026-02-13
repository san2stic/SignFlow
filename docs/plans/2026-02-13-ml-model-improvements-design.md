# ML Model Improvements — Adaptive Temporal + Feature Engineering

**Date:** 2026-02-13
**Status:** Approved
**Approach:** A — Adaptive Temporal + Feature Engineering

## Context

Training logs show train_loss=0.0, val_loss=0.0, accuracy=100% from epoch 1 — the model memorizes instead of learning. Root causes:
- Window too short (30 frames / 1 sec) for compound signs (up to 8 sec)
- Raw coordinates only (225 features) — no motion information
- Model too large (~600k params) for dataset size (10-30 videos per sign)
- No compound sign handling (multi-movement gestures)

## Section 1: Enriched Feature Engineering

New file `backend/app/ml/feature_engineering.py`.

Features per frame:

| Feature | Dimension | Description |
|---------|-----------|-------------|
| Raw coordinates | 225 | Existing x,y,z positions |
| Velocities | 225 | Frame-to-frame differences (dx, dy, dz) |
| Inter-hand distances | 5 | Wrist-wrist, key fingertip distances |
| Wrist/elbow angles | 4 | Arm joint flexion |
| Hand shape features | 10 | Openness, finger spread, fist vs open |

**Total: ~469 features per frame**

Function: `compute_enriched_features(raw_landmarks: np.ndarray) -> np.ndarray`
- Input: `[num_frames, 225]`
- Output: `[num_frames, ~469]`
- First frame velocities = 0

## Section 2: Adaptive Temporal Handling

Replace fixed `seq_len=30` sliding window with full-video temporal resampling.

**Strategy:** Take entire video, resample to fixed `max_tokens=64` frames via linear interpolation.

```
Short video  (30 frames / 1 sec)  → pad to 64
Medium video (90 frames / 3 sec)  → resample to 64
Long video   (240 frames / 8 sec) → resample to 64
```

Function: `temporal_resample(sequence: np.ndarray, target_len: int = 64) -> np.ndarray`

Changes in `LandmarkDataset`:
- Remove sliding window for training (each video = 1 complete sample)
- Apply `temporal_resample()` then `compute_enriched_features()`

Changes in `SignFlowInferencePipeline`:
- Buffer: `deque(maxlen=180)` (6 sec at 30fps)
- Resample to 64 tokens before inference
- End-of-sign detection via motion energy returning to zero

## Section 3: Reduced Model Architecture

| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| d_model | 256 | 128 | Halved, sufficient for ~469 features |
| nhead | 8 | 4 | 128/4 = 32 dim/head |
| num_layers | 4 | 2 | Less memorization |
| dim_feedforward | 512 | 256 | Proportional to d_model |
| dropout | 0.1 | 0.3 | Aggressive regularization |
| feature_dropout | 0.1 | 0.2 | Idem |
| pooling_dropout | 0.1 | 0.3 | Idem |

**~150k parameters** (down from ~600k)

Training config changes:

| Parameter | Old | New |
|-----------|-----|-----|
| label_smoothing | 0.05 | 0.1 |
| weight_decay | 0.01 | 0.05 |
| warmup_epochs | 5 | 3 |
| early_stopping_patience | 10 | 15 |
| use_mixup | False (few-shot) | True always |
| mixup_alpha | 0.2 | 0.3 |

Augmentation changes:
- `num_augmentations_per_sample`: 3 → 5
- Apply in full-retrain mode too (not just few-shot)
- New `temporal_crop()`: random 70-100% sub-segment, resample to 64

## Section 4: Inference State Machine

Replace continuous per-frame prediction with sign-level prediction.

```
IDLE (hands at rest)
  → motion_energy > threshold
  → RECORDING: accumulate frames (max 180)

RECORDING
  → motion_energy < threshold for 10+ consecutive frames
  → OR buffer full (180 frames)
  → INFERRING

INFERRING
  → temporal_resample(buffer, 64)
  → compute_enriched_features()
  → model.forward() → prediction
  → Return to IDLE
```

One inference per complete sign instead of per frame.

## Section 5: Files Changed

| File | Change |
|------|--------|
| `ml/feature_engineering.py` | **NEW** — velocities, distances, angles |
| `ml/model.py` | New defaults (d_model=128, layers=2, dropout=0.3) |
| `ml/dataset.py` | Add `temporal_resample()`, remove sliding window for training, call `compute_enriched_features()` |
| `ml/augmentation.py` | Add `temporal_crop()`, enable augmentation in full-retrain |
| `ml/pipeline.py` | State machine, buffer 180, resample+enriched before inference |
| `ml/trainer.py` | Regularization defaults (label_smoothing=0.1, weight_decay=0.05, mixup always) |
| `ml/fewshot.py` | Adjust num_features for new format |
| `schemas/training.py` | `sequence_length` default 64, remove stride from schema |
| `services/training_service.py` | Pass new configs, remove sliding window logic |

## Backward Compatibility

- Old checkpoints (num_features=225, d_model=256) remain loadable
- Existing .npy files reusable — enrichment happens at training/inference time
- Frontend unchanged — sends same raw landmarks via WebSocket
- Database schema unchanged
- Old models keep working; new models require full re-training
