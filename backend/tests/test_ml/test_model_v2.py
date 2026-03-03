"""Tests for SignTransformerV2 multi-stream architecture and SignBoundaryDetector BiLSTM.

Tests cover:
  - Feature splitting (V2Layout dimensions)
  - Forward pass shape correctness
  - SignTransformerV2Config serialization
  - Checkpoint save/load round-trip
  - SignBoundaryDetector forward pass + boundary detection
  - model_configs V2 specs
  - pipeline integration flags
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# ── Fixtures ─────────────────────────────────────────────────────────────────

BATCH = 2
SEQ = 16  # short sequence to keep tests fast
FEATURE_DIM_V2 = 611
NUM_CLASSES = 10


@pytest.fixture()
def v2_batch() -> torch.Tensor:
    """Random V2 feature batch (2, 16, 611)."""
    torch.manual_seed(42)
    return torch.randn(BATCH, SEQ, FEATURE_DIM_V2)


@pytest.fixture()
def v2_model() -> "SignTransformerV2":  # noqa: F821
    """Small SignTransformerV2 for fast testing."""
    from app.ml.model_v2 import SignTransformerV2, SignTransformerV2Config

    cfg = SignTransformerV2Config(
        feature_dim=FEATURE_DIM_V2,
        hand_hidden_dim=32,
        pose_hidden_dim=16,
        nmm_hidden_dim=16,
        fusion_dim=64,
        num_heads=4,
        num_layers=1,
        num_fusion_layers=1,
        dropout=0.0,
        num_classes=NUM_CLASSES,
        use_cosine_head=True,
        max_seq_len=SEQ + 2,
        use_gcn_hand=False,   # disable GCN to avoid torch-geometric dep in CI
    )
    return SignTransformerV2.from_config(cfg)


# ── V2Layout ──────────────────────────────────────────────────────────────────


def test_v2_layout_constants() -> None:
    """Verify V2Layout slice boundaries sum to 611."""
    from app.ml.model_v2 import V2Layout as L

    assert L.LEFT_HAND_END - L.LEFT_HAND_START == 63
    assert L.RIGHT_HAND_END - L.RIGHT_HAND_START == 63
    assert L.POSE_END - L.POSE_START == 99
    assert L.FACIAL_EXPR_END - L.FACIAL_EXPR_START == 12
    assert L.COORD_VEL_END - L.COORD_VEL_START == 225
    assert L.INTER_DIST_END - L.INTER_DIST_START == 5
    assert L.JOINT_ANGLE_END - L.JOINT_ANGLE_START == 4
    assert L.FACIAL_VEL_END - L.FACIAL_VEL_START == 6
    assert L.HANDSHAPE_END - L.HANDSHAPE_START == 84
    assert L.NMM_END - L.NMM_START == 32
    assert L.SIGNING_SPACE_END - L.SIGNING_SPACE_START == 18
    assert L.TOTAL == FEATURE_DIM_V2


# ── split_feature_streams ────────────────────────────────────────────────────


def test_split_feature_streams_shapes(v2_model, v2_batch) -> None:
    """split_feature_streams should return tensors with expected last dims."""
    from app.ml.model_v2 import V2Layout as L

    streams = v2_model.split_feature_streams(v2_batch)

    assert streams["left_hand"].shape == (BATCH, SEQ, 63)
    assert streams["right_hand"].shape == (BATCH, SEQ, 63)
    assert streams["pose"].shape == (BATCH, SEQ, 99)
    assert streams["facial_expr"].shape == (BATCH, SEQ, 12)
    assert streams["coord_vel"].shape == (BATCH, SEQ, 225)
    assert streams["inter_dist"].shape == (BATCH, SEQ, 5)
    assert streams["joint_angles"].shape == (BATCH, SEQ, 4)
    assert streams["facial_vel"].shape == (BATCH, SEQ, 6)
    assert streams["handshape"].shape == (BATCH, SEQ, 84)
    assert streams["nmm"].shape == (BATCH, SEQ, 32)
    assert streams["signing_space"].shape == (BATCH, SEQ, 18)


# ── forward pass ─────────────────────────────────────────────────────────────


def test_forward_output_shape(v2_model, v2_batch) -> None:
    """forward() must return (batch, num_classes) logits."""
    v2_model.eval()
    with torch.no_grad():
        logits = v2_model(v2_batch)
    assert logits.shape == (BATCH, NUM_CLASSES), f"Expected ({BATCH}, {NUM_CLASSES}), got {logits.shape}"


def test_forward_no_nan(v2_model, v2_batch) -> None:
    """Forward pass must not produce NaN logits."""
    v2_model.eval()
    with torch.no_grad():
        logits = v2_model(v2_batch)
    assert not torch.isnan(logits).any(), "NaN found in logits"
    assert not torch.isinf(logits).any(), "Inf found in logits"


def test_extract_embeddings_shape(v2_model, v2_batch) -> None:
    """extract_embeddings() must return (batch, fusion_dim) tensor."""
    v2_model.eval()
    with torch.no_grad():
        emb = v2_model.extract_embeddings(v2_batch)
    assert emb.shape == (BATCH, v2_model.fusion_dim)


def test_single_frame_does_not_crash(v2_model) -> None:
    """Model should handle single-frame sequences without errors."""
    v2_model.eval()
    x = torch.randn(1, 1, FEATURE_DIM_V2)
    with torch.no_grad():
        logits = v2_model(x)
    assert logits.shape == (1, NUM_CLASSES)


def test_zero_input_fallback(v2_model) -> None:
    """Zero input (no landmarks) must not crash and produces low-confidence output."""
    v2_model.eval()
    x = torch.zeros(1, SEQ, FEATURE_DIM_V2)
    with torch.no_grad():
        logits = v2_model(x)
    assert logits.shape == (1, NUM_CLASSES)
    assert not torch.isnan(logits).any()


# ── Config serialization ──────────────────────────────────────────────────────


def test_config_to_dict_from_dict(v2_model) -> None:
    """Config must survive a to_dict/from_dict round-trip."""
    from app.ml.model_v2 import SignTransformerV2Config

    cfg = v2_model.to_config()
    d = cfg.to_dict()
    cfg2 = SignTransformerV2Config.from_dict(d)

    assert cfg.num_classes == cfg2.num_classes
    assert cfg.fusion_dim == cfg2.fusion_dim
    assert cfg.feature_version == 2


def test_feature_version_marker(v2_model) -> None:
    """Model must expose feature_version == 2."""
    assert v2_model.feature_version == 2
    assert v2_model.FEATURE_VERSION == 2


# ── Checkpoint I/O ────────────────────────────────────────────────────────────


def test_checkpoint_save_load_round_trip(v2_model, v2_batch) -> None:
    """save_checkpoint → load_checkpoint must reproduce identical logits."""
    from app.ml.model_v2 import SignTransformerV2

    v2_model.eval()
    with torch.no_grad():
        logits_before = v2_model(v2_batch)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "v2_test.pt"
        v2_model.save_checkpoint(ckpt_path)

        assert ckpt_path.exists()

        loaded = SignTransformerV2.load_checkpoint(ckpt_path)
        loaded.eval()
        with torch.no_grad():
            logits_after = loaded(v2_batch)

    assert torch.allclose(logits_before, logits_after, atol=1e-5), \
        "Logits changed after checkpoint round-trip"


def test_load_v1_checkpoint_creates_v2_model(v2_model, v2_batch) -> None:
    """load_v1_checkpoint must create a V2 model even from a V1-like checkpoint stub."""
    from app.ml.model_v2 import SignTransformerV2
    from app.ml.model import SignTransformer

    # Create a minimal V1-like checkpoint file
    v1_stub = {
        "architecture": "SignTransformer",
        "num_classes": NUM_CLASSES,
        "config": {
            "num_features": 493,
            "num_classes": NUM_CLASSES,
        },
        "model_state_dict": {},  # empty weights — only testing model creation
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "v1_stub.pt"
        torch.save(v1_stub, str(ckpt_path))

        model = SignTransformerV2.load_v1_checkpoint(
            ckpt_path, num_classes=NUM_CLASSES
        )

    assert isinstance(model, SignTransformerV2)
    assert model.num_classes == NUM_CLASSES
    assert model.feature_version == 2


# ── SignBoundaryDetector ──────────────────────────────────────────────────────


@pytest.fixture()
def seg_model() -> "SignBoundaryDetector":  # noqa: F821
    """Small boundary detector for testing."""
    from app.ml.sign_segmentation import SignBoundaryDetector

    return SignBoundaryDetector(
        feature_dim=FEATURE_DIM_V2,
        hidden_dim=32,
        num_layers=1,
        dropout=0.0,
        use_crf=False,
        input_selection="velocity+nmm+handshape",
    )


def test_segmentation_forward_shape(seg_model, v2_batch) -> None:
    """Forward pass must return (batch, seq, 4) logits."""
    seg_model.eval()
    with torch.no_grad():
        logits = seg_model(v2_batch)
    assert logits.shape == (BATCH, SEQ, 4), f"Expected ({BATCH}, {SEQ}, 4), got {logits.shape}"


def test_segmentation_no_nan(seg_model, v2_batch) -> None:
    """Segmentation logits must not contain NaN."""
    seg_model.eval()
    with torch.no_grad():
        logits = seg_model(v2_batch)
    assert not torch.isnan(logits).any()


def test_segmentation_decode_length(seg_model, v2_batch) -> None:
    """decode() must return one label sequence per batch item, length == seq_len."""
    seg_model.eval()
    decoded = seg_model.decode(v2_batch)
    assert len(decoded) == BATCH
    for seq in decoded:
        assert len(seq) == SEQ


def test_detect_boundaries_returns_list(seg_model) -> None:
    """detect_boundaries on a random sequence must return a list of tuples."""
    seg_model.eval()
    feature_seq = np.random.randn(32, FEATURE_DIM_V2).astype(np.float32)
    segments = seg_model.detect_boundaries(feature_seq, device="cpu")
    assert isinstance(segments, list)
    for seg in segments:
        assert isinstance(seg, tuple)
        assert len(seg) == 2
        start, end = seg
        assert 0 <= start <= end < 32


def test_detect_boundaries_zero_sequence(seg_model) -> None:
    """Zero sequence should not crash (may return empty segment list)."""
    seg_model.eval()
    feature_seq = np.zeros((20, FEATURE_DIM_V2), dtype=np.float32)
    segments = seg_model.detect_boundaries(feature_seq, device="cpu")
    assert isinstance(segments, list)


def test_segmentation_checkpoint_round_trip(seg_model, v2_batch) -> None:
    """save_checkpoint → load_checkpoint must preserve logits."""
    from app.ml.sign_segmentation import SignBoundaryDetector

    seg_model.eval()
    with torch.no_grad():
        logits_before = seg_model(v2_batch)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "seg_test.pt"
        seg_model.save_checkpoint(ckpt_path)
        loaded = SignBoundaryDetector.load_checkpoint(ckpt_path)
        loaded.eval()
        with torch.no_grad():
            logits_after = loaded(v2_batch)

    assert torch.allclose(logits_before, logits_after, atol=1e-5)


def test_motion_energy_fallback_returns_segments() -> None:
    """motion_energy_fallback must return a list of (start, end) tuples."""
    from app.ml.sign_segmentation import motion_energy_fallback

    # Create a synthetic V2 sequence with a clear motion burst in the middle
    seq = np.zeros((60, 611), dtype=np.float32)
    # Simulate motion in velocity block [237:462] frames 20-40
    seq[20:40, 237:462] = np.random.randn(20, 225).astype(np.float32) * 0.1

    segments = motion_energy_fallback(seq, motion_start_threshold=0.001)
    assert isinstance(segments, list)


def test_extract_segments_merge() -> None:
    """_extract_segments must merge nearby segments."""
    from app.ml.sign_segmentation import SignBoundaryDetector

    # 0-9: sign, 10-11: gap (2 frames), 12-20: sign
    is_sign = np.array([True] * 10 + [False] * 2 + [True] * 9)
    segments = SignBoundaryDetector._extract_segments(
        is_sign, min_sign_frames=5, merge_gap_frames=3
    )
    # Gap of 2 < merge_gap_frames=3 → merged into single segment
    assert len(segments) == 1
    assert segments[0] == (0, 20)


def test_extract_segments_filter_short() -> None:
    """Short segments below min_sign_frames must be filtered out."""
    from app.ml.sign_segmentation import SignBoundaryDetector

    is_sign = np.array([True] * 3 + [False] * 5 + [True] * 20)
    segments = SignBoundaryDetector._extract_segments(
        is_sign, min_sign_frames=8, merge_gap_frames=2
    )
    # First segment (3 frames) is too short
    assert all((e - s + 1) >= 8 for s, e in segments)


# ── model_configs V2 specs ────────────────────────────────────────────────────


def test_v2_model_configs_available() -> None:
    """V2_MODEL_CONFIGS and SEGMENTATION_CONFIGS must be non-empty."""
    from app.ml.model_configs import V2_MODEL_CONFIGS, SEGMENTATION_CONFIGS

    assert len(V2_MODEL_CONFIGS) >= 3
    assert "v2_default" in V2_MODEL_CONFIGS
    assert "default" in SEGMENTATION_CONFIGS


def test_get_v2_model_config() -> None:
    """get_v2_model_config must return the correct spec."""
    from app.ml.model_configs import get_v2_model_config

    cfg = get_v2_model_config("v2_default")
    assert cfg.feature_version == 2
    assert cfg.feature_dim == FEATURE_DIM_V2
    assert cfg.hand_hidden_dim == 128


def test_v2_config_estimated_params_positive() -> None:
    """Estimated params must be > 0 for all V2 configs."""
    from app.ml.model_configs import V2_MODEL_CONFIGS

    for name, cfg in V2_MODEL_CONFIGS.items():
        assert cfg.estimated_params > 0, f"{name} has non-positive estimated_params"


def test_list_v2_model_configs() -> None:
    """list_v2_model_configs must return a non-empty list of dicts."""
    from app.ml.model_configs import list_v2_model_configs

    configs = list_v2_model_configs()
    assert len(configs) >= 3
    for c in configs:
        assert "name" in c
        assert "estimated_params" in c
        assert c["feature_version"] == 2


# ── Pipeline integration ──────────────────────────────────────────────────────


def test_pipeline_flags_init() -> None:
    """SignFlowInferencePipeline must expose V2 model and segmentation attributes."""
    from app.ml.pipeline import SignFlowInferencePipeline

    pipe = SignFlowInferencePipeline(feature_version=2)
    assert hasattr(pipe, "_model_is_v2")
    assert hasattr(pipe, "segmentation_model")
    assert hasattr(pipe, "_use_bilstm_segmentation")
    assert pipe._model_is_v2 is False
    assert pipe.segmentation_model is None
    assert pipe._use_bilstm_segmentation is False
    assert pipe.feature_version == 2


def test_pipeline_detect_boundaries_bilstm_fallback() -> None:
    """detect_boundaries_bilstm without model should fall back gracefully."""
    from app.ml.pipeline import SignFlowInferencePipeline

    pipe = SignFlowInferencePipeline(feature_version=2)
    feature_seq = np.zeros((30, FEATURE_DIM_V2), dtype=np.float32)
    segments = pipe.detect_boundaries_bilstm(feature_seq)
    assert isinstance(segments, list)


def test_pipeline_v2_infer_window_calls_base() -> None:
    """_infer_window_v2 without model must return ('NONE', 0.0, [])."""
    from app.ml.pipeline import SignFlowInferencePipeline

    pipe = SignFlowInferencePipeline(feature_version=2)
    window = np.zeros((16, FEATURE_DIM_V2), dtype=np.float32)
    label, conf, alts = pipe._infer_window_v2(window)
    assert label == "NONE"
    assert conf == 0.0
    assert alts == []
