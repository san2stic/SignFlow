"""Tests du pipeline V2 — features 611 dims, SignTransformerV2, segmentation, grammaire.

Covers:
  - ENRICHED_FEATURE_DIM_V2 == 611 et ENRICHED_FEATURE_DIM == 493
  - normalize_landmarks_v2() et extract_features_v2() produisent des vecteurs 611 dims
  - normalize_landmarks (V1) produit des vecteurs 493 dims (rétrocompat)
  - SignTransformerV2 avec un batch (2, 32, 611) — shape et absence de NaN
  - SignBoundaryDetector.detect_boundaries() avec séquence synthétique
  - Pipeline end-to-end avec mock landmarks (feature_version=2)
  - enable_grammar_translation() s'active sans erreur
  - enable_conversation_context() s'active sans erreur
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# ── Fixtures ─────────────────────────────────────────────────────────────────

BATCH = 2
SEQ = 32
FEATURE_DIM_V1 = 493
FEATURE_DIM_V2 = 611
NUM_CLASSES = 5

# Landmarks synthétiques (21 points xyz)
_HAND_21 = [[float(i * 0.01), float(i * 0.01), 0.0] for i in range(21)]
# Pose synthétique (33 points xyz)
_POSE_33 = [[float(i * 0.01), float(i * 0.01), 0.0] for i in range(33)]
# Face synthétique (468 points xyz)
_FACE_468 = [[float(i * 0.001), float(i * 0.001), 0.0] for i in range(468)]

# Payload landmarks complet compatible extract_features_v2
MOCK_LANDMARKS_PAYLOAD = {
    "hands": {"left": _HAND_21, "right": _HAND_21},
    "pose": _POSE_33,
    "face": _FACE_468,
}


# ---------------------------------------------------------------------------
# Tests dimensions constantes
# ---------------------------------------------------------------------------


def test_enriched_feature_dim_v2_equals_611() -> None:
    """ENRICHED_FEATURE_DIM_V2 doit valoir exactement 611."""
    from app.ml.features import ENRICHED_FEATURE_DIM_V2
    assert ENRICHED_FEATURE_DIM_V2 == 611


def test_enriched_feature_dim_v1_equals_493() -> None:
    """ENRICHED_FEATURE_DIM (V1) doit valoir exactement 493 pour la rétrocompatibilité."""
    from app.ml.feature_engineering import ENRICHED_FEATURE_DIM
    assert ENRICHED_FEATURE_DIM == 493


def test_v2_feature_dim_greater_than_v1() -> None:
    """V2 doit avoir plus de features que V1 (611 > 493)."""
    from app.ml.features import ENRICHED_FEATURE_DIM_V2
    from app.ml.feature_engineering import ENRICHED_FEATURE_DIM
    assert ENRICHED_FEATURE_DIM_V2 > ENRICHED_FEATURE_DIM


# ---------------------------------------------------------------------------
# Tests normalize_landmarks_v2 — vecteur 611 dims
# ---------------------------------------------------------------------------


def test_normalize_landmarks_v2_output_dim() -> None:
    """normalize_landmarks_v2() doit produire un vecteur de 611 dims."""
    from app.ml.features import normalize_landmarks_v2, FrameLandmarks
    frame = FrameLandmarks(
        left_hand=_HAND_21,
        right_hand=_HAND_21,
        pose=_POSE_33,
        face=_FACE_468,
    )
    feat = normalize_landmarks_v2(frame)
    assert feat.shape == (611,), f"Expected (611,), got {feat.shape}"


def test_normalize_landmarks_v2_no_nan() -> None:
    """normalize_landmarks_v2() ne doit pas produire de NaN."""
    from app.ml.features import normalize_landmarks_v2, FrameLandmarks
    frame = FrameLandmarks(
        left_hand=_HAND_21,
        right_hand=_HAND_21,
        pose=_POSE_33,
        face=_FACE_468,
    )
    feat = normalize_landmarks_v2(frame)
    assert not np.isnan(feat).any(), "NaN détecté dans les features V2"
    assert not np.isinf(feat).any(), "Inf détecté dans les features V2"


def test_normalize_landmarks_v2_zero_landmarks_no_crash() -> None:
    """normalize_landmarks_v2() avec landmarks zeros ne doit pas crasher."""
    from app.ml.features import normalize_landmarks_v2, FrameLandmarks
    frame = FrameLandmarks(
        left_hand=[[0.0, 0.0, 0.0]] * 21,
        right_hand=[[0.0, 0.0, 0.0]] * 21,
        pose=[[0.0, 0.0, 0.0]] * 33,
        face=None,
    )
    feat = normalize_landmarks_v2(frame)
    assert feat.shape == (611,)
    assert not np.isnan(feat).any()


def test_normalize_landmarks_v2_dtype_float32() -> None:
    """Le vecteur V2 doit être float32."""
    from app.ml.features import normalize_landmarks_v2, FrameLandmarks
    frame = FrameLandmarks(
        left_hand=_HAND_21,
        right_hand=_HAND_21,
        pose=_POSE_33,
        face=_FACE_468,
    )
    feat = normalize_landmarks_v2(frame)
    assert feat.dtype == np.float32


# ---------------------------------------------------------------------------
# Tests extract_features_v2 — interface dict → 611 dims
# ---------------------------------------------------------------------------


def test_extract_features_v2_output_dim() -> None:
    """extract_features_v2() avec payload complet doit produire 611 dims."""
    from app.ml.features import extract_features_v2
    feat = extract_features_v2(MOCK_LANDMARKS_PAYLOAD)
    assert feat.shape == (611,), f"Expected (611,), got {feat.shape}"


def test_extract_features_v2_empty_payload() -> None:
    """extract_features_v2() avec payload vide ne doit pas crasher."""
    from app.ml.features import extract_features_v2
    feat = extract_features_v2({})
    assert feat.shape == (611,)
    assert not np.isnan(feat).any()


def test_extract_features_v2_consistent_with_normalize_v2() -> None:
    """extract_features_v2 et normalize_landmarks_v2 doivent produire le même résultat."""
    from app.ml.features import extract_features_v2, normalize_landmarks_v2, FrameLandmarks
    feat_dict = extract_features_v2(MOCK_LANDMARKS_PAYLOAD)
    frame = FrameLandmarks(
        left_hand=_HAND_21,
        right_hand=_HAND_21,
        pose=_POSE_33,
        face=_FACE_468,
    )
    feat_direct = normalize_landmarks_v2(frame)
    np.testing.assert_allclose(feat_dict, feat_direct, rtol=1e-5)


# ---------------------------------------------------------------------------
# Tests SignTransformerV2 — batch (2, 32, 611)
# ---------------------------------------------------------------------------


@pytest.fixture()
def v2_model():
    """Petit SignTransformerV2 pour les tests."""
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
        use_cosine_head=False,
        max_seq_len=SEQ + 4,
        use_gcn_hand=False,
    )
    return SignTransformerV2.from_config(cfg)


def test_v2_model_forward_shape(v2_model) -> None:
    """SignTransformerV2.forward() doit retourner (batch, num_classes)."""
    torch.manual_seed(0)
    x = torch.randn(BATCH, SEQ, FEATURE_DIM_V2)
    v2_model.eval()
    with torch.no_grad():
        out = v2_model(x)
    assert out.shape == (BATCH, NUM_CLASSES)


def test_v2_model_forward_no_nan(v2_model) -> None:
    """Le forward pass ne doit pas produire de NaN."""
    torch.manual_seed(0)
    x = torch.randn(BATCH, SEQ, FEATURE_DIM_V2)
    v2_model.eval()
    with torch.no_grad():
        out = v2_model(x)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_v2_model_feature_version_marker(v2_model) -> None:
    """SignTransformerV2 doit exposer feature_version == 2."""
    assert v2_model.feature_version == 2
    assert v2_model.FEATURE_VERSION == 2


# ---------------------------------------------------------------------------
# Tests SignBoundaryDetector
# ---------------------------------------------------------------------------


@pytest.fixture()
def seg_model():
    """Petit SignBoundaryDetector pour les tests."""
    from app.ml.sign_segmentation import SignBoundaryDetector
    return SignBoundaryDetector(
        feature_dim=FEATURE_DIM_V2,
        hidden_dim=32,
        num_layers=1,
        dropout=0.0,
        use_crf=False,
        input_selection="velocity+nmm+handshape",
    )


def test_boundary_detector_detect_boundaries_synthetic(seg_model) -> None:
    """detect_boundaries() sur une séquence synthétique doit retourner des tuples (start, end)."""
    seg_model.eval()
    np.random.seed(42)
    # Séquence avec un signe simulé (mouvement dans frames 10-25)
    seq = np.zeros((SEQ, FEATURE_DIM_V2), dtype=np.float32)
    seq[10:25, 237:462] = np.random.randn(15, 225).astype(np.float32) * 0.5
    segments = seg_model.detect_boundaries(seq, device="cpu")
    assert isinstance(segments, list)
    for seg in segments:
        assert isinstance(seg, tuple)
        assert len(seg) == 2
        start, end = seg
        assert 0 <= start <= end < SEQ


def test_boundary_detector_zero_sequence_no_crash(seg_model) -> None:
    """Une séquence nulle ne doit pas faire crasher detect_boundaries."""
    seg_model.eval()
    seq = np.zeros((SEQ, FEATURE_DIM_V2), dtype=np.float32)
    segments = seg_model.detect_boundaries(seq, device="cpu")
    assert isinstance(segments, list)


# ---------------------------------------------------------------------------
# Tests pipeline end-to-end avec mock landmarks
# ---------------------------------------------------------------------------


def test_pipeline_v2_init_feature_version() -> None:
    """SignFlowInferencePipeline(feature_version=2) doit s'initialiser sans erreur."""
    from app.ml.pipeline import SignFlowInferencePipeline
    pipe = SignFlowInferencePipeline(feature_version=2)
    assert pipe.feature_version == 2


def test_pipeline_v2_extract_features_v2_returns_correct_dims() -> None:
    """extract_features_v2() doit retourner 611 dims depuis un payload de landmarks."""
    from app.ml.features import extract_features_v2
    feats = extract_features_v2(MOCK_LANDMARKS_PAYLOAD)
    assert feats.shape == (611,)
    assert not np.isnan(feats).any()


def test_pipeline_v2_infer_window_without_model_returns_none() -> None:
    """_infer_window_v2 sans modèle chargé doit retourner ('NONE', 0.0, [])."""
    from app.ml.pipeline import SignFlowInferencePipeline
    pipe = SignFlowInferencePipeline(feature_version=2)
    window = np.zeros((16, FEATURE_DIM_V2), dtype=np.float32)
    label, conf, alts = pipe._infer_window_v2(window)
    assert label == "NONE"
    assert conf == 0.0
    assert alts == []


# ---------------------------------------------------------------------------
# Tests enable_grammar_translation
# ---------------------------------------------------------------------------


def test_enable_grammar_translation_no_error() -> None:
    """enable_grammar_translation() doit s'activer sans lever d'exception."""
    from app.ml.pipeline import SignFlowInferencePipeline
    pipe = SignFlowInferencePipeline(feature_version=2)
    # Ne doit pas lever d'exception
    pipe.enable_grammar_translation()
    assert hasattr(pipe, "grammar_translator")


def test_grammar_translation_produces_output() -> None:
    """Le traducteur grammatical doit produire une chaîne non vide sur des signes valides."""
    from app.ml.grammar.lsfb_translator import LSFBToFrenchTranslator
    translator = LSFBToFrenchTranslator()
    # translate_buffer accepte le format natif du pipeline d'inférence
    buffer = [
        {"label": "JE", "confidence": 0.9},
        {"label": "ALLER", "confidence": 0.8},
        {"label": "MAISON", "confidence": 0.85},
    ]
    result = translator.translate_buffer(buffer)
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# Tests enable_conversation_context
# ---------------------------------------------------------------------------


def test_enable_conversation_context_no_error() -> None:
    """enable_conversation_context() doit s'activer sans lever d'exception."""
    from app.ml.pipeline import SignFlowInferencePipeline
    pipe = SignFlowInferencePipeline(feature_version=2)
    pipe.enable_conversation_context()
    assert hasattr(pipe, "conversation_context")


def test_conversation_context_accessible_after_enable() -> None:
    """Le ConversationContext doit être accessible après enable_conversation_context()."""
    from app.ml.pipeline import SignFlowInferencePipeline
    from app.ml.conversation_context import ConversationContext
    pipe = SignFlowInferencePipeline(feature_version=2)
    pipe.enable_conversation_context()
    assert isinstance(pipe.conversation_context, ConversationContext)
