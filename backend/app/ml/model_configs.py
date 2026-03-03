"""Predefined model configurations for different capacities and use cases.

V1 configs use SignTransformer with ENRICHED_FEATURE_DIM (493).
V2 configs use SignTransformerV2 with ENRICHED_FEATURE_DIM_V2 (611).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.ml.feature_engineering import ENRICHED_FEATURE_DIM
from app.ml.features import ENRICHED_FEATURE_DIM_V2


@dataclass
class ModelConfig:
    """Configuration for a SignTransformer model architecture."""

    name: str
    description: str
    num_features: int
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float = 0.2
    feature_dropout: float = 0.15
    pooling_dropout: float = 0.2
    use_cls_token: bool = True
    token_dropout: float = 0.05
    temporal_smoothing: float = 0.1
    use_multiscale_stem: bool = True
    use_cosine_head: bool = True
    relative_bias_max_distance: int = 64
    cosine_head_weight: float = 0.35

    @property
    def estimated_params(self) -> int:
        """
        Estimate the number of trainable parameters.

        This is approximate and doesn't account for all layers.
        """
        # Embedding: num_features * d_model
        embedding_params = self.num_features * self.d_model

        # Transformer layers (approximate)
        # Each layer has:
        # - Multi-head attention: 4 * d_model^2 (Q, K, V, O projections)
        # - Feed-forward: 2 * d_model * dim_feedforward
        # - Layer norms: 4 * d_model
        attn_params_per_layer = 4 * (self.d_model ** 2)
        ff_params_per_layer = 2 * self.d_model * self.dim_feedforward
        norm_params_per_layer = 4 * self.d_model

        layer_params = (attn_params_per_layer + ff_params_per_layer + norm_params_per_layer)
        transformer_params = self.num_layers * layer_params

        # Classification head: d_model * num_classes (estimated)
        # Note: actual num_classes unknown at config time
        head_params = self.d_model * 100  # Assume ~100 classes

        total = embedding_params + transformer_params + head_params
        return int(total)

    def to_model_kwargs(self, num_classes: int) -> dict[str, Any]:
        """
        Convert config to model initialization kwargs.

        Args:
            num_classes: Number of output classes

        Returns:
            Dictionary of kwargs for SignTransformer.__init__
        """
        return {
            "num_features": self.num_features,
            "num_classes": num_classes,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "feature_dropout": self.feature_dropout,
            "pooling_dropout": self.pooling_dropout,
            "use_cls_token": self.use_cls_token,
            "token_dropout": self.token_dropout,
            "temporal_smoothing": self.temporal_smoothing,
            "use_multiscale_stem": self.use_multiscale_stem,
            "use_cosine_head": self.use_cosine_head,
            "relative_bias_max_distance": self.relative_bias_max_distance,
            "cosine_head_weight": self.cosine_head_weight,
        }


# =======================
# Predefined Configurations
# =======================

# BASELINE: Original small model (~150k params)
# Good for: Fast iteration, CPU training, low-resource environments
MODEL_CONFIG_BASELINE = ModelConfig(
    name="baseline",
    description="Original small model (~150k params) - Fast training on CPU/MPS",
    num_features=ENRICHED_FEATURE_DIM,
    d_model=192,
    nhead=6,
    num_layers=4,
    dim_feedforward=768,
    dropout=0.2,
    feature_dropout=0.15,
    pooling_dropout=0.2,
)

# MEDIUM: Moderate capacity (~600k params)
# Good for: Better accuracy with reasonable training time
MODEL_CONFIG_MEDIUM = ModelConfig(
    name="medium",
    description="Medium model (~600k params) - Balanced accuracy/speed",
    num_features=ENRICHED_FEATURE_DIM,
    d_model=256,
    nhead=8,
    num_layers=6,
    dim_feedforward=1024,
    dropout=0.25,
    feature_dropout=0.15,
    pooling_dropout=0.2,
)

# LARGE: High capacity (~1.5M params) - Phase 1.2 target
# Good for: Maximum accuracy, GPU training recommended
MODEL_CONFIG_LARGE = ModelConfig(
    name="large",
    description="Large model (~1.5M params) - High accuracy, GPU recommended",
    num_features=ENRICHED_FEATURE_DIM,
    d_model=384,
    nhead=8,
    num_layers=6,
    dim_feedforward=1536,
    dropout=0.3,
    feature_dropout=0.15,
    pooling_dropout=0.25,
)

# XLARGE: Extra large capacity (~3.5M params)
# Good for: State-of-the-art accuracy, requires GPU
MODEL_CONFIG_XLARGE = ModelConfig(
    name="xlarge",
    description="Extra large model (~3.5M params) - Maximum accuracy, requires GPU",
    num_features=ENRICHED_FEATURE_DIM,
    d_model=512,
    nhead=8,
    num_layers=8,
    dim_feedforward=2048,
    dropout=0.3,
    feature_dropout=0.15,
    pooling_dropout=0.25,
)

# LIGHTWEIGHT: Minimal model for edge deployment (~50k params)
# Good for: Edge devices, mobile, real-time constraints
MODEL_CONFIG_LIGHTWEIGHT = ModelConfig(
    name="lightweight",
    description="Lightweight model (~50k params) - Edge deployment, mobile",
    num_features=ENRICHED_FEATURE_DIM,
    d_model=128,
    nhead=4,
    num_layers=2,
    dim_feedforward=512,
    dropout=0.15,
    feature_dropout=0.1,
    pooling_dropout=0.15,
)


# Registry of all available V1 configurations
MODEL_CONFIGS = {
    "baseline": MODEL_CONFIG_BASELINE,
    "lightweight": MODEL_CONFIG_LIGHTWEIGHT,
    "medium": MODEL_CONFIG_MEDIUM,
    "large": MODEL_CONFIG_LARGE,
    "xlarge": MODEL_CONFIG_XLARGE,
}


# =======================
# V2 Configurations (SignTransformerV2 multi-stream)
# =======================


@dataclass
class SignTransformerV2ConfigSpec:
    """Lightweight config spec for V2 model variants (mirrors SignTransformerV2Config)."""

    name: str
    description: str
    feature_dim: int = ENRICHED_FEATURE_DIM_V2  # 611
    hand_hidden_dim: int = 128
    pose_hidden_dim: int = 64
    nmm_hidden_dim: int = 64
    fusion_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    num_fusion_layers: int = 2
    dropout: float = 0.1
    use_cosine_head: bool = True
    max_seq_len: int = 64
    use_gcn_hand: bool = True
    num_gcn_layers: int = 2
    cosine_head_weight: float = 0.35
    feature_version: int = 2

    @property
    def estimated_params(self) -> int:
        """Rough parameter estimate (R&D guidance only)."""
        # Hand stream: 2*(21*3*hidden + hidden^2*num_layers*4 + hidden*dim_feedforward*2)
        hand = 2 * (63 * self.hand_hidden_dim + self.num_layers * (
            4 * self.hand_hidden_dim ** 2 + 2 * self.hand_hidden_dim * self.hand_hidden_dim * 4
        ))
        pose = 126 * self.pose_hidden_dim + self.num_layers * (
            4 * self.pose_hidden_dim ** 2 + 2 * self.pose_hidden_dim * self.pose_hidden_dim * 4
        )
        nmm = 50 * self.nmm_hidden_dim + self.num_layers * (
            4 * self.nmm_hidden_dim ** 2 + 2 * self.nmm_hidden_dim * self.nmm_hidden_dim * 4
        )
        concat_dim = self.hand_hidden_dim + self.pose_hidden_dim + self.nmm_hidden_dim
        fusion = concat_dim * self.fusion_dim
        head = self.fusion_dim * 100  # assume 100 classes
        return int(hand + pose + nmm + fusion + head)

    def to_v2_config_kwargs(self, num_classes: int = 100) -> dict[str, Any]:
        """Return kwargs for SignTransformerV2Config or SignTransformerV2.__init__."""
        return {
            "feature_dim": self.feature_dim,
            "hand_hidden_dim": self.hand_hidden_dim,
            "pose_hidden_dim": self.pose_hidden_dim,
            "nmm_hidden_dim": self.nmm_hidden_dim,
            "fusion_dim": self.fusion_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "num_fusion_layers": self.num_fusion_layers,
            "dropout": self.dropout,
            "num_classes": num_classes,
            "use_cosine_head": self.use_cosine_head,
            "max_seq_len": self.max_seq_len,
            "use_gcn_hand": self.use_gcn_hand,
            "num_gcn_layers": self.num_gcn_layers,
            "cosine_head_weight": self.cosine_head_weight,
        }


# V2 LIGHTWEIGHT: ~2M params — Edge/mobile deployment
SIGN_TRANSFORMER_V2_CONFIG_LIGHTWEIGHT = SignTransformerV2ConfigSpec(
    name="v2_lightweight",
    description="V2 lightweight multi-stream (~2M params) — Edge/mobile",
    hand_hidden_dim=64,
    pose_hidden_dim=32,
    nmm_hidden_dim=32,
    fusion_dim=128,
    num_heads=4,
    num_layers=2,
    num_fusion_layers=1,
    dropout=0.1,
)

# V2 DEFAULT: ~5M params — Standard usage
SIGN_TRANSFORMER_V2_CONFIG = SignTransformerV2ConfigSpec(
    name="v2_default",
    description="V2 default multi-stream (~5M params) — Balanced accuracy/speed",
    hand_hidden_dim=128,
    pose_hidden_dim=64,
    nmm_hidden_dim=64,
    fusion_dim=256,
    num_heads=8,
    num_layers=4,
    num_fusion_layers=2,
    dropout=0.1,
)

# V2 LARGE: ~12M params — Higher accuracy, GPU recommended
SIGN_TRANSFORMER_V2_CONFIG_LARGE = SignTransformerV2ConfigSpec(
    name="v2_large",
    description="V2 large multi-stream (~12M params) — High accuracy, GPU recommended",
    hand_hidden_dim=256,
    pose_hidden_dim=128,
    nmm_hidden_dim=128,
    fusion_dim=512,
    num_heads=8,
    num_layers=6,
    num_fusion_layers=3,
    dropout=0.1,
)

# =======================
# Segmentation configs
# =======================


@dataclass
class SegmentationConfigSpec:
    """Config spec for SignBoundaryDetector variants."""

    name: str
    description: str
    feature_dim: int = ENRICHED_FEATURE_DIM_V2  # 611
    input_selection: str = "velocity+nmm+handshape"
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    use_crf: bool = False
    min_sign_frames: int = 8
    merge_gap_frames: int = 3
    boundary_threshold: float = 0.5

    @property
    def estimated_params(self) -> int:
        """Rough BiLSTM parameter estimate."""
        # BiLSTM: 4 * (input+hidden)*hidden*2 (bidirectional)
        from app.ml.sign_segmentation import _get_input_dim
        input_dim = _get_input_dim(self.input_selection, self.feature_dim)
        proj = input_dim * self.hidden_dim
        lstm_in = self.hidden_dim
        lstm = 4 * (lstm_in + self.hidden_dim) * self.hidden_dim * 2  # bidirectional
        lstm *= self.num_layers
        out = 2 * self.hidden_dim * 4   # output projection → 4 labels
        return int(proj + lstm + out)

    def to_segmentation_kwargs(self) -> dict[str, Any]:
        """Return kwargs for SignBoundaryDetector.__init__."""
        return {
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "use_crf": self.use_crf,
            "input_selection": self.input_selection,
        }


# Standard segmentation config (no CRF, compact input)
SEGMENTATION_CONFIG = SegmentationConfigSpec(
    name="segmentation_default",
    description="SignBoundaryDetector default (~800K params) — BiLSTM BIEO tagger",
    hidden_dim=128,
    num_layers=2,
    use_crf=False,
)

# CRF variant for constrained decoding
SEGMENTATION_CONFIG_CRF = SegmentationConfigSpec(
    name="segmentation_crf",
    description="SignBoundaryDetector with CRF head (~800K params + CRF)",
    hidden_dim=128,
    num_layers=2,
    use_crf=True,
)


# Extended registries
V2_MODEL_CONFIGS: dict[str, SignTransformerV2ConfigSpec] = {
    "v2_lightweight": SIGN_TRANSFORMER_V2_CONFIG_LIGHTWEIGHT,
    "v2_default": SIGN_TRANSFORMER_V2_CONFIG,
    "v2_large": SIGN_TRANSFORMER_V2_CONFIG_LARGE,
}

SEGMENTATION_CONFIGS: dict[str, SegmentationConfigSpec] = {
    "default": SEGMENTATION_CONFIG,
    "crf": SEGMENTATION_CONFIG_CRF,
}


def get_model_config(name: str) -> ModelConfig:
    """
    Get a predefined V1 model configuration by name.

    Args:
        name: Configuration name (baseline, lightweight, medium, large, xlarge)

    Returns:
        ModelConfig instance

    Raises:
        ValueError: If configuration name is not found
    """
    if name not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model config '{name}'. Available: {available}")

    return MODEL_CONFIGS[name]


def get_v2_model_config(name: str = "v2_default") -> SignTransformerV2ConfigSpec:
    """Get a predefined V2 model configuration by name.

    Args:
        name: Configuration name (v2_lightweight, v2_default, v2_large)

    Returns:
        SignTransformerV2ConfigSpec instance
    """
    if name not in V2_MODEL_CONFIGS:
        available = ", ".join(V2_MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown V2 config '{name}'. Available: {available}")
    return V2_MODEL_CONFIGS[name]


def get_segmentation_config(name: str = "default") -> SegmentationConfigSpec:
    """Get a predefined segmentation configuration by name."""
    if name not in SEGMENTATION_CONFIGS:
        available = ", ".join(SEGMENTATION_CONFIGS.keys())
        raise ValueError(f"Unknown segmentation config '{name}'. Available: {available}")
    return SEGMENTATION_CONFIGS[name]


def list_model_configs() -> list[dict[str, Any]]:
    """
    List all available model configurations with their specs.

    Returns:
        List of config dictionaries with name, description, and estimated params
    """
    return [
        {
            "name": config.name,
            "description": config.description,
            "d_model": config.d_model,
            "num_layers": config.num_layers,
            "nhead": config.nhead,
            "dim_feedforward": config.dim_feedforward,
            "estimated_params": config.estimated_params,
        }
        for config in MODEL_CONFIGS.values()
    ]


def list_v2_model_configs() -> list[dict[str, Any]]:
    """List all V2 model configurations."""
    return [
        {
            "name": c.name,
            "description": c.description,
            "hand_hidden_dim": c.hand_hidden_dim,
            "pose_hidden_dim": c.pose_hidden_dim,
            "nmm_hidden_dim": c.nmm_hidden_dim,
            "fusion_dim": c.fusion_dim,
            "num_layers": c.num_layers,
            "estimated_params": c.estimated_params,
            "feature_version": c.feature_version,
        }
        for c in V2_MODEL_CONFIGS.values()
    ]
