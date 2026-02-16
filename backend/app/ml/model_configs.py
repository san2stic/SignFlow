"""Predefined model configurations for different capacities and use cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.ml.feature_engineering import ENRICHED_FEATURE_DIM


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


# Registry of all available configurations
MODEL_CONFIGS = {
    "baseline": MODEL_CONFIG_BASELINE,
    "lightweight": MODEL_CONFIG_LIGHTWEIGHT,
    "medium": MODEL_CONFIG_MEDIUM,
    "large": MODEL_CONFIG_LARGE,
    "xlarge": MODEL_CONFIG_XLARGE,
}


def get_model_config(name: str) -> ModelConfig:
    """
    Get a predefined model configuration by name.

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
