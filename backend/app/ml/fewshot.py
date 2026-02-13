"""Few-shot fine-tuning helpers for incremental SignFlow training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import structlog
import torch
from torch import nn

from app.ml.model import SignTransformer

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class FewShotPreparation:
    """Prepared model metadata for few-shot fine-tuning."""

    model: SignTransformer
    loaded_from_checkpoint: bool
    frozen_layer_count: int


def expand_classifier_head(model: SignTransformer, target_num_classes: int) -> bool:
    """
    Expand model classifier head while preserving existing class weights.

    Args:
        model: Existing transformer model.
        target_num_classes: Desired number of output classes.

    Returns:
        True if classifier was expanded, False if unchanged.
    """
    if target_num_classes <= model.num_classes:
        return False

    old_classifier = model.classifier
    new_classifier = nn.Linear(old_classifier.in_features, target_num_classes)

    with torch.no_grad():
        old_out = old_classifier.out_features
        new_classifier.weight[:old_out].copy_(old_classifier.weight)
        if old_classifier.bias is not None and new_classifier.bias is not None:
            new_classifier.bias[:old_out].copy_(old_classifier.bias)

    model.classifier = new_classifier
    model.num_classes = target_num_classes
    return True


def freeze_transformer_encoder_layers(model: SignTransformer, freeze_until_layer: int = 3) -> int:
    """
    Freeze first encoder layers for few-shot adaptation.

    Args:
        model: Transformer model.
        freeze_until_layer: Freeze layers [0, freeze_until_layer).

    Returns:
        Number of frozen encoder layers.
    """
    frozen = 0
    for index, layer in enumerate(model.encoder.layers):
        should_freeze = index < freeze_until_layer
        for parameter in layer.parameters():
            parameter.requires_grad = not should_freeze
        if should_freeze:
            frozen += 1
    return frozen


def prepare_few_shot_model(
    *,
    checkpoint_path: str | Path | None,
    num_features: int,
    num_classes: int,
    d_model: int = 256,
    device: str = "cpu",
    freeze_until_layer: int = 3,
) -> FewShotPreparation:
    """
    Build model for few-shot training from active checkpoint or fresh initialization.

    - Loads active model weights if checkpoint exists.
    - Expands classifier head to include new classes.
    - Freezes early transformer layers for stable adaptation.
    """
    loaded_from_checkpoint = False

    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
        model = SignTransformer(
            num_features=checkpoint.get("num_features", num_features),
            num_classes=checkpoint.get("num_classes", num_classes),
            d_model=checkpoint.get("d_model", d_model),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        loaded_from_checkpoint = True
    else:
        model = SignTransformer(
            num_features=num_features,
            num_classes=num_classes,
            d_model=d_model,
        )

    # Enable gradients for training (checkpoint may come from inference mode).
    for parameter in model.parameters():
        parameter.requires_grad = True

    target_num_classes = max(num_classes, model.num_classes)
    expanded = expand_classifier_head(model, target_num_classes=target_num_classes)
    frozen_count = freeze_transformer_encoder_layers(model, freeze_until_layer=freeze_until_layer)
    model.to(torch.device(device))

    logger.info(
        "few_shot_model_prepared",
        loaded_from_checkpoint=loaded_from_checkpoint,
        expanded_classifier=expanded,
        num_classes=model.num_classes,
        frozen_layer_count=frozen_count,
    )

    return FewShotPreparation(
        model=model,
        loaded_from_checkpoint=loaded_from_checkpoint,
        frozen_layer_count=frozen_count,
    )
