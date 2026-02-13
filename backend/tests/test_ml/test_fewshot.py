"""Few-shot model preparation tests."""

from __future__ import annotations

from pathlib import Path

import torch

from app.ml.fewshot import (
    expand_classifier_head,
    freeze_transformer_encoder_layers,
    prepare_few_shot_model,
)
from app.ml.model import SignTransformer


def test_expand_classifier_head_preserves_existing_weights() -> None:
    """Expanding the head should keep old class weights untouched."""
    model = SignTransformer(num_features=225, num_classes=2, d_model=32)

    with torch.no_grad():
        model.classifier.weight.fill_(0.5)
        model.classifier.bias.fill_(0.1)
        original_weight = model.classifier.weight.detach().clone()
        original_bias = model.classifier.bias.detach().clone()

    expanded = expand_classifier_head(model, target_num_classes=4)

    assert expanded is True
    assert model.classifier.out_features == 4
    assert torch.equal(model.classifier.weight[:2], original_weight)
    assert torch.equal(model.classifier.bias[:2], original_bias)


def test_freeze_transformer_encoder_layers_freezes_first_three() -> None:
    """Few-shot freezing should affect encoder layers 0-2 and keep layer 3 trainable."""
    model = SignTransformer(num_features=225, num_classes=3)
    frozen_count = freeze_transformer_encoder_layers(model, freeze_until_layer=3)

    assert frozen_count == 3
    assert all(not parameter.requires_grad for parameter in model.encoder.layers[0].parameters())
    assert all(not parameter.requires_grad for parameter in model.encoder.layers[1].parameters())
    assert all(not parameter.requires_grad for parameter in model.encoder.layers[2].parameters())
    assert all(parameter.requires_grad for parameter in model.encoder.layers[3].parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())


def test_prepare_few_shot_model_loads_checkpoint_and_expands_head(tmp_path: Path) -> None:
    """Preparation should load existing checkpoint and append class output when needed."""
    base = SignTransformer(num_features=225, num_classes=2, d_model=32)
    checkpoint_path = tmp_path / "model_v1.pt"
    torch.save(
        {
            "model_state_dict": base.state_dict(),
            "num_classes": 2,
            "num_features": 225,
            "d_model": 32,
        },
        checkpoint_path,
    )

    prepared = prepare_few_shot_model(
        checkpoint_path=checkpoint_path,
        num_features=225,
        num_classes=3,
        d_model=256,
        device="cpu",
        freeze_until_layer=3,
    )

    assert prepared.loaded_from_checkpoint is True
    assert prepared.frozen_layer_count == 3
    assert prepared.model.num_classes == 3
    assert prepared.model.classifier.out_features == 3
