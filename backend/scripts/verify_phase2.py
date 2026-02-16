#!/usr/bin/env python3
"""Verify Phase 2 implementation: Spatial-Temporal Architecture.

Tests:
1. SpatialLandmarkEncoder with GCN
2. TemporalConvPooling for variable-length sequences
3. SpatialTemporalTransformer end-to-end
4. MaskedLandmarkModel for pretraining

Usage:
    cd backend
    python scripts/verify_phase2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import structlog
import torch

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml.spatial_encoder import (
    SpatialLandmarkEncoder,
    SpatialTemporalTransformer,
    build_combined_graph,
    build_hand_graph,
)
from app.ml.temporal_pooling import (
    HybridTemporalPooling,
    LearnedTemporalPooling,
    TemporalConvPooling,
)
from app.ml.pretraining import MaskedLandmarkConfig, MaskedLandmarkModel

logger = structlog.get_logger(__name__)


def test_graph_construction():
    """Test skeleton graph construction."""
    logger.info("test_graph_construction", status="starting")

    # Test hand graph
    hand_edge_index, hand_num_nodes = build_hand_graph()
    assert hand_num_nodes == 21, f"Expected 21 hand nodes, got {hand_num_nodes}"
    assert hand_edge_index.shape[0] == 2, "Edge index should have 2 rows"
    logger.info(
        "hand_graph_ok",
        num_nodes=hand_num_nodes,
        num_edges=hand_edge_index.shape[1],
    )

    # Test combined graph
    combined_edge_index, combined_num_nodes = build_combined_graph()
    assert combined_num_nodes == 75, f"Expected 75 nodes, got {combined_num_nodes}"
    logger.info(
        "combined_graph_ok",
        num_nodes=combined_num_nodes,
        num_edges=combined_edge_index.shape[1],
    )

    logger.info("test_graph_construction", status="passed")


def test_spatial_encoder():
    """Test SpatialLandmarkEncoder forward pass."""
    logger.info("test_spatial_encoder", status="starting")

    batch_size = 4
    seq_len = 64
    num_landmarks = 75
    landmark_dim = 3

    # Create encoder
    encoder = SpatialLandmarkEncoder(
        num_landmarks=num_landmarks,
        landmark_dim=landmark_dim,
        hidden_dim=128,
        output_dim=256,
        num_gcn_layers=2,
        dropout=0.1,
    )

    # Random input
    x = torch.randn(batch_size, seq_len, num_landmarks * landmark_dim)

    # Forward pass
    try:
        output = encoder(x)
        assert output.shape == (
            batch_size,
            seq_len,
            256,
        ), f"Expected shape ({batch_size}, {seq_len}, 256), got {output.shape}"
        logger.info("spatial_encoder_forward_ok", output_shape=tuple(output.shape))
    except Exception as e:
        logger.error("spatial_encoder_forward_failed", error=str(e))
        raise

    logger.info("test_spatial_encoder", status="passed")


def test_temporal_pooling():
    """Test TemporalConvPooling with variable-length sequences."""
    logger.info("test_temporal_pooling", status="starting")

    batch_size = 4
    input_dim = 256
    output_length = 64

    # Variable-length sequences
    seq_lengths = [80, 120, 50, 100]
    max_len = max(seq_lengths)

    # Create pooler
    pooler = TemporalConvPooling(
        input_dim=input_dim,
        hidden_dim=128,
        output_length=output_length,
        num_scales=3,
        dropout=0.1,
    )

    # Padded input
    x = torch.randn(batch_size, max_len, input_dim)
    lengths = torch.tensor(seq_lengths)

    # Forward pass
    try:
        output = pooler(x, lengths)
        assert output.shape == (
            batch_size,
            output_length,
            input_dim,
        ), f"Expected shape ({batch_size}, {output_length}, {input_dim}), got {output.shape}"
        logger.info("temporal_pooling_ok", output_shape=tuple(output.shape))
    except Exception as e:
        logger.error("temporal_pooling_failed", error=str(e))
        raise

    # Test attention pooling
    attention_pooler = LearnedTemporalPooling(
        input_dim=input_dim,
        output_length=output_length,
        num_heads=4,
    )

    try:
        output = attention_pooler(x)
        assert output.shape == (batch_size, output_length, input_dim)
        logger.info("attention_pooling_ok", output_shape=tuple(output.shape))
    except Exception as e:
        logger.error("attention_pooling_failed", error=str(e))
        raise

    # Test hybrid pooling
    hybrid_pooler = HybridTemporalPooling(
        input_dim=input_dim,
        intermediate_length=128,
        output_length=output_length,
    )

    try:
        output = hybrid_pooler(x, lengths)
        assert output.shape == (batch_size, output_length, input_dim)
        logger.info("hybrid_pooling_ok", output_shape=tuple(output.shape))
    except Exception as e:
        logger.error("hybrid_pooling_failed", error=str(e))
        raise

    logger.info("test_temporal_pooling", status="passed")


def test_spatial_temporal_transformer():
    """Test SpatialTemporalTransformer end-to-end."""
    logger.info("test_spatial_temporal_transformer", status="starting")

    batch_size = 4
    seq_len = 64
    num_landmarks = 75
    landmark_dim = 3
    num_classes = 100

    # Create model
    model = SpatialTemporalTransformer(
        num_landmarks=num_landmarks,
        landmark_dim=landmark_dim,
        num_classes=num_classes,
        spatial_hidden_dim=128,
        spatial_output_dim=256,
        num_gcn_layers=2,
        d_model=384,
        nhead=8,
        num_transformer_layers=6,
        dim_feedforward=1536,
        dropout=0.3,
    )

    # Random input
    x = torch.randn(batch_size, seq_len, num_landmarks * landmark_dim)

    # Forward pass
    try:
        logits = model(x)
        assert logits.shape == (
            batch_size,
            num_classes,
        ), f"Expected shape ({batch_size}, {num_classes}), got {logits.shape}"
        logger.info("spatial_temporal_forward_ok", logits_shape=tuple(logits.shape))
    except Exception as e:
        logger.error("spatial_temporal_forward_failed", error=str(e))
        raise

    # Test embeddings extraction
    try:
        embeddings = model.extract_embeddings(x)
        assert embeddings.shape == (batch_size, 384), f"Expected shape ({batch_size}, 384), got {embeddings.shape}"
        logger.info("embeddings_extraction_ok", embeddings_shape=tuple(embeddings.shape))
    except Exception as e:
        logger.error("embeddings_extraction_failed", error=str(e))
        raise

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info("model_parameters", count=num_params, formatted=f"{num_params:,}")

    logger.info("test_spatial_temporal_transformer", status="passed")


def test_masked_landmark_model():
    """Test MaskedLandmarkModel for pretraining."""
    logger.info("test_masked_landmark_model", status="starting")

    batch_size = 4
    seq_len = 64
    num_landmarks = 75
    landmark_dim = 3
    num_classes = 100

    # Create base encoder
    encoder = SpatialTemporalTransformer(
        num_landmarks=num_landmarks,
        landmark_dim=landmark_dim,
        num_classes=num_classes,
        spatial_hidden_dim=128,
        spatial_output_dim=256,
        num_gcn_layers=2,
        d_model=384,
        nhead=8,
        num_transformer_layers=6,
        dim_feedforward=1536,
        dropout=0.3,
    )

    # Create masked landmark model
    config = MaskedLandmarkConfig(
        mask_prob=0.15,
        num_landmarks=num_landmarks,
        landmark_dim=landmark_dim,
    )

    model = MaskedLandmarkModel(encoder, config, encoder_output_dim=384)

    # Random input
    x = torch.randn(batch_size, seq_len, num_landmarks * landmark_dim)

    # Forward pass with loss
    try:
        outputs = model(x, return_loss=True)
        assert "loss" in outputs, "Missing 'loss' in outputs"
        assert "reconstruction_loss" in outputs, "Missing 'reconstruction_loss'"
        assert "temporal_smoothness_loss" in outputs, "Missing 'temporal_smoothness_loss'"
        logger.info(
            "masked_landmark_forward_ok",
            loss=float(outputs["loss"]),
            reconstruction_loss=float(outputs["reconstruction_loss"]),
            temporal_smoothness_loss=float(outputs["temporal_smoothness_loss"]),
        )
    except Exception as e:
        logger.error("masked_landmark_forward_failed", error=str(e))
        raise

    # Test backward pass
    try:
        loss = outputs["loss"]
        loss.backward()
        logger.info("masked_landmark_backward_ok")
    except Exception as e:
        logger.error("masked_landmark_backward_failed", error=str(e))
        raise

    logger.info("test_masked_landmark_model", status="passed")


def main():
    """Run all Phase 2 verification tests."""
    logger.info("phase2_verification", status="starting")

    try:
        # Check torch-geometric availability
        try:
            import torch_geometric

            logger.info("torch_geometric_available", version=torch_geometric.__version__)
        except ImportError:
            logger.error(
                "torch_geometric_not_installed",
                msg="Install with: pip install torch-geometric",
            )
            return 1

        # Run tests
        test_graph_construction()
        test_spatial_encoder()
        test_temporal_pooling()
        test_spatial_temporal_transformer()
        test_masked_landmark_model()

        logger.info("phase2_verification", status="ALL TESTS PASSED ✓")
        return 0

    except Exception as e:
        logger.error("phase2_verification_failed", error=str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
