"""Spatial encoders for landmark data using Graph Convolutional Networks.

This module provides graph-based spatial feature learning for hand and pose landmarks.
Instead of relying on hand-crafted features, GCNs learn spatial relationships
between joints directly from the anatomical skeleton structure.
"""

from __future__ import annotations

from typing import Literal

import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = structlog.get_logger(__name__)

# Optional torch-geometric import - graceful degradation
try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.utils import add_self_loops

    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    logger.warning(
        "torch_geometric_not_installed",
        msg="torch-geometric not installed, spatial encoding unavailable",
    )


# ======================
# Hand Skeleton Graph
# ======================

# MediaPipe hand landmarks indices (21 points per hand)
# 0: WRIST
# 1-4: THUMB (CMC, MCP, IP, TIP)
# 5-8: INDEX (MCP, PIP, DIP, TIP)
# 9-12: MIDDLE (MCP, PIP, DIP, TIP)
# 13-16: RING (MCP, PIP, DIP, TIP)
# 17-20: PINKY (MCP, PIP, DIP, TIP)

HAND_EDGES = [
    # Wrist to finger bases
    (0, 1),
    (0, 5),
    (0, 9),
    (0, 13),
    (0, 17),
    # Thumb chain
    (1, 2),
    (2, 3),
    (3, 4),
    # Index chain
    (5, 6),
    (6, 7),
    (7, 8),
    # Middle chain
    (9, 10),
    (10, 11),
    (11, 12),
    # Ring chain
    (13, 14),
    (14, 15),
    (15, 16),
    # Pinky chain
    (17, 18),
    (18, 19),
    (19, 20),
]

# MediaPipe pose landmarks indices (33 points)
# Simplified upper body skeleton
POSE_EDGES = [
    # Torso
    (11, 12),  # Shoulders
    (11, 23),  # Left shoulder to hip
    (12, 24),  # Right shoulder to hip
    (23, 24),  # Hips
    # Arms
    (11, 13),  # Left shoulder to elbow
    (13, 15),  # Left elbow to wrist
    (12, 14),  # Right shoulder to elbow
    (14, 16),  # Right elbow to wrist
]


def build_hand_graph() -> tuple[Tensor, int]:
    """
    Build edge index tensor for hand skeleton graph.

    Returns:
        edge_index: [2, num_edges] tensor of directed edges
        num_nodes: Number of nodes (21 for hand)
    """
    edges = HAND_EDGES + [(j, i) for i, j in HAND_EDGES]  # Bidirectional
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index, 21


def build_pose_graph() -> tuple[Tensor, int]:
    """
    Build edge index tensor for pose skeleton graph.

    Returns:
        edge_index: [2, num_edges] tensor of directed edges
        num_nodes: Number of nodes (33 for pose)
    """
    edges = POSE_EDGES + [(j, i) for i, j in POSE_EDGES]  # Bidirectional
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index, 33


def build_combined_graph() -> tuple[Tensor, int]:
    """
    Build combined graph for left_hand (21) + right_hand (21) + pose (33).

    Offset indices for each modality:
    - Left hand: 0-20
    - Right hand: 21-41
    - Pose: 42-74

    Returns:
        edge_index: [2, num_edges] tensor
        num_nodes: Total number of nodes (75)
    """
    left_hand_edges, _ = build_hand_graph()
    right_hand_edges, _ = build_hand_graph()
    pose_edges, _ = build_pose_graph()

    # Offset right hand and pose indices
    right_hand_edges = right_hand_edges + 21
    pose_edges = pose_edges + 42

    # Concatenate all edges
    edge_index = torch.cat([left_hand_edges, right_hand_edges, pose_edges], dim=1)

    # Add connections between wrists and pose (optional cross-modality edges)
    # Left wrist (0) to left wrist in pose (15+42=57)
    # Right wrist (21) to right wrist in pose (16+42=58)
    cross_edges = torch.tensor([[0, 21], [57, 58]], dtype=torch.long)
    cross_edges_reverse = torch.tensor([[57, 58], [0, 21]], dtype=torch.long)
    edge_index = torch.cat([edge_index, cross_edges, cross_edges_reverse], dim=1)

    return edge_index, 75


# ======================
# Graph Convolutional Layers
# ======================


class GraphConvLayer(nn.Module):
    """
    Single graph convolution layer using torch-geometric.

    Applies message passing along skeleton edges to aggregate neighbor features.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "torch-geometric is required for GraphConvLayer. "
                "Install with: pip install torch-geometric"
            )

        self.conv = GCNConv(in_channels, out_channels, add_self_loops=True, normalize=True)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Args:
            x: [num_nodes, in_channels] node features
            edge_index: [2, num_edges] graph structure

        Returns:
            [num_nodes, out_channels] updated features
        """
        h = self.conv(x, edge_index)
        h = self.norm(h)
        h = F.gelu(h)
        h = self.dropout(h)
        return h


# ======================
# Spatial Encoder
# ======================


class SpatialLandmarkEncoder(nn.Module):
    """
    Graph-based spatial encoder for landmarks.

    Encodes raw (x, y, z) coordinates into learned spatial features using GCN layers
    that respect anatomical skeleton structure.

    Architecture:
        Input: [batch, seq, num_landmarks * 3] raw coordinates
        Graph: Skeleton edges for hands/pose
        GCN: 2 layers (input_dim → hidden_dim → output_dim)
        Output: [batch, seq, output_dim] learned spatial features
    """

    def __init__(
        self,
        *,
        num_landmarks: int = 75,  # 21 left + 21 right + 33 pose
        landmark_dim: int = 3,  # (x, y, z)
        hidden_dim: int = 128,
        output_dim: int = 256,
        num_gcn_layers: int = 2,
        dropout: float = 0.1,
        graph_type: Literal["combined", "hand", "pose"] = "combined",
    ):
        super().__init__()
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "torch-geometric is required for SpatialLandmarkEncoder. "
                "Install with: pip install torch-geometric"
            )

        self.num_landmarks = num_landmarks
        self.landmark_dim = landmark_dim
        self.output_dim = output_dim

        # Build graph structure
        if graph_type == "combined":
            edge_index, num_nodes = build_combined_graph()
        elif graph_type == "hand":
            edge_index, num_nodes = build_hand_graph()
        elif graph_type == "pose":
            edge_index, num_nodes = build_pose_graph()
        else:
            raise ValueError(f"Unknown graph_type: {graph_type}")

        if num_nodes != num_landmarks:
            logger.warning(
                "landmark_count_mismatch",
                num_landmarks=num_landmarks,
                graph_nodes=num_nodes,
                msg="Number of landmarks doesn't match graph structure",
            )

        self.register_buffer("edge_index", edge_index)

        # Input projection
        self.input_proj = nn.Linear(landmark_dim, hidden_dim)

        # GCN layers
        self.gcn_layers = nn.ModuleList()
        in_dim = hidden_dim
        for _ in range(num_gcn_layers):
            self.gcn_layers.append(GraphConvLayer(in_dim, hidden_dim, dropout=dropout))
            in_dim = hidden_dim

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode landmark sequences with graph convolutions.

        Args:
            x: [batch, seq, num_landmarks * landmark_dim] raw landmarks

        Returns:
            [batch, seq, output_dim] spatial features
        """
        batch_size, seq_len, _ = x.shape

        # Reshape to [batch * seq, num_landmarks, landmark_dim]
        x = x.view(batch_size * seq_len, self.num_landmarks, self.landmark_dim)

        # Project to hidden dim: [batch * seq, num_landmarks, hidden_dim]
        h = self.input_proj(x)

        # Flatten for GCN: [batch * seq * num_landmarks, hidden_dim]
        h = h.view(batch_size * seq_len * self.num_landmarks, -1)

        # Apply GCN layers
        # Need to expand edge_index for batched graph
        edge_index_batch = self._create_batched_edge_index(
            batch_size * seq_len, device=x.device
        )

        for gcn_layer in self.gcn_layers:
            h = gcn_layer(h, edge_index_batch)

        # Reshape back: [batch * seq, num_landmarks, hidden_dim]
        h = h.view(batch_size * seq_len, self.num_landmarks, -1)

        # Global pooling over landmarks (mean pooling)
        h = h.mean(dim=1)  # [batch * seq, hidden_dim]

        # Output projection
        h = self.output_proj(h)  # [batch * seq, output_dim]

        # Reshape to sequence: [batch, seq, output_dim]
        h = h.view(batch_size, seq_len, self.output_dim)

        return h

    def _create_batched_edge_index(self, num_graphs: int, device: torch.device) -> Tensor:
        """
        Create batched edge index for multiple independent graphs.

        Args:
            num_graphs: Number of independent graphs (batch_size * seq_len)
            device: Device to place tensor on

        Returns:
            [2, num_edges * num_graphs] batched edge index
        """
        edge_index = self.edge_index.to(device)
        num_nodes = self.num_landmarks

        # Create offset for each graph in batch
        offsets = torch.arange(num_graphs, device=device) * num_nodes
        offsets = offsets.view(-1, 1, 1)

        # Expand edge index for each graph
        edge_index_expanded = edge_index.unsqueeze(0).expand(num_graphs, -1, -1)
        edge_index_batched = edge_index_expanded + offsets

        # Flatten: [2, num_edges * num_graphs]
        edge_index_batched = edge_index_batched.transpose(0, 1).reshape(2, -1)

        return edge_index_batched


# ======================
# Combined Spatial-Temporal Model
# ======================


class SpatialTemporalTransformer(nn.Module):
    """
    Hybrid model combining spatial GCN encoder with temporal transformer.

    Architecture:
        1. SpatialLandmarkEncoder: Raw landmarks → Spatial features (GCN)
        2. Temporal Transformer: Spatial features → Sequence encoding
        3. Classification head: Sequence encoding → Logits

    This replaces hand-crafted features (469-dim) with learned spatial features.
    """

    def __init__(
        self,
        *,
        num_landmarks: int = 75,
        landmark_dim: int = 3,
        num_classes: int,
        spatial_hidden_dim: int = 128,
        spatial_output_dim: int = 256,
        num_gcn_layers: int = 2,
        d_model: int = 384,
        nhead: int = 8,
        num_transformer_layers: int = 6,
        dim_feedforward: int = 1536,
        dropout: float = 0.3,
        graph_type: Literal["combined", "hand", "pose"] = "combined",
    ):
        super().__init__()
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "torch-geometric is required for SpatialTemporalTransformer. "
                "Install with: pip install torch-geometric"
            )

        self.num_landmarks = num_landmarks
        self.num_classes = num_classes
        self.d_model = d_model

        # Spatial encoder (GCN)
        self.spatial_encoder = SpatialLandmarkEncoder(
            num_landmarks=num_landmarks,
            landmark_dim=landmark_dim,
            hidden_dim=spatial_hidden_dim,
            output_dim=spatial_output_dim,
            num_gcn_layers=num_gcn_layers,
            dropout=dropout,
        )

        # Project spatial features to transformer dimension
        self.spatial_to_transformer = nn.Sequential(
            nn.Linear(spatial_output_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model)

        # CLS token for sequence classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        # Temporal transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
            enable_nested_tensor=False,
        )

        # Output layers
        self.output_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [batch, seq, num_landmarks * landmark_dim] raw landmarks

        Returns:
            [batch, num_classes] logits
        """
        # Spatial encoding with GCN
        spatial_features = self.spatial_encoder(x)  # [batch, seq, spatial_output_dim]

        # Project to transformer dimension
        h = self.spatial_to_transformer(spatial_features)  # [batch, seq, d_model]

        # Add CLS token
        batch_size = h.size(0)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        h = torch.cat([cls_token, h], dim=1)  # [batch, seq+1, d_model]

        # Add positional encoding
        h = self.positional_encoding(h)

        # Temporal transformer
        h = self.transformer(h)  # [batch, seq+1, d_model]

        # Extract CLS token representation
        cls_representation = h[:, 0]  # [batch, d_model]

        # Classification
        cls_representation = self.output_norm(cls_representation)
        logits = self.classifier(cls_representation)

        return logits

    def extract_embeddings(self, x: Tensor) -> Tensor:
        """
        Extract embeddings before classification head.

        Useful for metric learning or few-shot learning.

        Args:
            x: [batch, seq, num_landmarks * landmark_dim] raw landmarks

        Returns:
            [batch, d_model] embeddings
        """
        spatial_features = self.spatial_encoder(x)
        h = self.spatial_to_transformer(spatial_features)

        batch_size = h.size(0)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        h = torch.cat([cls_token, h], dim=1)

        h = self.positional_encoding(h)
        h = self.transformer(h)

        cls_representation = h[:, 0]
        return self.output_norm(cls_representation)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        import math

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        """Add positional information to sequence embeddings."""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]
