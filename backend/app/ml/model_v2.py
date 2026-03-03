"""SignTransformerV2 — Architecture multi-stream pour LSFB V2.

Multi-stream architecture with 3 specialized streams fused with cross-modal attention:
  - Hand stream: GCN-fallback MLP on 42 hand landmarks (21L + 21R)
  - Pose stream: MLP on 33 pose landmarks body coords
  - NMM stream: MLP on 32 non-manual marker dimensions (facial AUs + head/gaze)

Feature layout V2 (611 dims from features.py):
  [0:63]    left_hand xyz (normalized)
  [63:126]  right_hand xyz (normalized)
  [126:225] pose xyz (normalized)
  [225:237] compact facial expression (12 dims)
  [237:462] coordinate velocities (225 dims)
  [462:467] inter-hand distances (5 dims)
  [467:471] joint angles (4 dims)
  [471:477] facial expression velocity (6 dims)
  [477:561] handshape features (84 dims = 2 × 42)
  [561:593] NMM features (32 dims)
  [593:611] signing space features (18 dims)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from app.ml.features import ENRICHED_FEATURE_DIM_V2

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional torch-geometric import for GCN support
# ---------------------------------------------------------------------------
try:
    from app.ml.spatial_encoder import (
        HAND_EDGES,
        GraphConvLayer,
        build_hand_graph,
    )
    _SPATIAL_ENCODER_AVAILABLE = True
except Exception:
    _SPATIAL_ENCODER_AVAILABLE = False
    logger.warning(
        "spatial_encoder_unavailable",
        msg="spatial_encoder import failed — hand GCN will fall back to MLP",
    )


# ---------------------------------------------------------------------------
# Layout constants (V2 feature vector)
# ---------------------------------------------------------------------------


class V2Layout:
    """Slice constants for V2 feature vector (611 dims)."""

    LEFT_HAND_START = 0
    LEFT_HAND_END = 63       # 21 joints × 3 = 63

    RIGHT_HAND_START = 63
    RIGHT_HAND_END = 126     # 21 joints × 3 = 63

    POSE_START = 126
    POSE_END = 225           # 33 joints × 3 = 99

    FACIAL_EXPR_START = 225
    FACIAL_EXPR_END = 237    # 12 dims

    COORD_VEL_START = 237
    COORD_VEL_END = 462      # 225 dims

    INTER_DIST_START = 462
    INTER_DIST_END = 467     # 5 dims

    JOINT_ANGLE_START = 467
    JOINT_ANGLE_END = 471    # 4 dims

    FACIAL_VEL_START = 471
    FACIAL_VEL_END = 477     # 6 dims

    HANDSHAPE_START = 477
    HANDSHAPE_END = 561      # 84 dims (2 × 42)

    NMM_START = 561
    NMM_END = 593            # 32 dims

    SIGNING_SPACE_START = 593
    SIGNING_SPACE_END = 611  # 18 dims

    TOTAL = 611


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class SignTransformerV2Config:
    """Configuration for SignTransformerV2 multi-stream architecture."""

    feature_dim: int = ENRICHED_FEATURE_DIM_V2  # 611
    hand_hidden_dim: int = 128          # dim of hand stream encoder output
    pose_hidden_dim: int = 64           # dim of pose stream encoder output
    nmm_hidden_dim: int = 64            # dim of NMM stream encoder output
    fusion_dim: int = 256               # dim after cross-modal fusion
    num_heads: int = 8                  # attention heads in temporal + cross-modal
    num_layers: int = 4                 # Transformer layers per stream
    num_fusion_layers: int = 2          # cross-modal attention layers
    dropout: float = 0.1
    num_classes: int = 100
    use_cosine_head: bool = True
    max_seq_len: int = 64
    use_gcn_hand: bool = True           # use GCN for hand stream (falls back to MLP if unavailable)
    num_gcn_layers: int = 2
    cosine_head_weight: float = 0.35    # blend weight (0=linear, 1=cosine)
    feature_version: int = 2            # model metadata marker

    # Extra fields for estimated_params usage
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to plain dict (no nested mutable)."""
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
            "num_classes": self.num_classes,
            "use_cosine_head": self.use_cosine_head,
            "max_seq_len": self.max_seq_len,
            "use_gcn_hand": self.use_gcn_hand,
            "num_gcn_layers": self.num_gcn_layers,
            "cosine_head_weight": self.cosine_head_weight,
            "feature_version": self.feature_version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SignTransformerV2Config":
        """Deserialize from plain dict."""
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in known and k != "extra_metadata"}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Positional encoding (shared)
# ---------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]  # type: ignore[index]


# ---------------------------------------------------------------------------
# Stream encoders
# ---------------------------------------------------------------------------


class HandStreamEncoder(nn.Module):
    """Encode left + right hand coords with optional per-frame GCN then temporal Transformer.

    Input:  (batch, seq, 42 × 3 = 126) = left_hand (63) + right_hand (63)
    Also uses handshape features (84 dims) if available in auxiliary input.
    Output: (batch, seq, hand_hidden_dim)
    """

    def __init__(
        self,
        hand_hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 64,
        use_gcn: bool = True,
        num_gcn_layers: int = 2,
    ) -> None:
        super().__init__()
        self.use_gcn = use_gcn and _SPATIAL_ENCODER_AVAILABLE
        self.hand_hidden_dim = hand_hidden_dim

        # Hand coord input: 63 + 63 = 126 raw coords
        # plus handshape: 84 dims → total 210 or fall back to 126
        # We project to hidden_dim regardless
        self._coord_in = 126   # left + right xyz
        self._hs_in = 84       # handshape features

        if self.use_gcn:
            try:
                # GCN for 21 landmarks per hand
                from app.ml.spatial_encoder import GraphConvLayer
                self._gcn_left = self._build_hand_gcn(num_gcn_layers, dropout)
                self._gcn_right = self._build_hand_gcn(num_gcn_layers, dropout)
                self._gcn_out_dim = 3 * 21  # after node pooling 3 keeps -> projected below
                proj_in = 2 * hand_hidden_dim + self._hs_in  # 2 hands × h_dim + handshape
            except Exception:
                self.use_gcn = False
                proj_in = self._coord_in + self._hs_in
        else:
            proj_in = self._coord_in + self._hs_in

        # Input projection: concat hand coords (or GCN output) + handshape → hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(proj_in if self.use_gcn else (self._coord_in + self._hs_in), hand_hidden_dim),
            nn.LayerNorm(hand_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Temporal Transformer
        _nhead = num_heads
        if hand_hidden_dim % _nhead != 0:
            _candidates = [h for h in (8, 4, 2, 1) if hand_hidden_dim % h == 0]
            _nhead = _candidates[0]

        self.pos_enc = PositionalEncoding(hand_hidden_dim, max_len=max_seq_len + 2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hand_hidden_dim,
            nhead=_nhead,
            dim_feedforward=hand_hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(hand_hidden_dim)

    @staticmethod
    def _build_hand_gcn(num_gcn_layers: int, dropout: float) -> nn.ModuleList:
        """Build a ModuleList of GCN layers for one hand (21 nodes, 3 features)."""
        from app.ml.spatial_encoder import GraphConvLayer

        layers = nn.ModuleList()
        in_ch = 3  # xyz per node
        for _ in range(num_gcn_layers):
            layers.append(GraphConvLayer(in_ch, in_ch, dropout=dropout))
        return layers

    def _apply_hand_gcn(
        self,
        gcn_layers: nn.ModuleList,
        hand_coords: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """Apply GCN layers to (batch*seq, 21, 3) hand landmark tensor.

        Returns (batch*seq, 21*3) flat features after message passing.
        """
        bs_seq, n_lm, n_feat = hand_coords.shape  # (B*T, 21, 3)
        # Flatten for GCN: (B*T*21, 3)
        h = hand_coords.view(bs_seq * n_lm, n_feat)

        # Expand edge_index for batch
        num_nodes = n_lm
        offsets = torch.arange(bs_seq, device=h.device) * num_nodes
        offsets = offsets.view(-1, 1, 1)
        edge_expanded = edge_index.unsqueeze(0).expand(bs_seq, -1, -1).clone()
        edge_expanded = edge_expanded + offsets
        edge_batched = edge_expanded.transpose(0, 1).reshape(2, -1)

        for gcn_layer in gcn_layers:
            h = gcn_layer(h, edge_batched)

        # Back to (B*T, 21, 3) -> (B*T, 63)
        return h.view(bs_seq, n_lm * n_feat)

    def forward(
        self,
        left_hand: Tensor,    # (batch, seq, 63)
        right_hand: Tensor,   # (batch, seq, 63)
        handshape: Tensor,    # (batch, seq, 84)
    ) -> Tensor:
        """Return (batch, seq, hand_hidden_dim)."""
        batch, seq, _ = left_hand.shape

        if self.use_gcn:
            try:
                edge_index, _ = build_hand_graph()
                edge_index = edge_index.to(left_hand.device)

                left_nodes = left_hand.view(batch * seq, 21, 3)
                right_nodes = right_hand.view(batch * seq, 21, 3)

                left_gcn = self._apply_hand_gcn(self._gcn_left, left_nodes, edge_index)
                right_gcn = self._apply_hand_gcn(self._gcn_right, right_nodes, edge_index)

                # Project GCN output to hand_hidden_dim
                # left_gcn, right_gcn: (B*T, 63)
                # Reshape for projection
                left_proj = left_gcn.view(batch, seq, -1)
                right_proj = right_gcn.view(batch, seq, -1)

                # GCN output: (batch, seq, 63+63) + handshape (84) = 210
                # input_proj expects proj_in = 2*hand_hidden_dim + 84 when GCN
                # But GCN output is 63 not hand_hidden_dim — remap projection
                # Simplest: concatenate raw GCN out + handshape
                combined = torch.cat([left_proj, right_proj, handshape], dim=-1)
            except Exception:
                self.use_gcn = False
                combined = torch.cat([left_hand, right_hand, handshape], dim=-1)
        else:
            combined = torch.cat([left_hand, right_hand, handshape], dim=-1)

        # Project to hand_hidden_dim
        h = self.input_proj(combined)   # (batch, seq, hand_hidden_dim)
        h = self.pos_enc(h)
        h = self.transformer(h)
        return self.output_norm(h)


class PoseStreamEncoder(nn.Module):
    """Encode pose body coords + joint angles with MLP + Transformer.

    Input:  pose (99 dims) + inter_dist (5) + joint_angles (4) + signing_space (18) = 126
    Output: (batch, seq, pose_hidden_dim)
    """

    def __init__(
        self,
        pose_hidden_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 64,
    ) -> None:
        super().__init__()
        # 99 pose + 5 inter_dist + 4 joint_angles + 18 signing_space = 126
        pose_in = 99 + 5 + 4 + 18

        _nhead = num_heads
        if pose_hidden_dim % _nhead != 0:
            _candidates = [h for h in (4, 2, 1) if pose_hidden_dim % h == 0]
            _nhead = _candidates[0]

        self.input_proj = nn.Sequential(
            nn.Linear(pose_in, pose_hidden_dim),
            nn.LayerNorm(pose_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_enc = PositionalEncoding(pose_hidden_dim, max_len=max_seq_len + 2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=pose_hidden_dim,
            nhead=_nhead,
            dim_feedforward=pose_hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(pose_hidden_dim)

    def forward(
        self,
        pose: Tensor,          # (batch, seq, 99)
        inter_dist: Tensor,    # (batch, seq, 5)
        joint_angles: Tensor,  # (batch, seq, 4)
        signing_space: Tensor, # (batch, seq, 18)
    ) -> Tensor:
        """Return (batch, seq, pose_hidden_dim)."""
        combined = torch.cat([pose, inter_dist, joint_angles, signing_space], dim=-1)
        h = self.input_proj(combined)
        h = self.pos_enc(h)
        h = self.transformer(h)
        return self.output_norm(h)


class NMMStreamEncoder(nn.Module):
    """Encode Non-Manual Markers (NMM) — facial AUs + head pose + gaze.

    Input:  nmm (32) + facial_expr (12) + facial_vel (6) = 50 dims
    Output: (batch, seq, nmm_hidden_dim)
    """

    def __init__(
        self,
        nmm_hidden_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 64,
    ) -> None:
        super().__init__()
        nmm_in = 32 + 12 + 6  # NMM + facial_expr + facial_vel = 50

        _nhead = num_heads
        if nmm_hidden_dim % _nhead != 0:
            _candidates = [h for h in (4, 2, 1) if nmm_hidden_dim % h == 0]
            _nhead = _candidates[0]

        self.input_proj = nn.Sequential(
            nn.Linear(nmm_in, nmm_hidden_dim),
            nn.LayerNorm(nmm_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_enc = PositionalEncoding(nmm_hidden_dim, max_len=max_seq_len + 2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=nmm_hidden_dim,
            nhead=_nhead,
            dim_feedforward=nmm_hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(nmm_hidden_dim)

    def forward(
        self,
        nmm: Tensor,          # (batch, seq, 32)
        facial_expr: Tensor,  # (batch, seq, 12)
        facial_vel: Tensor,   # (batch, seq, 6)
    ) -> Tensor:
        """Return (batch, seq, nmm_hidden_dim)."""
        combined = torch.cat([nmm, facial_expr, facial_vel], dim=-1)
        h = self.input_proj(combined)
        h = self.pos_enc(h)
        h = self.transformer(h)
        return self.output_norm(h)


# ---------------------------------------------------------------------------
# Cross-modal attention fusion
# ---------------------------------------------------------------------------


class CrossModalAttentionLayer(nn.Module):
    """Single cross-modal attention: query from stream A, key/value from stream B."""

    def __init__(self, query_dim: int, kv_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        # Ensure num_heads divides both dims after projection
        _nhead = num_heads
        if query_dim % _nhead != 0:
            _nhead = [h for h in (4, 2, 1) if query_dim % h == 0][0]

        # Project kv to same dim as query
        self.kv_proj = nn.Linear(kv_dim, query_dim) if kv_dim != query_dim else nn.Identity()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=_nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(query_dim)
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim * 4, query_dim),
        )
        self.norm2 = nn.LayerNorm(query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: Tensor, key_value: Tensor) -> Tensor:
        """
        Args:
            query:     (batch, seq, query_dim)
            key_value: (batch, seq, kv_dim)
        Returns: (batch, seq, query_dim)
        """
        kv = self.kv_proj(key_value)
        attended, _ = self.cross_attn(query=query, key=kv, value=kv)
        query = self.norm(query + self.dropout(attended))
        query = self.norm2(query + self.dropout(self.ffn(query)))
        return query


class CrossModalFusion(nn.Module):
    """Multi-layer cross-modal attention fusion for hand + pose + NMM streams.

    Applies cross-attention in both directions:
      - hand ↔ NMM  (co-articulation: which sign + which mouth shape)
      - hand ↔ pose (hand location in signing space context)

    Then projects concatenation to fusion_dim.
    """

    def __init__(
        self,
        hand_dim: int,
        pose_dim: int,
        nmm_dim: int,
        fusion_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.hand_nmm_layers = nn.ModuleList([
            CrossModalAttentionLayer(hand_dim, nmm_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.nmm_hand_layers = nn.ModuleList([
            CrossModalAttentionLayer(nmm_dim, hand_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.hand_pose_layers = nn.ModuleList([
            CrossModalAttentionLayer(hand_dim, pose_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        concat_dim = hand_dim + pose_dim + nmm_dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(concat_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hand: Tensor,  # (batch, seq, hand_dim)
        pose: Tensor,  # (batch, seq, pose_dim)
        nmm: Tensor,   # (batch, seq, nmm_dim)
    ) -> Tensor:
        """Return (batch, seq, fusion_dim)."""
        # hand attends to NMM (co-articulation)
        h = hand
        n = nmm
        for h_layer, n_layer in zip(self.hand_nmm_layers, self.nmm_hand_layers):
            h = h_layer(h, n)
            n = n_layer(n, h)

        # hand attends to pose (signing space)
        for layer in self.hand_pose_layers:
            h = layer(h, pose)

        # Final concat + projection
        fused = torch.cat([h, pose, n], dim=-1)
        return self.fusion_proj(fused)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class SignTransformerV2(nn.Module):
    """Multi-stream Transformer for LSFB V2 sign classification.

    Architecture:
        ┌─────────────────────────────────────────────────────────┐
        │   Input V2 features (611 dims) per frame                │
        └──────┬─────────────┬───────────────┬────────────────────┘
               ↓             ↓               ↓
        ┌────────────┐ ┌───────────┐ ┌─────────────────┐
        │ Hand Stream│ │Pose Stream│ │   NMM Stream    │
        │ GCN/MLP    │ │  MLP      │ │   MLP           │
        │ 126+84 dim │ │ 126 dim   │ │   50 dim        │
        └─────┬──────┘ └─────┬─────┘ └──────┬──────────┘
              ↓              ↓               ↓
        ┌─────────────────────────────────────────────────────┐
        │       Temporal Transformer per stream (parallel)    │
        └──────────────────────┬──────────────────────────────┘
                               ↓
        ┌─────────────────────────────────────────────────────┐
        │       Cross-Modal Attention Fusion                  │
        │    (hand ↔ NMM, hand ↔ pose)                       │
        └──────────────────────┬──────────────────────────────┘
                               ↓
        ┌─────────────────────────────────────────────────────┐
        │   CLS-pooling + Cosine/Linear Classification Head  │
        └─────────────────────────────────────────────────────┘
    """

    # Metadata marker for checkpoint compatibility
    FEATURE_VERSION: int = 2

    def __init__(
        self,
        *,
        num_classes: int,
        config: SignTransformerV2Config | None = None,
        # Individual kwargs (overridden by config when provided)
        feature_dim: int = ENRICHED_FEATURE_DIM_V2,
        hand_hidden_dim: int = 128,
        pose_hidden_dim: int = 64,
        nmm_hidden_dim: int = 64,
        fusion_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        num_fusion_layers: int = 2,
        dropout: float = 0.1,
        use_cosine_head: bool = True,
        max_seq_len: int = 64,
        use_gcn_hand: bool = True,
        num_gcn_layers: int = 2,
        cosine_head_weight: float = 0.35,
    ) -> None:
        super().__init__()

        if config is not None:
            feature_dim = config.feature_dim
            hand_hidden_dim = config.hand_hidden_dim
            pose_hidden_dim = config.pose_hidden_dim
            nmm_hidden_dim = config.nmm_hidden_dim
            fusion_dim = config.fusion_dim
            num_heads = config.num_heads
            num_layers = config.num_layers
            num_fusion_layers = config.num_fusion_layers
            dropout = config.dropout
            num_classes = config.num_classes
            use_cosine_head = config.use_cosine_head
            max_seq_len = config.max_seq_len
            use_gcn_hand = config.use_gcn_hand
            num_gcn_layers = config.num_gcn_layers
            cosine_head_weight = config.cosine_head_weight

        # Store hyperparams as public attributes for checkpoint compatibility
        self.num_classes = num_classes
        self.num_features = feature_dim  # alias used by V1 compat
        self.feature_dim = feature_dim
        self.hand_hidden_dim = hand_hidden_dim
        self.pose_hidden_dim = pose_hidden_dim
        self.nmm_hidden_dim = nmm_hidden_dim
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_fusion_layers = num_fusion_layers
        self.dropout = dropout
        self.use_cosine_head = use_cosine_head
        self.max_seq_len = max_seq_len
        self.cosine_head_weight = float(max(0.0, min(1.0, cosine_head_weight)))
        self.feature_version = self.FEATURE_VERSION

        # Velocity stream: use coord velocities (225 dims) as auxiliary input
        # Injected into hand + pose streams via concat before projection
        # Here we encode it separately then add to hand/pose via a learned gate
        self.vel_encoder = nn.Sequential(
            nn.Linear(225, hand_hidden_dim),
            nn.LayerNorm(hand_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Stream encoders ──────────────────────────────────────────────────
        self.hand_encoder = HandStreamEncoder(
            hand_hidden_dim=hand_hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            use_gcn=use_gcn_hand,
            num_gcn_layers=num_gcn_layers,
        )

        self.pose_encoder = PoseStreamEncoder(
            pose_hidden_dim=pose_hidden_dim,
            num_layers=num_layers,
            num_heads=max(1, min(num_heads, pose_hidden_dim // 8)) if pose_hidden_dim >= 8 else 1,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        self.nmm_encoder = NMMStreamEncoder(
            nmm_hidden_dim=nmm_hidden_dim,
            num_layers=num_layers,
            num_heads=max(1, min(num_heads, nmm_hidden_dim // 8)) if nmm_hidden_dim >= 8 else 1,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        # ── Cross-modal fusion ───────────────────────────────────────────────
        self.cross_modal_fusion = CrossModalFusion(
            hand_dim=hand_hidden_dim,
            pose_dim=pose_hidden_dim,
            nmm_dim=nmm_hidden_dim,
            fusion_dim=fusion_dim,
            num_layers=num_fusion_layers,
            num_heads=max(1, min(4, fusion_dim // 32)),
            dropout=dropout,
        )

        # ── CLS token + pooling ──────────────────────────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        # Gated pooling between CLS and mean representation
        self.pooling_gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, 1),
            nn.Sigmoid(),
        )
        self.pooling_pos_enc = PositionalEncoding(fusion_dim, max_len=max_seq_len + 2)
        self.output_norm = nn.LayerNorm(fusion_dim)
        self.pooling_dropout = nn.Dropout(dropout)

        # ── Classification heads ─────────────────────────────────────────────
        self.classifier = nn.Linear(fusion_dim, num_classes)
        if use_cosine_head:
            self.metric_projection = nn.Linear(fusion_dim, fusion_dim)
            self.class_prototypes = nn.Parameter(torch.randn(num_classes, fusion_dim))
            self.cosine_logit_scale = nn.Parameter(torch.tensor(10.0))
            nn.init.normal_(self.class_prototypes, mean=0.0, std=0.02)

    # ── Feature splitting ────────────────────────────────────────────────────

    def split_feature_streams(self, x: Tensor) -> dict[str, Tensor]:
        """Split V2 (611-dim) vector into named stream tensors.

        Args:
            x: (batch, seq, 611)
        Returns:
            Dict with keys: left_hand, right_hand, pose, facial_expr,
                            coord_vel, inter_dist, joint_angles, facial_vel,
                            handshape, nmm, signing_space
        """
        L = V2Layout
        return {
            "left_hand":    x[..., L.LEFT_HAND_START:L.LEFT_HAND_END],       # (B,T,63)
            "right_hand":   x[..., L.RIGHT_HAND_START:L.RIGHT_HAND_END],     # (B,T,63)
            "pose":         x[..., L.POSE_START:L.POSE_END],                  # (B,T,99)
            "facial_expr":  x[..., L.FACIAL_EXPR_START:L.FACIAL_EXPR_END],   # (B,T,12)
            "coord_vel":    x[..., L.COORD_VEL_START:L.COORD_VEL_END],       # (B,T,225)
            "inter_dist":   x[..., L.INTER_DIST_START:L.INTER_DIST_END],     # (B,T,5)
            "joint_angles": x[..., L.JOINT_ANGLE_START:L.JOINT_ANGLE_END],   # (B,T,4)
            "facial_vel":   x[..., L.FACIAL_VEL_START:L.FACIAL_VEL_END],     # (B,T,6)
            "handshape":    x[..., L.HANDSHAPE_START:L.HANDSHAPE_END],       # (B,T,84)
            "nmm":          x[..., L.NMM_START:L.NMM_END],                   # (B,T,32)
            "signing_space":x[..., L.SIGNING_SPACE_START:L.SIGNING_SPACE_END],  # (B,T,18)
        }

    # ── Forward ──────────────────────────────────────────────────────────────

    def extract_embeddings(self, x: Tensor) -> Tensor:
        """Encode (batch, seq, 611) → (batch, fusion_dim) pooled embedding."""
        streams = self.split_feature_streams(x)

        # Encode each stream
        hand_out = self.hand_encoder(
            left_hand=streams["left_hand"],
            right_hand=streams["right_hand"],
            handshape=streams["handshape"],
        )  # (B, T, hand_hidden_dim)

        pose_out = self.pose_encoder(
            pose=streams["pose"],
            inter_dist=streams["inter_dist"],
            joint_angles=streams["joint_angles"],
            signing_space=streams["signing_space"],
        )  # (B, T, pose_hidden_dim)

        nmm_out = self.nmm_encoder(
            nmm=streams["nmm"],
            facial_expr=streams["facial_expr"],
            facial_vel=streams["facial_vel"],
        )  # (B, T, nmm_hidden_dim)

        # Add velocity signal as additive gate on hand stream
        vel_gate = torch.sigmoid(self.vel_encoder(streams["coord_vel"]))
        hand_out = hand_out * vel_gate

        # Cross-modal fusion
        fused = self.cross_modal_fusion(hand_out, pose_out, nmm_out)  # (B, T, fusion_dim)

        # CLS token prepend → positional encoding
        batch_size = fused.size(0)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        fused_with_cls = torch.cat([cls_token, fused], dim=1)  # (B, T+1, fusion_dim)
        fused_with_cls = self.pooling_pos_enc(fused_with_cls)

        # Gated CLS + mean pooling
        cls_repr = fused_with_cls[:, 0]     # (B, fusion_dim)
        seq_repr = fused_with_cls[:, 1:]    # (B, T, fusion_dim)

        # Masked mean pool (ignore near-zero frames)
        active = (x.abs().sum(dim=-1) > 1e-6).float()  # (B, T)
        weights = active.unsqueeze(-1)
        weighted_sum = (seq_repr * weights).sum(dim=1)
        active_count = active.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_repr = weighted_sum / active_count  # (B, fusion_dim)

        gate = self.pooling_gate(torch.cat([cls_repr, mean_repr], dim=-1))  # (B, 1)
        pooled = gate * cls_repr + (1.0 - gate) * mean_repr
        pooled = self.output_norm(pooled)
        return self.pooling_dropout(pooled)

    def forward(self, x: Tensor) -> Tensor:
        """Compute logits from (batch, seq, 611) input.

        Args:
            x: Feature tensor (batch, seq_len, feature_dim=611)
        Returns:
            logits: (batch, num_classes)
        """
        pooled = self.extract_embeddings(x)
        linear_logits = self.classifier(pooled)

        if not self.use_cosine_head:
            return linear_logits

        embeddings = F.normalize(self.metric_projection(pooled), dim=-1)
        prototypes = F.normalize(self.class_prototypes, dim=-1)
        scale = torch.clamp(self.cosine_logit_scale, min=1.0, max=50.0)
        cosine_logits = scale * torch.matmul(embeddings, prototypes.transpose(0, 1))

        blend = self.cosine_head_weight
        return (1.0 - blend) * linear_logits + blend * cosine_logits

    def set_inference_mode(self) -> None:
        """Set model to inference mode (no gradients)."""
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    # ── Config helpers ───────────────────────────────────────────────────────

    def to_config(self) -> SignTransformerV2Config:
        """Export current architecture as a config dataclass."""
        return SignTransformerV2Config(
            feature_dim=self.feature_dim,
            hand_hidden_dim=self.hand_hidden_dim,
            pose_hidden_dim=self.pose_hidden_dim,
            nmm_hidden_dim=self.nmm_hidden_dim,
            fusion_dim=self.fusion_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            num_fusion_layers=self.num_fusion_layers,
            dropout=self.dropout,
            num_classes=self.num_classes,
            use_cosine_head=self.use_cosine_head,
            max_seq_len=self.max_seq_len,
            cosine_head_weight=self.cosine_head_weight,
        )

    @classmethod
    def from_config(cls, config: SignTransformerV2Config) -> "SignTransformerV2":
        """Instantiate from a SignTransformerV2Config."""
        return cls(num_classes=config.num_classes, config=config)

    # ── Checkpoint I/O ───────────────────────────────────────────────────────

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model + config to a .pt checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.to_config().to_dict(),
            "num_classes": self.num_classes,
            "feature_version": self.FEATURE_VERSION,
            "architecture": "SignTransformerV2",
        }
        torch.save(checkpoint, str(path))
        logger.info("checkpoint_saved", path=str(path), feature_version=self.FEATURE_VERSION)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
        strict: bool = True,
    ) -> "SignTransformerV2":
        """Load SignTransformerV2 from a saved checkpoint."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(str(path), map_location=device, weights_only=False)

        if checkpoint.get("architecture") != "SignTransformerV2":
            raise ValueError(
                f"Checkpoint at {path} is not a SignTransformerV2 checkpoint "
                f"(architecture={checkpoint.get('architecture')}). "
                "Use load_v1_checkpoint() to migrate from V1."
            )

        config_dict = checkpoint.get("config", {})
        config_dict["num_classes"] = checkpoint.get("num_classes", config_dict.get("num_classes", 100))
        config = SignTransformerV2Config.from_dict(config_dict)

        model = cls.from_config(config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        model.to(device)
        logger.info("checkpoint_loaded", path=str(path), num_classes=model.num_classes)
        return model

    @classmethod
    def load_v1_checkpoint(
        cls,
        path: str | Path,
        num_classes: int | None = None,
        v2_config: SignTransformerV2Config | None = None,
        device: str | torch.device = "cpu",
    ) -> "SignTransformerV2":
        """Create a new V2 model, optionally transferring V1 weights where shapes match.

        V1 architecture is fundamentally different; only the cosine scale and any
        identically-named parameters are transferred. The rest is randomly initialized.

        Args:
            path: Path to a V1 SignTransformer checkpoint.
            num_classes: Number of output classes. Inferred from checkpoint if None.
            v2_config: V2 config to use. If None, defaults are used.
            device: Target device.
        """
        path = Path(path)
        checkpoint = torch.load(str(path), map_location=device, weights_only=False)
        ckpt_num_classes = checkpoint.get("num_classes", num_classes or 100)
        if num_classes is None:
            num_classes = ckpt_num_classes

        if v2_config is None:
            v2_config = SignTransformerV2Config(num_classes=num_classes)
        else:
            v2_config.num_classes = num_classes

        model = cls.from_config(v2_config)

        # Attempt soft weight transfer (shape-compatible keys only)
        v1_state = checkpoint.get("model_state_dict", checkpoint)
        v2_state = model.state_dict()
        transferred, skipped = 0, 0
        for k, v in v1_state.items():
            if k in v2_state and v2_state[k].shape == v.shape:
                v2_state[k] = v
                transferred += 1
            else:
                skipped += 1

        model.load_state_dict(v2_state, strict=False)
        model.to(device)
        logger.info(
            "v1_checkpoint_adapted",
            path=str(path),
            transferred_params=transferred,
            skipped_params=skipped,
            num_classes=num_classes,
        )
        return model

    @property
    def input_dim(self) -> int:
        """Alias used by pipeline.py for version auto-detection."""
        return self.feature_dim
