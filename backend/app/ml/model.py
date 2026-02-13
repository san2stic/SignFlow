"""Transformer model for landmark sequence classification."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from app.ml.feature_engineering import ENRICHED_FEATURE_DIM


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        """Add positional information to sequence embeddings."""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class SignTransformer(nn.Module):
    """Transformer encoder for sign classification from landmark sequences."""

    def __init__(
        self,
        *,
        num_features: int = ENRICHED_FEATURE_DIM,  # was 225
        num_classes: int,
        d_model: int = 128,           # was 256
        nhead: int = 4,               # was 8
        num_layers: int = 2,          # was 4
        dim_feedforward: int = 256,   # was 512
        dropout: float = 0.3,         # was 0.1
        feature_dropout: float = 0.2, # was 0.1
        pooling_dropout: float = 0.3, # was 0.1
        use_cls_token: bool = True,
    ) -> None:
        super().__init__()
        # Store architecture parameters
        self.num_features = num_features
        self.num_classes = num_classes
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.use_cls_token = use_cls_token

        self.input_norm = nn.LayerNorm(num_features)
        self.embedding = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.GELU(),
            nn.Dropout(feature_dropout),
        )
        self.positional_encoding = PositionalEncoding(d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.pooling_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        self.pooling_dropout = nn.Dropout(pooling_dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """Compute logits from [batch, seq, features] input tensor."""
        pooled = self.extract_embeddings(x)
        return self.classifier(pooled)

    def extract_embeddings(self, x: Tensor) -> Tensor:
        """Return pooled transformer embeddings before classification head."""
        raw_input = x
        active_mask = self._active_mask(raw_input)

        encoded = self.input_norm(raw_input)
        encoded = self.embedding(encoded)

        if self.use_cls_token:
            batch_size = encoded.size(0)
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            encoded = torch.cat([cls_token, encoded], dim=1)

        encoded = self.positional_encoding(encoded)
        key_padding_mask = self._build_key_padding_mask(active_mask, device=encoded.device)
        encoded = self.encoder(encoded, src_key_padding_mask=key_padding_mask)
        encoded = self.output_norm(encoded)

        if self.use_cls_token:
            cls_representation = encoded[:, 0]
            sequence_encoded = encoded[:, 1:]
        else:
            cls_representation = encoded.mean(dim=1)
            sequence_encoded = encoded

        mean_representation = self.masked_temporal_pool(sequence_encoded, raw_input)
        fusion_weight = self.pooling_gate(
            torch.cat([cls_representation, mean_representation], dim=-1)
        )
        fused = fusion_weight * cls_representation + (1.0 - fusion_weight) * mean_representation
        return self.pooling_dropout(fused)

    def _build_key_padding_mask(self, active_mask: Tensor, device: torch.device) -> Tensor:
        """
        Build key padding mask for transformer.

        `True` values are ignored by attention.
        """
        sequence_padding = ~active_mask
        if self.use_cls_token:
            cls_padding = torch.zeros((active_mask.size(0), 1), dtype=torch.bool, device=device)
            return torch.cat([cls_padding, sequence_padding.to(device)], dim=1)
        return sequence_padding.to(device)

    @staticmethod
    def _active_mask(raw_input: Tensor, eps: float = 1e-6) -> Tensor:
        """Detect active frames (non-zero landmarks) for padding-aware attention."""
        if raw_input.ndim != 3:
            raise ValueError("Expected raw_input with shape [batch, seq, features]")
        return raw_input.abs().sum(dim=-1) > eps

    @staticmethod
    def masked_temporal_pool(encoded: Tensor, raw_input: Tensor, eps: float = 1e-6) -> Tensor:
        """
        Pool encoded sequence while ignoring inactive (near-zero) frames.

        Inactive frames appear when landmarks are missing and zero-padded. Ignoring
        them improves robustness by preventing these frames from diluting the pooled
        representation used for classification.
        """
        if encoded.ndim != 3 or raw_input.ndim != 3:
            raise ValueError("Expected encoded/raw_input with shape [batch, seq, features]")

        if encoded.shape[:2] != raw_input.shape[:2]:
            raise ValueError("encoded and raw_input must share batch and sequence dimensions")

        # Frame is considered active when any landmark feature is non-zero.
        active_mask = (raw_input.abs().sum(dim=-1) > eps).float()  # [batch, seq]
        weights = active_mask.unsqueeze(-1)  # [batch, seq, 1]
        weighted_sum = (encoded * weights).sum(dim=1)  # [batch, d_model]
        active_count = active_mask.sum(dim=1, keepdim=True)  # [batch, 1]

        # If all frames are inactive, fall back to standard mean pooling.
        pooled_active = weighted_sum / active_count.clamp_min(1.0)
        pooled_mean = encoded.mean(dim=1)
        has_active = (active_count > 0).expand_as(pooled_active)
        return torch.where(has_active, pooled_active, pooled_mean)

    def set_inference_mode(self) -> None:
        """Set model to inference mode with no gradients."""
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
