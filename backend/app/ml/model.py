"""Lightweight transformer model for landmark sequence classification."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


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
        num_features: int = 225,  # Default: 21*3*2 (hands) + 33*3 (pose) = 225
        num_classes: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Store architecture parameters
        self.num_features = num_features
        self.num_classes = num_classes
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        self.embedding = nn.Linear(num_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Compute logits from [batch, seq, features] input tensor."""
        pooled = self.extract_embeddings(x)
        return self.classifier(pooled)

    def extract_embeddings(self, x: Tensor) -> Tensor:
        """Return pooled transformer embeddings before classification head."""
        raw_input = x
        encoded = self.embedding(raw_input)
        encoded = self.positional_encoding(encoded)
        encoded = self.encoder(encoded)
        return self.masked_temporal_pool(encoded, raw_input)

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
