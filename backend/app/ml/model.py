"""Transformer model for landmark sequence classification."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
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
        num_features: int = ENRICHED_FEATURE_DIM,
        num_classes: int,
        d_model: int = 192,
        nhead: int = 6,
        num_layers: int = 4,
        dim_feedforward: int = 768,
        dropout: float = 0.2,
        feature_dropout: float = 0.15,
        pooling_dropout: float = 0.2,
        use_cls_token: bool = True,
        token_dropout: float = 0.05,
        temporal_smoothing: float = 0.1,
        use_multiscale_stem: bool = True,
        use_cosine_head: bool = True,
        relative_bias_max_distance: int = 64,
        cosine_head_weight: float = 0.35,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            fallback_heads = [head for head in (8, 6, 4, 3, 2, 1) if d_model % head == 0]
            nhead = fallback_heads[0] if fallback_heads else 1

        # Store architecture parameters
        self.num_features = num_features
        self.num_classes = num_classes
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.feature_dropout = feature_dropout
        self.pooling_dropout_value = pooling_dropout
        self.use_cls_token = use_cls_token
        self.token_dropout = max(0.0, min(0.5, float(token_dropout)))
        self.temporal_smoothing = max(0.0, min(1.0, float(temporal_smoothing)))
        self.use_multiscale_stem = bool(use_multiscale_stem)
        self.use_cosine_head = bool(use_cosine_head)
        self.relative_bias_max_distance = max(1, int(relative_bias_max_distance))
        self.cosine_head_weight = float(max(0.0, min(1.0, cosine_head_weight)))

        self.input_norm = nn.LayerNorm(num_features)
        self.embedding = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.GELU(),
            nn.Dropout(feature_dropout),
        )
        if self.use_multiscale_stem:
            base = d_model // 3
            channels = [base, base, d_model - (2 * base)]
            kernels = [3, 5, 7]
            self.temporal_stem = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv1d(num_features, out_channels, kernel_size=kernel, padding=kernel // 2),
                        nn.GELU(),
                    )
                    for kernel, out_channels in zip(kernels, channels)
                ]
            )
            self.stem_fusion = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(feature_dropout),
            )

        self.positional_encoding = PositionalEncoding(d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.relative_attention_bias = nn.Parameter(
            torch.zeros((2 * self.relative_bias_max_distance) + 1)
        )

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
        self.metric_projection = nn.Linear(d_model, d_model)
        self.class_prototypes = nn.Parameter(torch.randn(num_classes, d_model))
        self.cosine_logit_scale = nn.Parameter(torch.tensor(10.0))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.class_prototypes, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """Compute logits from [batch, seq, features] input tensor."""
        pooled = self.extract_embeddings(x)
        linear_logits = self.classifier(pooled)
        if not self.use_cosine_head:
            return linear_logits

        embeddings = F.normalize(self.metric_projection(pooled), dim=-1)
        prototypes = F.normalize(self.class_prototypes, dim=-1)
        scale = torch.clamp(self.cosine_logit_scale, min=1.0, max=50.0)
        cosine_logits = scale * torch.matmul(embeddings, prototypes.transpose(0, 1))
        blend = self.cosine_head_weight
        return ((1.0 - blend) * linear_logits) + (blend * cosine_logits)

    def extract_embeddings(self, x: Tensor) -> Tensor:
        """Return pooled transformer embeddings before classification head."""
        raw_input = x
        active_mask = self._active_mask(raw_input)

        encoded = self.input_norm(raw_input)
        encoded = self.embedding(encoded)
        if self.use_multiscale_stem:
            stem_features = self._apply_multiscale_temporal_stem(raw_input)
            encoded = self.stem_fusion(torch.cat([encoded, stem_features], dim=-1))
        encoded = self._apply_token_dropout(encoded, active_mask)

        if self.use_cls_token:
            batch_size = encoded.size(0)
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            encoded = torch.cat([cls_token, encoded], dim=1)

        encoded = self.positional_encoding(encoded)
        encoded = self._apply_temporal_smoothing(encoded)
        key_padding_mask = self._build_key_padding_mask(active_mask, device=encoded.device)
        attention_bias = self._build_relative_attention_bias(encoded.size(1), device=encoded.device)
        encoded = self.encoder(
            encoded,
            mask=attention_bias,
            src_key_padding_mask=key_padding_mask,
        )
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

    def _apply_multiscale_temporal_stem(self, raw_input: Tensor) -> Tensor:
        """Encode temporal patterns at multiple receptive fields."""
        if raw_input.ndim != 3 or raw_input.size(1) < 2:
            return self.embedding(self.input_norm(raw_input))
        sequence_first = raw_input.transpose(1, 2)
        features = [branch(sequence_first) for branch in self.temporal_stem]
        merged = torch.cat(features, dim=1)
        return merged.transpose(1, 2)

    def _build_relative_attention_bias(self, seq_len: int, device: torch.device) -> Tensor | None:
        """Build shared relative attention bias matrix for transformer self-attention."""
        if seq_len <= 1:
            return None

        positions = torch.arange(seq_len, device=device)
        relative = positions[:, None] - positions[None, :]
        relative = torch.clamp(
            relative,
            min=-self.relative_bias_max_distance,
            max=self.relative_bias_max_distance,
        )
        relative = relative + self.relative_bias_max_distance
        bias = self.relative_attention_bias[relative].clone()
        if self.use_cls_token:
            bias[0, :] = 0.0
            bias[:, 0] = 0.0
        return bias

    def _apply_token_dropout(self, encoded: Tensor, active_mask: Tensor) -> Tensor:
        """Randomly drop active temporal tokens during training for robustness."""
        if not self.training or self.token_dropout <= 0.0:
            return encoded
        if encoded.ndim != 3 or active_mask.ndim != 2:
            return encoded

        drop_prob = self.token_dropout
        random_mask = torch.rand(
            (encoded.size(0), encoded.size(1)),
            device=encoded.device,
        ) < drop_prob
        drop_mask = random_mask & active_mask
        if not torch.any(drop_mask):
            return encoded

        return encoded.masked_fill(drop_mask.unsqueeze(-1), 0.0)

    def _apply_temporal_smoothing(self, encoded: Tensor) -> Tensor:
        """Inject local temporal smoothing before attention for motion continuity."""
        if self.temporal_smoothing <= 0.0 or encoded.ndim != 3 or encoded.size(1) < 3:
            return encoded

        if self.use_cls_token:
            cls_token = encoded[:, :1]
            sequence = encoded[:, 1:]
            smoothed = F.avg_pool1d(
                sequence.transpose(1, 2),
                kernel_size=3,
                stride=1,
                padding=1,
            ).transpose(1, 2)
            mixed = (1.0 - self.temporal_smoothing) * sequence + self.temporal_smoothing * smoothed
            return torch.cat([cls_token, mixed], dim=1)

        smoothed = F.avg_pool1d(
            encoded.transpose(1, 2),
            kernel_size=3,
            stride=1,
            padding=1,
        ).transpose(1, 2)
        return (1.0 - self.temporal_smoothing) * encoded + self.temporal_smoothing * smoothed

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
