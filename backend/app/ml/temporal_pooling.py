"""Temporal pooling strategies for variable-length sequences.

This module provides temporal convolution networks (TCN) and adaptive pooling
to handle variable-length sign language sequences without losing temporal dynamics.

Replaces naive temporal resampling with learned temporal aggregation that preserves
speed variations critical for sign language understanding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TemporalConvBlock(nn.Module):
    """
    Single temporal convolution block with residual connection.

    Uses dilated convolutions to capture multi-scale temporal patterns.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2  # Same padding

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)

        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [batch, channels, seq_len]

        Returns:
            [batch, channels, seq_len]
        """
        identity = self.residual(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = F.gelu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + identity  # Residual connection
        out = F.gelu(out)

        return out


class TemporalConvPooling(nn.Module):
    """
    Multi-scale temporal convolution network with adaptive pooling.

    Architecture:
        - Multi-scale dilated convolutions (dilation 1, 2, 4)
        - Captures short-term, medium-term, and long-term temporal dependencies
        - Adaptive pooling to fixed output length (e.g., 64 tokens)
        - Preserves dynamic temporal information unlike naive interpolation

    This replaces dataset.py:temporal_resample() which loses speed variation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_length: int = 64,
        kernel_size: int = 3,
        num_scales: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_length = output_length

        # Input projection
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

        # Multi-scale temporal convolution blocks
        self.conv_blocks = nn.ModuleList()
        dilations = [2**i for i in range(num_scales)]  # [1, 2, 4, ...]

        for dilation in dilations:
            self.conv_blocks.append(
                TemporalConvBlock(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

        # Fusion layer to combine multi-scale features
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_dim * num_scales, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        # Adaptive pooling to fixed length
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_length)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_dim, input_dim, kernel_size=1),
            nn.LayerNorm([input_dim, output_length]),
        )

    def forward(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        """
        Pool variable-length sequences to fixed length.

        Args:
            x: [batch, seq_len, input_dim] input sequences
            lengths: [batch] actual lengths (optional, for masking)

        Returns:
            [batch, output_length, input_dim] pooled sequences
        """
        # Transpose to channel-first: [batch, input_dim, seq_len]
        x = x.transpose(1, 2)

        # Input projection
        h = self.input_proj(x)  # [batch, hidden_dim, seq_len]

        # Apply mask for padding if lengths provided
        if lengths is not None:
            mask = self._create_mask(h.size(0), h.size(2), lengths, device=h.device)
            h = h * mask

        # Multi-scale convolutions
        multi_scale_features = []
        for conv_block in self.conv_blocks:
            scale_features = conv_block(h)
            multi_scale_features.append(scale_features)

        # Concatenate multi-scale features
        h = torch.cat(multi_scale_features, dim=1)  # [batch, hidden_dim*num_scales, seq_len]

        # Fuse multi-scale features
        h = self.fusion(h)  # [batch, hidden_dim, seq_len]

        # Adaptive pooling to fixed length
        h = self.adaptive_pool(h)  # [batch, hidden_dim, output_length]

        # Output projection
        h = self.output_proj(h)  # [batch, input_dim, output_length]

        # Transpose back to [batch, output_length, input_dim]
        h = h.transpose(1, 2)

        return h

    @staticmethod
    def _create_mask(
        batch_size: int, seq_len: int, lengths: Tensor, device: torch.device
    ) -> Tensor:
        """
        Create binary mask for variable-length sequences.

        Args:
            batch_size: Batch size
            seq_len: Maximum sequence length
            lengths: [batch] actual lengths
            device: Device to place tensor on

        Returns:
            [batch, 1, seq_len] binary mask (1 for valid positions, 0 for padding)
        """
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        mask = (positions < lengths.unsqueeze(1)).unsqueeze(1).float()
        return mask


class LearnedTemporalPooling(nn.Module):
    """
    Attention-based temporal pooling for variable-length sequences.

    Uses self-attention to learn which frames are most important for classification,
    then pools adaptively based on learned importance weights.

    More flexible than TCN pooling but higher computational cost.
    """

    def __init__(
        self,
        input_dim: int,
        output_length: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_length = output_length

        # Learnable query tokens for pooling
        self.query_tokens = nn.Parameter(torch.randn(1, output_length, input_dim))
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)

        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer normalization
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        """
        Pool variable-length sequences with learned attention.

        Args:
            x: [batch, seq_len, input_dim] input sequences
            key_padding_mask: [batch, seq_len] bool mask (True for padding)

        Returns:
            [batch, output_length, input_dim] pooled sequences
        """
        batch_size = x.size(0)

        # Expand query tokens for batch
        queries = self.query_tokens.expand(batch_size, -1, -1)

        # Cross-attention: queries attend to input sequence
        pooled, _ = self.cross_attention(
            query=queries,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
        )

        # Layer normalization
        pooled = self.norm(pooled)

        return pooled


class HybridTemporalPooling(nn.Module):
    """
    Hybrid pooling combining TCN and attention-based methods.

    Uses TCN for coarse temporal aggregation, then attention for fine-grained selection.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        intermediate_length: int = 128,
        output_length: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tcn_pooling = TemporalConvPooling(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_length=intermediate_length,
            dropout=dropout,
        )

        self.attention_pooling = LearnedTemporalPooling(
            input_dim=input_dim,
            output_length=output_length,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim] input sequences
            lengths: [batch] actual lengths (optional)

        Returns:
            [batch, output_length, input_dim] pooled sequences
        """
        # First stage: TCN coarse pooling
        h = self.tcn_pooling(x, lengths)

        # Second stage: Attention fine-grained pooling
        h = self.attention_pooling(h)

        return h


def create_temporal_pooler(
    input_dim: int,
    target_length: int,
    method: str = "tcn",
    **kwargs,
) -> nn.Module:
    """
    Factory function to create temporal pooling modules.

    Args:
        input_dim: Input feature dimension
        target_length: Desired output length
        method: Pooling method ("tcn", "attention", "hybrid")
        **kwargs: Additional arguments for pooling module

    Returns:
        Temporal pooling module
    """
    if method == "tcn":
        return TemporalConvPooling(
            input_dim=input_dim, output_length=target_length, **kwargs
        )
    elif method == "attention":
        return LearnedTemporalPooling(
            input_dim=input_dim, output_length=target_length, **kwargs
        )
    elif method == "hybrid":
        return HybridTemporalPooling(
            input_dim=input_dim, output_length=target_length, **kwargs
        )
    else:
        raise ValueError(f"Unknown pooling method: {method}")
