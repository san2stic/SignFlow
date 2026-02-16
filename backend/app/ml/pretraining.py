"""Self-supervised pretraining strategies for landmark encoders.

Implements Masked Landmark Modeling (MLM) - similar to BERT's masked language modeling
but for 3D landmark sequences. This allows pretraining on unlabeled video data before
fine-tuning on labeled sign language datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = structlog.get_logger(__name__)


@dataclass
class MaskedLandmarkConfig:
    """Configuration for masked landmark modeling pretraining."""

    mask_prob: float = 0.15  # Probability of masking a landmark
    mask_token_prob: float = 0.8  # Of masked tokens, 80% replaced with mask token
    random_token_prob: float = 0.1  # 10% replaced with random values
    unchanged_prob: float = 0.1  # 10% left unchanged

    num_landmarks: int = 75  # Total number of landmarks
    landmark_dim: int = 3  # (x, y, z)

    reconstruction_loss_weight: float = 1.0
    temporal_smoothness_weight: float = 0.1  # Encourage temporal consistency

    def __post_init__(self):
        """Validate probabilities sum to 1.0."""
        total = self.mask_token_prob + self.random_token_prob + self.unchanged_prob
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Mask token probs must sum to 1.0, got {total}. "
                f"mask_token={self.mask_token_prob}, "
                f"random_token={self.random_token_prob}, "
                f"unchanged={self.unchanged_prob}"
            )


class MaskedLandmarkModel(nn.Module):
    """
    Wrapper for self-supervised pretraining with masked landmark modeling.

    Architecture:
        1. Mask random landmarks in input sequence
        2. Encode with spatial-temporal encoder
        3. Reconstruct masked landmarks with dedicated head
        4. Train on MSE loss between original and reconstructed landmarks

    This allows pretraining on unlabeled videos before fine-tuning on labeled data.
    """

    def __init__(
        self,
        encoder: nn.Module,
        config: MaskedLandmarkConfig,
        encoder_output_dim: int = 256,
    ):
        """
        Args:
            encoder: Spatial-temporal encoder (e.g., SpatialTemporalTransformer)
                     Must have extract_embeddings(x) -> [batch, d_model] method
            config: Masking configuration
            encoder_output_dim: Output dimension of encoder embeddings
        """
        super().__init__()
        self.encoder = encoder
        self.config = config

        # Learnable mask token
        self.mask_token = nn.Parameter(
            torch.randn(1, 1, config.num_landmarks * config.landmark_dim)
        )
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

        # Reconstruction head
        # Maps encoder embeddings back to landmark space
        self.reconstruction_head = nn.Sequential(
            nn.Linear(encoder_output_dim, encoder_output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(encoder_output_dim * 2, config.num_landmarks * config.landmark_dim),
        )

    def forward(
        self, x: Tensor, return_loss: bool = True
    ) -> dict[str, Tensor] | Tensor:
        """
        Forward pass with optional masking and loss computation.

        Args:
            x: [batch, seq, num_landmarks * landmark_dim] raw landmarks
            return_loss: If True, return loss dict. If False, return reconstructed landmarks.

        Returns:
            If return_loss=True:
                {
                    "loss": total loss,
                    "reconstruction_loss": MSE loss,
                    "temporal_smoothness_loss": temporal consistency loss,
                    "masked_predictions": reconstructed masked landmarks
                }
            If return_loss=False:
                [batch, seq, num_landmarks * landmark_dim] reconstructed landmarks
        """
        batch_size, seq_len, feature_dim = x.shape

        # Save original for loss computation
        original_landmarks = x.clone()

        # Apply masking
        masked_input, mask_indices = self._apply_masking(x)

        # Encode masked input
        # NOTE: Assumes encoder has extract_embeddings method
        if hasattr(self.encoder, "extract_embeddings"):
            embeddings = self.encoder.extract_embeddings(masked_input)
        else:
            # Fallback: use forward pass (may not work for all encoders)
            logger.warning(
                "encoder_missing_extract_embeddings",
                msg="Encoder missing extract_embeddings method, using forward pass",
            )
            embeddings = self.encoder(masked_input)

        # Reconstruct landmarks
        reconstructed = self.reconstruction_head(embeddings)  # [batch, feature_dim]

        # Expand to sequence dimension
        # Note: This simple version reconstructs from global representation
        # More advanced: use sequence-to-sequence reconstruction
        reconstructed = reconstructed.unsqueeze(1).expand(-1, seq_len, -1)

        if not return_loss:
            return reconstructed

        # Compute reconstruction loss only on masked positions
        reconstruction_loss = self._compute_reconstruction_loss(
            original_landmarks, reconstructed, mask_indices
        )

        # Temporal smoothness loss (encourage smooth motion)
        temporal_smoothness_loss = self._compute_temporal_smoothness_loss(reconstructed)

        # Total loss
        total_loss = (
            self.config.reconstruction_loss_weight * reconstruction_loss
            + self.config.temporal_smoothness_weight * temporal_smoothness_loss
        )

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "temporal_smoothness_loss": temporal_smoothness_loss,
            "masked_predictions": reconstructed,
        }

    def _apply_masking(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Apply random masking to input landmarks.

        Strategy (BERT-style):
        - mask_prob% of landmarks are selected for masking
        - Of these:
            - mask_token_prob% replaced with learnable mask token
            - random_token_prob% replaced with random values
            - unchanged_prob% left unchanged (for robustness)

        Args:
            x: [batch, seq, feature_dim] input landmarks

        Returns:
            masked_input: [batch, seq, feature_dim] masked landmarks
            mask_indices: [batch, seq] bool tensor (True for masked positions)
        """
        batch_size, seq_len, feature_dim = x.shape
        device = x.device

        # Random mask selection (per frame)
        mask_prob = self.config.mask_prob
        mask_indices = torch.rand(batch_size, seq_len, device=device) < mask_prob

        # Copy input for masking
        masked_input = x.clone()

        # For each masked position, decide replacement strategy
        num_masked = mask_indices.sum()
        if num_masked == 0:
            return masked_input, mask_indices

        # Generate random numbers for each masked position
        replacement_probs = torch.rand(num_masked, device=device)

        # Cumulative probabilities for strategy selection
        mask_token_threshold = self.config.mask_token_prob
        random_token_threshold = mask_token_threshold + self.config.random_token_prob

        # Strategy 1: Replace with mask token
        use_mask_token = replacement_probs < mask_token_threshold
        mask_token_expanded = self.mask_token.expand(batch_size, seq_len, -1)
        masked_input[mask_indices] = torch.where(
            use_mask_token.unsqueeze(-1).expand(-1, feature_dim),
            mask_token_expanded[mask_indices],
            masked_input[mask_indices],
        )

        # Strategy 2: Replace with random values
        use_random = (replacement_probs >= mask_token_threshold) & (
            replacement_probs < random_token_threshold
        )
        random_values = torch.randn_like(masked_input[mask_indices]) * 0.1
        masked_input[mask_indices] = torch.where(
            use_random.unsqueeze(-1).expand(-1, feature_dim),
            random_values,
            masked_input[mask_indices],
        )

        # Strategy 3: Leave unchanged (implicit - already in masked_input)

        return masked_input, mask_indices

    def _compute_reconstruction_loss(
        self, original: Tensor, reconstructed: Tensor, mask_indices: Tensor
    ) -> Tensor:
        """
        Compute MSE loss only on masked positions.

        Args:
            original: [batch, seq, feature_dim] original landmarks
            reconstructed: [batch, seq, feature_dim] reconstructed landmarks
            mask_indices: [batch, seq] bool mask

        Returns:
            Scalar MSE loss
        """
        # Compute MSE only on masked positions
        masked_original = original[mask_indices]
        masked_reconstructed = reconstructed[mask_indices]

        if masked_original.numel() == 0:
            # No masked positions
            return torch.tensor(0.0, device=original.device)

        loss = F.mse_loss(masked_reconstructed, masked_original)
        return loss

    def _compute_temporal_smoothness_loss(self, reconstructed: Tensor) -> Tensor:
        """
        Encourage temporal smoothness in reconstructed landmarks.

        Computes L2 norm of frame-to-frame differences.

        Args:
            reconstructed: [batch, seq, feature_dim] reconstructed landmarks

        Returns:
            Scalar smoothness loss
        """
        if reconstructed.size(1) <= 1:
            return torch.tensor(0.0, device=reconstructed.device)

        # Frame-to-frame differences
        diffs = reconstructed[:, 1:] - reconstructed[:, :-1]

        # L2 norm of differences
        smoothness_loss = (diffs**2).mean()

        return smoothness_loss


class SequenceToSequenceMaskedLandmarkModel(nn.Module):
    """
    Advanced masked landmark model with sequence-to-sequence reconstruction.

    Unlike MaskedLandmarkModel which reconstructs from global representation,
    this version uses a decoder to reconstruct each frame independently.

    Better for capturing fine-grained temporal dynamics.
    """

    def __init__(
        self,
        encoder: nn.Module,
        config: MaskedLandmarkConfig,
        encoder_output_dim: int = 256,
        decoder_hidden_dim: int = 512,
        decoder_num_layers: int = 2,
    ):
        super().__init__()
        self.encoder = encoder
        self.config = config

        # Mask token
        self.mask_token = nn.Parameter(
            torch.randn(1, 1, config.num_landmarks * config.landmark_dim)
        )
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

        # Transformer decoder for sequence-to-sequence reconstruction
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=encoder_output_dim,
            nhead=8,
            dim_feedforward=decoder_hidden_dim,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_num_layers)

        # Output projection to landmark space
        self.output_proj = nn.Linear(
            encoder_output_dim, config.num_landmarks * config.landmark_dim
        )

    def forward(self, x: Tensor, return_loss: bool = True) -> dict[str, Tensor] | Tensor:
        """
        Forward pass with sequence-to-sequence reconstruction.

        Args:
            x: [batch, seq, num_landmarks * landmark_dim]
            return_loss: Whether to return loss dict

        Returns:
            Loss dict or reconstructed sequence
        """
        original_landmarks = x.clone()

        # Apply masking
        masked_input, mask_indices = self._apply_masking(x)

        # Encode full sequence
        # NOTE: This requires encoder to return sequence representations
        # Not just global pooled representation
        if hasattr(self.encoder, "encode_sequence"):
            encoded_sequence = self.encoder.encode_sequence(masked_input)
        else:
            logger.error(
                "encoder_missing_encode_sequence",
                msg="Encoder must have encode_sequence method for seq2seq reconstruction",
            )
            raise NotImplementedError(
                "Encoder must implement encode_sequence(x) -> [batch, seq, d_model]"
            )

        # Decode sequence
        # Use masked input as decoder input (teacher forcing during training)
        decoded_sequence = self.decoder(
            tgt=encoded_sequence,
            memory=encoded_sequence,
        )

        # Project to landmark space
        reconstructed = self.output_proj(decoded_sequence)

        if not return_loss:
            return reconstructed

        # Compute losses
        reconstruction_loss = self._compute_reconstruction_loss(
            original_landmarks, reconstructed, mask_indices
        )

        temporal_smoothness_loss = self._compute_temporal_smoothness_loss(reconstructed)

        total_loss = (
            self.config.reconstruction_loss_weight * reconstruction_loss
            + self.config.temporal_smoothness_weight * temporal_smoothness_loss
        )

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "temporal_smoothness_loss": temporal_smoothness_loss,
            "masked_predictions": reconstructed,
        }

    def _apply_masking(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Apply BERT-style masking (same as MaskedLandmarkModel)."""
        batch_size, seq_len, feature_dim = x.shape
        device = x.device

        mask_indices = torch.rand(batch_size, seq_len, device=device) < self.config.mask_prob
        masked_input = x.clone()

        num_masked = mask_indices.sum()
        if num_masked == 0:
            return masked_input, mask_indices

        replacement_probs = torch.rand(num_masked, device=device)
        mask_token_threshold = self.config.mask_token_prob
        random_token_threshold = mask_token_threshold + self.config.random_token_prob

        use_mask_token = replacement_probs < mask_token_threshold
        mask_token_expanded = self.mask_token.expand(batch_size, seq_len, -1)
        masked_input[mask_indices] = torch.where(
            use_mask_token.unsqueeze(-1).expand(-1, feature_dim),
            mask_token_expanded[mask_indices],
            masked_input[mask_indices],
        )

        use_random = (replacement_probs >= mask_token_threshold) & (
            replacement_probs < random_token_threshold
        )
        random_values = torch.randn_like(masked_input[mask_indices]) * 0.1
        masked_input[mask_indices] = torch.where(
            use_random.unsqueeze(-1).expand(-1, feature_dim),
            random_values,
            masked_input[mask_indices],
        )

        return masked_input, mask_indices

    def _compute_reconstruction_loss(
        self, original: Tensor, reconstructed: Tensor, mask_indices: Tensor
    ) -> Tensor:
        """MSE loss on masked positions."""
        masked_original = original[mask_indices]
        masked_reconstructed = reconstructed[mask_indices]

        if masked_original.numel() == 0:
            return torch.tensor(0.0, device=original.device)

        return F.mse_loss(masked_reconstructed, masked_original)

    def _compute_temporal_smoothness_loss(self, reconstructed: Tensor) -> Tensor:
        """Temporal smoothness loss."""
        if reconstructed.size(1) <= 1:
            return torch.tensor(0.0, device=reconstructed.device)

        diffs = reconstructed[:, 1:] - reconstructed[:, :-1]
        return (diffs**2).mean()
