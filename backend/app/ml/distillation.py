"""Knowledge-distillation helpers for teacher/student training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from app.ml.model import SignTransformer


@dataclass
class DistillationConfig:
    """Distillation loss controls."""

    temperature: float = 4.0
    alpha: float = 0.5


class DistillationTrainer:
    """Utility class to combine hard and soft targets during student training."""

    def __init__(self, *, temperature: float = 4.0, alpha: float = 0.5) -> None:
        self.config = DistillationConfig(
            temperature=float(max(1e-3, temperature)),
            alpha=float(np.clip(alpha, 0.0, 1.0)),
        )

    def distillation_loss(
        self,
        *,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence from teacher probabilities to student probabilities."""
        temperature = self.config.temperature
        student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        return kl * (temperature ** 2)

    def combined_loss(
        self,
        *,
        hard_loss: torch.Tensor,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Blend supervised hard loss and teacher-guided soft loss."""
        soft_loss = self.distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
        )
        alpha = self.config.alpha
        return ((1.0 - alpha) * hard_loss) + (alpha * soft_loss)


def build_student_model(
    teacher_model: SignTransformer,
    *,
    d_model: int = 192,
    num_layers: int = 4,
    nhead: int = 6,
    dim_feedforward: int = 768,
) -> SignTransformer:
    """Build a compact student initialized from teacher topology metadata."""
    student = SignTransformer(
        num_features=int(teacher_model.num_features),
        num_classes=int(teacher_model.num_classes),
        d_model=int(d_model),
        nhead=int(nhead),
        num_layers=int(num_layers),
        dim_feedforward=int(dim_feedforward),
        dropout=float(getattr(teacher_model, "dropout", 0.2)),
        feature_dropout=float(getattr(teacher_model, "feature_dropout", 0.15)),
        pooling_dropout=float(getattr(teacher_model, "pooling_dropout_value", 0.2)),
        use_cls_token=bool(getattr(teacher_model, "use_cls_token", True)),
        token_dropout=float(getattr(teacher_model, "token_dropout", 0.05)),
        temporal_smoothing=float(getattr(teacher_model, "temporal_smoothing", 0.1)),
        use_multiscale_stem=bool(getattr(teacher_model, "use_multiscale_stem", True)),
        use_cosine_head=bool(getattr(teacher_model, "use_cosine_head", True)),
        relative_bias_max_distance=int(getattr(teacher_model, "relative_bias_max_distance", 64)),
        cosine_head_weight=float(getattr(teacher_model, "cosine_head_weight", 0.35)),
    )
    return student
