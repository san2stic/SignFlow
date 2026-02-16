"""Tests for distillation helpers."""

from __future__ import annotations

import torch

from app.ml.distillation import DistillationTrainer, build_student_model
from app.ml.model import SignTransformer


def test_distillation_trainer_combines_hard_and_soft_losses() -> None:
    """Combined loss should be positive and include soft-target guidance."""
    trainer = DistillationTrainer(temperature=4.0, alpha=0.5)
    student_logits = torch.tensor([[2.5, 0.4, -0.2]], dtype=torch.float32)
    teacher_logits = torch.tensor([[1.7, 0.9, -0.3]], dtype=torch.float32)
    hard_loss = torch.tensor(0.8, dtype=torch.float32)

    soft = trainer.distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
    )
    combined = trainer.combined_loss(
        hard_loss=hard_loss,
        student_logits=student_logits,
        teacher_logits=teacher_logits,
    )

    assert float(soft.item()) > 0.0
    assert float(combined.item()) > 0.0
    assert abs(float(combined.item()) - 0.8) > 1e-4


def test_build_student_model_preserves_io_topology() -> None:
    """Student model should keep teacher input/output spaces."""
    teacher = SignTransformer(num_features=469, num_classes=7, d_model=384, nhead=8, num_layers=6)
    student = build_student_model(
        teacher,
        d_model=192,
        nhead=6,
        num_layers=4,
        dim_feedforward=768,
    )

    assert student.num_features == teacher.num_features
    assert student.num_classes == teacher.num_classes
    assert student.d_model == 192
