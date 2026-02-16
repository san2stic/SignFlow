"""Tests for shadow-mode comparison evaluator."""

from __future__ import annotations

from app.api.shadow_mode import ShadowModeEvaluator
from app.ml.pipeline import Prediction


def _prediction(label: str, confidence: float) -> Prediction:
    return Prediction(
        prediction=label,
        confidence=confidence,
        alternatives=[],
        sentence_buffer="",
        is_sentence_complete=False,
        decision_diagnostics={"status": "accepted"},
    )


def test_shadow_mode_detects_high_confidence_disagreement() -> None:
    """Evaluator should flag high-confidence disagreement when labels diverge."""
    evaluator = ShadowModeEvaluator(min_confidence=0.6)
    result = evaluator.compare(
        primary=_prediction("bonjour", 0.91),
        shadow=_prediction("merci", 0.87),
    )

    assert result.disagreed is True
    assert result.high_confidence_disagreement is True
    snapshot = evaluator.snapshot()
    assert snapshot["disagreements"] == 1
    assert snapshot["high_confidence_disagreements"] == 1


def test_shadow_mode_ignores_recording_state_disagreement() -> None:
    """RECORDING pseudo-state should not be counted as semantic disagreement."""
    evaluator = ShadowModeEvaluator(min_confidence=0.6)
    result = evaluator.compare(
        primary=_prediction("RECORDING", 0.0),
        shadow=_prediction("bonjour", 0.82),
    )

    assert result.disagreed is False
    assert result.high_confidence_disagreement is False
