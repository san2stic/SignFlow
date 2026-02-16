"""Shadow mode comparison utilities."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import structlog

from app.ml.pipeline import Prediction

logger = structlog.get_logger(__name__)


@dataclass
class ShadowComparison:
    """Result of one primary vs shadow prediction comparison."""

    disagreed: bool
    high_confidence_disagreement: bool
    confidence_gap: float
    primary_prediction: str
    shadow_prediction: str


class ShadowModeEvaluator:
    """Compare primary and shadow predictions for silent model evaluation."""

    def __init__(
        self,
        *,
        min_confidence: float = 0.6,
        max_recent: int = 500,
    ) -> None:
        self.min_confidence = float(max(0.0, min(1.0, min_confidence)))
        self.recent: deque[ShadowComparison] = deque(maxlen=max(50, int(max_recent)))
        self.total = 0
        self.disagreements = 0
        self.high_confidence_disagreements = 0

    def compare(self, *, primary: Prediction, shadow: Prediction) -> ShadowComparison:
        """Compare a primary and shadow prediction result."""
        self.total += 1
        primary_label = str(primary.prediction)
        shadow_label = str(shadow.prediction)
        confidence_gap = abs(float(primary.confidence) - float(shadow.confidence))

        comparable = (
            primary_label not in {"RECORDING"}
            and shadow_label not in {"RECORDING"}
        )
        disagreed = comparable and primary_label != shadow_label
        high_confidence_disagreement = (
            disagreed
            and float(primary.confidence) >= self.min_confidence
            and float(shadow.confidence) >= self.min_confidence
        )

        if disagreed:
            self.disagreements += 1
        if high_confidence_disagreement:
            self.high_confidence_disagreements += 1

        result = ShadowComparison(
            disagreed=bool(disagreed),
            high_confidence_disagreement=bool(high_confidence_disagreement),
            confidence_gap=float(confidence_gap),
            primary_prediction=primary_label,
            shadow_prediction=shadow_label,
        )
        self.recent.append(result)
        return result

    def snapshot(self) -> dict[str, float | int]:
        """Return aggregate shadow-evaluation counters."""
        disagreement_rate = float(self.disagreements / self.total) if self.total else 0.0
        high_conf_rate = (
            float(self.high_confidence_disagreements / self.total)
            if self.total
            else 0.0
        )
        return {
            "total": int(self.total),
            "disagreements": int(self.disagreements),
            "high_confidence_disagreements": int(self.high_confidence_disagreements),
            "disagreement_rate": round(disagreement_rate, 6),
            "high_confidence_disagreement_rate": round(high_conf_rate, 6),
        }
