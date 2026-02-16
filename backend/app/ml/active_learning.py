"""Active-learning utilities for uncertainty sampling during inference."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from time import time
from typing import Literal
from uuid import uuid4

import numpy as np

ActiveLearningStrategy = Literal["entropy", "margin", "combined"]


@dataclass(frozen=True)
class ActiveLearningSample:
    """One uncertain prediction candidate queued for human annotation."""

    id: str
    created_at: float
    uncertainty: float
    strategy: ActiveLearningStrategy
    prediction: str
    confidence: float
    margin: float
    entropy: float
    alternatives: list[dict[str, float]]
    sentence_buffer: str
    frame_idx: int | None = None
    timestamp: float | None = None
    route: str = "production"
    model_version: str = "unknown"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable payload for APIs and websocket hints."""
        return {
            "id": self.id,
            "created_at": round(float(self.created_at), 6),
            "uncertainty": round(float(self.uncertainty), 4),
            "strategy": self.strategy,
            "prediction": self.prediction,
            "confidence": round(float(self.confidence), 4),
            "margin": round(float(self.margin), 4),
            "entropy": round(float(self.entropy), 4),
            "alternatives": self.alternatives,
            "sentence_buffer": self.sentence_buffer,
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
            "route": self.route,
            "model_version": self.model_version,
        }


def _normalize_confidence(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _top2_margin(confidence: float, alternatives: list[dict[str, float]]) -> float:
    top1 = _normalize_confidence(confidence)
    top2 = 0.0
    for alt in alternatives:
        score = alt.get("confidence", 0.0)
        try:
            top2 = max(top2, _normalize_confidence(float(score)))
        except (TypeError, ValueError):
            continue
    return float(np.clip(top1 - top2, 0.0, 1.0))


def _approx_normalized_entropy(confidence: float, alternatives: list[dict[str, float]]) -> float:
    """Approximate entropy from top-1 + alternatives (+remainder mass)."""
    probs: list[float] = [_normalize_confidence(confidence)]

    for alt in alternatives[:4]:
        score = alt.get("confidence", 0.0)
        try:
            probs.append(_normalize_confidence(float(score)))
        except (TypeError, ValueError):
            continue

    probs = [p for p in probs if p > 0.0]
    total = float(sum(probs))
    if total <= 1e-9:
        return 0.0

    if total < 1.0:
        probs.append(1.0 - total)
    elif total > 1.0:
        probs = [p / total for p in probs]

    arr = np.asarray(probs, dtype=np.float32)
    entropy = -float(np.sum(arr * np.log(arr + 1e-12)))
    max_entropy = float(np.log(max(arr.size, 2)))
    return float(np.clip(entropy / max_entropy, 0.0, 1.0))


def uncertainty_score(
    *,
    strategy: ActiveLearningStrategy,
    confidence: float,
    alternatives: list[dict[str, float]],
) -> tuple[float, float, float]:
    """
    Compute uncertainty score in [0, 1] from prediction confidences.

    Returns:
        tuple: (uncertainty, margin, entropy)
    """
    normalized_conf = _normalize_confidence(confidence)
    margin = _top2_margin(normalized_conf, alternatives)
    entropy = _approx_normalized_entropy(normalized_conf, alternatives)

    if strategy == "entropy":
        score = entropy
    elif strategy == "margin":
        score = 1.0 - margin
    else:
        score = (
            (0.5 * (1.0 - normalized_conf))
            + (0.3 * entropy)
            + (0.2 * (1.0 - margin))
        )

    return float(np.clip(score, 0.0, 1.0)), margin, entropy


class ActiveLearningQueue:
    """In-memory queue storing most uncertain samples for annotation workflows."""

    def __init__(
        self,
        *,
        strategy: ActiveLearningStrategy = "combined",
        min_uncertainty: float = 0.6,
        max_size: int = 2000,
        top_n: int = 250,
        cooldown_seconds: float = 1.5,
    ) -> None:
        self.strategy: ActiveLearningStrategy = strategy
        self.min_uncertainty = float(np.clip(min_uncertainty, 0.0, 1.0))
        self.max_size = max(1, int(max_size))
        self.top_n = max(1, int(top_n))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))

        self._lock = Lock()
        self._samples_by_id: dict[str, ActiveLearningSample] = {}
        self._arrival_order: list[str] = []
        self._last_enqueue_time = 0.0
        self._last_signature = ""

    def consider(
        self,
        *,
        prediction: str,
        confidence: float,
        alternatives: list[dict[str, float]] | None = None,
        sentence_buffer: str = "",
        frame_idx: int | None = None,
        timestamp: float | None = None,
        route: str = "production",
        model_version: str = "unknown",
    ) -> ActiveLearningSample | None:
        """Evaluate one prediction and queue it if uncertainty is high enough."""
        if prediction == "RECORDING":
            return None

        alt_values = alternatives or []
        if prediction == "NONE" and confidence <= 0.0 and not alt_values:
            return None

        uncertainty, margin, entropy = uncertainty_score(
            strategy=self.strategy,
            confidence=confidence,
            alternatives=alt_values,
        )
        if uncertainty < self.min_uncertainty:
            return None

        now = time()
        signature = f"{prediction}:{round(float(confidence), 3)}:{round(margin, 3)}"
        with self._lock:
            if (
                self.cooldown_seconds > 0.0
                and signature == self._last_signature
                and (now - self._last_enqueue_time) < self.cooldown_seconds
            ):
                return None

            sample = ActiveLearningSample(
                id=str(uuid4()),
                created_at=now,
                uncertainty=uncertainty,
                strategy=self.strategy,
                prediction=prediction,
                confidence=_normalize_confidence(confidence),
                margin=margin,
                entropy=entropy,
                alternatives=alt_values[:4],
                sentence_buffer=sentence_buffer,
                frame_idx=frame_idx,
                timestamp=timestamp,
                route=route,
                model_version=model_version,
            )
            self._samples_by_id[sample.id] = sample
            self._arrival_order.append(sample.id)
            self._last_enqueue_time = now
            self._last_signature = signature
            self._trim_unlocked()
            return sample

    def top_uncertain(self, *, limit: int | None = None) -> list[ActiveLearningSample]:
        """Return highest-uncertainty candidates, most recent first on ties."""
        with self._lock:
            items = list(self._samples_by_id.values())

        if not items:
            return []

        cap = self.top_n if limit is None else max(1, min(int(limit), self.top_n))
        ranked = sorted(
            items,
            key=lambda item: (item.uncertainty, item.created_at),
            reverse=True,
        )
        return ranked[:cap]

    def resolve(self, sample_id: str) -> ActiveLearningSample | None:
        """Remove one queued sample after annotation/decision."""
        with self._lock:
            sample = self._samples_by_id.pop(sample_id, None)
            if sample is None:
                return None
            self._arrival_order = [item for item in self._arrival_order if item != sample_id]
            return sample

    def snapshot(self) -> dict[str, object]:
        """Expose queue internals for debugging and monitoring."""
        with self._lock:
            queue_size = len(self._samples_by_id)
            oldest_ts = (
                self._samples_by_id[self._arrival_order[0]].created_at
                if self._arrival_order
                else None
            )
        return {
            "strategy": self.strategy,
            "min_uncertainty": self.min_uncertainty,
            "queue_size": queue_size,
            "max_size": self.max_size,
            "top_n": self.top_n,
            "cooldown_seconds": self.cooldown_seconds,
            "oldest_sample_timestamp": oldest_ts,
        }

    def _trim_unlocked(self) -> None:
        while len(self._arrival_order) > self.max_size:
            oldest_id = self._arrival_order.pop(0)
            self._samples_by_id.pop(oldest_id, None)
