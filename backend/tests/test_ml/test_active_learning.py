"""Tests for active-learning uncertainty sampling."""

from __future__ import annotations

from app.ml.active_learning import ActiveLearningQueue, uncertainty_score


def test_uncertainty_score_margin_strategy_prioritizes_small_top2_margin() -> None:
    """Margin strategy should produce high uncertainty when top-1 and top-2 are close."""
    uncertainty, margin, entropy = uncertainty_score(
        strategy="margin",
        confidence=0.58,
        alternatives=[{"sign": "lsfb_bonjour", "confidence": 0.55}],
    )

    assert 0.02 <= margin <= 0.04
    assert uncertainty > 0.9
    assert 0.0 <= entropy <= 1.0


def test_uncertainty_score_entropy_strategy_detects_flat_distribution() -> None:
    """Entropy strategy should be high for nearly uniform probabilities."""
    uncertainty, margin, entropy = uncertainty_score(
        strategy="entropy",
        confidence=0.34,
        alternatives=[
            {"sign": "lsfb_bonjour", "confidence": 0.33},
            {"sign": "lsfb_merci", "confidence": 0.33},
        ],
    )

    assert 0.0 <= margin <= 1.0
    assert entropy > 0.9
    assert uncertainty == entropy


def test_active_learning_queue_consider_and_resolve() -> None:
    """Queue should accept uncertain samples and remove them on resolve."""
    queue = ActiveLearningQueue(
        strategy="combined",
        min_uncertainty=0.6,
        cooldown_seconds=0.0,
        max_size=10,
        top_n=10,
    )

    not_selected = queue.consider(
        prediction="lsfb_bonjour",
        confidence=0.97,
        alternatives=[{"sign": "lsfb_merci", "confidence": 0.01}],
        sentence_buffer="bonjour",
    )
    assert not_selected is None

    selected = queue.consider(
        prediction="NONE",
        confidence=0.51,
        alternatives=[{"sign": "lsfb_bonjour", "confidence": 0.49}],
        sentence_buffer="",
        frame_idx=42,
    )
    assert selected is not None
    assert selected.frame_idx == 42

    top_items = queue.top_uncertain(limit=5)
    assert len(top_items) == 1
    assert top_items[0].id == selected.id

    resolved = queue.resolve(selected.id)
    assert resolved is not None
    assert resolved.id == selected.id
    assert queue.snapshot()["queue_size"] == 0


def test_active_learning_queue_trims_oldest_when_capacity_exceeded() -> None:
    """Queue should evict oldest entries once max_size is reached."""
    queue = ActiveLearningQueue(
        strategy="margin",
        min_uncertainty=0.0,
        cooldown_seconds=0.0,
        max_size=2,
        top_n=10,
    )

    sample_one = queue.consider(prediction="a", confidence=0.1, alternatives=[], sentence_buffer="")
    sample_two = queue.consider(prediction="b", confidence=0.2, alternatives=[], sentence_buffer="")
    sample_three = queue.consider(prediction="c", confidence=0.3, alternatives=[], sentence_buffer="")

    assert sample_one is not None
    assert sample_two is not None
    assert sample_three is not None
    assert queue.snapshot()["queue_size"] == 2
    assert queue.resolve(sample_one.id) is None
