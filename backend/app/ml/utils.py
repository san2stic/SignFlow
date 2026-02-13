"""ML helper functions."""

from __future__ import annotations


def estimate_remaining_time(progress: float) -> str:
    """Convert progress percentage into a simple ETA string."""
    if progress <= 0:
        return "~2m"
    if progress >= 100:
        return "0s"
    seconds = int((100 - progress) * 1.2)
    return f"~{seconds}s"
