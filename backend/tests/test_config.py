"""Configuration tests for TorchServe feature flags."""

from __future__ import annotations

from app.config import get_settings


def _reload_settings():
    get_settings.cache_clear()
    return get_settings()


def test_torchserve_settings_defaults(monkeypatch) -> None:
    """TorchServe settings should be safe-by-default (disabled)."""
    monkeypatch.delenv("USE_TORCHSERVE", raising=False)
    monkeypatch.delenv("TORCHSERVE_URL", raising=False)
    monkeypatch.delenv("TORCHSERVE_TIMEOUT_MS", raising=False)
    settings = _reload_settings()

    assert settings.use_torchserve is False
    assert settings.torchserve_url == "http://torchserve:8080"
    assert settings.torchserve_timeout_ms == 2000


def test_torchserve_settings_from_env(monkeypatch) -> None:
    """TorchServe settings should be configurable from environment variables."""
    monkeypatch.setenv("USE_TORCHSERVE", "true")
    monkeypatch.setenv("TORCHSERVE_URL", "http://localhost:18080")
    monkeypatch.setenv("TORCHSERVE_TIMEOUT_MS", "3500")

    settings = _reload_settings()
    assert settings.use_torchserve is True
    assert settings.torchserve_url == "http://localhost:18080"
    assert settings.torchserve_timeout_ms == 3500

    get_settings.cache_clear()


def test_active_learning_settings_defaults(monkeypatch) -> None:
    """Active learning should stay disabled by default."""
    monkeypatch.delenv("ACTIVE_LEARNING_ENABLED", raising=False)
    monkeypatch.delenv("ACTIVE_LEARNING_STRATEGY", raising=False)
    monkeypatch.delenv("ACTIVE_LEARNING_MIN_UNCERTAINTY", raising=False)
    settings = _reload_settings()

    assert settings.active_learning_enabled is False
    assert settings.active_learning_strategy == "combined"
    assert settings.active_learning_min_uncertainty == 0.6


def test_active_learning_settings_from_env(monkeypatch) -> None:
    """Active learning settings should be configurable from environment variables."""
    monkeypatch.setenv("ACTIVE_LEARNING_ENABLED", "true")
    monkeypatch.setenv("ACTIVE_LEARNING_STRATEGY", "entropy")
    monkeypatch.setenv("ACTIVE_LEARNING_MIN_UNCERTAINTY", "0.42")
    monkeypatch.setenv("ACTIVE_LEARNING_MAX_QUEUE", "500")
    monkeypatch.setenv("ACTIVE_LEARNING_TOP_N", "40")
    monkeypatch.setenv("ACTIVE_LEARNING_COOLDOWN_SECONDS", "2.0")

    settings = _reload_settings()
    assert settings.active_learning_enabled is True
    assert settings.active_learning_strategy == "entropy"
    assert settings.active_learning_min_uncertainty == 0.42
    assert settings.active_learning_max_queue == 500
    assert settings.active_learning_top_n == 40
    assert settings.active_learning_cooldown_seconds == 2.0

    get_settings.cache_clear()
