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
