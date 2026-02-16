"""Tests for canary model router."""

from __future__ import annotations

from dataclasses import dataclass

from app.api.router_v2 import ModelRouter


@dataclass
class _Session:
    model_version: str

    def reset(self) -> None:
        return None


class _Template:
    def __init__(self, *, model_version: str) -> None:
        self.model_version = model_version
        self.spawn_calls = 0

    def spawn_session(self) -> _Session:
        self.spawn_calls += 1
        return _Session(model_version=self.model_version)


def test_model_router_routes_to_canary_when_threshold_match() -> None:
    """Traffic under canary threshold should select canary model."""
    production = _Template(model_version="v_prod")
    canary = _Template(model_version="v_canary")

    router = ModelRouter(
        production_provider=lambda: production,
        model_loader=lambda model_id: canary if model_id == "canary-id" else None,
        canary_percentage=20.0,
        canary_model_id="canary-id",
        random_fn=lambda: 0.05,
    )
    session = router.spawn_sessions()

    assert session.route == "canary"
    assert session.primary.model_version == "v_canary"


def test_model_router_routes_to_production_when_threshold_miss() -> None:
    """Traffic above canary threshold should stay on production."""
    production = _Template(model_version="v_prod")
    canary = _Template(model_version="v_canary")

    router = ModelRouter(
        production_provider=lambda: production,
        model_loader=lambda model_id: canary if model_id == "canary-id" else None,
        canary_percentage=20.0,
        canary_model_id="canary-id",
        random_fn=lambda: 0.80,
    )
    session = router.spawn_sessions()

    assert session.route == "production"
    assert session.primary.model_version == "v_prod"


def test_model_router_spawns_shadow_session_when_enabled() -> None:
    """Shadow session should be created when configured with a different model."""
    production = _Template(model_version="v_prod")
    shadow = _Template(model_version="v_shadow")

    router = ModelRouter(
        production_provider=lambda: production,
        model_loader=lambda model_id: shadow if model_id == "shadow-id" else None,
        shadow_mode_enabled=True,
        shadow_model_id="shadow-id",
    )
    session = router.spawn_sessions()

    assert session.route == "production"
    assert session.shadow is not None
    assert session.shadow.model_version == "v_shadow"
