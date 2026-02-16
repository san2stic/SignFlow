"""Canary model router for inference-session traffic splitting."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import structlog

from app.ml.pipeline import SignFlowInferencePipeline

logger = structlog.get_logger(__name__)


@dataclass
class RoutingSession:
    """Primary/shadow sessions selected for one websocket connection."""

    primary: SignFlowInferencePipeline
    shadow: SignFlowInferencePipeline | None
    route: str


class ModelRouter:
    """
    Route inference traffic between production and canary models.

    The production template is resolved by `production_provider`, while canary/shadow
    templates are resolved lazily by `model_loader(model_id)`.
    """

    def __init__(
        self,
        *,
        production_provider: Callable[[], SignFlowInferencePipeline],
        model_loader: Callable[[str], SignFlowInferencePipeline | None],
        canary_percentage: float = 0.0,
        canary_model_id: str | None = None,
        shadow_mode_enabled: bool = False,
        shadow_model_id: str | None = None,
        random_fn: Callable[[], float] | None = None,
    ) -> None:
        self.production_provider = production_provider
        self.model_loader = model_loader
        self.canary_percentage = float(max(0.0, min(100.0, canary_percentage)))
        self.canary_model_id = str(canary_model_id).strip() if canary_model_id else None
        self.shadow_mode_enabled = bool(shadow_mode_enabled)
        self.shadow_model_id = str(shadow_model_id).strip() if shadow_model_id else None
        self.random_fn = random_fn or random.random
        self._canary_template: SignFlowInferencePipeline | None = None
        self._shadow_template: SignFlowInferencePipeline | None = None

    @property
    def canary_enabled(self) -> bool:
        """Whether canary split routing is enabled."""
        return bool(self.canary_model_id and self.canary_percentage > 0.0)

    @property
    def shadow_enabled(self) -> bool:
        """Whether shadow mode routing is enabled."""
        return bool(self.shadow_mode_enabled and self.shadow_model_id)

    def reload(self) -> None:
        """Drop cached canary/shadow templates (production stays provider-owned)."""
        self._canary_template = None
        self._shadow_template = None

    def spawn_sessions(self) -> RoutingSession:
        """Create routed per-connection sessions."""
        primary_template, route = self._select_primary_template()
        primary_session = primary_template.spawn_session()
        primary_session.reset()

        shadow_session = self._build_shadow_session(primary_template)
        return RoutingSession(primary=primary_session, shadow=shadow_session, route=route)

    def _select_primary_template(self) -> tuple[SignFlowInferencePipeline, str]:
        production_template = self.production_provider()
        if not self.canary_enabled:
            return production_template, "production"

        canary_template = self._load_canary_template()
        if canary_template is None:
            return production_template, "production"

        threshold = self.canary_percentage / 100.0
        sample = float(self.random_fn())
        if sample < threshold:
            return canary_template, "canary"
        return production_template, "production"

    def _build_shadow_session(
        self,
        primary_template: SignFlowInferencePipeline,
    ) -> SignFlowInferencePipeline | None:
        if not self.shadow_enabled:
            return None

        shadow_template = self._load_shadow_template()
        if shadow_template is None:
            return None

        if getattr(shadow_template, "model_version", None) == getattr(primary_template, "model_version", None):
            return None

        session = shadow_template.spawn_session()
        session.reset()
        return session

    def _load_canary_template(self) -> SignFlowInferencePipeline | None:
        if not self.canary_model_id:
            return None
        if self._canary_template is None:
            self._canary_template = self.model_loader(self.canary_model_id)
            if self._canary_template is None:
                logger.warning("canary_model_not_available", model_id=self.canary_model_id)
        return self._canary_template

    def _load_shadow_template(self) -> SignFlowInferencePipeline | None:
        if not self.shadow_model_id:
            return None
        if self._shadow_template is None:
            self._shadow_template = self.model_loader(self.shadow_model_id)
            if self._shadow_template is None:
                logger.warning("shadow_model_not_available", model_id=self.shadow_model_id)
        return self._shadow_template
