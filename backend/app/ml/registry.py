"""MLflow Model Registry wrapper for model lifecycle operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

try:
    import mlflow
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except Exception:  # noqa: BLE001
    mlflow = None
    MlflowClient = None
    MLFLOW_AVAILABLE = False


class ModelRegistry:
    """Wrapper over MLflow Model Registry with safe no-op fallback."""

    def __init__(
        self,
        *,
        model_name: str = "signflow-model",
        tracking_uri: str | None = None,
        enabled: bool = True,
    ) -> None:
        self.model_name = model_name
        self.enabled = bool(enabled and MLFLOW_AVAILABLE)
        self.tracking_uri = tracking_uri
        self.client = None

        if not self.enabled:
            return

        try:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            self.client = MlflowClient(tracking_uri=tracking_uri)
        except Exception as exc:  # noqa: BLE001
            logger.warning("model_registry_init_failed", error=str(exc))
            self.enabled = False
            self.client = None

    def register_model(
        self,
        *,
        model_path: str,
        version_name: str,
        run_id: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Register one local model artifact as a registry version."""
        if not self.enabled or self.client is None:
            return {"registered": False, "reason": "registry_disabled"}

        path = Path(model_path).expanduser().resolve()
        if not path.exists():
            return {
                "registered": False,
                "reason": "model_path_not_found",
                "model_path": str(path),
            }

        try:
            self._ensure_registered_model_exists()

            model_version = self.client.create_model_version(
                name=self.model_name,
                source=path.as_uri(),
                run_id=run_id,
            )
            registry_version = str(model_version.version)
            merged_tags = {
                "signflow_version": str(version_name),
                **(tags or {}),
            }
            for key, value in merged_tags.items():
                self.client.set_model_version_tag(
                    name=self.model_name,
                    version=registry_version,
                    key=str(key),
                    value=str(value),
                )

            return {
                "registered": True,
                "model_name": self.model_name,
                "registry_version": registry_version,
                "source": path.as_uri(),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("model_registry_register_failed", error=str(exc), model_name=self.model_name)
            return {
                "registered": False,
                "reason": "registry_error",
                "error": str(exc),
            }

    def promote_to_staging(self, *, registry_version: str) -> dict[str, Any]:
        """Promote one registry version to Staging stage."""
        return self._transition_stage(registry_version=registry_version, stage="Staging")

    def promote_to_production(self, *, registry_version: str) -> dict[str, Any]:
        """Promote one registry version to Production stage."""
        return self._transition_stage(registry_version=registry_version, stage="Production")

    def _transition_stage(self, *, registry_version: str, stage: str) -> dict[str, Any]:
        if not self.enabled or self.client is None:
            return {"promoted": False, "reason": "registry_disabled"}

        try:
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=str(registry_version),
                stage=stage,
                archive_existing_versions=False,
            )
            return {
                "promoted": True,
                "model_name": self.model_name,
                "registry_version": str(registry_version),
                "stage": stage,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "model_registry_promotion_failed",
                error=str(exc),
                model_name=self.model_name,
                registry_version=registry_version,
                stage=stage,
            )
            return {
                "promoted": False,
                "reason": "registry_error",
                "error": str(exc),
            }

    def _ensure_registered_model_exists(self) -> None:
        """Create registered model if needed (ignore already exists)."""
        assert self.client is not None
        try:
            self.client.create_registered_model(self.model_name)
        except Exception as exc:  # noqa: BLE001
            message = str(exc).lower()
            if "already exists" in message or "resource already exists" in message:
                return
            raise


def create_default_registry(
    *,
    enabled: bool = False,
    model_name: str = "signflow-model",
    tracking_uri: str | None = None,
) -> ModelRegistry:
    """Create a registry instance with safe defaults."""
    return ModelRegistry(
        model_name=model_name,
        tracking_uri=tracking_uri,
        enabled=enabled,
    )
