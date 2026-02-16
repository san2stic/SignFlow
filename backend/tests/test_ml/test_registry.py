"""Tests for MLflow model registry wrapper."""

from __future__ import annotations

from pathlib import Path

import app.ml.registry as registry_module
from app.ml.registry import ModelRegistry


def test_registry_returns_disabled_when_not_enabled(tmp_path: Path) -> None:
    """Registry wrapper should no-op safely when disabled."""
    model_file = tmp_path / "model.pt"
    model_file.write_bytes(b"dummy")

    registry = ModelRegistry(enabled=False)
    result = registry.register_model(
        model_path=str(model_file),
        version_name="v1",
    )

    assert result["registered"] is False
    assert result["reason"] == "registry_disabled"


def test_registry_registers_and_promotes_with_fake_client(monkeypatch, tmp_path: Path) -> None:
    """Registry wrapper should call client methods for register/promote flow."""
    calls: dict[str, object] = {
        "created_model": None,
        "created_version": None,
        "tags": [],
        "transitions": [],
    }

    class _FakeVersion:
        version = "12"

    class _FakeClient:
        def __init__(self, tracking_uri=None):
            self.tracking_uri = tracking_uri

        def create_registered_model(self, name: str) -> None:
            calls["created_model"] = name

        def create_model_version(self, name: str, source: str, run_id: str | None):
            calls["created_version"] = {"name": name, "source": source, "run_id": run_id}
            return _FakeVersion()

        def set_model_version_tag(self, name: str, version: str, key: str, value: str) -> None:
            tags = calls["tags"]
            assert isinstance(tags, list)
            tags.append((name, version, key, value))

        def transition_model_version_stage(
            self,
            *,
            name: str,
            version: str,
            stage: str,
            archive_existing_versions: bool,
        ) -> None:
            transitions = calls["transitions"]
            assert isinstance(transitions, list)
            transitions.append((name, version, stage, archive_existing_versions))

    class _FakeMlflow:
        @staticmethod
        def set_tracking_uri(_uri: str | None) -> None:
            return None

    monkeypatch.setattr(registry_module, "MLFLOW_AVAILABLE", True)
    monkeypatch.setattr(registry_module, "MlflowClient", _FakeClient)
    monkeypatch.setattr(registry_module, "mlflow", _FakeMlflow)

    model_file = tmp_path / "model.pt"
    model_file.write_bytes(b"model")

    registry = ModelRegistry(
        model_name="signflow-registry",
        tracking_uri="file:///tmp/mlruns",
        enabled=True,
    )
    result = registry.register_model(
        model_path=str(model_file),
        version_name="v42",
        run_id="session-123",
        tags={"owner": "qa"},
    )
    assert result["registered"] is True
    assert result["registry_version"] == "12"

    promote = registry.promote_to_staging(registry_version="12")
    assert promote["promoted"] is True

    assert calls["created_model"] == "signflow-registry"
    created_version = calls["created_version"]
    assert isinstance(created_version, dict)
    assert created_version["run_id"] == "session-123"
