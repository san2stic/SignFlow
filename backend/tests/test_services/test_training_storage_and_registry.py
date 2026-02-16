"""Training storage and registry integration behavior tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import app.services.training_service as training_service_module
from app.config import get_settings
from app.services.training_service import TrainingService


def test_resolve_models_dir_falls_back_when_configured_path_is_unwritable(monkeypatch, tmp_path: Path) -> None:
    """Service should fallback to backend/data/models when configured model dir cannot be created."""
    configured_dir = (tmp_path / "protected-models").resolve()
    fallback_dir = Path(training_service_module.__file__).resolve().parents[2] / "data" / "models"
    settings_stub = SimpleNamespace(model_dir=str(configured_dir))

    real_mkdir = Path.mkdir

    def fake_mkdir(self: Path, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        if self == configured_dir:
            raise PermissionError("permission denied")
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(training_service_module.Path, "mkdir", fake_mkdir, raising=True)

    resolved = TrainingService._resolve_models_dir(settings_stub)  # type: ignore[arg-type]

    assert resolved == fallback_dir


def test_register_model_registry_uses_mlflow_run_id(monkeypatch, tmp_path: Path) -> None:
    """Registry wrapper should receive the MLflow run id, not the training session id."""
    calls: dict[str, object] = {}

    class _FakeRegistry:
        def register_model(
            self,
            *,
            model_path: str,
            version_name: str,
            run_id: str | None = None,
            tags: dict[str, str] | None = None,
        ) -> dict[str, object]:
            calls["model_path"] = model_path
            calls["version_name"] = version_name
            calls["run_id"] = run_id
            calls["tags"] = dict(tags or {})
            return {"registered": False, "reason": "disabled-for-test"}

        def promote_to_staging(self, *, registry_version: str) -> dict[str, object]:
            raise AssertionError("staging promotion should not run for disabled registration")

    monkeypatch.setattr(
        training_service_module,
        "create_default_registry",
        lambda **_kwargs: _FakeRegistry(),
    )

    settings = get_settings()
    monkeypatch.setattr(settings, "mlflow_registry_enabled", True)
    monkeypatch.setattr(settings, "mlflow_registry_model_name", "signflow-model")
    monkeypatch.setattr(settings, "mlflow_registry_auto_promote_staging", False)
    monkeypatch.setattr(settings, "mlflow_tracking_uri", "file:///tmp/mlruns")

    model_file = tmp_path / "model.pt"
    model_file.write_bytes(b"model")

    TrainingService._register_model_registry_version(
        model_path=model_file,
        signflow_version="v42",
        training_session_id="training-session-123",
        mlflow_run_id="mlflow-run-789",
        local_model_version_id="local-model-456",
        parent_version="v41",
    )

    assert calls["run_id"] == "mlflow-run-789"
    assert calls["version_name"] == "v42"
    assert calls["model_path"] == str(model_file)
    tags = calls["tags"]
    assert isinstance(tags, dict)
    assert tags["training_session_id"] == "training-session-123"
    assert tags["local_model_version_id"] == "local-model-456"
    assert tags["parent_version"] == "v41"
