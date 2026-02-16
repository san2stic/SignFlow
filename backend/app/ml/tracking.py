"""MLflow tracking integration for experiment management and model versioning."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Optional MLflow import - gracefully degrade if not available
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("mlflow_not_installed", msg="MLflow not installed, tracking disabled")


class MLFlowTracker:
    """
    MLflow experiment tracker for training runs.

    Features:
    - Automatic experiment tracking
    - Hyperparameter logging
    - Metrics logging per epoch
    - Model artifact storage
    - Graceful degradation if MLflow unavailable
    """

    def __init__(
        self,
        experiment_name: str = "signflow-training",
        tracking_uri: str | None = None,
        enabled: bool = True,
    ):
        """
        Initialize MLflow tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: URI for MLflow tracking server (defaults to local filesystem)
            enabled: Whether to enable tracking (can disable for testing)
        """
        self.enabled = enabled and MLFLOW_AVAILABLE
        self.experiment_name = experiment_name
        self._active_run = None

        if not self.enabled:
            if not MLFLOW_AVAILABLE:
                logger.info("mlflow_disabled", reason="not_installed")
            else:
                logger.info("mlflow_disabled", reason="explicitly_disabled")
            return

        # Set tracking URI
        if tracking_uri is None:
            # Default to local filesystem in data/models/mlruns
            from app.config import get_settings
            settings = get_settings()
            tracking_uri = f"file://{settings.model_dir}/mlruns"

        mlflow.set_tracking_uri(tracking_uri)
        logger.info("mlflow_tracking_uri_set", uri=tracking_uri)

        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info("mlflow_experiment_created", name=experiment_name, id=experiment_id)
            else:
                experiment_id = experiment.experiment_id
                logger.info("mlflow_experiment_found", name=experiment_name, id=experiment_id)

            mlflow.set_experiment(experiment_name)
        except Exception as e:
            logger.error("mlflow_experiment_setup_failed", error=str(e))
            self.enabled = False

    def start(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        run_id: str | None = None,
    ) -> str | None:
        """Start one MLflow run and keep it active until ``end`` is called."""
        if not self.enabled:
            return None
        if self._active_run is not None:
            return self._active_run.info.run_id

        try:
            if run_id:
                self._active_run = mlflow.start_run(run_id=run_id)
            else:
                self._active_run = mlflow.start_run(run_name=run_name, tags=tags)
            run_id = self._active_run.info.run_id
            logger.info("mlflow_run_started", run_id=run_id, name=run_name)
            return run_id
        except Exception as e:
            logger.error("mlflow_run_failed", error=str(e))
            self._active_run = None
            return None

    def end(self) -> None:
        """End the currently active MLflow run, if any."""
        if not self.enabled or self._active_run is None:
            return

        run_id = self._active_run.info.run_id
        try:
            mlflow.end_run()
            logger.info("mlflow_run_ended", run_id=run_id)
        except Exception as e:
            logger.error("mlflow_end_run_failed", error=str(e))
        finally:
            self._active_run = None

    @contextlib.contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        run_id: str | None = None,
    ):
        """
        Start an MLflow run as a context manager.

        Args:
            run_name: Optional name for the run
            tags: Optional tags to attach to the run

        Yields:
            MLflow run object (or None if disabled)
        """
        if not self.enabled:
            yield None
            return

        self.start(run_name=run_name, tags=tags, run_id=run_id)
        try:
            yield self._active_run
        finally:
            self.end()

    def log_params(self, params: dict[str, Any]) -> None:
        """
        Log hyperparameters.

        Args:
            params: Dictionary of hyperparameters to log
        """
        if not self.enabled or self._active_run is None:
            return

        try:
            # MLflow requires string values for params
            str_params = {k: str(v) for k, v in params.items()}
            mlflow.log_params(str_params)
            logger.debug("mlflow_params_logged", count=len(params))
        except Exception as e:
            logger.error("mlflow_log_params_failed", error=str(e))

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """
        Log metrics for the current step/epoch.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number (e.g., epoch number)
        """
        if not self.enabled or self._active_run is None:
            return

        try:
            mlflow.log_metrics(metrics, step=step)
            logger.debug("mlflow_metrics_logged", count=len(metrics), step=step)
        except Exception as e:
            logger.error("mlflow_log_metrics_failed", error=str(e), step=step)

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        """
        Log a file artifact.

        Args:
            local_path: Path to the local file
            artifact_path: Optional subdirectory in the artifact store
        """
        if not self.enabled or self._active_run is None:
            return

        try:
            mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
            logger.info("mlflow_artifact_logged", path=str(local_path))
        except Exception as e:
            logger.error("mlflow_log_artifact_failed", error=str(e), path=str(local_path))

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: str | None = None,
    ) -> None:
        """
        Log a PyTorch model.

        Args:
            model: PyTorch model to log
            artifact_path: Path within the run's artifact directory
            registered_model_name: If provided, register model in MLflow Model Registry
        """
        if not self.enabled or self._active_run is None:
            return

        try:
            import torch

            # Save model state dict temporarily
            temp_path = Path("/tmp/mlflow_model_temp.pt")
            torch.save(model.state_dict(), temp_path)

            # Log as artifact
            mlflow.log_artifact(str(temp_path), artifact_path=artifact_path)

            # Clean up temp file
            temp_path.unlink(missing_ok=True)

            logger.info("mlflow_model_logged", artifact_path=artifact_path)

            # Register model if requested
            if registered_model_name:
                self._register_model(artifact_path, registered_model_name)

        except Exception as e:
            logger.error("mlflow_log_model_failed", error=str(e))

    def _register_model(self, artifact_path: str, model_name: str) -> None:
        """Register model in MLflow Model Registry."""
        if not self.enabled or self._active_run is None:
            return

        try:
            client = MlflowClient()
            run_id = self._active_run.info.run_id
            model_uri = f"runs:/{run_id}/{artifact_path}"

            # Register model version
            model_version = client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=run_id
            )

            logger.info(
                "mlflow_model_registered",
                name=model_name,
                version=model_version.version,
                run_id=run_id
            )
        except Exception as e:
            logger.error("mlflow_register_model_failed", error=str(e), name=model_name)

    def log_dict(self, dictionary: dict, filename: str) -> None:
        """
        Log a dictionary as JSON artifact.

        Args:
            dictionary: Dictionary to log
            filename: Name of the JSON file
        """
        if not self.enabled or self._active_run is None:
            return

        try:
            mlflow.log_dict(dictionary, filename)
            logger.debug("mlflow_dict_logged", filename=filename)
        except Exception as e:
            logger.error("mlflow_log_dict_failed", error=str(e), filename=filename)

    def set_tags(self, tags: dict[str, str]) -> None:
        """
        Set tags on the current run.

        Args:
            tags: Dictionary of tags
        """
        if not self.enabled or self._active_run is None:
            return

        try:
            mlflow.set_tags(tags)
            logger.debug("mlflow_tags_set", count=len(tags))
        except Exception as e:
            logger.error("mlflow_set_tags_failed", error=str(e))

    @property
    def run_id(self) -> str | None:
        """Get the current run ID."""
        if self._active_run is None:
            return None
        return self._active_run.info.run_id

    @property
    def is_active(self) -> bool:
        """Check if a run is currently active."""
        return self._active_run is not None


def create_default_tracker(enabled: bool = True) -> MLFlowTracker:
    """
    Create a default MLflow tracker instance.

    Args:
        enabled: Whether to enable tracking

    Returns:
        MLFlowTracker instance
    """
    return MLFlowTracker(
        experiment_name="signflow-training",
        enabled=enabled
    )
