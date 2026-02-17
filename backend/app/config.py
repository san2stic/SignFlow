"""Application settings and environment configuration."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_JWT_SECRET = "your-secret-key-change-in-production-minimum-32-chars"
DEFAULT_JWT_SECRET_LEGACY = "your-secret-key-change-in-production-minimum-32-chars-required-for-security"
DEFAULT_JWT_SECRET_EXAMPLE = "replace-with-a-strong-random-secret-min-32-chars"
MIN_JWT_SECRET_LENGTH = 32


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    env: str = "development"
    app_name: str = "SignFlow API"
    api_v1_prefix: str = "/api/v1"

    database_url: str = "sqlite:///./data/signflow.db"
    redis_url: str = "redis://localhost:6379/0"
    search_backend: Literal["sql", "elasticsearch"] = "sql"
    elasticsearch_url: str = Field(default="http://localhost:9200")
    elasticsearch_index: str = Field(default="signflow-signs")
    elasticsearch_timeout_ms: int = Field(default=2000, ge=100, le=60000)
    elasticsearch_reindex_on_startup: bool = False
    elasticsearch_fail_open: bool = True
    elasticsearch_verify_certs: bool = False

    model_dir: str = "/app/data/models"
    video_dir: str = "/app/data/videos"
    export_dir: str = "/app/data/exports"

    # S3 / MinIO storage (USE_S3_STORAGE=true en production serveur)
    use_s3_storage: bool = False
    s3_endpoint_url: str = "http://minio:9000"
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_bucket_videos: str = "signflow-videos"
    s3_bucket_models: str = "signflow-models"
    s3_presigned_url_expiry: int = Field(default=3600, ge=60, le=86400)
    s3_region: str = "us-east-1"

    trusted_hosts: str = Field(default="localhost,127.0.0.1,testserver")
    enable_docs: bool | None = None
    max_upload_mb: int = Field(default=50, ge=1, le=1024)
    max_video_seconds: int = Field(default=10, ge=1, le=60)
    max_request_mb: int = Field(default=64, ge=1, le=2048)
    max_dictionary_import_mb: int = Field(default=30, ge=1, le=2048)
    max_dictionary_import_files: int = Field(default=1000, ge=1, le=20000)
    max_dictionary_import_file_mb: int = Field(default=15, ge=1, le=512)
    max_dictionary_import_uncompressed_mb: int = Field(default=200, ge=1, le=8192)
    max_dictionary_import_compression_ratio: float = Field(default=150.0, ge=1.0, le=5000.0)
    rate_limit_per_minute: int = Field(default=120, ge=1, le=10000)
    write_rate_limit_per_minute: int = Field(default=120, ge=1, le=10000)
    ws_messages_per_minute: int = Field(default=900, ge=60, le=10000)
    ws_max_connections_per_ip: int = Field(default=3, ge=1, le=100)
    training_use_celery: bool = False
    translate_seq_len: int = Field(default=64, ge=8, le=256)
    translate_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    translate_inference_num_views: int = Field(default=1, ge=1, le=8)
    translate_inference_temperature: float = Field(default=1.0, ge=0.1, le=3.0)
    translate_max_view_disagreement: float = Field(default=0.35, ge=0.01, le=1.0)
    translate_tta_enable_mirror: bool = True
    translate_tta_enable_temporal_jitter: bool = True
    translate_tta_enable_spatial_noise: bool = True
    translate_tta_temporal_jitter_ratio: float = Field(default=0.05, ge=0.0, le=0.3)
    translate_tta_spatial_noise_std: float = Field(default=0.005, ge=0.0, le=0.1)
    use_torchserve: bool = False
    torchserve_url: str = Field(default="http://torchserve:8080")
    torchserve_timeout_ms: int = Field(default=2000, ge=100, le=30000)
    canary_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    canary_model_id: str | None = None
    shadow_mode_enabled: bool = False
    shadow_model_id: str | None = None
    shadow_min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    inference_metrics_enabled: bool = True
    drift_detection_enabled: bool = True
    drift_window_size: int = Field(default=1000, ge=64, le=50000)
    drift_check_every: int = Field(default=100, ge=1, le=10000)
    drift_min_samples: int = Field(default=200, ge=32, le=50000)
    drift_p_value_threshold: float = Field(default=0.05, ge=0.000001, le=1.0)
    drift_mean_shift_threshold: float = Field(default=0.12, ge=0.0, le=1.0)
    active_learning_enabled: bool = False
    active_learning_strategy: Literal["entropy", "margin", "combined"] = "combined"
    active_learning_min_uncertainty: float = Field(default=0.6, ge=0.0, le=1.0)
    active_learning_max_queue: int = Field(default=2000, ge=10, le=200000)
    active_learning_top_n: int = Field(default=250, ge=1, le=10000)
    active_learning_cooldown_seconds: float = Field(default=1.5, ge=0.0, le=60.0)
    mlflow_registry_enabled: bool = False
    mlflow_registry_model_name: str = "signflow-model"
    mlflow_registry_auto_promote_staging: bool = True
    mlflow_tracking_uri: str | None = None

    # JWT Authentication
    jwt_secret_key: str = Field(default=DEFAULT_JWT_SECRET)
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = Field(default=10080, ge=1, le=525600)  # 7 days default

    @model_validator(mode="after")
    def validate_security_settings(self) -> "Settings":
        """Validate auth-related settings before app startup."""
        jwt_secret = self.jwt_secret_key.strip()
        if len(jwt_secret) < MIN_JWT_SECRET_LENGTH:
            raise ValueError(
                f"JWT_SECRET_KEY must be at least {MIN_JWT_SECRET_LENGTH} characters."
            )

        if self.env.lower() == "production":
            known_placeholders = {
                DEFAULT_JWT_SECRET,
                DEFAULT_JWT_SECRET_LEGACY,
                DEFAULT_JWT_SECRET_EXAMPLE,
            }
            if jwt_secret in known_placeholders:
                raise ValueError(
                    "JWT_SECRET_KEY uses a known placeholder value and is not allowed in production."
                )
            if any(marker in jwt_secret for marker in ("change-in-production", "replace-with-a-strong-random-secret")):
                raise ValueError(
                    "JWT_SECRET_KEY appears to be a template value and is not allowed in production."
                )

        return self

    @property
    def trusted_host_list(self) -> list[str]:
        """Parse comma-separated trusted hostnames for Host header validation."""
        hosts = [host.strip() for host in self.trusted_hosts.split(",") if host.strip()]
        return hosts or ["localhost", "127.0.0.1"]

    @property
    def docs_enabled(self) -> bool:
        """Enable docs by default in non-production environments only."""
        if self.enable_docs is not None:
            return bool(self.enable_docs)
        return self.env.lower() != "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
