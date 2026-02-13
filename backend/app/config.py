"""Application settings and environment configuration."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    env: str = "development"
    app_name: str = "SignFlow API"
    api_v1_prefix: str = "/api/v1"

    database_url: str = "sqlite:///./data/signflow.db"
    redis_url: str = "redis://localhost:6379/0"

    model_dir: str = "/app/data/models"
    video_dir: str = "/app/data/videos"
    export_dir: str = "/app/data/exports"

    cors_origins: str = Field(default="http://localhost:3000")
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

    @property
    def cors_origin_list(self) -> list[str]:
        """Parse comma-separated CORS origins."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

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
