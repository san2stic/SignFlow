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
    max_upload_mb: int = 50
    max_video_seconds: int = 10
    rate_limit_per_minute: int = 120
    training_use_celery: bool = False

    @property
    def cors_origin_list(self) -> list[str]:
        """Parse comma-separated CORS origins."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
