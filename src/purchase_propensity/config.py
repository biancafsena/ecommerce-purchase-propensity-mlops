"""Application settings loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated settings shared across the application."""

    project_name: str = "ecommerce-purchase-propensity-mlops"
    environment: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    random_seed: int = Field(default=42, ge=0)

    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    mlflow_experiment_name: str = "ecommerce-purchase-propensity"
    mlflow_registered_model_name: str = "purchase-propensity-classifier"
    mlflow_model_alias: str = "champion"

    model_decision_threshold: float = Field(default=0.50, ge=0.0, le=1.0)

    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8080, ge=1, le=65535)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached and validated settings instance."""
    return Settings()
