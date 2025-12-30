"""Runtime configuration utilities for FashionFlow."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "fashionflow-api"
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    llm_provider: Literal["openai", "azure", "mock"] = "mock"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    azure_openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_deployment: str | None = None

    vectorstore_path: Path = Path(".data/chroma")
    vectorstore_collection: str = "fashionflow-products"

    agent_strategy: Literal["multi_agent", "rule_based"] = "multi_agent"
    telemetry_endpoint: str | None = None
    feature_flags: dict[str, bool] = {}

    class Config:
        frozen = True


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance for dependency injection."""

    return Settings()
