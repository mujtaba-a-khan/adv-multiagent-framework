"""Application settings loaded from environment variables via Pydantic."""

from __future__ import annotations

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the adversarial framework.

    All fields can be overridden via environment variables prefixed with ``ADV_``.
    For example, ``ADV_OLLAMA_BASE_URL=http://gpu-server:11434``.
    """

    model_config = SettingsConfigDict(
        env_prefix="ADV_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def _fix_asyncpg_ssl(self) -> Settings:
        """asyncpg uses ``ssl=`` not ``sslmode=`` — translate automatically."""
        for field in ("database_url", "database_url_direct"):
            val = getattr(self, field)
            if val and "sslmode=" in val:
                object.__setattr__(self, field, val.replace("sslmode=", "ssl="))
        return self

    # Ollama (Primary LLM Provider)

    ollama_base_url: str = "http://localhost:11434"

    # Default model assignments per agent role (Ollama model tags)
    attacker_model: str = "phi4-reasoning:14b"
    analyzer_model: str = "phi4-reasoning:14b"
    defender_model: str = "qwen3:8b"
    target_model: str = "llama3:8b"

    # Optional Commercial Providers

    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""

    # Database (Neon Serverless PostgreSQL)

    database_url: str = ""
    database_url_direct: str = ""  # Direct (non-pooled) for Alembic migrations

    # Redis

    redis_url: str = "redis://localhost:6379/0"

    # Safety & Budget Limits

    max_turns: int = 20
    max_cost_usd: float = 50.0
    daily_max_cost_usd: float = 200.0
    max_errors: int = 5

    # Logging

    log_level: str = "INFO"
    log_json: bool = False  # True for production structured JSON logs

    # Application

    environment: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["http://localhost:3000"]


def get_settings() -> Settings:
    """Factory for cached settings instance."""
    return Settings()
