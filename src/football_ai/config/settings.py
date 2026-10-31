"""Application settings loaded from environment variables / `.env`.

Secrets are stored as :class:`pydantic.SecretStr` so they never appear in
``repr()``, logs or tracebacks.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiFootballSettings(BaseSettings):
    """Configuration of the API-Football (api-sports.io) connector."""

    model_config = SettingsConfigDict(
        env_prefix="API_FOOTBALL_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    key: SecretStr | None = None
    base_url: str = "https://v3.football.api-sports.io"

    # Quotas (free plan defaults). The server headers, when present, take precedence.
    daily_limit: int = Field(default=100, ge=1)
    per_minute_limit: int = Field(default=10, ge=1)
    quota_reserve: int = Field(default=5, ge=0)

    # HTTP behaviour
    timeout_seconds: float = Field(default=20.0, gt=0)
    max_retries: int = Field(default=4, ge=0, le=10)
    backoff_base_seconds: float = Field(default=1.0, ge=0)
    backoff_max_seconds: float = Field(default=60.0, ge=0)

    timezone: str = "UTC"

    # Cache TTLs in seconds
    cache_ttl_leagues: int = Field(default=7 * 24 * 3600, ge=0)
    cache_ttl_fixtures_past: int = Field(default=7 * 24 * 3600, ge=0)
    cache_ttl_fixtures_recent: int = Field(default=15 * 60, ge=0)
    cache_ttl_historical: int = Field(default=30 * 24 * 3600, ge=0)
    cache_ttl_season_current: int = Field(default=6 * 3600, ge=0)

    @field_validator("key", mode="before")
    @classmethod
    def _empty_key_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def is_configured(self) -> bool:
        return self.key is not None and bool(self.key.get_secret_value().strip())


class Settings(BaseSettings):
    """Top-level application settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "console"
    data_dir: Path = Path("var")
    competitions_file: Path = Path("configs/competitions.toml")
    team_aliases_file: Path = Path("configs/team_aliases.toml")
    schedule_file: Path = Path("configs/schedule.toml")
    plans_file: Path = Path("configs/plans.toml")
    database_url: str = "postgresql+psycopg://football:football@localhost:5432/football_ai"
    database_echo: bool = False

    api_football: ApiFootballSettings = Field(default_factory=ApiFootballSettings)

    # Backend API
    api_keys: SecretStr | None = None  # comma-separated keys accepted in X-API-Key
    api_rate_limit_per_minute: int = Field(default=60, ge=1)
    api_cors_origins: str = ""  # comma-separated
    production_model_priority: str = "ensemble,dixon_coles,elo"
    redis_url: str | None = None  # shared rate limiting across API workers
    metrics_enabled: bool = True
    metrics_token: SecretStr | None = None  # optional bearer token protecting /metrics

    # Telegram bot
    telegram_bot_token: SecretStr | None = None
    telegram_allowed_users: str = ""  # comma-separated user ids; empty = everyone
    telegram_rate_limit_per_minute: int = Field(default=20, ge=1)
    bot_backend: Literal["service", "http"] = "service"
    bot_api_url: str = "http://localhost:8000"
    bot_api_key: SecretStr | None = None
    telegram_admin_chat_id: int | None = None  # alerts destination (optional)

    @property
    def allowed_telegram_users(self) -> set[int]:
        return {int(x) for x in self.telegram_allowed_users.split(",") if x.strip().isdigit()}

    @property
    def accepted_api_keys(self) -> set[str]:
        if self.api_keys is None:
            return set()
        return {k.strip() for k in self.api_keys.get_secret_value().split(",") if k.strip()}

    @property
    def model_priority(self) -> tuple[str, ...]:
        return tuple(n.strip() for n in self.production_model_priority.split(",") if n.strip())

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def state_dir(self) -> Path:
        return self.data_dir / "state"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
