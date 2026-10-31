"""Loading of the competition catalogue (``configs/competitions.toml``)."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from football_ai.core.exceptions import ConfigurationError


class CompetitionConfig(BaseModel):
    """A competition the system covers, independent of any data provider."""

    key: str = Field(pattern=r"^[a-z0-9_]+$")
    name: str
    country: str
    type: str
    priority: int = 1
    provider_ids: dict[str, int]

    def provider_id(self, provider: str) -> int | None:
        return self.provider_ids.get(provider)


def load_competitions(path: Path) -> list[CompetitionConfig]:
    """Load and validate the competition catalogue."""
    if not path.is_file():
        raise ConfigurationError(f"Competitions file not found: {path}")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        items = [CompetitionConfig.model_validate(c) for c in raw.get("competitions", [])]
    except (tomllib.TOMLDecodeError, ValidationError) as exc:
        raise ConfigurationError(f"Invalid competitions file {path}: {exc}") from exc

    keys = [c.key for c in items]
    duplicates = {k for k in keys if keys.count(k) > 1}
    if duplicates:
        raise ConfigurationError(f"Duplicate competition keys: {sorted(duplicates)}")
    if not items:
        raise ConfigurationError(f"No competitions defined in {path}")
    return items
