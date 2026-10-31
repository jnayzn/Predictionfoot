"""Manual team alias overrides (``configs/team_aliases.toml``)."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, ValidationError

from football_ai.core.exceptions import ConfigurationError


class TeamAlias(BaseModel):
    provider: str
    name: str
    canonical: str


def load_aliases(path: Path) -> list[TeamAlias]:
    if not path.is_file():
        return []
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        return [TeamAlias.model_validate(a) for a in raw.get("aliases", [])]
    except (tomllib.TOMLDecodeError, ValidationError) as exc:
        raise ConfigurationError(f"Invalid aliases file {path}: {exc}") from exc
