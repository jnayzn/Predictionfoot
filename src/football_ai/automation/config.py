"""Scheduler configuration (``configs/schedule.toml``)."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from football_ai.core.exceptions import ConfigurationError


class JobConfig(BaseModel):
    cron: str
    enabled: bool = True
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("cron")
    @classmethod
    def _five_fields(cls, value: str) -> str:
        if len(value.split()) != 5:
            raise ValueError(f"cron must have 5 fields: {value!r}")
        return value


class ScheduleConfig(BaseModel):
    timezone: str = "UTC"
    jobs: dict[str, JobConfig]


def load_schedule(path: Path) -> ScheduleConfig:
    if not path.is_file():
        raise ConfigurationError(f"schedule file not found: {path}")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        jobs = {}
        for name, values in raw.get("jobs", {}).items():
            values = dict(values)
            cron = values.pop("cron")
            enabled = values.pop("enabled", True)
            jobs[name] = JobConfig(cron=cron, enabled=enabled, options=values)
        return ScheduleConfig(timezone=raw.get("timezone", "UTC"), jobs=jobs)
    except (tomllib.TOMLDecodeError, ValidationError, KeyError) as exc:
        raise ConfigurationError(f"invalid schedule file {path}: {exc}") from exc
