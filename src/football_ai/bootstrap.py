"""Composition root: wires settings, storage, quota, client and provider."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from sqlalchemy.orm import Session, sessionmaker

from football_ai.config.competitions import CompetitionConfig, load_competitions
from football_ai.config.settings import Settings
from football_ai.core.logging import redaction_filter
from football_ai.data.collectors.api_football import (
    ApiFootballClient,
    ApiFootballProvider,
    QuotaManager,
)
from football_ai.data.storage.raw_archive import RawArchive
from football_ai.data.storage.response_cache import ResponseCache
from football_ai.database.connection import create_db_engine, create_session_factory


def build_quota_manager(settings: Settings) -> QuotaManager:
    cfg = settings.api_football
    return QuotaManager(
        daily_limit=cfg.daily_limit,
        per_minute_limit=cfg.per_minute_limit,
        reserve=cfg.quota_reserve,
        state_file=settings.state_dir / "api_football_quota.json",
    )


@dataclass(slots=True)
class ApiFootballContext:
    client: ApiFootballClient
    provider: ApiFootballProvider
    competitions: list[CompetitionConfig]


@asynccontextmanager
async def api_football_context(settings: Settings) -> AsyncIterator[ApiFootballContext]:
    cfg = settings.api_football
    if cfg.key is not None:
        redaction_filter.register(cfg.key.get_secret_value())
    competitions = load_competitions(settings.competitions_file)
    client = ApiFootballClient(
        cfg,
        build_quota_manager(settings),
        cache=ResponseCache(settings.cache_dir),
        archive=RawArchive(settings.raw_dir),
    )
    async with client:
        yield ApiFootballContext(
            client=client,
            provider=ApiFootballProvider(client, competitions, cfg),
            competitions=competitions,
        )


def build_session_factory(settings: Settings) -> sessionmaker[Session]:
    return create_session_factory(
        create_db_engine(settings.database_url, echo=settings.database_echo)
    )
