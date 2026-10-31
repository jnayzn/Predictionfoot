"""API-Football implementation of :class:`FootballDataProvider`.

Quota strategy (free plan = 100 requests/day):

* ``discover_competitions`` -> **1 request** (``GET /leagues`` without filter)
  returns every league with all its seasons and coverage flags; the result is
  cached for a week. The six configured competitions are resolved locally.
* ``get_fixtures_by_date`` -> **1 request** (``GET /fixtures?date=...``)
  returns all fixtures worldwide for that day; we filter our competitions
  locally instead of issuing one request per league.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from football_ai.config.competitions import CompetitionConfig
from football_ai.config.settings import ApiFootballSettings
from football_ai.core.exceptions import ConfigurationError
from football_ai.data.collectors.api_football.client import ApiFootballClient
from football_ai.data.collectors.api_football.parsers import (
    PROVIDER,
    ParseError,
    parse_fixture,
    parse_league,
)
from football_ai.data.collectors.base import CompetitionDiscovery, FixtureBatch
from football_ai.data.validators.fixtures import validate_competition_mapping, validate_fixture
from football_ai.data.validators.issues import DataQualityIssue, Severity, has_errors
from football_ai.domain.models import Competition, Fixture

logger = logging.getLogger(__name__)


class ApiFootballProvider:
    name = PROVIDER

    def __init__(
        self,
        client: ApiFootballClient,
        competitions: list[CompetitionConfig],
        settings: ApiFootballSettings,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._client = client
        self._settings = settings
        self._now = now
        self._competitions = [c for c in competitions if c.provider_id(PROVIDER) is not None]
        skipped = {c.key for c in competitions} - {c.key for c in self._competitions}
        if skipped:
            logger.warning("competitions_without_provider_id", extra={"keys": sorted(skipped)})
        self._by_league_id: dict[int, CompetitionConfig] = {
            c.provider_ids[PROVIDER]: c for c in self._competitions
        }
        try:
            self._tz = ZoneInfo(settings.timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ConfigurationError(
                f"Invalid API_FOOTBALL_TIMEZONE {settings.timezone!r}"
            ) from exc

    @property
    def competitions(self) -> list[CompetitionConfig]:
        return list(self._competitions)

    # ------------------------------------------------------------ discovery
    async def discover_competitions(self) -> CompetitionDiscovery:
        resp = await self._client.get("leagues", cache_ttl=self._settings.cache_ttl_leagues)
        items: list[Any] = resp.response if isinstance(resp.response, list) else []
        result = CompetitionDiscovery(from_cache=resp.from_cache)

        by_id: dict[int, dict[str, Any]] = {}
        by_name: dict[tuple[str, str], dict[str, Any]] = {}
        for item in items:
            league = item.get("league") if isinstance(item, dict) else None
            if not isinstance(league, dict) or league.get("id") is None:
                continue
            try:
                by_id[int(league["id"])] = item
            except (TypeError, ValueError):
                continue
            country = (item.get("country") or {}).get("name") or ""
            by_name[(str(league.get("name", "")).casefold(), str(country).casefold())] = item

        for config in self._competitions:
            league_id = config.provider_ids[PROVIDER]
            raw = by_id.get(league_id)
            if raw is None:
                raw = by_name.get((config.name.casefold(), config.country.casefold()))
                if raw is not None:
                    result.issues.append(
                        DataQualityIssue(
                            "competition",
                            config.key,
                            "provider_id",
                            Severity.WARNING,
                            f"id {league_id} not found; matched by name/country to "
                            f"id {raw['league']['id']} - update configs/competitions.toml",
                        )
                    )
            if raw is None:
                result.missing.append(config.key)
                continue
            try:
                competition = parse_league(raw, key=config.key)
            except ParseError as exc:
                result.issues.append(
                    DataQualityIssue("competition", config.key, "payload", Severity.ERROR, str(exc))
                )
                result.missing.append(config.key)
                continue
            issues = validate_competition_mapping(competition, config)
            result.issues.extend(issues)
            if has_errors(issues):
                result.missing.append(config.key)
                continue
            result.competitions.append(competition)

        logger.info(
            "competitions_discovered",
            extra={
                "found": [c.key for c in result.competitions],
                "missing": result.missing,
                "issues": len(result.issues),
                "from_cache": result.from_cache,
            },
        )
        return result

    @staticmethod
    def current_season(competition: Competition) -> int | None:
        season = competition.current_season
        return season.year if season else None

    # ------------------------------------------------------------- fixtures
    def _fixtures_ttl(self, day: date) -> int:
        today = self._now().astimezone(self._tz).date()
        if day < today - timedelta(days=2):
            return self._settings.cache_ttl_fixtures_past
        return self._settings.cache_ttl_fixtures_recent

    async def get_fixtures_by_date(
        self, day: date, competition_keys: list[str] | None = None
    ) -> FixtureBatch:
        wanted: set[int] = set(self._by_league_id)
        if competition_keys:
            unknown = set(competition_keys) - {c.key for c in self._competitions}
            if unknown:
                raise ConfigurationError(f"Unknown competition keys: {sorted(unknown)}")
            wanted = {
                lid for lid, c in self._by_league_id.items() if c.key in set(competition_keys)
            }

        resp = await self._client.get(
            "fixtures",
            {"date": day.isoformat(), "timezone": self._settings.timezone},
            cache_ttl=self._fixtures_ttl(day),
        )
        items: list[Any] = resp.response if isinstance(resp.response, list) else []
        batch = FixtureBatch(day=day, total_received=len(items), from_cache=resp.from_cache)
        now = self._now()
        seen: dict[int, Fixture] = {}

        for item in items:
            league_id = self._league_id(item)
            if league_id is None or league_id not in wanted:
                batch.other_competitions_skipped += 1
                continue
            config = self._by_league_id[league_id]
            try:
                fixture = parse_fixture(item, competition_key=config.key)
            except (ParseError, ValueError) as exc:
                fid = str(((item.get("fixture") or {}).get("id")) or "?")
                batch.issues.append(
                    DataQualityIssue("fixture", fid, "payload", Severity.ERROR, str(exc))
                )
                batch.rejected += 1
                continue
            issues = validate_fixture(fixture, now=now)
            batch.issues.extend(issues)
            if has_errors(issues):
                batch.rejected += 1
                continue
            if fixture.provider_id in seen:
                batch.duplicates_removed += 1
                if seen[fixture.provider_id] != fixture:
                    batch.issues.append(
                        DataQualityIssue(
                            "fixture",
                            str(fixture.provider_id),
                            "duplicate",
                            Severity.WARNING,
                            "conflicting duplicates in payload; last occurrence kept",
                        )
                    )
            seen[fixture.provider_id] = fixture

        batch.fixtures = sorted(
            seen.values(), key=lambda f: (f.kickoff_utc, f.competition_key or "", f.provider_id)
        )
        logger.info(
            "fixtures_by_date",
            extra={
                "day": day.isoformat(),
                "received": batch.total_received,
                "kept": len(batch.fixtures),
                "rejected": batch.rejected,
                "duplicates": batch.duplicates_removed,
                "from_cache": batch.from_cache,
            },
        )
        return batch

    @staticmethod
    def _league_id(item: Any) -> int | None:
        if not isinstance(item, dict):
            return None
        league = item.get("league")
        if not isinstance(league, dict):
            return None
        try:
            return int(league["id"])
        except (KeyError, TypeError, ValueError):
            return None
