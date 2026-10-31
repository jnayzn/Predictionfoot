"""API-Football implementation of :class:`FootballDataProvider`.

Quota strategy (free plan = 100 requests/day):

* ``discover_competitions`` -> **1 request** (``GET /leagues`` without filter)
  returns every league with all its seasons and coverage flags; the result is
  cached for a week. The six configured competitions are resolved locally.
* ``get_fixtures_by_date`` -> **1 request** (``GET /fixtures?date=...``)
  returns all fixtures worldwide for that day; we filter our competitions
  locally instead of issuing one request per league.
* ``get_season_fixtures`` / ``get_teams`` / ``get_standings`` / ``get_injuries``
  -> **1 request** per competition-season each (cached 30 days once finished).
* ``get_fixture_details`` -> statistics + lineups + player stats for up to
  **20 fixtures per request** (``GET /fixtures?ids=a-b-c``) instead of
  3 requests per fixture.
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
    parse_fixture_details,
    parse_injury,
    parse_league,
    parse_odds,
    parse_standings,
    parse_team,
)
from football_ai.data.collectors.base import CompetitionDiscovery, DetailsBatch, FixtureBatch
from football_ai.data.validators.fixtures import validate_competition_mapping, validate_fixture
from football_ai.data.validators.issues import DataQualityIssue, Severity, has_errors
from football_ai.domain.models import (
    Competition,
    Fixture,
    Injury,
    OddsQuote,
    StandingRow,
    Team,
)

logger = logging.getLogger(__name__)

# Maximum number of ids accepted by ``/fixtures?ids=`` (provider documentation).
DETAILS_CHUNK_SIZE = 20


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
            int(c.provider_ids[PROVIDER]): c for c in self._competitions
        }
        try:
            self._tz = ZoneInfo(settings.timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ConfigurationError(
                f"Invalid API_FOOTBALL_TIMEZONE {settings.timezone!r}"
            ) from exc

    @property
    def client(self) -> ApiFootballClient:
        return self._client

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
            league_id = int(config.provider_ids[PROVIDER])
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

    def season_ttl(self, season: int) -> int:
        """Long cache for finished seasons, short cache for the current/recent ones."""
        # A European season starting in year N ends by July of N+1.
        finished = self._now().date() > date(season + 1, 7, 31)
        if finished:
            return self._settings.cache_ttl_historical
        return self._settings.cache_ttl_season_current

    def config_for(self, competition_key: str) -> CompetitionConfig:
        for config in self._competitions:
            if config.key == competition_key:
                return config
        raise ConfigurationError(f"Unknown competition key: {competition_key!r}")

    def league_id(self, competition_key: str) -> int:
        return int(self.config_for(competition_key).provider_ids[PROVIDER])

    @property
    def covered_league_ids(self) -> set[int]:
        return set(self._by_league_id)

    def _wanted_league_ids(self, competition_keys: list[str] | None) -> set[int]:
        if not competition_keys:
            return set(self._by_league_id)
        return {self.league_id(k) for k in competition_keys}

    async def get_fixtures_by_date(
        self, day: date, competition_keys: list[str] | None = None
    ) -> FixtureBatch:
        wanted = self._wanted_league_ids(competition_keys)
        resp = await self._client.get(
            "fixtures",
            {"date": day.isoformat(), "timezone": self._settings.timezone},
            cache_ttl=self._fixtures_ttl(day),
        )
        batch = self.build_fixture_batch(resp.response, wanted, from_cache=resp.from_cache)
        batch.day = day
        batch.label = f"date={day.isoformat()}"
        self._log_batch(batch)
        return batch

    async def get_season_fixtures(self, competition_key: str, season: int) -> FixtureBatch:
        """All fixtures of one competition season - 1 request."""
        league_id = self.league_id(competition_key)
        resp = await self._client.get(
            "fixtures",
            {"league": league_id, "season": season, "timezone": self._settings.timezone},
            cache_ttl=self.season_ttl(season),
        )
        batch = self.build_fixture_batch(resp.response, {league_id}, from_cache=resp.from_cache)
        batch.label = f"{competition_key}:{season}"
        self._log_batch(batch)
        return batch

    async def get_fixture_details(
        self, fixture_ids: list[int], *, cache_ttl: int | None = None
    ) -> DetailsBatch:
        """Statistics, lineups and player stats - up to 20 fixtures per request."""
        result = DetailsBatch()
        unique_ids = list(dict.fromkeys(fixture_ids))
        for start in range(0, len(unique_ids), DETAILS_CHUNK_SIZE):
            chunk = unique_ids[start : start + DETAILS_CHUNK_SIZE]
            resp = await self._client.get(
                "fixtures",
                {"ids": "-".join(str(i) for i in chunk), "timezone": self._settings.timezone},
                cache_ttl=cache_ttl,
            )
            result.requests += 0 if resp.from_cache else 1
            parsed = self.parse_details_payload(resp.response)
            result.details.extend(parsed.details)
            result.issues.extend(parsed.issues)
        return result

    def parse_details_payload(self, response: Any) -> DetailsBatch:
        result = DetailsBatch()
        for item in response if isinstance(response, list) else []:
            league_id = self._league_id(item)
            config = self._by_league_id.get(league_id) if league_id is not None else None
            try:
                details = parse_fixture_details(
                    item, competition_key=config.key if config else None
                )
            except (ParseError, ValueError) as exc:
                result.issues.append(self._payload_issue("fixture", item, exc))
                continue
            issues = validate_fixture(details.fixture, now=self._now())
            result.issues.extend(issues)
            if has_errors(issues):
                continue
            result.details.append(details)
        return result

    # ------------------------------------------------------- reference data
    async def get_teams(self, competition_key: str, season: int) -> list[Team]:
        resp = await self._client.get(
            "teams",
            {"league": self.league_id(competition_key), "season": season},
            cache_ttl=self.season_ttl(season),
        )
        return self.parse_teams_payload(resp.response)

    @staticmethod
    def parse_teams_payload(response: Any) -> list[Team]:
        teams: dict[int, Team] = {}
        for item in response if isinstance(response, list) else []:
            try:
                team = parse_team(item)
            except (ParseError, ValueError) as exc:
                logger.warning("team_parse_error", extra={"error": str(exc)})
                continue
            teams[team.provider_id] = team
        return list(teams.values())

    async def get_standings(self, competition_key: str, season: int) -> list[StandingRow]:
        resp = await self._client.get(
            "standings",
            {"league": self.league_id(competition_key), "season": season},
            cache_ttl=self.season_ttl(season),
        )
        return self.parse_standings_payload(resp.response)

    @staticmethod
    def parse_standings_payload(response: Any) -> list[StandingRow]:
        rows: list[StandingRow] = []
        for item in response if isinstance(response, list) else []:
            try:
                rows.extend(parse_standings(item))
            except (ParseError, ValueError) as exc:
                logger.warning("standings_parse_error", extra={"error": str(exc)})
        return rows

    async def get_injuries(self, competition_key: str, season: int) -> list[Injury]:
        resp = await self._client.get(
            "injuries",
            {
                "league": self.league_id(competition_key),
                "season": season,
                "timezone": self._settings.timezone,
            },
            cache_ttl=self.season_ttl(season),
        )
        return self.parse_injuries_payload(resp.response)

    @staticmethod
    def parse_injuries_payload(response: Any) -> list[Injury]:
        injuries: list[Injury] = []
        for item in response if isinstance(response, list) else []:
            try:
                injuries.append(parse_injury(item))
            except (ParseError, ValueError) as exc:
                logger.warning("injury_parse_error", extra={"error": str(exc)})
        return injuries

    async def get_fixture_odds(self, fixture_id: int) -> list[OddsQuote]:
        """Pre-match odds of one fixture (paginated by the provider)."""
        pages = await self._client.get_all_pages(
            "odds", {"fixture": fixture_id}, cache_ttl=self._settings.cache_ttl_fixtures_recent
        )
        quotes: list[OddsQuote] = []
        for page in pages:
            quotes.extend(self.parse_odds_payload(page.response))
        return quotes

    @staticmethod
    def parse_odds_payload(response: Any) -> list[OddsQuote]:
        quotes: list[OddsQuote] = []
        for item in response if isinstance(response, list) else []:
            try:
                quotes.extend(parse_odds(item))
            except (ParseError, ValueError) as exc:
                logger.warning("odds_parse_error", extra={"error": str(exc)})
        return quotes

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _payload_issue(entity: str, item: Any, exc: Exception) -> DataQualityIssue:
        fid = "?"
        if isinstance(item, dict):
            fid = str(((item.get("fixture") or {}).get("id")) or "?")
        return DataQualityIssue(entity, fid, "payload", Severity.ERROR, str(exc))

    def build_fixture_batch(
        self, response: Any, wanted: set[int], *, from_cache: bool = False
    ) -> FixtureBatch:
        items: list[Any] = response if isinstance(response, list) else []
        batch = FixtureBatch(total_received=len(items), from_cache=from_cache)
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
                batch.issues.append(self._payload_issue("fixture", item, exc))
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
        return batch

    @staticmethod
    def _log_batch(batch: FixtureBatch) -> None:
        logger.info(
            "fixtures_batch",
            extra={
                "scope": batch.label,
                "received": batch.total_received,
                "kept": len(batch.fixtures),
                "rejected": batch.rejected,
                "duplicates": batch.duplicates_removed,
                "from_cache": batch.from_cache,
            },
        )

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
