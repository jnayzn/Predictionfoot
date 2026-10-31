"""API-Football ETL operations. Each method = one Extract -> Transform -> Validate -> Load run.

Extraction, transformation and validation happen in the provider (which only
returns validated domain objects); loading happens here, inside one database
transaction per operation, through the repositories.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from sqlalchemy.orm import Session, sessionmaker

from football_ai.data.collectors.api_football.provider import ApiFootballProvider
from football_ai.data.validators.issues import DataQualityIssue
from football_ai.database.connection import session_scope
from football_ai.database.repositories import (
    FixtureRepository,
    MatchDataRepository,
    ReferenceRepository,
    UpsertStats,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LoadReport:
    operation: str
    scope: str
    items: int = 0
    fixtures: UpsertStats = field(default_factory=UpsertStats)
    issues: list[DataQualityIssue] = field(default_factory=list)
    requests: int = 0

    def summary(self) -> str:
        f = self.fixtures
        return (
            f"{self.operation}[{self.scope}] items={self.items} "
            f"fixtures(+{f.inserted} ~{f.updated} ={f.unchanged} corrections={f.corrections}) "
            f"issues={len(self.issues)} requests={self.requests}"
        )


class ApiFootballEtl:
    def __init__(
        self,
        provider: ApiFootballProvider,
        session_factory: sessionmaker[Session],
        *,
        today: date | None = None,
    ) -> None:
        self.provider = provider
        self.session_factory = session_factory
        self._today = today

    @property
    def today(self) -> date:
        return self._today or datetime.now(UTC).date()

    async def sync_competitions(self) -> LoadReport:
        discovery = await self.provider.discover_competitions()
        report = LoadReport("competitions", "all", issues=list(discovery.issues))
        with session_scope(self.session_factory) as session:
            refs = ReferenceRepository(session)
            for competition in discovery.competitions:
                refs.upsert_competition(competition)
        report.items = len(discovery.competitions)
        return report

    async def sync_fixtures_by_date(self, day: date) -> LoadReport:
        batch = await self.provider.get_fixtures_by_date(day)
        report = LoadReport("fixtures_by_date", day.isoformat(), issues=list(batch.issues))
        with session_scope(self.session_factory) as session:
            report.fixtures = FixtureRepository(session).upsert_fixtures(batch.fixtures)
        report.items = len(batch.fixtures)
        return report

    async def import_season_fixtures(self, key: str, season: int) -> LoadReport:
        batch = await self.provider.get_season_fixtures(key, season)
        report = LoadReport("fixtures", f"{key}:{season}", issues=list(batch.issues))
        with session_scope(self.session_factory) as session:
            report.fixtures = FixtureRepository(session).upsert_fixtures(batch.fixtures)
        report.items = len(batch.fixtures)
        return report

    async def import_teams(self, key: str, season: int) -> LoadReport:
        teams = await self.provider.get_teams(key, season)
        with session_scope(self.session_factory) as session:
            refs = ReferenceRepository(session)
            for team in teams:
                refs.upsert_team(team)
        return LoadReport("teams", f"{key}:{season}", items=len(teams))

    async def import_standings(self, key: str, season: int) -> LoadReport:
        rows = await self.provider.get_standings(key, season)
        with session_scope(self.session_factory) as session:
            MatchDataRepository(session).save_standings(
                rows, snapshot_date=self.today, league_key=key
            )
        return LoadReport("standings", f"{key}:{season}", items=len(rows))

    async def import_injuries(self, key: str, season: int) -> LoadReport:
        injuries = await self.provider.get_injuries(key, season)
        with session_scope(self.session_factory) as session:
            MatchDataRepository(session).save_injuries(injuries, league_key=key)
        return LoadReport("injuries", f"{key}:{season}", items=len(injuries))

    def pending_detail_ids(self, key: str, season: int, limit: int | None = None) -> list[int]:
        league_id = self.provider.league_id(key)
        with session_scope(self.session_factory) as session:
            refs = ReferenceRepository(session)
            league = refs.ensure_league(self.provider.name, league_id, key, key=key)
            return FixtureRepository(session, refs).finished_without_details(
                self.provider.name, league.id, season, limit
            )

    async def import_fixture_details(self, fixture_ids: list[int], scope: str) -> LoadReport:
        batch = await self.provider.get_fixture_details(fixture_ids)
        report = LoadReport("details", scope, issues=list(batch.issues), requests=batch.requests)
        with session_scope(self.session_factory) as session:
            report.fixtures = MatchDataRepository(session).save_fixture_details(batch.details)
        report.items = len(batch.details)
        return report

    async def import_fixture_odds(self, fixture_id: int) -> LoadReport:
        quotes = await self.provider.get_fixture_odds(fixture_id)
        with session_scope(self.session_factory) as session:
            inserted = MatchDataRepository(session).save_odds(quotes)
        return LoadReport("odds", str(fixture_id), items=inserted)
