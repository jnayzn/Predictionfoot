"""End-to-end ETL tests: mocked API-Football (respx) -> provider -> repositories -> DB."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from football_ai.config.competitions import CompetitionConfig
from football_ai.config.settings import ApiFootballSettings
from football_ai.data.collectors.api_football.client import ApiFootballClient
from football_ai.data.collectors.api_football.provider import ApiFootballProvider
from football_ai.data.pipeline.etl import ApiFootballEtl
from football_ai.data.pipeline.football_data_uk import load_csv_season
from football_ai.data.pipeline.historical import HistoricalImporter
from football_ai.data.pipeline.replay import RawReplayer
from football_ai.database import models as db
from football_ai.database.connection import session_scope
from football_ai.database.models import Base
from tests.conftest import BASE_URL, all_leagues_payload, envelope, fixture_item
from tests.payloads import add_details, injury_item, standings_item, team_item
from tests.unit.test_football_data_uk import CSV_WITH_TIME

NOW = datetime(2026, 10, 31, 12, 0, tzinfo=UTC)
TEAMS = [(33, "Alpha"), (34, "Beta"), (35, "Gamma"), (36, "Delta"), (37, "Epsilon"), (38, "Zeta")]


def season_fixtures(league_id: int, season: int, n: int = 25) -> dict[int, dict[str, Any]]:
    items: dict[int, dict[str, Any]] = {}
    start = datetime(season, 8, 10, 15, tzinfo=UTC)
    for i in range(n):
        home, away = TEAMS[i % 6], TEAMS[(i + 1 + i // 6) % 6]
        if home == away:
            away = TEAMS[(i + 2) % 6]
        fid = league_id * 1_000_000 + season * 100 + i
        items[fid] = fixture_item(
            fid,
            league_id,
            home,
            away,
            status="FT",
            goals=(i % 3, (i + 1) % 2),
            date=(start + timedelta(days=7 * i)).isoformat(),
            season=season,
        )
    return items


class FakeApi:
    """Dispatches API-Football endpoints. ``plan_restricted`` seasons answer errors.plan."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.plan_restricted: set[int] = set()
        self.daily_limit_reached = False
        self.fixtures: dict[int, dict[str, Any]] = {}
        for season in (2024, 2025):
            self.fixtures.update(season_fixtures(39, season))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        endpoint = request.url.path.strip("/")
        params = dict(request.url.params)
        self.calls.append(f"{endpoint}?{sorted(params.items())}")
        if self.daily_limit_reached:
            return httpx.Response(200, json=envelope(endpoint, [], errors={"requests": "limit"}))
        season = int(params.get("season", 0))
        if season in self.plan_restricted:
            return httpx.Response(
                200, json=envelope(endpoint, [], errors={"plan": "Free plans do not have access"})
            )
        if endpoint == "leagues":
            return httpx.Response(200, json=all_leagues_payload())
        if endpoint == "fixtures" and "ids" in params:
            ids = [int(x) for x in params["ids"].split("-")]
            assert len(ids) <= 20
            items = []
            for fid in ids:
                item = self.fixtures[fid]
                teams = item["teams"]
                items.append(
                    add_details(
                        item,
                        (teams["home"]["id"], teams["home"]["name"]),
                        (teams["away"]["id"], teams["away"]["name"]),
                    )
                )
            return httpx.Response(200, json=envelope(endpoint, items))
        if endpoint == "fixtures":
            items = [
                f
                for f in self.fixtures.values()
                if f["league"]["id"] == int(params["league"]) and f["league"]["season"] == season
            ]
            return httpx.Response(200, json=envelope(endpoint, items))
        if endpoint == "teams":
            return httpx.Response(200, json=envelope(endpoint, [team_item(t, n) for t, n in TEAMS]))
        if endpoint == "standings":
            rows = [(t, n, rank + 1, 30 - rank) for rank, (t, n) in enumerate(TEAMS)]
            return httpx.Response(200, json=envelope(endpoint, [standings_item(39, season, rows)]))
        if endpoint == "injuries":
            fid = next(iter(self.fixtures))
            return httpx.Response(
                200, json=envelope(endpoint, [injury_item(990, TEAMS[0], 39, season, fid)])
            )
        return httpx.Response(404)


@pytest.fixture
def fake_api() -> Any:
    api = FakeApi()
    with respx.mock(base_url=BASE_URL, assert_all_called=False) as mock:
        mock.route().mock(side_effect=api)
        yield api


@pytest.fixture
def make_etl(
    make_client: Callable[..., ApiFootballClient],
    competitions: list[CompetitionConfig],
    api_settings: ApiFootballSettings,
    session_factory: sessionmaker[Session],
) -> Callable[..., ApiFootballEtl]:
    def _make(settings: ApiFootballSettings | None = None) -> ApiFootballEtl:
        cfg = settings or api_settings
        provider = ApiFootballProvider(make_client(cfg), competitions, cfg, now=lambda: NOW)
        return ApiFootballEtl(provider, session_factory, today=NOW.date())

    return _make


def count(session_factory: sessionmaker[Session], model: type[Base]) -> int:
    with session_scope(session_factory) as s:
        return s.scalar(select(func.count()).select_from(model)) or 0


async def test_full_season_import_then_resume(
    fake_api: FakeApi,
    make_etl: Callable[..., ApiFootballEtl],
    session_factory: sessionmaker[Session],
) -> None:
    importer = HistoricalImporter(make_etl(), max_requests=50)
    report = await importer.run(["premier_league"], [2025])

    assert report.stopped_reason is None and report.jobs_failed == []
    assert sorted(j.split(":")[0] for j in report.jobs_done) == [
        "details",
        "fixtures",
        "injuries",
        "standings",
        "teams",
    ]
    # leagues 1 + fixtures 1 + teams 1 + standings 1 + injuries 1 + details ceil(25/20)=2
    assert report.requests_used == 7
    assert count(session_factory, db.Fixture) == 25
    assert count(session_factory, db.ActualResult) == 25
    assert count(session_factory, db.TeamStatistics) == 50
    assert count(session_factory, db.Standing) == 6
    assert count(session_factory, db.Injury) == 1
    assert count(session_factory, db.Season) > 0

    # second run: finished season already imported -> nothing fetched (leagues from cache)
    again = await HistoricalImporter(make_etl(), max_requests=50).run(["premier_league"], [2025])
    assert again.requests_used == 0
    assert len(again.jobs_skipped) == 5 and again.jobs_done == []


async def test_budget_stops_and_next_run_resumes(
    fake_api: FakeApi,
    make_etl: Callable[..., ApiFootballEtl],
    session_factory: sessionmaker[Session],
) -> None:
    first = await HistoricalImporter(make_etl(), max_requests=3).run(["premier_league"], [2025])
    assert first.stopped_reason == "request budget exhausted"
    assert first.requests_used == 3  # leagues + fixtures + teams
    assert [j.split(":")[0] for j in first.jobs_done] == ["fixtures", "teams"]

    second = await HistoricalImporter(make_etl(), max_requests=4).run(["premier_league"], [2025])
    # standings + injuries + 2 detail chunks = 4 requests, leagues served from cache
    assert second.requests_used == 4
    assert second.stopped_reason is None
    assert count(session_factory, db.TeamStatistics) == 50


async def test_details_are_resumed_chunk_by_chunk(
    fake_api: FakeApi,
    make_etl: Callable[..., ApiFootballEtl],
    session_factory: sessionmaker[Session],
) -> None:
    await HistoricalImporter(make_etl(), max_requests=2).run(
        ["premier_league"], [2025], steps=["fixtures"]
    )
    partial = await HistoricalImporter(make_etl(), max_requests=1).run(
        ["premier_league"], [2025], steps=["details"]
    )
    assert partial.stopped_reason == "request budget exhausted"
    with session_scope(session_factory) as s:
        done = s.scalar(select(func.count()).where(db.Fixture.details_fetched_at.is_not(None)))
    assert done == 20
    final = await HistoricalImporter(make_etl(), max_requests=5).run(
        ["premier_league"], [2025], steps=["details"]
    )
    assert final.requests_used == 1 and final.jobs_done == ["details:premier_league:2025"]


async def test_plan_restriction_skips_season_but_continues(
    fake_api: FakeApi,
    make_etl: Callable[..., ApiFootballEtl],
    session_factory: sessionmaker[Session],
) -> None:
    fake_api.plan_restricted = {2024}
    report = await HistoricalImporter(make_etl(), max_requests=50).run(
        ["premier_league"], [2024, 2025], steps=["fixtures", "teams"]
    )
    assert report.jobs_failed == ["fixtures:premier_league:2024 (plan restriction)"]
    assert set(report.jobs_done) == {"fixtures:premier_league:2025", "teams:premier_league:2025"}
    with session_scope(session_factory) as s:
        failed = s.scalar(select(db.IngestionJob).where(db.IngestionJob.status == "failed"))
        assert failed is not None and "Free plans" in (failed.last_error or "")


async def test_daily_quota_stops_run(
    fake_api: FakeApi, make_etl: Callable[..., ApiFootballEtl]
) -> None:
    fake_api.daily_limit_reached = True
    report = await HistoricalImporter(make_etl(), max_requests=50).run(["premier_league"], [2025])
    assert report.stopped_reason is not None and "quota" in report.stopped_reason


async def test_replay_rebuilds_database_without_requests(
    fake_api: FakeApi,
    make_etl: Callable[..., ApiFootballEtl],
    session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    etl = make_etl()
    await HistoricalImporter(etl, max_requests=50).run(["premier_league"], [2025])
    before = {
        m: count(session_factory, m)
        for m in (db.Fixture, db.TeamStatistics, db.Standing, db.LineupPlayer, db.Team, db.Season)
    }
    calls = len(fake_api.calls)

    with session_scope(session_factory) as s:  # wipe data tables
        for table in reversed(Base.metadata.sorted_tables):
            s.execute(table.delete())
    assert count(session_factory, db.Fixture) == 0

    report = RawReplayer(tmp_path / "raw", etl.provider, session_factory).run()
    assert len(fake_api.calls) == calls  # zero network call
    assert report.files == 7
    after = {m: count(session_factory, m) for m in before}
    assert after == before


async def test_sync_fixtures_by_date(
    make_etl: Callable[..., ApiFootballEtl], session_factory: sessionmaker[Session]
) -> None:
    respx_payload = envelope("fixtures", [fixture_item(1, 39, (33, "A"), (34, "B"))])
    with respx.mock(base_url=BASE_URL) as mock:
        mock.get("/fixtures").mock(return_value=httpx.Response(200, json=respx_payload))
        report = await make_etl().sync_fixtures_by_date(date(2026, 10, 31))
    assert report.fixtures.inserted == 1
    assert count(session_factory, db.Fixture) == 1


def test_csv_season_load(
    session_factory: sessionmaker[Session], competitions: list[CompetitionConfig]
) -> None:
    config = next(c for c in competitions if c.key == "premier_league")
    for _ in range(2):  # idempotent
        report = load_csv_season(CSV_WITH_TIME.encode(), config, 2024, session_factory)
    assert report.fixtures.unchanged == 2
    assert count(session_factory, db.Fixture) == 2
    assert count(session_factory, db.ActualResult) == 2
    assert count(session_factory, db.TeamStatistics) == 4
    assert count(session_factory, db.Odds) > 20
    with session_scope(session_factory) as s:
        league = s.scalar(select(db.League))
        assert league is not None and (league.provider, league.key) == (
            "football_data_uk",
            "premier_league",
        )


async def test_two_seasons_and_request_accounting(
    fake_api: FakeApi,
    make_etl: Callable[..., ApiFootballEtl],
    session_factory: sessionmaker[Session],
) -> None:
    report = await HistoricalImporter(make_etl(), max_requests=50).run(
        ["premier_league"], [2024, 2025], steps=["fixtures"]
    )
    assert count(session_factory, db.Fixture) == 50
    assert [load.requests for load in report.loads] == [1, 1, 1]  # leagues + 2 seasons
