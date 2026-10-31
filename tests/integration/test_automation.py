"""Scheduled jobs: configuration, runner bookkeeping, API jobs within budget,
prediction refresh on new information, alerts."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from football_ai.automation.config import load_schedule
from football_ai.automation.jobs import JOBS, JobContext, JobOutcome, run_job
from football_ai.automation.scheduler import build_scheduler
from football_ai.bootstrap import ApiFootballContext
from football_ai.config.competitions import CompetitionConfig
from football_ai.config.settings import ApiFootballSettings, Settings
from football_ai.core.exceptions import ConfigurationError
from football_ai.data.collectors.api_football.client import ApiFootballClient
from football_ai.data.collectors.api_football.provider import ApiFootballProvider
from football_ai.database import models as db
from football_ai.database.connection import session_scope
from football_ai.prediction.engine import PredictionEngine
from tests.conftest import ROOT
from tests.integration.test_pipeline import FakeApi
from tests.integration.test_pipeline import fake_api as fake_api  # re-exported fixture

NOW = datetime(2025, 11, 1, 12, tzinfo=UTC)  # FakeApi seasons: 2024, 2025


class RecordingAlerts:
    def __init__(self) -> None:
        self.sent: list[tuple[str, list[str]]] = []

    async def send(self, title: str, lines: list[str]) -> None:
        self.sent.append((title, lines))


def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "var",  # type: ignore[call-arg]
        competitions_file=ROOT / "configs" / "competitions.toml",
        team_aliases_file=ROOT / "configs" / "team_aliases.toml",
    )


@pytest.fixture
def make_ctx(
    make_client: Callable[..., ApiFootballClient],
    competitions: list[CompetitionConfig],
    api_settings: ApiFootballSettings,
    tmp_path: Path,
) -> Callable[..., JobContext]:
    def _make(
        factory: sessionmaker[Session],
        *,
        now: datetime = NOW,
        cfg: ApiFootballSettings | None = None,
        engine: PredictionEngine | None = None,
    ) -> JobContext:
        api_cfg = cfg or api_settings

        @asynccontextmanager
        async def api_context() -> AsyncIterator[ApiFootballContext]:
            client = make_client(api_cfg)
            provider = ApiFootballProvider(client, competitions, api_cfg, now=lambda: now)
            async with client:
                yield ApiFootballContext(client, provider, competitions)

        return JobContext(
            settings=settings(tmp_path),
            session_factory=factory,
            engine=engine or PredictionEngine(factory),
            api_context=api_context,
            alerts=RecordingAlerts(),
            now=lambda: now,
        )

    return _make


def competitions_config() -> list[CompetitionConfig]:
    from football_ai.config.competitions import load_competitions

    return load_competitions(ROOT / "configs" / "competitions.toml")


def count(factory: sessionmaker[Session], model: type[db.Base], *where: Any) -> int:
    with session_scope(factory) as s:
        stmt = select(func.count()).select_from(model)
        for w in where:
            stmt = stmt.where(w)
        return s.scalar(stmt) or 0


# ------------------------------------------------------------- configuration
def test_schedule_file_and_scheduler(
    make_ctx: Callable[..., JobContext], session_factory: sessionmaker[Session]
) -> None:
    schedule = load_schedule(ROOT / "configs" / "schedule.toml")
    assert set(schedule.jobs) <= set(JOBS) and schedule.timezone == "Europe/Paris"
    assert schedule.jobs["odds"].options["max_requests"] == 15
    scheduler = build_scheduler(make_ctx(session_factory), schedule)
    jobs = {j.id: j for j in scheduler.get_jobs()}
    assert set(jobs) == {n for n, c in schedule.jobs.items() if c.enabled}
    assert jobs["sync_fixtures"].max_instances == 1 and jobs["sync_fixtures"].coalesce
    assert "hour='7,13,19'" in str(jobs["sync_fixtures"].trigger)


def test_invalid_schedules(
    tmp_path: Path, make_ctx: Callable[..., JobContext], session_factory: sessionmaker[Session]
) -> None:
    bad = tmp_path / "s.toml"
    bad.write_text('[jobs.sync_fixtures]\ncron = "0 7 * *"\n')
    with pytest.raises(ConfigurationError):
        load_schedule(bad)
    bad.write_text('[jobs.mine_bitcoin]\ncron = "0 7 * * *"\n')
    with pytest.raises(ConfigurationError):
        build_scheduler(make_ctx(session_factory), load_schedule(bad))


# -------------------------------------------------------------------- runner
async def test_runner_records_success_and_failure(
    make_ctx: Callable[..., JobContext],
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = make_ctx(session_factory)

    async def boom(ctx: JobContext, opts: dict[str, Any]) -> JobOutcome:
        raise RuntimeError("collector exploded")

    monkeypatch.setitem(JOBS, "boom", boom)
    outcome = await run_job("boom", ctx)
    assert outcome.status == "failed"
    assert ctx.alerts.sent and "boom" in ctx.alerts.sent[0][0]  # type: ignore[attr-defined]
    ok = await run_job("evaluate_predictions", ctx)
    assert ok.status == "success"
    with session_scope(session_factory) as s:
        runs = {r.job: r for r in s.scalars(select(db.ScheduledJobRun))}
    assert runs["boom"].status == "failed" and "collector exploded" in (runs["boom"].error or "")
    assert runs["evaluate_predictions"].finished_at is not None


# ------------------------------------------------------------------ API jobs
async def test_api_jobs_respect_budget(
    fake_api: FakeApi,
    make_ctx: Callable[..., JobContext],
    session_factory: sessionmaker[Session],
    api_settings: ApiFootballSettings,
) -> None:
    ctx = make_ctx(session_factory)
    first_day = min(f["fixture"]["date"][:10] for f in fake_api.fixtures.values())
    day_ctx = make_ctx(session_factory, now=datetime.fromisoformat(first_day + "T23:00:00+00:00"))
    synced = await run_job("sync_fixtures", day_ctx)
    assert synced.status == "success" and synced.requests_used == 5  # today, +3 days, yesterday
    assert count(session_factory, db.Fixture) >= 1

    # import two seasons, then details within a budget of 1 request (20 fixtures)
    hist = await run_job(
        "historical_import",
        ctx,
        load_schedule(ROOT / "configs" / "schedule.toml")
        .jobs["historical_import"]
        .model_copy(update={"options": {"seasons": "2024-2025", "max_requests": 4}}),
    )
    assert hist.status == "partial" and hist.requests_used == 4
    from football_ai.automation.config import JobConfig

    late = make_ctx(session_factory, now=datetime(2026, 9, 1, tzinfo=UTC))
    details = await run_job(
        "match_details",
        late,
        JobConfig(cron="0 0 * * *", options={"days_back": 800, "max_requests": 1}),
    )
    pending = details.details["pending"]
    assert details.requests_used == 1  # budget of 1 request respected
    assert details.details["loaded"] == min(20, pending)
    assert details.status == ("partial" if pending > 20 else "success")

    # quota nearly exhausted -> job stops without exceeding it
    tight = api_settings.model_copy(update={"daily_limit": 3, "quota_reserve": 1})
    starved = await run_job("refresh_results", make_ctx(session_factory, cfg=tight))
    assert starved.status in ("partial", "success") and starved.requests_used <= 2


async def test_odds_and_injuries_jobs(
    fake_api: FakeApi,
    make_ctx: Callable[..., JobContext],
    session_factory: sessionmaker[Session],
) -> None:
    from football_ai.automation.config import JobConfig

    ctx = make_ctx(session_factory)
    await run_job(
        "historical_import",
        ctx,
        JobConfig(cron="0 0 * * *", options={"seasons": "2025", "max_requests": 3}),
    )
    with session_scope(session_factory) as s:
        fixture = s.scalars(select(db.Fixture)).first()
        assert fixture is not None
        fixture.status_group, fixture.status_short = "scheduled", "NS"
        kickoff = fixture.kickoff_utc
        league = s.scalar(select(db.League).where(db.League.key == "premier_league"))
        assert league is not None
        season = s.scalar(
            select(db.Season).where(db.Season.league_id == league.id, db.Season.year == 2025)
        )
        assert season is not None  # synthetic provider marks 2026 as current
    near = make_ctx(session_factory, now=kickoff - timedelta(hours=5))
    result = await run_job("odds", near)
    assert result.requests_used == 1 and result.details["quotes"] == 8
    inj = await run_job("injuries", near)
    # one request per competition with a known current season (all six here)
    assert set(inj.details["competitions"]) == {c.key for c in competitions_config()}
    assert inj.requests_used == 6


# --------------------------------------------------------- internal jobs
async def test_prediction_job_stores_only_new_information(
    world: dict[str, Any], engine: PredictionEngine, make_ctx: Callable[..., JobContext]
) -> None:
    factory = world["factory"]
    ctx = make_ctx(factory, now=datetime.now(UTC), engine=engine)
    before = count(factory, db.Prediction)
    first = await run_job("predictions", ctx)
    assert first.details["upcoming"] == len(world["seeded"].upcoming_match_ids)
    stored_first = count(factory, db.Prediction) - before
    assert stored_first >= len(world["seeded"].upcoming_match_ids) - 1
    again = await run_job("predictions", ctx)
    assert again.details["stored"] == 0  # same information -> no new version
    # new information: an injury list now covers the match -> features change
    match_id = world["seeded"].upcoming_match_ids[0]
    with session_scope(factory) as s:
        match = s.get(db.Match, match_id)
        assert match is not None
        s.add(
            db.Odds(
                fixture_id=match.primary_fixture_id,
                provider="api_football",
                bookmaker="NewBook",
                market="1x2",
                selection="home",
                odd=1.2,
                odds_type="pre_match",
            )
        )
        s.add(
            db.Odds(
                fixture_id=match.primary_fixture_id,
                provider="api_football",
                bookmaker="NewBook",
                market="1x2",
                selection="draw",
                odd=7.0,
                odds_type="pre_match",
            )
        )
        s.add(
            db.Odds(
                fixture_id=match.primary_fixture_id,
                provider="api_football",
                bookmaker="NewBook",
                market="1x2",
                selection="away",
                odd=13.0,
                odds_type="pre_match",
            )
        )
    refreshed = await run_job("predictions", ctx)
    assert refreshed.details["stored"] == 1


async def test_quality_check_alerts(
    world: dict[str, Any], engine: PredictionEngine, make_ctx: Callable[..., JobContext]
) -> None:
    factory = world["factory"]
    ctx = make_ctx(factory, now=datetime.now(UTC), engine=engine)
    with session_scope(factory) as s:
        s.add(
            db.IngestionJob(
                provider="api_football",
                job_type="fixtures",
                scope="ligue_1:2025",
                status="failed",
                last_error="HTTP 500",
                attempts=1,
                requests_used=1,
                items_loaded=0,
            )
        )
    outcome = await run_job("quality_check", ctx)
    assert outcome.details["alerts"] >= 1
    titles = [t for t, _ in ctx.alerts.sent]  # type: ignore[attr-defined]
    assert any("Contrôle qualité" in t for t in titles)
    assert list((ctx.settings.data_dir / "reports").glob("quality_*.json"))
    skipped = await run_job("daily_report", ctx)
    assert skipped.status == "skipped"
    built = await run_job("build_matches", ctx)
    assert "premier_league" in built.details
