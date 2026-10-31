"""Scheduled jobs. Each job is small, idempotent and safe to re-run.

API-consuming jobs check the remaining daily quota and their own
``max_requests`` before every call; when the budget is exhausted they stop
cleanly (status ``partial``) and the next run continues.
"""

from __future__ import annotations

import logging
import traceback
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from football_ai.automation.alerts import AlertSink, LogAlertSink
from football_ai.automation.config import JobConfig
from football_ai.bootstrap import ApiFootballContext
from football_ai.config.settings import Settings
from football_ai.core.exceptions import DataProviderError, QuotaExceededError
from football_ai.data.cleaning.canonical import CanonicalBuilder
from football_ai.data.collectors.api_football.parsers import PROVIDER
from football_ai.data.collectors.api_football.provider import DETAILS_CHUNK_SIZE
from football_ai.data.pipeline.etl import ApiFootballEtl
from football_ai.data.pipeline.historical import HistoricalImporter
from football_ai.data.quality.checks import load_frames, run_checks
from football_ai.data.quality.report import save_report
from football_ai.database import models as db
from football_ai.database.connection import session_scope
from football_ai.database.repositories import PredictionRepository
from football_ai.prediction.engine import PredictionEngine

logger = logging.getLogger(__name__)

ApiContextFactory = Callable[[], AbstractAsyncContextManager[ApiFootballContext]]


@dataclass(slots=True)
class JobOutcome:
    status: str = "success"  # success | partial | skipped | failed
    requests_used: int = 0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class JobContext:
    settings: Settings
    session_factory: sessionmaker[Session]
    engine: PredictionEngine
    api_context: ApiContextFactory
    alerts: AlertSink = field(default_factory=LogAlertSink)
    bot: Any = None  # aiogram Bot (daily report)
    backend: Any = None  # BotBackend (daily report)
    now: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))


JobFunction = Callable[[JobContext, dict[str, Any]], Awaitable[JobOutcome]]


def _budget(ctx_api: ApiFootballContext, max_requests: int | None, used: int) -> int:
    available = ctx_api.client.quota.snapshot().available
    if max_requests is not None:
        available = min(available, max_requests - used)
    return max(0, available)


# ------------------------------------------------------------- API jobs
async def sync_fixtures(ctx: JobContext, opts: dict[str, Any]) -> JobOutcome:
    """Fixtures + results from yesterday to today + days_ahead (most imminent first)."""
    days_ahead = int(opts.get("days_ahead", 3))
    today = ctx.now().date()
    days = [
        today,
        *(today + timedelta(days=d) for d in range(1, days_ahead + 1)),
        today - timedelta(days=1),
    ]
    return await _sync_days(ctx, days)


async def refresh_results(ctx: JobContext, opts: dict[str, Any]) -> JobOutcome:
    days_back = int(opts.get("days_back", 2))
    today = ctx.now().date()
    return await _sync_days(ctx, [today - timedelta(days=d) for d in range(0, days_back + 1)])


async def _sync_days(ctx: JobContext, days: list[Any]) -> JobOutcome:
    outcome = JobOutcome(details={"days": [], "corrections": 0, "fixtures": 0})
    async with ctx.api_context() as api:
        etl = ApiFootballEtl(api.provider, ctx.session_factory, today=ctx.now().date())
        start = api.client.requests_sent
        for day in days:
            if _budget(api, None, 0) <= 0:
                outcome.status = "partial"
                break
            report = await etl.sync_fixtures_by_date(day)
            outcome.details["days"].append(day.isoformat())
            outcome.details["fixtures"] += report.items
            outcome.details["corrections"] += report.fixtures.corrections
        outcome.requests_used = api.client.requests_sent - start
    return outcome


async def match_details(ctx: JobContext, opts: dict[str, Any]) -> JobOutcome:
    """Statistics/lineups/players of recently finished matches (20 per request)."""
    days_back = int(opts.get("days_back", 7))
    max_requests = int(opts.get("max_requests", 10))
    since = ctx.now() - timedelta(days=days_back)
    with session_scope(ctx.session_factory) as session:
        pending = list(
            session.scalars(
                select(db.Fixture.provider_id)
                .where(
                    db.Fixture.provider == PROVIDER,
                    db.Fixture.status_group == "finished",
                    db.Fixture.details_fetched_at.is_(None),
                    db.Fixture.kickoff_utc >= since,
                )
                .order_by(db.Fixture.kickoff_utc.desc())
            )
        )
    outcome = JobOutcome(details={"pending": len(pending), "loaded": 0})
    if not pending:
        outcome.status = "skipped"
        return outcome
    async with ctx.api_context() as api:
        etl = ApiFootballEtl(api.provider, ctx.session_factory, today=ctx.now().date())
        start = api.client.requests_sent
        for i in range(0, len(pending), DETAILS_CHUNK_SIZE):
            if _budget(api, max_requests, api.client.requests_sent - start) <= 0:
                outcome.status = "partial"
                break
            report = await etl.import_fixture_details(pending[i : i + DETAILS_CHUNK_SIZE], "recent")
            outcome.details["loaded"] += report.items
        outcome.requests_used = api.client.requests_sent - start
    return outcome


async def injuries(ctx: JobContext, opts: dict[str, Any]) -> JobOutcome:
    outcome = JobOutcome(details={"competitions": {}})
    with session_scope(ctx.session_factory) as session:
        current = {
            key: year
            for key, year in session.execute(
                select(db.League.key, db.Season.year)
                .join(db.Season, db.Season.league_id == db.League.id)
                .where(db.League.provider == PROVIDER, db.Season.is_current.is_(True))
            ).all()
            if key
        }
    if not current:
        outcome.status = "skipped"
        outcome.details["reason"] = "no current season known - run sync-competitions"
        return outcome
    async with ctx.api_context() as api:
        etl = ApiFootballEtl(api.provider, ctx.session_factory, today=ctx.now().date())
        start = api.client.requests_sent
        for key, season in sorted(current.items()):
            if _budget(api, None, 0) <= 0:
                outcome.status = "partial"
                break
            report = await etl.import_injuries(key, season)
            outcome.details["competitions"][key] = report.items
        outcome.requests_used = api.client.requests_sent - start
    return outcome


async def odds(ctx: JobContext, opts: dict[str, Any]) -> JobOutcome:
    hours_ahead = int(opts.get("hours_ahead", 30))
    max_requests = int(opts.get("max_requests", 15))
    now = ctx.now()
    with session_scope(ctx.session_factory) as session:
        upcoming = list(
            session.scalars(
                select(db.Fixture.provider_id)
                .where(
                    db.Fixture.provider == PROVIDER,
                    db.Fixture.status_group == "scheduled",
                    db.Fixture.kickoff_utc > now,
                    db.Fixture.kickoff_utc <= now + timedelta(hours=hours_ahead),
                )
                .order_by(db.Fixture.kickoff_utc)
            )
        )
    outcome = JobOutcome(details={"upcoming": len(upcoming), "quotes": 0})
    if not upcoming:
        outcome.status = "skipped"
        return outcome
    async with ctx.api_context() as api:
        etl = ApiFootballEtl(api.provider, ctx.session_factory, today=now.date())
        start = api.client.requests_sent
        for fixture_id in upcoming:
            if _budget(api, max_requests, api.client.requests_sent - start) <= 0:
                outcome.status = "partial"
                break
            outcome.details["quotes"] += (await etl.import_fixture_odds(fixture_id)).items
        outcome.requests_used = api.client.requests_sent - start
    return outcome


async def historical_import(ctx: JobContext, opts: dict[str, Any]) -> JobOutcome:
    from football_ai.cli.common import parse_seasons

    seasons = parse_seasons(str(opts.get("seasons", "2018-2025")))
    async with ctx.api_context() as api:
        budget = min(int(opts.get("max_requests", 40)), api.client.quota.snapshot().available)
        if budget <= 0:
            return JobOutcome(status="skipped", details={"reason": "no quota left"})
        importer = HistoricalImporter(
            ApiFootballEtl(api.provider, ctx.session_factory, today=ctx.now().date()),
            max_requests=budget,
        )
        keys = [c.key for c in api.provider.competitions]
        report = await importer.run(keys, seasons)
    return JobOutcome(
        status="partial" if report.stopped_reason else "success",
        requests_used=report.requests_used,
        details={
            "done": len(report.jobs_done),
            "failed": report.jobs_failed[:10],
            "stopped": report.stopped_reason,
        },
    )


# ------------------------------------------------------- internal jobs
async def build_matches(ctx: JobContext, opts: dict[str, Any]) -> JobOutcome:
    from football_ai.config.aliases import load_aliases
    from football_ai.config.competitions import load_competitions

    aliases = load_aliases(ctx.settings.team_aliases_file)
    details: dict[str, Any] = {}
    for comp in load_competitions(ctx.settings.competitions_file):
        with session_scope(ctx.session_factory) as session:
            report = CanonicalBuilder(session, aliases=aliases).build(comp.key)
        details[comp.key] = {
            "inserted": report.matches_inserted,
            "updated": report.matches_updated,
            "conflicts": report.conflicts,
        }
    ctx.engine.invalidate()
    return JobOutcome(details=details)


async def predictions(ctx: JobContext, opts: dict[str, Any]) -> JobOutcome:
    hours_ahead = int(opts.get("hours_ahead", 72))
    now = ctx.now()
    with session_scope(ctx.session_factory) as session:
        ids = list(
            session.scalars(
                select(db.Match.id).where(
                    db.Match.status_group == "scheduled",
                    db.Match.kickoff_utc > now,
                    db.Match.kickoff_utc <= now + timedelta(hours=hours_ahead),
                )
            )
        )
    if not ids:
        return JobOutcome(status="skipped", details={"upcoming": 0})
    ctx.engine.invalidate()  # fresh data + latest registered model
    _, stored = ctx.engine.predict_and_store(ids)
    return JobOutcome(details={"upcoming": len(ids), "stored": stored})


async def evaluate_predictions(ctx: JobContext, opts: dict[str, Any]) -> JobOutcome:
    with session_scope(ctx.session_factory) as session:
        created = PredictionRepository(session).evaluate_pending()
    return JobOutcome(details={"evaluated": created})


async def daily_report(ctx: JobContext, opts: dict[str, Any]) -> JobOutcome:
    if ctx.bot is None or ctx.backend is None:
        return JobOutcome(status="skipped", details={"reason": "Telegram bot not configured"})
    from football_ai.bot.notifications import send_daily_report

    stats = await send_daily_report(ctx.bot, ctx.backend, ctx.now().date())
    return JobOutcome(
        details={
            "subscribers": stats.subscribers,
            "sent": stats.sent,
            "deactivated": stats.deactivated,
            "failed": stats.failed,
        }
    )


async def quality_check(ctx: JobContext, opts: dict[str, Any]) -> JobOutcome:
    """Data-quality report + collection/job failures of the last 24 h -> alerts."""
    since = ctx.now() - timedelta(hours=24)
    with session_scope(ctx.session_factory) as session:
        report = run_checks(load_frames(session), now=ctx.now(), scope="scheduled")
        save_report(report, session, ctx.settings.data_dir / "reports")
        failed_ingestion = list(
            session.scalars(
                select(db.IngestionJob).where(
                    db.IngestionJob.status == "failed", db.IngestionJob.updated_at >= since
                )
            )
        )
        failed_runs = list(
            session.scalars(
                select(db.ScheduledJobRun).where(
                    db.ScheduledJobRun.status == "failed", db.ScheduledJobRun.started_at >= since
                )
            )
        )
    problems = [
        f"{f.check}: {f.count} ({f.message})" for f in report.findings if f.severity == "error"
    ]
    stale = next((f for f in report.findings if f.check == "stale_results"), None)
    if stale is not None:
        problems.append(f"stale_results: {stale.count} matches without result")
    problems += [
        f"ingestion {j.job_type}:{j.scope} failed: {(j.last_error or '')[:120]}"
        for j in failed_ingestion
    ]
    problems += [f"job {r.job} failed: {(r.error or '')[:120]}" for r in failed_runs]
    if problems:
        await ctx.alerts.send("Contrôle qualité : anomalies détectées", problems)
    return JobOutcome(
        status="success",
        details={"errors": report.errors, "warnings": report.warnings, "alerts": len(problems)},
    )


JOBS: dict[str, JobFunction] = {
    "sync_fixtures": sync_fixtures,
    "refresh_results": refresh_results,
    "match_details": match_details,
    "injuries": injuries,
    "odds": odds,
    "build_matches": build_matches,
    "predictions": predictions,
    "daily_report": daily_report,
    "evaluate_predictions": evaluate_predictions,
    "quality_check": quality_check,
    "historical_import": historical_import,
}


# --------------------------------------------------------------- runner
async def run_job(name: str, ctx: JobContext, config: JobConfig | None = None) -> JobOutcome:
    """Run one job, record it in ``scheduled_job_runs`` and alert on failure."""
    fn = JOBS[name]
    opts = dict(config.options) if config else {}
    with session_scope(ctx.session_factory) as session:
        run = db.ScheduledJobRun(job=name, status="running", started_at=ctx.now())
        session.add(run)
        session.flush()
        run_id = run.id
    try:
        outcome = await fn(ctx, opts)
        error = None
    except QuotaExceededError as exc:
        outcome, error = JobOutcome(status="partial", details={"quota": str(exc)}), None
    except (DataProviderError, Exception) as exc:  # a job must never kill the scheduler
        outcome, error = JobOutcome(status="failed"), f"{type(exc).__name__}: {exc}"
        logger.exception("job_failed", extra={"job": name})
        await ctx.alerts.send(
            f"Tâche « {name} » en échec", [error, traceback.format_exc(limit=3)[-500:]]
        )
    with session_scope(ctx.session_factory) as session:
        finished = session.get(db.ScheduledJobRun, run_id)
        if finished is not None:
            run = finished
            run.status = outcome.status
            run.finished_at = datetime.now(UTC)
            run.requests_used = outcome.requests_used
            run.details = outcome.details
            run.error = error
    logger.info(
        "job_finished",
        extra={"job": name, "status": outcome.status, "requests": outcome.requests_used},
    )
    return outcome
