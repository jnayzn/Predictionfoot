"""Resumable, budget-aware historical import, season by season.

Strategy for a free plan (100 requests/day):

1. ``competitions`` (1 request, cached a week) fills leagues/seasons/coverage.
2. Steps run in priority order **across all competition-seasons** before moving
   to the next step, so the most useful data (fixtures/results) arrives first:
   ``fixtures`` -> ``teams`` -> ``standings`` -> ``injuries`` -> ``details``.
3. Every (step, competition, season) is an ``ingestion_jobs`` row. A finished
   season that was imported successfully is never fetched again; the current
   season is refreshed when older than ``refresh_after``.
4. The run stops cleanly when the request budget or the daily quota is
   exhausted; the next run resumes where it stopped.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

from football_ai.core.exceptions import (
    DataProviderError,
    PlanRestrictionError,
    QuotaExceededError,
)
from football_ai.data.collectors.api_football.parsers import PROVIDER
from football_ai.data.collectors.api_football.provider import DETAILS_CHUNK_SIZE
from football_ai.data.pipeline.etl import ApiFootballEtl, LoadReport
from football_ai.database.connection import session_scope
from football_ai.database.repositories import JobRepository, ReferenceRepository

logger = logging.getLogger(__name__)

STEPS: tuple[str, ...] = ("fixtures", "teams", "standings", "injuries", "details")
# coverage flag (from /leagues) required for a step; None = always attempted
STEP_COVERAGE: dict[str, tuple[str, ...]] = {
    "standings": ("standings",),
    "injuries": ("injuries",),
    "details": ("fixtures.statistics_fixtures", "fixtures.lineups"),
}


@dataclass(slots=True)
class ImportReport:
    requests_used: int = 0
    jobs_done: list[str] = field(default_factory=list)
    jobs_skipped: list[str] = field(default_factory=list)
    jobs_failed: list[str] = field(default_factory=list)
    loads: list[LoadReport] = field(default_factory=list)
    stopped_reason: str | None = None


def season_is_finished(season: int, today: date) -> bool:
    return today > date(season + 1, 7, 31)


class HistoricalImporter:
    def __init__(
        self,
        etl: ApiFootballEtl,
        *,
        max_requests: int,
        refresh_after: timedelta = timedelta(hours=12),
    ) -> None:
        self.etl = etl
        self.max_requests = max_requests
        self.refresh_after = refresh_after
        self._client = etl.provider.client  # request accounting only
        self._start_requests = 0

    # ------------------------------------------------------------ accounting
    @property
    def _used(self) -> int:
        return self._client.requests_sent - self._start_requests

    def _remaining(self) -> int:
        available = self._client.quota.snapshot().available
        return max(0, min(self.max_requests - self._used, available))

    # ------------------------------------------------------------------ plan
    def _should_skip(self, step: str, key: str, season: int) -> str | None:
        scope = f"{key}:{season}"
        today = self.etl.today
        with session_scope(self.etl.session_factory) as session:
            jobs = JobRepository(session)
            finished_at = jobs.finished_at(PROVIDER, step, scope)
            if finished_at is not None:
                if season_is_finished(season, today):
                    return "already imported"
                if datetime.now(UTC) - finished_at < self.refresh_after:
                    return "recently refreshed"
            required = STEP_COVERAGE.get(step)
            if required:
                league = ReferenceRepository(session).league_by_key(key, PROVIDER)
                coverage = (
                    ReferenceRepository(session).season_coverage(league.id, season)
                    if league
                    else None
                )
                if coverage is not None and not any(coverage.get(flag) for flag in required):
                    return f"not covered by provider ({'/'.join(required)})"
        return None

    def _record(
        self, step: str, scope: str, *, requests: int, items: int = 0, error: str | None = None
    ) -> None:
        with session_scope(self.etl.session_factory) as session:
            jobs = JobRepository(session)
            job = jobs.get_or_create(PROVIDER, step, scope)
            jobs.start(job)
            if error is None:
                jobs.succeed(job, requests=requests, items=items)
            else:
                jobs.fail(job, error, requests=requests)

    # ------------------------------------------------------------------- run
    async def run(
        self,
        competition_keys: Sequence[str],
        seasons: Sequence[int],
        steps: Sequence[str] = STEPS,
    ) -> ImportReport:
        unknown = set(steps) - set(STEPS)
        if unknown:
            raise ValueError(f"unknown steps: {sorted(unknown)}")
        self._start_requests = self._client.requests_sent
        report = ImportReport()
        restricted: set[tuple[str, int]] = set()
        try:
            before = self._used
            competitions = await self.etl.sync_competitions()
            competitions.requests = self._used - before
            report.loads.append(competitions)
            for step in [s for s in STEPS if s in steps]:
                for season in sorted(seasons, reverse=True):  # most recent first
                    for key in competition_keys:
                        if (key, season) in restricted:
                            continue
                        if not await self._execute(step, key, season, report, restricted):
                            report.stopped_reason = "request budget exhausted"
                            return report
        except QuotaExceededError as exc:
            report.stopped_reason = f"daily quota exhausted: {exc}"
        finally:
            report.requests_used = self._used
            logger.info(
                "historical_import_finished",
                extra={
                    "requests": report.requests_used,
                    "done": len(report.jobs_done),
                    "skipped": len(report.jobs_skipped),
                    "failed": len(report.jobs_failed),
                    "stopped": report.stopped_reason,
                },
            )
        return report

    async def _execute(
        self,
        step: str,
        key: str,
        season: int,
        report: ImportReport,
        restricted: set[tuple[str, int]],
    ) -> bool:
        """Run one job. Returns False when the run must stop (budget exhausted)."""
        scope = f"{key}:{season}"
        reason = self._should_skip(step, key, season)
        if reason:
            report.jobs_skipped.append(f"{step}:{scope} ({reason})")
            return True
        if self._remaining() <= 0:
            return False
        before = self._used
        try:
            load = await self._run_step(step, key, season)
        except PlanRestrictionError as exc:
            restricted.add((key, season))
            self._record(step, scope, requests=self._used - before, error=str(exc))
            report.jobs_failed.append(f"{step}:{scope} (plan restriction)")
            return True
        except QuotaExceededError:
            raise
        except DataProviderError as exc:
            self._record(step, scope, requests=self._used - before, error=str(exc))
            report.jobs_failed.append(f"{step}:{scope} ({exc})")
            return True
        if load is None:  # details partially fetched, budget exhausted
            return False
        load.requests = self._used - before
        self._record(step, scope, requests=load.requests, items=load.items)
        report.loads.append(load)
        report.jobs_done.append(f"{step}:{scope}")
        return True

    async def _run_step(self, step: str, key: str, season: int) -> LoadReport | None:
        if step == "fixtures":
            return await self.etl.import_season_fixtures(key, season)
        if step == "teams":
            return await self.etl.import_teams(key, season)
        if step == "standings":
            return await self.etl.import_standings(key, season)
        if step == "injuries":
            return await self.etl.import_injuries(key, season)
        return await self._run_details(key, season)

    async def _run_details(self, key: str, season: int) -> LoadReport | None:
        """Fetch details by chunks of 20 within budget. ``None`` = not finished yet."""
        total = LoadReport("details", f"{key}:{season}")
        while True:
            remaining = self._remaining()
            pending = self.etl.pending_detail_ids(key, season)
            if not pending:
                return total
            if remaining <= 0:
                return None
            chunk_count = min(remaining, math.ceil(len(pending) / DETAILS_CHUNK_SIZE))
            ids = pending[: chunk_count * DETAILS_CHUNK_SIZE]
            load = await self.etl.import_fixture_details(ids, total.scope)
            total.items += load.items
            total.requests += load.requests
            total.issues.extend(load.issues)
            total.fixtures.merge(load.fixtures)
            if load.items == 0:
                logger.warning("details_no_progress", extra={"scope": total.scope, "ids": len(ids)})
                return total
