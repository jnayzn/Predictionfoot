"""Phase 12 commands: scheduler, one-off job runs, run history."""

from __future__ import annotations

import argparse

from sqlalchemy import select

from football_ai.automation.config import load_schedule
from football_ai.automation.jobs import JOBS
from football_ai.bootstrap import build_session_factory
from football_ai.cli.common import EXIT_ERROR, EXIT_OK
from football_ai.config.settings import Settings
from football_ai.database import models as db
from football_ai.database.connection import session_scope


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("scheduler", help="Run every scheduled job (configs/schedule.toml)")
    p.set_defaults(handler=cmd_scheduler)
    p = sub.add_parser("run-job", help="Run one scheduled job now")
    p.add_argument("job", choices=sorted(JOBS))
    p.set_defaults(handler=cmd_run_job)
    p = sub.add_parser("job-runs", help="Recent scheduled job executions")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(handler=cmd_job_runs)


async def cmd_scheduler(args: argparse.Namespace, settings: Settings) -> int:
    from football_ai.automation.scheduler import run_scheduler

    await run_scheduler(settings)
    return EXIT_OK


async def cmd_run_job(args: argparse.Namespace, settings: Settings) -> int:
    from football_ai.automation.jobs import run_job
    from football_ai.automation.scheduler import build_context

    schedule = load_schedule(settings.schedule_file)
    ctx = build_context(settings, with_bot=args.job == "daily_report")
    try:
        outcome = await run_job(args.job, ctx, schedule.jobs.get(args.job))
    finally:
        if ctx.bot is not None:
            await ctx.bot.session.close()
    print(f"{args.job}: {outcome.status} requests={outcome.requests_used} {outcome.details}")
    return EXIT_ERROR if outcome.status == "failed" else EXIT_OK


def cmd_job_runs(args: argparse.Namespace, settings: Settings) -> int:
    with session_scope(build_session_factory(settings)) as session:
        runs = session.scalars(
            select(db.ScheduledJobRun)
            .order_by(db.ScheduledJobRun.started_at.desc())
            .limit(args.limit)
        )
        for r in runs:
            print(
                f"{r.started_at:%Y-%m-%d %H:%M} {r.job:<22} {r.status:<8} req={r.requests_used:<3} "
                f"{r.error or r.details}"
            )
    return EXIT_OK
