"""APScheduler wiring: every enabled job of ``configs/schedule.toml`` on its cron."""

from __future__ import annotations

import asyncio
import logging
import signal
from contextlib import suppress

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from football_ai.automation.alerts import (
    AlertSink,
    CompositeAlertSink,
    LogAlertSink,
    TelegramAlertSink,
)
from football_ai.automation.config import ScheduleConfig, load_schedule
from football_ai.automation.jobs import JOBS, JobContext, run_job
from football_ai.bootstrap import api_football_context, build_session_factory
from football_ai.config.settings import Settings
from football_ai.core.exceptions import ConfigurationError
from football_ai.prediction.engine import PredictionEngine

logger = logging.getLogger(__name__)


def build_scheduler(ctx: JobContext, schedule: ScheduleConfig) -> AsyncIOScheduler:
    unknown = set(schedule.jobs) - set(JOBS)
    if unknown:
        raise ConfigurationError(f"unknown jobs in schedule: {sorted(unknown)}")
    scheduler = AsyncIOScheduler(timezone=schedule.timezone)
    for name, config in schedule.jobs.items():
        if not config.enabled:
            continue
        scheduler.add_job(
            run_job,
            CronTrigger.from_crontab(config.cron, timezone=schedule.timezone),
            args=[name, ctx, config],
            id=name,
            name=name,
            max_instances=1,  # never two concurrent runs of the same job
            coalesce=True,  # missed runs collapse into one
            misfire_grace_time=1800,
        )
    return scheduler


def build_context(settings: Settings, *, with_bot: bool = True) -> JobContext:
    factory = build_session_factory(settings)
    engine = PredictionEngine(factory, model_priority=settings.model_priority)
    sinks: list[AlertSink] = [LogAlertSink()]
    bot = backend = None
    if with_bot and settings.telegram_bot_token is not None:
        from football_ai.bot.main import build_backend, build_bot

        bot = build_bot(settings)
        backend = build_backend(settings)
        if settings.telegram_admin_chat_id is not None:
            sinks.append(TelegramAlertSink(bot, settings.telegram_admin_chat_id))
    return JobContext(
        settings=settings,
        session_factory=factory,
        engine=engine,
        api_context=lambda: api_football_context(settings),
        alerts=CompositeAlertSink(*sinks),
        bot=bot,
        backend=backend,
    )


async def run_scheduler(settings: Settings) -> None:
    schedule = load_schedule(settings.schedule_file)
    ctx = build_context(settings)
    scheduler = build_scheduler(ctx, schedule)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)
    scheduler.start()
    for job in scheduler.get_jobs():
        logger.info("job_scheduled", extra={"job": job.id, "next_run": str(job.next_run_time)})
    try:
        await stop.wait()
    finally:
        scheduler.shutdown(wait=False)
        if ctx.bot is not None:
            await ctx.bot.session.close()
