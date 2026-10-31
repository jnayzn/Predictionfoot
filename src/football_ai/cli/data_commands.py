"""Phase 2 commands: database management and ETL (historical import, sync, replay, CSV)."""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import func, select

from football_ai.bootstrap import api_football_context, build_session_factory
from football_ai.cli.api_commands import print_quota
from football_ai.cli.common import EXIT_ERROR, EXIT_OK, parse_date, parse_seasons
from football_ai.config.competitions import load_competitions
from football_ai.config.settings import Settings
from football_ai.core.exceptions import ConfigurationError, DataProviderError
from football_ai.data.collectors.football_data_uk.collector import FootballDataUkCollector
from football_ai.data.pipeline.etl import ApiFootballEtl, LoadReport
from football_ai.data.pipeline.football_data_uk import load_csv_season
from football_ai.data.pipeline.historical import STEPS, HistoricalImporter
from football_ai.data.pipeline.replay import RawReplayer
from football_ai.database import models as db
from football_ai.database.connection import session_scope
from football_ai.database.repositories import JobRepository

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("db-upgrade", help="Apply database migrations (alembic upgrade head)")
    p.set_defaults(handler=lambda args, settings: cmd_db_upgrade(settings))
    p = sub.add_parser("db-stats", help="Row counts of the main tables")
    p.set_defaults(handler=lambda args, settings: cmd_db_stats(settings))
    p = sub.add_parser("jobs", help="List historical ingestion jobs")
    p.set_defaults(handler=lambda args, settings: cmd_jobs(settings))

    p = sub.add_parser("sync-competitions", help="Load competitions/seasons into the database")
    p.set_defaults(handler=lambda args, settings: cmd_sync_competitions(settings))

    p = sub.add_parser("sync-fixtures", help="Load fixtures of a day into the database")
    p.add_argument("--date", type=parse_date, default=None, help="YYYY-MM-DD (default today)")
    p.set_defaults(
        handler=lambda args, settings: cmd_sync_fixtures(settings, args.date or date.today())
    )

    p = sub.add_parser("import-history", help="Budgeted, resumable historical import")
    p.add_argument("--competition", action="append", dest="competitions", help="(repeatable)")
    p.add_argument("--seasons", type=parse_seasons, required=True, help="e.g. 2021-2024")
    p.add_argument("--steps", default=",".join(STEPS), help=f"subset of {','.join(STEPS)}")
    p.add_argument("--max-requests", type=int, default=50, help="request budget of this run")
    p.set_defaults(handler=cmd_import_history)

    p = sub.add_parser("import-csv", help="Import football-data.co.uk CSV seasons (no API key)")
    p.add_argument("--competition", action="append", dest="competitions", help="(repeatable)")
    p.add_argument("--seasons", type=parse_seasons, required=True, help="e.g. 2005-2024")
    p.add_argument("--file", type=Path, help="local CSV file (single competition and season)")
    p.set_defaults(handler=cmd_import_csv)

    p = sub.add_parser("replay", help="Rebuild the database from the raw JSON archive (0 request)")
    p.set_defaults(handler=lambda args, settings: cmd_replay(settings))


def _print_load(report: LoadReport) -> None:
    print(report.summary())
    for issue in report.issues[:20]:
        print(f"  ISSUE: {issue}")
    if len(report.issues) > 20:
        print(f"  ... {len(report.issues) - 20} more issues")


def cmd_db_upgrade(settings: Settings) -> int:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.cmd_opts = argparse.Namespace(x=[f"url={settings.database_url}"])
    command.upgrade(cfg, "head")
    print("Database schema is up to date.")
    return EXIT_OK


def cmd_db_stats(settings: Settings) -> int:
    factory = build_session_factory(settings)
    tables = [
        db.League, db.Season, db.Team, db.Player, db.Fixture, db.ActualResult,
        db.TeamStatistics, db.PlayerStatistics, db.Lineup, db.Injury, db.Standing, db.Odds,
        db.Prediction, db.ModelVersion, db.DataChangeLog, db.IngestionJob,
        db.CanonicalTeam, db.TeamLink, db.Match, db.MatchSource, db.DataQualityReport,
    ]  # fmt: skip
    with session_scope(factory) as session:
        for model in tables:
            count = session.scalar(select(func.count()).select_from(model)) or 0
            print(f"{model.__tablename__:<24} {count:>10}")
    return EXIT_OK


def cmd_jobs(settings: Settings) -> int:
    with session_scope(build_session_factory(settings)) as session:
        for job in JobRepository(session).list():
            finished = job.finished_at.isoformat(timespec="minutes") if job.finished_at else "-"
            print(
                f"{job.provider:<16} {job.job_type:<10} {job.scope:<24} {job.status:<8} "
                f"items={job.items_loaded:<6} requests={job.requests_used:<4} {finished}"
                + (f"  error={job.last_error[:80]}" if job.last_error else "")
            )
    return EXIT_OK


async def cmd_sync_competitions(settings: Settings) -> int:
    async with api_football_context(settings) as ctx:
        report = await ApiFootballEtl(
            ctx.provider, build_session_factory(settings)
        ).sync_competitions()
    _print_load(report)
    print_quota(settings)
    return EXIT_OK


async def cmd_sync_fixtures(settings: Settings, day: date) -> int:
    async with api_football_context(settings) as ctx:
        etl = ApiFootballEtl(ctx.provider, build_session_factory(settings))
        report = await etl.sync_fixtures_by_date(day)
    _print_load(report)
    print_quota(settings)
    return EXIT_OK


async def cmd_import_history(args: argparse.Namespace, settings: Settings) -> int:
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    async with api_football_context(settings) as ctx:
        keys = args.competitions or [c.key for c in ctx.provider.competitions]
        for key in keys:
            ctx.provider.config_for(key)  # validates keys early
        importer = HistoricalImporter(
            ApiFootballEtl(ctx.provider, build_session_factory(settings)),
            max_requests=args.max_requests,
        )
        report = await importer.run(keys, args.seasons, steps)
    for load in report.loads:
        _print_load(load)
    for label, items in (
        ("SKIPPED", report.jobs_skipped),
        ("FAILED", report.jobs_failed),
    ):
        for item in items:
            print(f"{label}: {item}")
    print(
        f"done={len(report.jobs_done)} skipped={len(report.jobs_skipped)} "
        f"failed={len(report.jobs_failed)} requests={report.requests_used}"
    )
    if report.stopped_reason:
        print(f"STOPPED: {report.stopped_reason} - run the same command again later to resume.")
    print_quota(settings)
    return EXIT_OK if not report.jobs_failed else EXIT_ERROR


async def cmd_import_csv(args: argparse.Namespace, settings: Settings) -> int:
    competitions = {c.key: c for c in load_competitions(settings.competitions_file)}
    keys = args.competitions or [
        k for k, c in competitions.items() if c.provider_id("football_data_uk")
    ]
    unknown = set(keys) - set(competitions)
    if unknown:
        raise ConfigurationError(f"Unknown competition keys: {sorted(unknown)}")
    if args.file and (len(keys) != 1 or len(args.seasons) != 1):
        raise ConfigurationError("--file requires exactly one --competition and one season")
    factory = build_session_factory(settings)
    collector = FootballDataUkCollector(settings.raw_dir)
    status = EXIT_OK
    try:
        for key in keys:
            config = competitions[key]
            division = config.provider_id("football_data_uk")
            if division is None:
                print(f"SKIPPED: {key} (not available on football-data.co.uk)")
                continue
            for season in args.seasons:
                try:
                    content = (
                        args.file.read_bytes()
                        if args.file
                        else await collector.fetch_season(str(division), season)
                    )
                except DataProviderError as exc:
                    print(f"FAILED: {key}:{season} ({exc})")
                    status = EXIT_ERROR
                    continue
                _print_load(load_csv_season(content, config, season, factory))
    finally:
        await collector.aclose()
    return status


async def cmd_replay(settings: Settings) -> int:
    async with api_football_context(settings) as ctx:
        report = RawReplayer(settings.raw_dir, ctx.provider, build_session_factory(settings)).run()
    print(f"files={report.files} skipped={report.skipped_files} items={report.items}")
    return EXIT_OK
