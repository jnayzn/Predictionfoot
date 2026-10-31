"""Command-line interface (Phase 1).

Commands::

    football-ai quota                       # local quota state, no network
    football-ai status                      # account status (does not consume quota)
    football-ai discover [--json]           # resolve the 6 competitions (1 request, cached 7 days)
    football-ai fixtures --date YYYY-MM-DD  # fixtures of the day (1 request, cached)
            [--league premier_league ...] [--json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Sequence
from datetime import date

from football_ai.bootstrap import api_football_context, build_quota_manager
from football_ai.config.settings import Settings
from football_ai.core.exceptions import (
    AuthenticationError,
    ConfigurationError,
    DataProviderError,
    PlanRestrictionError,
    QuotaExceededError,
)
from football_ai.core.logging import configure_logging

logger = logging.getLogger("football_ai.cli")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONFIG = 2
EXIT_QUOTA = 3
EXIT_AUTH = 4


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}, expected YYYY-MM-DD") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="football-ai", description="Football AI Predictor CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("quota", help="Show the local API quota state (no network call)")
    sub.add_parser("status", help="Show the API-Football account status (free call)")

    p_disc = sub.add_parser("discover", help="Discover the configured competitions and seasons")
    p_disc.add_argument("--json", action="store_true", help="Output JSON")

    p_fix = sub.add_parser("fixtures", help="List fixtures of the covered competitions for a day")
    p_fix.add_argument("--date", type=_parse_date, default=None, help="YYYY-MM-DD (default today)")
    p_fix.add_argument(
        "--league", action="append", dest="leagues", help="Competition key (repeatable)"
    )
    p_fix.add_argument("--json", action="store_true", help="Output JSON")
    return parser


def _print_quota(settings: Settings) -> None:
    snap = build_quota_manager(settings).snapshot()
    server = snap.server_remaining if snap.server_remaining is not None else "n/a"
    print(
        f"Quota {snap.day} (UTC): used {snap.used}/{snap.daily_limit}, "
        f"available {snap.available} (reserve {snap.reserve}), "
        f"server remaining {server}, "
        f"rate limit {snap.minute_limit}/min"
    )


async def _cmd_status(settings: Settings) -> int:
    async with api_football_context(settings) as ctx:
        resp = await ctx.client.status()
    data = resp.response if isinstance(resp.response, dict) else {}
    print(json.dumps(data, indent=2, ensure_ascii=False))
    _print_quota(settings)
    return EXIT_OK


async def _cmd_discover(settings: Settings, as_json: bool) -> int:
    async with api_football_context(settings) as ctx:
        result = await ctx.provider.discover_competitions()
    if as_json:
        print(
            json.dumps(
                {
                    "competitions": [c.model_dump(mode="json") for c in result.competitions],
                    "missing": result.missing,
                    "issues": [str(i) for i in result.issues],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print(f"{'key':<18} {'id':>5}  {'name':<24} {'country':<10} seasons      current")
        for c in result.competitions:
            years = [s.year for s in c.seasons]
            span = f"{min(years)}-{max(years)}" if years else "-"
            current = c.current_season.year if c.current_season else "-"
            print(
                f"{c.key:<18} {c.provider_id:>5}  {c.name:<24} {c.country:<10} {span:<12} {current}"
            )
        for key in result.missing:
            print(f"MISSING: {key}")
        for issue in result.issues:
            print(f"ISSUE: {issue}")
        print(f"(source: {'cache' if result.from_cache else 'API'})")
    _print_quota(settings)
    return EXIT_OK if not result.missing else EXIT_ERROR


async def _cmd_fixtures(
    settings: Settings, day: date, leagues: list[str] | None, as_json: bool
) -> int:
    async with api_football_context(settings) as ctx:
        batch = await ctx.provider.get_fixtures_by_date(day, leagues)
    if as_json:
        print(
            json.dumps(
                {
                    "date": day.isoformat(),
                    "fixtures": [f.model_dump(mode="json") for f in batch.fixtures],
                    "issues": [str(i) for i in batch.issues],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print(f"Fixtures on {day.isoformat()} ({len(batch.fixtures)} in covered competitions)")
        for f in batch.fixtures:
            score = (
                f"{f.goals.home}-{f.goals.away}"
                if f.goals.home is not None and f.goals.away is not None
                else "vs"
            )
            print(
                f"  {f.kickoff_utc:%H:%M}Z [{f.competition_key}] "
                f"{f.home_team.name} {score} {f.away_team.name} ({f.status_short}) #{f.provider_id}"
            )
        for issue in batch.issues:
            print(f"ISSUE: {issue}")
        print(
            f"(received {batch.total_received}, "
            f"other competitions {batch.other_competitions_skipped}, "
            f"rejected {batch.rejected}, duplicates {batch.duplicates_removed}, "
            f"source: {'cache' if batch.from_cache else 'API'})"
        )
    _print_quota(settings)
    return EXIT_OK


async def _run(args: argparse.Namespace, settings: Settings) -> int:
    if args.command == "quota":
        _print_quota(settings)
        return EXIT_OK
    if args.command == "status":
        return await _cmd_status(settings)
    if args.command == "discover":
        return await _cmd_discover(settings, args.json)
    if args.command == "fixtures":
        return await _cmd_fixtures(settings, args.date or date.today(), args.leagues, args.json)
    raise AssertionError(f"unknown command {args.command}")  # pragma: no cover


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = Settings()
    except Exception as exc:  # pydantic ValidationError
        print(f"Configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    configure_logging(settings.log_level, settings.log_format)
    try:
        return asyncio.run(_run(args, settings))
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except QuotaExceededError as exc:
        print(f"Quota exceeded: {exc}", file=sys.stderr)
        return EXIT_QUOTA
    except AuthenticationError as exc:
        print(f"Authentication error: {exc}", file=sys.stderr)
        return EXIT_AUTH
    except PlanRestrictionError as exc:
        print(f"Plan restriction: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except DataProviderError as exc:
        print(f"Data provider error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
