"""Command-line interface: ``football-ai <command>`` (see ``football-ai --help``)."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import sys
from collections.abc import Sequence

from football_ai.cli import (
    api_commands,
    data_commands,
    feature_commands,
    quality_commands,
)
from football_ai.cli.common import (
    EXIT_AUTH,
    EXIT_CONFIG,
    EXIT_ERROR,
    EXIT_OK,
    EXIT_QUOTA,
)
from football_ai.config.settings import Settings
from football_ai.core.exceptions import (
    AuthenticationError,
    ConfigurationError,
    DataProviderError,
    PlanRestrictionError,
    QuotaExceededError,
)
from football_ai.core.logging import configure_logging

__all__ = [
    "EXIT_AUTH",
    "EXIT_CONFIG",
    "EXIT_ERROR",
    "EXIT_OK",
    "EXIT_QUOTA",
    "build_parser",
    "main",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="football-ai", description="Football AI Predictor CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    api_commands.register(sub)
    data_commands.register(sub)
    quality_commands.register(sub)
    feature_commands.register(sub)
    return parser


async def _dispatch(args: argparse.Namespace, settings: Settings) -> int:
    result = args.handler(args, settings)
    if inspect.isawaitable(result):
        result = await result
    return int(result)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = Settings()
    except Exception as exc:  # pydantic ValidationError
        print(f"Configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    configure_logging(settings.log_level, settings.log_format)
    try:
        return asyncio.run(_dispatch(args, settings))
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
