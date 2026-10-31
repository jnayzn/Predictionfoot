"""Phase 11 command: run the Telegram bot."""

from __future__ import annotations

import argparse

from football_ai.cli.common import EXIT_OK
from football_ai.config.settings import Settings


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("bot", help="Run the Telegram bot (long polling)")
    p.set_defaults(handler=cmd_bot)


async def cmd_bot(args: argparse.Namespace, settings: Settings) -> int:
    from football_ai.bot.main import run_bot

    await run_bot(settings)
    return EXIT_OK
