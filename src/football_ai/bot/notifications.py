"""Daily report pushed to subscribers (called by the scheduler, Phase 12)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

from aiogram import Bot
from aiogram.exceptions import TelegramForbiddenError, TelegramRetryAfter

from football_ai.bot.backend import BotBackend
from football_ai.bot.formatting import DISCLAIMER, format_prediction_short, truncate

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ReportStats:
    subscribers: int = 0
    sent: int = 0
    deactivated: int = 0
    failed: int = 0


async def build_daily_report(backend: BotBackend, day: date, competitions: list[str]) -> str | None:
    fixtures = [m for m in await backend.fixtures(day) if m.status == "scheduled"]
    if competitions:
        fixtures = [m for m in fixtures if m.competition in competitions]
    if not fixtures:
        return None
    lines = [f"<b>📅 Rapport du {day:%d/%m/%Y}</b>", ""]
    for m in fixtures[:20]:
        lines.append(format_prediction_short(await backend.prediction(m.id)))
    lines += ["", "Analyse complète : /analyze &lt;id&gt; ou /today", "", DISCLAIMER]
    return truncate("\n".join(lines))


async def send_daily_report(bot: Bot, backend: BotBackend, day: date) -> ReportStats:
    stats = ReportStats()
    cache: dict[tuple[str, ...], str | None] = {}
    for sub in await backend.subscribers():
        stats.subscribers += 1
        key = tuple(sorted(sub.competitions))
        if key not in cache:
            cache[key] = await build_daily_report(backend, day, list(key))
        text = cache[key]
        if text is None:
            continue
        try:
            await bot.send_message(sub.chat_id, text)
            stats.sent += 1
        except TelegramForbiddenError:  # user blocked the bot: stop sending
            await backend.unsubscribe(sub.chat_id)
            stats.deactivated += 1
        except TelegramRetryAfter as exc:
            logger.warning("telegram_flood_control", extra={"retry_after": exc.retry_after})
            stats.failed += 1
        except Exception:
            logger.exception("telegram_send_failed", extra={"chat_id": sub.chat_id})
            stats.failed += 1
    return stats
