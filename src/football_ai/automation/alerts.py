"""Operational alerts (collection errors, data quality, failed jobs)."""

from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger("football_ai.alerts")


class AlertSink(Protocol):
    async def send(self, title: str, lines: list[str]) -> None: ...


class LogAlertSink:
    async def send(self, title: str, lines: list[str]) -> None:
        logger.error("alert", extra={"title": title, "details": lines})


class TelegramAlertSink:
    """Sends alerts to an administrator chat (TELEGRAM_ADMIN_CHAT_ID)."""

    def __init__(self, bot: object, chat_id: int) -> None:
        self.bot = bot
        self.chat_id = chat_id

    async def send(self, title: str, lines: list[str]) -> None:
        from html import escape

        text = f"🚨 <b>{escape(title)}</b>\n" + "\n".join(f"• {escape(x)}" for x in lines[:20])
        try:
            await self.bot.send_message(self.chat_id, text[:4000])  # type: ignore[attr-defined]
        except Exception:
            logger.exception("alert_delivery_failed")


class CompositeAlertSink:
    def __init__(self, *sinks: AlertSink) -> None:
        self.sinks = sinks

    async def send(self, title: str, lines: list[str]) -> None:
        for sink in self.sinks:
            await sink.send(title, lines)
