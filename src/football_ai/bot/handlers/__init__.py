"""Command and callback handlers.

Each module exposes ``build_router()``: a NEW router per dispatcher (an aiogram
router can only be attached to one dispatcher).
"""

from __future__ import annotations

from aiogram import Router

from football_ai.bot.handlers import analysis, general, matches, subscriptions, teams


def routers() -> list[Router]:
    return [m.build_router() for m in (general, matches, analysis, teams, subscriptions)]
