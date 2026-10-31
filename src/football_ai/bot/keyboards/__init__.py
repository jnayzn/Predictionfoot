"""Inline keyboards and typed callback data."""

from __future__ import annotations

from aiogram.filters.callback_data import CallbackData
from aiogram.types import InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

from football_ai.backend.schemas.api import CompetitionOut, MatchOut, TeamRefOut
from football_ai.bot.formatting import COMPETITION_EMOJI


class CompetitionCB(CallbackData, prefix="comp"):
    key: str


class MatchCB(CallbackData, prefix="match"):
    id: int
    view: str = "analysis"  # analysis | markets


class TeamCB(CallbackData, prefix="team"):
    id: int


class MenuCB(CallbackData, prefix="menu"):
    action: str  # today | leagues | performance | help


def main_menu() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="📅 Matchs du jour", callback_data=MenuCB(action="today"))
    kb.button(text="🏆 Compétitions", callback_data=MenuCB(action="leagues"))
    kb.button(text="📈 Performances", callback_data=MenuCB(action="performance"))
    kb.button(text="❓ Aide", callback_data=MenuCB(action="help"))
    kb.adjust(2)
    return kb.as_markup()


def competitions_keyboard(items: list[CompetitionOut]) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    for c in items:
        kb.button(
            text=f"{COMPETITION_EMOJI.get(c.key, '⚽')} {c.name}",
            callback_data=CompetitionCB(key=c.key),
        )
    kb.adjust(2)
    return kb.as_markup()


def matches_keyboard(matches: list[MatchOut], limit: int = 30) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    for m in matches[:limit]:
        kb.button(text=f"{m.home.name} – {m.away.name}"[:60], callback_data=MatchCB(id=m.id))
    kb.adjust(1)
    return kb.as_markup()


def analysis_keyboard(match_id: int) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="📊 Marchés détaillés", callback_data=MatchCB(id=match_id, view="markets"))
    kb.button(text="🔄 Actualiser", callback_data=MatchCB(id=match_id, view="analysis"))
    kb.button(text="⬅️ Menu", callback_data=MenuCB(action="leagues"))
    kb.adjust(2, 1)
    return kb.as_markup()


def teams_keyboard(teams: list[TeamRefOut]) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    for t in teams[:10]:
        kb.button(text=t.name[:60], callback_data=TeamCB(id=t.id))
    kb.adjust(1)
    return kb.as_markup()
