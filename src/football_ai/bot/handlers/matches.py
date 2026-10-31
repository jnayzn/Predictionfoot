"""/today, /leagues, /predictions, /results and competition selection."""

from __future__ import annotations

from datetime import UTC, date, datetime

from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import CallbackQuery, Message

from football_ai.bot.backend import BotBackend
from football_ai.bot.formatting import (
    DISCLAIMER,
    format_competitions,
    format_fixtures,
    format_prediction_short,
    format_results,
    truncate,
)
from football_ai.bot.keyboards import CompetitionCB, MenuCB, competitions_keyboard, matches_keyboard


def today() -> date:
    return datetime.now(UTC).date()


async def _send_today(message: Message, backend: BotBackend) -> None:
    day = today()
    fixtures = await backend.fixtures(day)
    markup = matches_keyboard(fixtures) if fixtures else None
    await message.answer(format_fixtures(fixtures, day, "Matchs"), reply_markup=markup)


async def today_command(message: Message, backend: BotBackend) -> None:
    await _send_today(message, backend)


async def menu_today(query: CallbackQuery, backend: BotBackend) -> None:
    if isinstance(query.message, Message):
        await _send_today(query.message, backend)
    await query.answer()


async def _send_leagues(message: Message, backend: BotBackend) -> None:
    items = await backend.competitions()
    await message.answer(format_competitions(items), reply_markup=competitions_keyboard(items))


async def leagues(message: Message, backend: BotBackend) -> None:
    await _send_leagues(message, backend)


async def menu_leagues(query: CallbackQuery, backend: BotBackend) -> None:
    if isinstance(query.message, Message):
        await _send_leagues(query.message, backend)
    await query.answer()


async def competition_selected(
    query: CallbackQuery, callback_data: CompetitionCB, backend: BotBackend
) -> None:
    day = today()
    fixtures = await backend.fixtures(day, callback_data.key)
    if isinstance(query.message, Message):
        markup = matches_keyboard(fixtures) if fixtures else None
        await query.message.answer(format_fixtures(fixtures, day, "Matchs"), reply_markup=markup)
    await query.answer()


async def predictions(message: Message, backend: BotBackend) -> None:
    day = today()
    fixtures = [m for m in await backend.fixtures(day) if m.status == "scheduled"]
    if not fixtures:
        await message.answer(f"Aucun match à venir aujourd'hui ({day:%d/%m/%Y}).")
        return
    blocks = [f"<b>Probabilités du {day:%d/%m/%Y}</b> (1 = domicile, N = nul, 2 = extérieur)", ""]
    for m in fixtures[:15]:
        blocks.append(format_prediction_short(await backend.prediction(m.id)))
    blocks += ["", DISCLAIMER]
    await message.answer(truncate("\n".join(blocks)), reply_markup=matches_keyboard(fixtures))


async def results(message: Message, command: CommandObject, backend: BotBackend) -> None:
    day = today()
    if command.args:
        try:
            day = date.fromisoformat(command.args.strip())
        except ValueError:
            await message.answer("Format de date attendu : AAAA-MM-JJ (ex. /results 2026-10-31).")
            return
    await message.answer(format_results(await backend.results(day), day))


def build_router() -> Router:
    router = Router(name="matches")
    router.message.register(today_command, Command("today"))
    router.callback_query.register(menu_today, MenuCB.filter(F.action == "today"))
    router.message.register(leagues, Command("leagues"))
    router.callback_query.register(menu_leagues, MenuCB.filter(F.action == "leagues"))
    router.callback_query.register(competition_selected, CompetitionCB.filter())
    router.message.register(predictions, Command("predictions"))
    router.message.register(results, Command("results"))
    return router
