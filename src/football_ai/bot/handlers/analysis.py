"""/analyze and match buttons (full analysis, detailed markets)."""

from __future__ import annotations

from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import CallbackQuery, Message

from football_ai.bot.backend import BotBackend
from football_ai.bot.formatting import format_analysis, format_markets
from football_ai.bot.keyboards import MatchCB, analysis_keyboard


async def analyze(message: Message, command: CommandObject, backend: BotBackend) -> None:
    arg = (command.args or "").strip()
    if not arg.isdigit():
        await message.answer(
            "Usage : /analyze &lt;id du match&gt; — ou choisissez un match via /today ou /leagues."
        )
        return
    analysis = await backend.analysis(int(arg))
    await message.answer(format_analysis(analysis), reply_markup=analysis_keyboard(int(arg)))


async def match_analysis(query: CallbackQuery, callback_data: MatchCB, backend: BotBackend) -> None:
    await query.answer("Analyse en cours…")
    analysis = await backend.analysis(callback_data.id)
    if isinstance(query.message, Message):
        await query.message.answer(
            format_analysis(analysis), reply_markup=analysis_keyboard(callback_data.id)
        )


async def match_markets(query: CallbackQuery, callback_data: MatchCB, backend: BotBackend) -> None:
    prediction = await backend.prediction(callback_data.id)
    if isinstance(query.message, Message):
        await query.message.answer(format_markets(prediction))
    await query.answer()


def build_router() -> Router:
    router = Router(name="analysis")
    router.message.register(analyze, Command("analyze"))
    router.callback_query.register(match_analysis, MatchCB.filter(F.view == "analysis"))
    router.callback_query.register(match_markets, MatchCB.filter(F.view == "markets"))
    return router
