"""/start, /help, /performance and the main menu."""

from __future__ import annotations

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, Message

from football_ai.bot.backend import BotBackend
from football_ai.bot.formatting import DISCLAIMER, HELP, format_performance
from football_ai.bot.keyboards import MenuCB, main_menu

WELCOME = (
    "<b>Bienvenue sur Football AI Predictor</b> ⚽\n\n"
    "J'analyse les matchs de Premier League, Liga, Bundesliga, Serie A, Ligue 1 et Ligue des "
    "champions et je calcule des <b>probabilités</b> à partir de données historiques et de "
    "modèles statistiques.\n\n" + DISCLAIMER
)


async def start(message: Message) -> None:
    await message.answer(WELCOME, reply_markup=main_menu())


async def help_command(message: Message) -> None:
    await message.answer(HELP)


async def performance(message: Message, backend: BotBackend) -> None:
    await message.answer(format_performance(await backend.performance()))


async def menu_help(query: CallbackQuery) -> None:
    if isinstance(query.message, Message):
        await query.message.answer(HELP)
    await query.answer()


async def menu_performance(query: CallbackQuery, backend: BotBackend) -> None:
    if isinstance(query.message, Message):
        await query.message.answer(format_performance(await backend.performance()))
    await query.answer()


def build_router() -> Router:
    router = Router(name="general")
    router.message.register(start, CommandStart())
    router.message.register(help_command, Command("help"))
    router.message.register(performance, Command("performance"))
    router.callback_query.register(menu_help, MenuCB.filter(F.action == "help"))
    router.callback_query.register(menu_performance, MenuCB.filter(F.action == "performance"))
    return router
