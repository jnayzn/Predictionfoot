"""/team and /compare."""

from __future__ import annotations

import re

from aiogram import Router
from aiogram.filters import Command, CommandObject
from aiogram.types import CallbackQuery, Message

from football_ai.backend.schemas.api import TeamRefOut
from football_ai.bot.backend import BotBackend
from football_ai.bot.formatting import format_compare, format_team
from football_ai.bot.keyboards import TeamCB, teams_keyboard

VS = re.compile(r"\s+(?:vs\.?|contre|-)\s+", re.IGNORECASE)


async def _resolve(backend: BotBackend, query: str) -> tuple[TeamRefOut | None, list[TeamRefOut]]:
    """Exact (case-insensitive) match first, otherwise the unique search result."""
    found = await backend.search_teams(query)
    exact = [t for t in found if t.name.casefold() == query.casefold()]
    if exact:
        return exact[0], found
    return (found[0] if len(found) == 1 else None), found


async def team(message: Message, command: CommandObject, backend: BotBackend) -> None:
    query = (command.args or "").strip()
    if len(query) < 2:
        await message.answer("Usage : /team &lt;nom de l'équipe&gt; (au moins 2 caractères).")
        return
    chosen, found = await _resolve(backend, query)
    if chosen is None:
        if not found:
            await message.answer("Aucune équipe trouvée.")
        else:
            await message.answer(
                "Plusieurs équipes correspondent :", reply_markup=teams_keyboard(found)
            )
        return
    await message.answer(format_team(chosen.name, await backend.team_matches(chosen.id)))


async def team_selected(query: CallbackQuery, callback_data: TeamCB, backend: BotBackend) -> None:
    matches = await backend.team_matches(callback_data.id)
    if isinstance(query.message, Message) and matches:
        first = matches[0]
        name = first.home.name if first.home.id == callback_data.id else first.away.name
        await query.message.answer(format_team(name, matches))
    await query.answer()


async def compare(message: Message, command: CommandObject, backend: BotBackend) -> None:
    parts = VS.split((command.args or "").strip(), maxsplit=1)
    if len(parts) != 2 or min(len(p) for p in parts) < 2:
        await message.answer("Usage : /compare &lt;équipe A&gt; vs &lt;équipe B&gt;")
        return
    resolved = [await _resolve(backend, p) for p in parts]
    for (chosen, found), name in zip(resolved, parts, strict=True):
        if chosen is None:
            hint = ", ".join(t.name for t in found[:5]) or "aucune"
            await message.answer(f"Équipe ambiguë ou inconnue : « {name} » (suggestions : {hint}).")
            return
    home, away = resolved[0][0], resolved[1][0]
    assert home is not None and away is not None
    await message.answer(format_compare(await backend.compare(home.id, away.id)))


def build_router() -> Router:
    router = Router(name="teams")
    router.message.register(team, Command("team"))
    router.callback_query.register(team_selected, TeamCB.filter())
    router.message.register(compare, Command("compare"))
    return router
