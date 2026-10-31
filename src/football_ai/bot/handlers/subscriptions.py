"""/subscribe and /unsubscribe (daily automatic report)."""

from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command, CommandObject
from aiogram.types import Message

from football_ai.bot.backend import BotBackend


async def subscribe(message: Message, command: CommandObject, backend: BotBackend) -> None:
    competitions = [c.strip() for c in (command.args or "").replace(",", " ").split() if c.strip()]
    username = message.from_user.username if message.from_user else None
    sub = await backend.subscribe(message.chat.id, username, competitions)
    scope = ", ".join(sub.competitions) if sub.competitions else "toutes les compétitions"
    await message.answer(
        f"✅ Rapport quotidien activé ({scope}). Désactivation : /unsubscribe\n"
        "Exemple de filtre : /subscribe ligue_1 champions_league"
    )


async def unsubscribe(message: Message, backend: BotBackend) -> None:
    done = await backend.unsubscribe(message.chat.id)
    await message.answer("Rapport quotidien désactivé." if done else "Vous n'étiez pas abonné.")


def build_router() -> Router:
    router = Router(name="subscriptions")
    router.message.register(subscribe, Command("subscribe"))
    router.message.register(unsubscribe, Command("unsubscribe"))
    return router
