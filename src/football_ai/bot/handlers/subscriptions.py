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


PRIVACY = (
    "<b>Données personnelles</b>\n\n"
    "Sans abonnement, le bot ne conserve <b>aucune</b> donnée vous concernant.\n"
    "Avec /subscribe, il conserve : l'identifiant du chat, votre nom d'utilisateur Telegram et "
    "les compétitions choisies, dans le seul but d'envoyer le rapport quotidien.\n"
    "Ces données ne sont ni vendues ni partagées.\n\n"
    "/unsubscribe arrête les envois ; /forget efface définitivement vos données."
)


async def privacy(message: Message) -> None:
    await message.answer(PRIVACY)


async def forget(message: Message, backend: BotBackend) -> None:
    erased = await backend.erase(message.chat.id)
    await message.answer(
        "Vos données ont été définitivement supprimées." if erased else "Aucune donnée à supprimer."
    )


def build_router() -> Router:
    router = Router(name="subscriptions")
    router.message.register(subscribe, Command("subscribe"))
    router.message.register(unsubscribe, Command("unsubscribe"))
    router.message.register(privacy, Command("privacy"))
    router.message.register(forget, Command("forget"))
    return router
