"""Bot middlewares: access control, per-user throttling, error handling."""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import CallbackQuery, Message, TelegramObject

from football_ai.bot.backend import BackendNotFoundError, BackendUnavailableError

logger = logging.getLogger(__name__)
Handler = Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]]


def _user_id(event: TelegramObject) -> int | None:
    user = getattr(event, "from_user", None)
    return user.id if user is not None else None


async def _reply(event: TelegramObject, text: str) -> None:
    if isinstance(event, Message):
        await event.answer(text)
    elif isinstance(event, CallbackQuery):
        await event.answer(text, show_alert=True)


class AccessMiddleware(BaseMiddleware):
    """Optional allow-list (private beta). Empty list = open to everyone."""

    def __init__(self, allowed: set[int]) -> None:
        self.allowed = allowed

    async def __call__(self, handler: Handler, event: TelegramObject, data: dict[str, Any]) -> Any:
        user = _user_id(event)
        if self.allowed and user not in self.allowed:
            await _reply(event, "Accès réservé. Contactez l'administrateur.")
            return None
        return await handler(event, data)


class ThrottlingMiddleware(BaseMiddleware):
    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.per_minute = per_minute
        self.clock = clock
        self.hits: dict[int, deque[float]] = defaultdict(deque)

    async def __call__(self, handler: Handler, event: TelegramObject, data: dict[str, Any]) -> Any:
        user = _user_id(event)
        if user is not None:
            now = self.clock()
            window = self.hits[user]
            while window and now - window[0] >= 60:
                window.popleft()
            if len(window) >= self.per_minute:
                await _reply(event, "Trop de requêtes, réessayez dans une minute.")
                return None
            window.append(now)
        return await handler(event, data)


class ErrorMiddleware(BaseMiddleware):
    async def __call__(self, handler: Handler, event: TelegramObject, data: dict[str, Any]) -> Any:
        try:
            return await handler(event, data)
        except BackendNotFoundError:
            await _reply(event, "Élément introuvable (match, équipe ou compétition).")
        except BackendUnavailableError as exc:
            logger.warning("bot_backend_unavailable", extra={"error": str(exc)})
            await _reply(event, "Service momentanément indisponible, réessayez plus tard.")
        except Exception:
            logger.exception("bot_unexpected_error")
            await _reply(event, "Une erreur inattendue est survenue.")
        return None
