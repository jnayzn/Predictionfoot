"""Bot assembly and entry point (``football-ai bot``)."""

from __future__ import annotations

import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import BotCommand

from football_ai.bot.backend import BotBackend, HttpBackend, ServiceBackend
from football_ai.bot.handlers import routers
from football_ai.bot.middlewares import AccessMiddleware, ErrorMiddleware, ThrottlingMiddleware
from football_ai.config.settings import Settings
from football_ai.core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

COMMANDS = [
    BotCommand(command="today", description="Matchs du jour"),
    BotCommand(command="leagues", description="Choisir une compétition"),
    BotCommand(command="analyze", description="Analyse complète d'un match"),
    BotCommand(command="predictions", description="Probabilités du jour"),
    BotCommand(command="team", description="Derniers matchs d'une équipe"),
    BotCommand(command="compare", description="Confrontation hypothétique"),
    BotCommand(command="results", description="Résultats et vérification"),
    BotCommand(command="performance", description="Performances du système"),
    BotCommand(command="subscribe", description="Rapport quotidien"),
    BotCommand(command="privacy", description="Données personnelles"),
    BotCommand(command="help", description="Aide"),
]


def build_dispatcher(backend: BotBackend, settings: Settings) -> Dispatcher:
    dp = Dispatcher()
    dp["backend"] = backend
    for observer in (dp.message, dp.callback_query):
        observer.outer_middleware(AccessMiddleware(settings.allowed_telegram_users))
        observer.outer_middleware(ThrottlingMiddleware(settings.telegram_rate_limit_per_minute))
        observer.middleware(ErrorMiddleware())
    for router in routers():
        dp.include_router(router)
    return dp


def build_backend(settings: Settings) -> BotBackend:
    if settings.bot_backend == "http":
        key = settings.bot_api_key.get_secret_value() if settings.bot_api_key else None
        return HttpBackend(settings.bot_api_url, key)
    from football_ai.backend.services.football_service import FootballService
    from football_ai.bootstrap import build_session_factory
    from football_ai.config.competitions import load_competitions
    from football_ai.prediction.engine import PredictionEngine

    factory = build_session_factory(settings)
    service = FootballService(
        factory,
        PredictionEngine(factory, model_priority=settings.model_priority),
        load_competitions(settings.competitions_file),
        reports_dir=settings.data_dir / "reports",
    )
    return ServiceBackend(service)


def build_bot(settings: Settings) -> Bot:
    if settings.telegram_bot_token is None:
        raise ConfigurationError("TELEGRAM_BOT_TOKEN is not set (see .env.example)")
    return Bot(
        token=settings.telegram_bot_token.get_secret_value(),
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )


async def run_bot(settings: Settings) -> None:
    from football_ai.core.logging import redaction_filter

    bot = build_bot(settings)
    if settings.telegram_bot_token is not None:
        redaction_filter.register(settings.telegram_bot_token.get_secret_value())
    dp = build_dispatcher(build_backend(settings), settings)
    await bot.set_my_commands(COMMANDS)
    logger.info("bot_started")
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()
