"""FastAPI application factory.

Run: ``uvicorn football_ai.backend.main:create_app --factory --host 0.0.0.0 --port 8000``
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from football_ai import __version__
from football_ai.backend.middleware import request_context
from football_ai.backend.routes.api import public, router
from football_ai.backend.security import RateLimiter
from football_ai.backend.services.football_service import FootballService, NotFoundError
from football_ai.bootstrap import build_session_factory
from football_ai.config.competitions import load_competitions
from football_ai.config.settings import Settings
from football_ai.core.exceptions import ConfigurationError, FootballAIError, InsufficientDataError
from football_ai.core.logging import configure_logging, redaction_filter
from football_ai.prediction.engine import NoProductionModelError, PredictionEngine

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, service: FootballService | None = None) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level, settings.log_format)
    if settings.app_env == "production" and not settings.accepted_api_keys:
        raise ConfigurationError("API_KEYS must be set in production")
    for key in settings.accepted_api_keys:
        redaction_filter.register(key)

    if service is None:
        factory = build_session_factory(settings)
        engine = PredictionEngine(factory, model_priority=settings.model_priority)
        service = FootballService(
            factory, engine, load_competitions(settings.competitions_file),
            reports_dir=settings.data_dir / "reports",
        )  # fmt: skip

    app = FastAPI(
        title="Football AI Predictor API",
        version=__version__,
        description=(
            "Probabilistic football match analysis. Every output is a probability, "
            "never a certainty, and not betting advice."
        ),
    )
    app.state.service = service
    app.state.api_keys = settings.accepted_api_keys
    app.state.rate_limiter = RateLimiter(settings.api_rate_limit_per_minute)
    app.middleware("http")(request_context)
    origins = [o.strip() for o in settings.api_cors_origins.split(",") if o.strip()]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET"])

    def _error(status: int, message: str, request: Request) -> JSONResponse:
        return JSONResponse(
            status_code=status,
            content={"error": message, "request_id": getattr(request.state, "request_id", None)},
        )

    @app.exception_handler(NotFoundError)
    async def not_found(request: Request, exc: NotFoundError) -> JSONResponse:
        return _error(404, str(exc), request)

    @app.exception_handler(NoProductionModelError)
    async def no_model(request: Request, exc: NoProductionModelError) -> JSONResponse:
        return _error(503, str(exc), request)

    @app.exception_handler(InsufficientDataError)
    async def no_data(request: Request, exc: InsufficientDataError) -> JSONResponse:
        return _error(422, str(exc), request)

    @app.exception_handler(FootballAIError)
    async def domain_error(request: Request, exc: FootballAIError) -> JSONResponse:
        logger.warning("api_domain_error", extra={"error": str(exc)})
        return _error(400, str(exc), request)

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("api_unexpected_error")
        return _error(500, "internal error", request)

    app.include_router(public)
    app.include_router(router)
    return app
