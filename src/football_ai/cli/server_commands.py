"""Phase 10 command: run the HTTP API."""

from __future__ import annotations

import argparse

from football_ai.cli.common import EXIT_OK
from football_ai.config.settings import Settings


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("serve", help="Run the FastAPI backend (uvicorn)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--workers", type=int, default=1)
    p.set_defaults(handler=cmd_serve)


def cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    import uvicorn

    from football_ai.core.exceptions import ConfigurationError

    # fail fast (clear exit code) instead of letting uvicorn respawn crashing workers
    if settings.app_env == "production" and not settings.accepted_api_keys:
        raise ConfigurationError("API_KEYS must be set when APP_ENV=production")

    uvicorn.run(
        "football_ai.backend.main:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_config=None,
    )
    return EXIT_OK
