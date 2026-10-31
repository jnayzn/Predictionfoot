"""Commercial API clients (hashed keys, plans, quotas, usage) and GDPR commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from football_ai.backend.main import create_app
from football_ai.backend.services.football_service import FootballService
from football_ai.bot.backend import ServiceBackend
from football_ai.cli import EXIT_OK, main
from football_ai.config.competitions import load_competitions
from football_ai.config.settings import Settings
from football_ai.database import models as db
from football_ai.database.connection import session_scope
from football_ai.database.repositories import ApiClientRepository
from football_ai.database.repositories.api_clients import Plan, hash_key, load_plans
from football_ai.prediction.engine import PredictionEngine
from tests.conftest import ROOT


@pytest.fixture
def service(world: dict[str, Any], engine: PredictionEngine) -> FootballService:
    return FootballService(
        world["factory"], engine, load_competitions(ROOT / "configs" / "competitions.toml")
    )


def app_client(service: FootballService, **extra: Any) -> TestClient:
    settings = Settings(
        _env_file=None,
        competitions_file=ROOT / "configs" / "competitions.toml",  # type: ignore[call-arg]
        api_keys="admin-key-123456",
        api_rate_limit_per_minute=1000,
        **extra,
    )
    return TestClient(create_app(settings, service))


def test_plans_file() -> None:
    plans = load_plans(ROOT / "configs" / "plans.toml")
    assert {"free", "pro", "business"} <= set(plans)
    assert plans["free"].daily_quota < plans["pro"].daily_quota < plans["business"].daily_quota


def test_keys_are_hashed_and_quota_enforced(
    world: dict[str, Any], service: FootballService
) -> None:
    with session_scope(world["factory"]) as s:
        row, key = ApiClientRepository(s).create("Media SA", Plan("trial", 3, 100))
        assert row.key_hash == hash_key(key) and key not in (row.key_hash, row.key_prefix)
        assert key.startswith("fai_") and len(key) > 40
    client = app_client(service)
    headers = {"X-API-Key": key}
    codes = [client.get("/v1/competitions", headers=headers).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
    usage = client.get("/v1/usage", headers={"X-API-Key": "admin-key-123456"}).json()
    assert usage["client"] == "admin"
    with session_scope(world["factory"]) as s:
        rows = ApiClientRepository(s).usage(row.id)
        assert rows[-1][2] == 4  # metered, including the rejected request
        stored = s.scalars(select(db.ApiClient.key_hash)).all()
        assert key not in stored


def test_plan_rate_limit_usage_endpoint_and_revocation(
    world: dict[str, Any], service: FootballService
) -> None:
    with session_scope(world["factory"]) as s:
        row, key = ApiClientRepository(s).create("Blogger", Plan("tiny", 1000, 2))
    client = app_client(service)
    headers = {"X-API-Key": key}
    assert [client.get("/v1/competitions", headers=headers).status_code for _ in range(3)] == [
        200,
        200,
        429,
    ]
    with session_scope(world["factory"]) as s:
        ApiClientRepository(s).revoke(row.key_prefix)
    fresh = app_client(service)  # new app = empty identity cache
    assert fresh.get("/v1/competitions", headers=headers).status_code == 401
    assert fresh.get("/v1/competitions", headers={"X-API-Key": "fai_unknown"}).status_code == 401


def test_usage_endpoint_for_clients(world: dict[str, Any], service: FootballService) -> None:
    with session_scope(world["factory"]) as s:
        _, key = ApiClientRepository(s).create("Analyst", Plan("pro", 5000, 60))
    client = app_client(service)
    client.get("/v1/competitions", headers={"X-API-Key": key})
    data = client.get("/v1/usage", headers={"X-API-Key": key}).json()
    assert data["plan"] == "pro" and data["daily_quota"] == 5000
    assert data["usage"][-1]["requests"] == 2


def test_cli_api_keys(
    world: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path = world["root"] / "api.db"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("PLANS_FILE", str(ROOT / "configs" / "plans.toml"))
    assert main(["api-keys", "create", "--name", "CLI Corp", "--plan", "business"]) == EXIT_OK
    out = capsys.readouterr().out
    key = out.split("store it securely): ")[1].strip()
    prefix = key[:12]
    assert main(["api-keys", "list"]) == EXIT_OK
    listing = capsys.readouterr().out
    assert "CLI Corp" in listing and prefix in listing and key not in listing
    assert main(["api-keys", "create", "--name", "x", "--plan", "platinum"]) != EXIT_OK
    assert main(["api-keys", "revoke", "--prefix", prefix]) == EXIT_OK
    assert main(["api-usage", "--days", "7"]) == EXIT_OK


async def test_gdpr_forget(service: FootballService) -> None:
    backend = ServiceBackend(service)
    await backend.subscribe(777, "someone", ["ligue_1"])
    assert await backend.erase(777) is True
    assert await backend.erase(777) is False
    assert 777 not in {s.chat_id for s in await backend.subscribers()}
    with session_scope(service.session_factory) as s:
        assert (
            s.scalar(select(db.TelegramSubscriber).where(db.TelegramSubscriber.chat_id == 777))
            is None
        )


async def test_bot_privacy_and_forget_commands(service: FootballService) -> None:
    from aiogram import Bot

    from football_ai.bot.main import build_dispatcher
    from tests.integration.test_bot import TOKEN, FakeSession, message, settings

    session = FakeSession()
    bot = Bot(TOKEN, session=session)
    dp = build_dispatcher(ServiceBackend(service), settings())
    await dp.feed_update(bot, message("/subscribe", 1))
    await dp.feed_update(bot, message("/privacy", 2))
    await dp.feed_update(bot, message("/forget", 3))
    await dp.feed_update(bot, message("/forget", 4))
    texts = session.texts()
    assert "aucune" in texts[1] and "/forget" in texts[1]
    assert "définitivement supprimées" in texts[2] and "Aucune donnée" in texts[3]
