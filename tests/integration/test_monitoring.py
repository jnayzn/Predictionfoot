"""Monitoring: drift (PSI/KS, performance), Prometheus metrics, Redis rate limiting."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from football_ai.automation.jobs import JobContext, run_job
from football_ai.backend.main import create_app
from football_ai.backend.security import RedisRateLimiter
from football_ai.backend.services.football_service import FootballService
from football_ai.config.competitions import load_competitions
from football_ai.config.settings import Settings
from football_ai.monitoring.drift import feature_drift, performance_drift, psi
from football_ai.prediction.engine import PredictionEngine
from tests.conftest import ROOT


def test_psi_stable_vs_shifted() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    assert psi(ref, rng.normal(0, 1, 5000)) < 0.02
    assert psi(ref, rng.normal(1.0, 1, 5000)) > 0.25
    assert np.isnan(psi(ref[:10], ref[:10]))  # not enough data


def test_feature_drift_report() -> None:
    rng = np.random.default_rng(1)
    ref = pd.DataFrame(
        {
            "home_elo": rng.normal(1500, 80, 400),
            "diff_ppg_5": rng.normal(0, 0.8, 400),
            "match_id": range(400),
        }
    )
    cur = pd.DataFrame(
        {
            "home_elo": rng.normal(1650, 80, 200),
            "diff_ppg_5": rng.normal(0, 0.8, 200),
            "match_id": range(200),
        }
    )
    report = feature_drift(ref, cur, now=datetime(2026, 1, 1, tzinfo=UTC))
    assert [f.feature for f in report.significant] == ["home_elo"]
    assert report.features[0].feature == "home_elo" and report.features[0].ks_pvalue < 1e-6
    assert report.to_dict()["max_psi"] == report.max_psi


def test_performance_drift() -> None:
    assert performance_drift([0.95] * 60, 0.90)["degraded"]
    assert not performance_drift([0.92] * 60, 0.90)["degraded"]  # within tolerance
    assert not performance_drift([2.0] * 10, 0.90)["degraded"]  # not enough evidence


class FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, int] = {}
        self.ttl: dict[str, int] = {}

    def incr(self, key: str) -> int:
        self.data[key] = self.data.get(key, 0) + 1
        return self.data[key]

    def expire(self, key: str, seconds: int) -> None:
        self.ttl[key] = seconds


def test_redis_rate_limiter_is_shared_and_windowed() -> None:
    redis = FakeRedis()
    clock = [120.0]
    worker_a = RedisRateLimiter(redis, 2, clock=lambda: clock[0])
    worker_b = RedisRateLimiter(redis, 2, clock=lambda: clock[0])
    worker_a.check("key:abc")
    worker_b.check("key:abc")  # same counter across workers
    with pytest.raises(HTTPException) as exc:
        worker_a.check("key:abc")
    assert exc.value.status_code == 429 and "Retry-After" in (exc.value.headers or {})
    assert all(v == 120 for v in redis.ttl.values())
    clock[0] += 60  # next window
    worker_a.check("key:abc")


def _client(world: dict[str, Any], engine: PredictionEngine, **extra: Any) -> TestClient:
    settings = Settings(
        _env_file=None,
        competitions_file=ROOT / "configs" / "competitions.toml",  # type: ignore[call-arg]
        api_keys="k-123456",
        data_dir=world["root"] / "var",
        **extra,
    )
    service = FootballService(
        world["factory"], engine, load_competitions(settings.competitions_file)
    )
    return TestClient(create_app(settings, service))


def test_metrics_endpoint(world: dict[str, Any], engine: PredictionEngine) -> None:
    client = _client(world, engine)
    client.get("/v1/competitions", headers={"X-API-Key": "k-123456"})
    client.get(
        f"/v1/matches/{world['seeded'].upcoming_match_ids[0]}", headers={"X-API-Key": "k-123456"}
    )
    body = client.get("/metrics").text
    assert (
        'football_ai_http_requests_total{method="GET",route="/v1/competitions",status="200"}'
        in body
    )
    # route templates, not raw ids (bounded cardinality)
    assert 'route="/v1/matches/{match_id}"' in body
    assert "football_ai_api_quota_limit" in body and "football_ai_model_info" in body
    assert "football_ai_predictions_stored" in body


def test_metrics_token_and_disable(world: dict[str, Any], engine: PredictionEngine) -> None:
    protected = _client(world, engine, metrics_token="metrics-secret")
    assert protected.get("/metrics").status_code == 401
    assert (
        protected.get("/metrics", headers={"Authorization": "Bearer metrics-secret"}).status_code
        == 200
    )
    disabled = _client(world, engine, metrics_enabled=False)
    assert disabled.get("/metrics").status_code == 404


async def test_drift_job(world: dict[str, Any], engine: PredictionEngine, tmp_path: Path) -> None:
    from contextlib import asynccontextmanager

    class Alerts:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send(self, title: str, lines: list[str]) -> None:
            self.sent.append(title)

    @asynccontextmanager
    async def no_api() -> Any:  # pragma: no cover - drift does not call the API
        raise AssertionError("no API call expected")
        yield

    settings = Settings(_env_file=None, data_dir=tmp_path / "var")  # type: ignore[call-arg]
    ctx = JobContext(
        settings=settings,
        session_factory=world["factory"],
        engine=engine,
        api_context=no_api,
        alerts=Alerts(),
        now=lambda: datetime.now(UTC),
    )
    from football_ai.automation.config import JobConfig

    outcome = await run_job(
        "drift_check",
        ctx,
        JobConfig(cron="0 5 * * 1", options={"recent_days": 120, "reference_days": 900}),
    )
    assert outcome.status == "success", outcome
    assert "max_psi" in outcome.details
    assert list((tmp_path / "var" / "reports").glob("drift_*.json"))
