"""Prediction engine, analysis, recommendation and the FastAPI backend."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from football_ai.analysis.match_analysis import analyse_match
from football_ai.analysis.recommendation import (
    RecommendationConfig,
    RecommendationStatus,
    recommend,
)
from football_ai.backend.main import create_app
from football_ai.backend.services.football_service import FootballService
from football_ai.config.competitions import load_competitions
from football_ai.config.settings import Settings
from football_ai.core.exceptions import ConfigurationError
from football_ai.database import models as db
from football_ai.database.connection import session_scope
from football_ai.prediction.confidence import Confidence, ConfidenceLevel
from football_ai.prediction.engine import PredictionEngine, UnknownMatchError
from football_ai.prediction.odds import load_match_odds
from tests.conftest import ROOT
from tests.db_seed import Seeded

API_KEY = "test-api-key-123456"


@pytest.fixture(scope="module")
def client(world: dict[str, Any], engine: PredictionEngine) -> TestClient:
    settings = Settings(
        _env_file=None,
        api_keys=API_KEY,
        api_rate_limit_per_minute=1000,  # type: ignore[call-arg]
        competitions_file=ROOT / "configs" / "competitions.toml",
        log_level="WARNING",
    )
    service = FootballService(
        world["factory"],
        engine,
        load_competitions(settings.competitions_file),
        reports_dir=world["root"] / "reports",
    )
    return TestClient(create_app(settings, service))


def auth() -> dict[str, str]:
    return {"X-API-Key": API_KEY}


# ------------------------------------------------------------------ engine
def test_engine_predictions(engine: PredictionEngine, world: dict[str, Any]) -> None:
    seeded: Seeded = world["seeded"]
    preds = engine.predict(seeded.upcoming_match_ids)
    for p in preds:
        assert sum(p.probabilities.values()) == pytest.approx(1.0)
        assert p.markets is not None and p.expected_goals is not None
        assert p.markets["1x2"]["home"] == pytest.approx(p.probabilities["home"], abs=1e-6)
        assert p.model_name == "ensemble" and p.member_probabilities
        assert p.home_team.startswith("Synthetic FC") and p.is_upcoming
        assert (
            "blessures et suspensions non couvertes par les sources" in p.confidence.uncertainties
        )
    with pytest.raises(UnknownMatchError):
        engine.predict([999999])


def test_hypothetical_match(engine: PredictionEngine, world: dict[str, Any]) -> None:
    teams = world["seeded"].team_ids
    p = engine.predict_hypothetical(teams[0], teams[1])
    reverse = engine.predict_hypothetical(teams[1], teams[0])
    assert sum(p.probabilities.values()) == pytest.approx(1.0)
    # home advantage: each team does better at home than away against the same opponent
    assert p.probabilities["home"] > reverse.probabilities["away"]
    with pytest.raises(ValueError):
        engine.predict_hypothetical(teams[0], teams[0])
    with pytest.raises(UnknownMatchError):
        engine.predict_hypothetical(teams[0], 424242)


def test_predict_and_store_only_upcoming(engine: PredictionEngine, world: dict[str, Any]) -> None:
    seeded: Seeded = world["seeded"]
    ids = seeded.upcoming_match_ids[:1] + seeded.played_match_ids[-1:]
    _, stored = engine.predict_and_store(ids)
    assert stored == 1
    with session_scope(world["factory"]) as s:
        row = s.scalar(select(db.Prediction))
        assert row is not None and row.created_at < row.kickoff_utc
        assert row.data_quality["uncertainties"]


# ---------------------------------------------------- analysis & recommendation
def test_analysis_texts_are_probabilistic(engine: PredictionEngine, world: dict[str, Any]) -> None:
    p = engine.predict(world["seeded"].upcoming_match_ids[:1])[0]
    analysis = analyse_match(p, engine.frames().matches)
    assert (
        "%" in analysis.summary and "certitude" in analysis.summary
    ) or "équilibré" in analysis.summary
    assert len(analysis.home.form_last5) == 5 and set(analysis.home.form_last5) <= set("VND")
    assert analysis.home.attack_index is not None
    assert all(f.favours in ("home", "away") for f in analysis.key_factors)


def test_recommendation_rules(engine: PredictionEngine, world: dict[str, Any]) -> None:
    p = engine.predict(world["seeded"].upcoming_match_ids[:1])[0]
    no_odds = recommend(p, None)
    assert no_odds.status is RecommendationStatus.NO_ODDS and no_odds.most_likely
    high = replace(p, confidence=Confidence(0.9, ConfidenceLevel.HIGH))
    fair = {
        "1x2": {k: round(1 / (v * 1.02), 3) for k, v in p.probabilities.items()},
    }
    # odds equal to the model's own probabilities (+2% margin): no value anywhere
    assert recommend(high, fair).status is RecommendationStatus.NONE
    # the market underrates the home side (book with a normal 4% margin) -> value
    ph, pd_, pa = p.probabilities["home"], p.probabilities["draw"], p.probabilities["away"]
    rest = 1.04 - 0.75 * ph
    generous = {
        "1x2": {
            "home": 1 / (0.75 * ph),
            "draw": 1 / (rest * pd_ / (pd_ + pa)),
            "away": 1 / (rest * pa / (pd_ + pa)),
        }
    }
    value = recommend(high, generous, RecommendationConfig(max_overround=1.5))
    assert value.status is RecommendationStatus.VALUE
    pick = value.selections[0]
    assert (pick.market, pick.selection) == ("1x2", "home")
    assert pick.market_probability < pick.adjusted_probability < pick.model_probability
    assert pick.expected_value > 0.03 and pick.reasons
    # low confidence: probabilities fully shrunk to the market -> never a recommendation
    low = replace(p, confidence=Confidence(0.2, ConfidenceLevel.LOW))
    refused = recommend(low, generous, RecommendationConfig(max_overround=1.5))
    assert refused.status is RecommendationStatus.NONE
    assert any("confiance faible" in r for r in refused.reasons)
    # an inconsistent book (implied probabilities summing below 1) is never trusted
    arbitrage = {"1x2": {"home": 3.0, "draw": 4.0, "away": 4.0}}
    stale = recommend(high, arbitrage, RecommendationConfig(max_overround=1.5))
    assert stale.status is RecommendationStatus.NONE
    assert any("incohérentes" in r for r in stale.reasons)


def test_match_odds_loader(world: dict[str, Any]) -> None:
    with session_scope(world["factory"]) as s:
        odds = load_match_odds(s, world["seeded"].upcoming_match_ids[0])
    assert set(odds["1x2"]) == {"home", "draw", "away"}
    assert sum(1 / v for v in odds["1x2"].values()) == pytest.approx(1.05, abs=0.01)


# --------------------------------------------------------------------- API
def test_health_is_public(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["model"] == "ensemble"


def test_api_key_required(client: TestClient) -> None:
    assert client.get("/v1/competitions").status_code == 401
    assert client.get("/v1/competitions", headers={"X-API-Key": "wrong"}).status_code == 401
    r = client.get("/v1/competitions", headers=auth())
    assert r.status_code == 200 and len(r.json()) == 6
    assert r.headers["X-Request-ID"]


def test_fixtures_prediction_analysis(client: TestClient, world: dict[str, Any]) -> None:
    match_id = world["seeded"].upcoming_match_ids[0]
    match = client.get(f"/v1/matches/{match_id}", headers=auth()).json()
    day = match["kickoff_utc"][:10]
    fixtures = client.get("/v1/fixtures", params={"day": day}, headers=auth()).json()
    assert match_id in {f["id"] for f in fixtures}
    pred = client.get(f"/v1/matches/{match_id}/prediction", headers=auth()).json()
    assert sum(pred["probabilities"].values()) == pytest.approx(1.0)
    assert pred["markets"]["over_under"]["2.5"]["over"] > 0
    assert "jamais des certitudes" in pred["disclaimer"]
    analysis = client.get(f"/v1/matches/{match_id}/analysis", headers=auth()).json()
    assert analysis["recommendation"]["status"] in ("value", "none")
    assert analysis["recommendation"]["disclaimer"]
    assert analysis["home"]["form_last5"]


def test_compare_teams_and_search(client: TestClient, world: dict[str, Any]) -> None:
    teams = world["seeded"].team_ids
    r = client.get("/v1/compare", params={"home": teams[0], "away": teams[1]}, headers=auth())
    assert r.status_code == 200 and r.json()["match"]["status"] == "hypothetical"
    found = client.get("/v1/teams", params={"search": "synthetic"}, headers=auth()).json()
    assert len(found) == 8
    history = client.get(f"/v1/teams/{teams[0]}/matches", params={"limit": 3}, headers=auth())
    assert len(history.json()) == 3


def test_input_validation_and_errors(client: TestClient) -> None:
    assert client.get("/v1/teams", params={"search": "a"}, headers=auth()).status_code == 422
    assert (
        client.get("/v1/fixtures", params={"competition": "DROP TABLE"}, headers=auth()).status_code
        == 422
    )
    assert (
        client.get("/v1/fixtures", params={"day": "not-a-date"}, headers=auth()).status_code == 422
    )
    r = client.get("/v1/matches/999999/prediction", headers=auth())
    assert r.status_code == 404 and r.json()["request_id"]
    assert (
        client.get("/v1/fixtures", params={"competition": "eredivisie"}, headers=auth()).status_code
        == 404
    )


def test_results_and_performance(client: TestClient, world: dict[str, Any]) -> None:
    with session_scope(world["factory"]) as s:
        last = s.scalar(
            select(db.Match)
            .where(db.Match.status_group == "finished")
            .order_by(db.Match.kickoff_utc.desc())
            .limit(1)
        )
        assert last is not None
        day = last.kickoff_utc.date().isoformat()
    results = client.get("/v1/results", params={"day": day}, headers=auth()).json()
    assert results and all(r["match"]["score"] is not None for r in results)
    perf = client.get("/v1/performance", headers=auth()).json()
    assert "live" in perf and "note" in perf


def test_rate_limit(world: dict[str, Any], engine: PredictionEngine) -> None:
    settings = Settings(
        _env_file=None,
        api_keys=API_KEY,
        api_rate_limit_per_minute=2,  # type: ignore[call-arg]
        competitions_file=ROOT / "configs" / "competitions.toml",
    )
    service = FootballService(
        world["factory"], engine, load_competitions(settings.competitions_file)
    )
    limited = TestClient(create_app(settings, service))
    codes = [limited.get("/v1/competitions", headers=auth()).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_production_requires_api_keys() -> None:
    with pytest.raises(ConfigurationError):
        create_app(Settings(_env_file=None, app_env="production"))  # type: ignore[call-arg]


def test_prediction_count(world: dict[str, Any]) -> None:
    with session_scope(world["factory"]) as s:
        assert (s.scalar(select(func.count()).select_from(db.Prediction)) or 0) >= 1
    assert datetime.now(UTC) - timedelta(days=1) < datetime.now(UTC)


def test_engine_memoizes_until_invalidated(world: dict[str, Any]) -> None:
    fresh = PredictionEngine(world["factory"])
    calls: list[int] = []
    original = fresh._compute

    def counting(ids: Any, frames: Any) -> Any:
        calls.append(len(ids))
        return original(ids, frames)

    fresh._compute = counting  # type: ignore[method-assign]
    ids = world["seeded"].upcoming_match_ids
    first = fresh.predict(ids[:1])
    assert calls == [len(ids)]  # one pass for every upcoming match
    assert fresh.predict(ids[1:2])[0].match_id == ids[1] and len(calls) == 1  # served from memory
    assert fresh.predict(ids[:1])[0] is first[0]
    fresh.invalidate()
    fresh.predict(ids[:1])
    assert len(calls) == 2
