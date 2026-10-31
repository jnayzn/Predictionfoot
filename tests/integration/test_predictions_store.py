"""Stored predictions: insert-only, pre-kick-off only, evaluated after the match."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from football_ai.data.collectors.api_football.parsers import parse_fixture
from football_ai.database import models as db
from football_ai.database.connection import session_scope
from football_ai.database.repositories import FixtureRepository, PredictionRepository
from football_ai.database.repositories.predictions import PredictionTooLateError
from football_ai.training.benchmark import load_market_benchmark
from tests.conftest import fixture_item

KICKOFF = datetime(2026, 11, 1, 15, tzinfo=UTC)


def _setup(session_factory: sessionmaker[Session]) -> tuple[int, int]:
    item = fixture_item(1, 39, (33, "A"), (34, "B"), date=KICKOFF.isoformat())
    with session_scope(session_factory) as s:
        stats = FixtureRepository(s).upsert_fixtures([parse_fixture(item)])
        model = db.ModelVersion(name="ensemble", version="v1", algorithm="test")
        s.add(model)
        s.flush()
        return stats.fixture_ids[1], model.id


def test_record_and_evaluate(session_factory: sessionmaker[Session]) -> None:
    fixture_id, model_id = _setup(session_factory)
    with session_scope(session_factory) as s:
        repo = PredictionRepository(s)
        repo.record(
            fixture_id=fixture_id,
            model_version_id=model_id,
            kickoff_utc=KICKOFF,
            probabilities=(0.5, 0.3, 0.2),
            expected_goals=(1.6, 1.0),
            created_at=KICKOFF - timedelta(hours=3),
        )
        with pytest.raises(PredictionTooLateError):
            repo.record(
                fixture_id=fixture_id,
                model_version_id=model_id,
                kickoff_utc=KICKOFF,
                probabilities=(0.5, 0.3, 0.2),
                created_at=KICKOFF,
            )
        assert repo.evaluate_pending() == 0  # match not played yet

    finished = fixture_item(
        1, 39, (33, "A"), (34, "B"), date=KICKOFF.isoformat(), status="FT", goals=(2, 1)
    )
    with session_scope(session_factory) as s:
        FixtureRepository(s).upsert_fixtures([parse_fixture(finished, competition_key=None)])
    with session_scope(session_factory) as s:
        repo = PredictionRepository(s)
        assert repo.evaluate_pending() == 1
        assert repo.evaluate_pending() == 0  # idempotent
        evaluation = s.scalar(select(db.PredictionEvaluation))
        assert evaluation is not None
        assert evaluation.correct and evaluation.actual_outcome == "H"
        assert evaluation.log_loss == pytest.approx(-__import__("math").log(0.5))
        assert evaluation.brier_score == pytest.approx(0.25 + 0.09 + 0.04)
        assert evaluation.goals_abs_error == pytest.approx(0.4)
        summary = repo.performance()
        assert summary.n == 1 and summary.accuracy == 1.0


def test_market_benchmark_uses_closing_odds_only(session_factory: sessionmaker[Session]) -> None:
    from football_ai.data.cleaning.canonical import CanonicalBuilder

    fixture_id, _ = _setup(session_factory)
    with session_scope(session_factory) as s:
        fixture = s.get(db.Fixture, fixture_id)
        assert fixture is not None
        fixture.league.key = "premier_league"
        for odds_type, values in (("closing", (2.0, 3.5, 4.0)), ("pre_match", (1.5, 4.0, 6.0))):
            for sel, odd in zip(("home", "draw", "away"), values, strict=True):
                s.add(
                    db.Odds(
                        fixture_id=fixture_id,
                        provider="api_football",
                        bookmaker="B",
                        market="1x2",
                        selection=sel,
                        odd=odd,
                        odds_type=odds_type,
                    )
                )
    with session_scope(session_factory) as s:
        CanonicalBuilder(s).build("premier_league")
    with session_scope(session_factory) as s:
        market = load_market_benchmark(s)
    assert len(market) == 1
    inv = [1 / 2.0, 1 / 3.5, 1 / 4.0]
    assert market.iloc[0]["market__home"] == pytest.approx(inv[0] / sum(inv))
