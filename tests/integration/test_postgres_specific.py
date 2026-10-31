"""PostgreSQL-only guarantees: real migrations, immutability trigger, constraints."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import Engine, inspect, text
from sqlalchemy.exc import DBAPIError, IntegrityError

from football_ai.database import models as db
from football_ai.database.connection import create_session_factory, session_scope
from tests.conftest import TEST_DATABASE_URL, _truncate_all, alembic_downgrade, alembic_upgrade

pytestmark = pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL not set")


def _seed_prediction(engine: Engine) -> int:
    factory = create_session_factory(engine)
    kickoff = datetime.now(UTC) + timedelta(days=1)
    with session_scope(factory) as s:
        league = db.League(provider="t", provider_id=1, name="L")
        home = db.Team(provider="t", provider_id=1, name="H")
        away = db.Team(provider="t", provider_id=2, name="A")
        s.add_all([league, home, away])
        s.flush()
        fixture = db.Fixture(
            provider="t",
            provider_id=1,
            league_id=league.id,
            season_year=2026,
            kickoff_utc=kickoff,
            status_short="NS",
            status_group="scheduled",
            home_team_id=home.id,
            away_team_id=away.id,
        )
        model = db.ModelVersion(name="m", version="1", algorithm="test")
        s.add_all([fixture, model])
        s.flush()
        pred = db.Prediction(
            fixture_id=fixture.id,
            model_version_id=model.id,
            kickoff_utc=kickoff,
            p_home=0.5,
            p_draw=0.3,
            p_away=0.2,
        )
        s.add(pred)
        s.flush()
        return pred.id


def test_predictions_are_immutable(pg_engine: Engine) -> None:
    _truncate_all(pg_engine)
    pred_id = _seed_prediction(pg_engine)
    for statement in (
        f"UPDATE predictions SET p_home = 0.6, p_draw = 0.2 WHERE id = {pred_id}",
        f"DELETE FROM predictions WHERE id = {pred_id}",
    ):
        with pytest.raises(DBAPIError, match="immutable"), pg_engine.begin() as conn:
            conn.execute(text(statement))


def test_prediction_constraints(pg_engine: Engine) -> None:
    _truncate_all(pg_engine)
    _seed_prediction(pg_engine)
    with pytest.raises(IntegrityError), pg_engine.begin() as conn:  # probabilities sum != 1
        conn.execute(
            text(
                "INSERT INTO predictions (fixture_id, model_version_id, created_at, kickoff_utc, "
                "p_home, p_draw, p_away, markets, data_quality) "
                "SELECT fixture_id, model_version_id, "
                "now(), kickoff_utc, 0.5, 0.5, 0.5, '{}', '{}' FROM predictions LIMIT 1"
            )
        )
    with pytest.raises(IntegrityError), pg_engine.begin() as conn:  # created after kick-off
        conn.execute(
            text(
                "INSERT INTO predictions (fixture_id, model_version_id, created_at, kickoff_utc, "
                "p_home, p_draw, p_away, markets, data_quality) "
                "SELECT fixture_id, model_version_id, "
                "kickoff_utc + interval '1 minute', kickoff_utc, 0.5, 0.3, 0.2, '{}', '{}' "
                "FROM predictions LIMIT 1"
            )
        )


def test_migration_downgrade_and_upgrade(pg_engine: Engine) -> None:
    assert TEST_DATABASE_URL
    alembic_downgrade(TEST_DATABASE_URL, "base")
    assert "fixtures" not in inspect(pg_engine).get_table_names()
    alembic_upgrade(TEST_DATABASE_URL, "head")
    tables = set(inspect(pg_engine).get_table_names())
    assert {t.name for t in db.Base.metadata.sorted_tables} <= tables
