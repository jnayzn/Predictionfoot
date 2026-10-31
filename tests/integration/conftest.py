"""Shared fixture: a database seeded with a SYNTHETIC league and a registered model."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from football_ai.database.connection import create_db_engine, create_session_factory
from football_ai.database.models import Base
from football_ai.prediction.engine import PredictionEngine
from tests.db_seed import seed_database


@pytest.fixture(scope="session")
def world(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    root = tmp_path_factory.mktemp("api")
    engine = create_db_engine(f"sqlite:///{root / 'api.db'}")
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    seeded = seed_database(factory, root / "models")
    yield {"factory": factory, "seeded": seeded, "root": root}
    engine.dispose()


@pytest.fixture(scope="session")
def engine(world: dict[str, Any]) -> PredictionEngine:
    return PredictionEngine(world["factory"])
