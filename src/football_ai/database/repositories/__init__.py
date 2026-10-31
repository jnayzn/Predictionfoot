"""Repositories: the only place where ORM queries are written."""

from football_ai.database.repositories.fixtures import FixtureRepository, UpsertStats
from football_ai.database.repositories.jobs import JobRepository
from football_ai.database.repositories.match_data import MatchDataRepository
from football_ai.database.repositories.reference import ReferenceRepository

__all__ = [
    "FixtureRepository",
    "JobRepository",
    "MatchDataRepository",
    "ReferenceRepository",
    "UpsertStats",
]
