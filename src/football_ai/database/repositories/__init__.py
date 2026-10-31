"""Repositories: the only place where ORM queries are written."""

from football_ai.database.repositories.api_clients import ApiClientRepository
from football_ai.database.repositories.fixtures import FixtureRepository, UpsertStats
from football_ai.database.repositories.jobs import JobRepository
from football_ai.database.repositories.match_data import MatchDataRepository
from football_ai.database.repositories.predictions import PredictionRepository
from football_ai.database.repositories.reference import ReferenceRepository
from football_ai.database.repositories.subscribers import SubscriberRepository

__all__ = [
    "ApiClientRepository",
    "FixtureRepository",
    "JobRepository",
    "MatchDataRepository",
    "PredictionRepository",
    "ReferenceRepository",
    "SubscriberRepository",
    "UpsertStats",
]
