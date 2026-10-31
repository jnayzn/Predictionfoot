"""Provider-agnostic collector interface.

The rest of the system (ETL, features, prediction engine) depends only on
this protocol and on :mod:`football_ai.domain.models`, never on a concrete
provider. Adding a provider = implementing this protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Protocol, runtime_checkable

from football_ai.data.validators.issues import DataQualityIssue
from football_ai.domain.models import Competition, Fixture, FixtureDetails


@dataclass(slots=True)
class CompetitionDiscovery:
    competitions: list[Competition] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    issues: list[DataQualityIssue] = field(default_factory=list)
    from_cache: bool = False


@dataclass(slots=True)
class FixtureBatch:
    day: date | None = None
    label: str = ""
    fixtures: list[Fixture] = field(default_factory=list)
    issues: list[DataQualityIssue] = field(default_factory=list)
    total_received: int = 0
    other_competitions_skipped: int = 0
    duplicates_removed: int = 0
    rejected: int = 0
    from_cache: bool = False


@dataclass(slots=True)
class DetailsBatch:
    details: list[FixtureDetails] = field(default_factory=list)
    issues: list[DataQualityIssue] = field(default_factory=list)
    requests: int = 0


@runtime_checkable
class FootballDataProvider(Protocol):
    """Interface every data provider must implement."""

    name: str

    async def discover_competitions(self) -> CompetitionDiscovery:
        """Resolve the configured competitions and their available seasons."""
        ...

    async def get_fixtures_by_date(
        self, day: date, competition_keys: list[str] | None = None
    ) -> FixtureBatch:
        """Return the fixtures of the covered competitions played on ``day``."""
        ...
