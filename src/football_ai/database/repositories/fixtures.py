"""Fixtures, actual results and correction tracking."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from football_ai.database import models as db
from football_ai.database.models import utcnow
from football_ai.database.repositories._utils import apply_changes, chunks
from football_ai.database.repositories.reference import ReferenceRepository
from football_ai.domain import models as dm
from football_ai.domain.models import FixtureStatusGroup

logger = logging.getLogger(__name__)

# Fields whose change on an already finished fixture is a *correction*.
_RESULT_FIELDS = ("home_goals", "away_goals", "ft_home_goals", "ft_away_goals", "status_short")


@dataclass(slots=True)
class UpsertStats:
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    corrections: int = 0
    fixture_ids: dict[int, int] = field(default_factory=dict)  # provider_id -> db id

    def merge(self, other: UpsertStats) -> None:
        self.inserted += other.inserted
        self.updated += other.updated
        self.unchanged += other.unchanged
        self.corrections += other.corrections
        self.fixture_ids.update(other.fixture_ids)


def match_outcome(home: int, away: int) -> str:
    return "H" if home > away else "A" if away > home else "D"


class FixtureRepository:
    def __init__(self, session: Session, refs: ReferenceRepository | None = None) -> None:
        self.session = session
        self.refs = refs or ReferenceRepository(session)

    # ----------------------------------------------------------------- queries
    def by_provider_ids(self, provider: str, provider_ids: Sequence[int]) -> dict[int, db.Fixture]:
        found: dict[int, db.Fixture] = {}
        for chunk in chunks(list(set(provider_ids))):
            rows = self.session.scalars(
                select(db.Fixture).where(
                    db.Fixture.provider == provider, db.Fixture.provider_id.in_(chunk)
                )
            )
            found.update({r.provider_id: r for r in rows})
        return found

    def finished_without_details(
        self, provider: str, league_id: int, season: int, limit: int | None = None
    ) -> list[int]:
        stmt = (
            select(db.Fixture.provider_id)
            .where(
                db.Fixture.provider == provider,
                db.Fixture.league_id == league_id,
                db.Fixture.season_year == season,
                db.Fixture.status_group == FixtureStatusGroup.FINISHED.value,
                db.Fixture.details_fetched_at.is_(None),
            )
            .order_by(db.Fixture.kickoff_utc)
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        return list(self.session.scalars(stmt))

    # ------------------------------------------------------------------ upsert
    def _values(self, fixture: dm.Fixture, league_id: int) -> dict[str, Any]:
        home = self.refs.ensure_team(fixture.provider, fixture.home_team)
        away = self.refs.ensure_team(fixture.provider, fixture.away_team)
        return {
            "league_id": league_id,
            "season_year": fixture.season,
            "round": fixture.round,
            "kickoff_utc": fixture.kickoff_utc,
            "kickoff_time_known": fixture.kickoff_time_known,
            "provider_timezone": fixture.provider_timezone,
            "status_short": fixture.status_short,
            "status_long": fixture.status_long,
            "status_group": fixture.status_group.value,
            "elapsed": fixture.elapsed,
            "venue_id": fixture.venue_id,
            "venue_name": fixture.venue_name,
            "venue_city": fixture.venue_city,
            "referee": fixture.referee,
            "home_team_id": home.id,
            "away_team_id": away.id,
            "home_goals": fixture.goals.home,
            "away_goals": fixture.goals.away,
            "ht_home_goals": fixture.score_halftime.home,
            "ht_away_goals": fixture.score_halftime.away,
            "ft_home_goals": fixture.score_fulltime.home,
            "ft_away_goals": fixture.score_fulltime.away,
            "et_home_goals": fixture.score_extratime.home,
            "et_away_goals": fixture.score_extratime.away,
            "pen_home_goals": fixture.score_penalty.home,
            "pen_away_goals": fixture.score_penalty.away,
        }

    def upsert_fixtures(self, fixtures: Sequence[dm.Fixture]) -> UpsertStats:
        stats = UpsertStats()
        if not fixtures:
            return stats
        by_provider: dict[str, list[dm.Fixture]] = {}
        for f in fixtures:
            by_provider.setdefault(f.provider, []).append(f)

        for provider, items in by_provider.items():
            existing = self.by_provider_ids(provider, [f.provider_id for f in items])
            for fixture in items:
                league = self.refs.ensure_league(
                    provider,
                    fixture.league_provider_id,
                    fixture.league_name or str(fixture.league_provider_id),
                    key=fixture.competition_key,
                )
                values = self._values(fixture, league.id)
                row = existing.get(fixture.provider_id)
                if row is None:
                    row = db.Fixture(provider=provider, provider_id=fixture.provider_id, **values)
                    self.session.add(row)
                    self.session.flush()
                    existing[fixture.provider_id] = row
                    stats.inserted += 1
                else:
                    was_finished = row.status_group == FixtureStatusGroup.FINISHED.value
                    old = {name: getattr(row, name) for name in _RESULT_FIELDS}
                    changed = apply_changes(row, values)
                    if not changed:
                        stats.unchanged += 1
                    else:
                        stats.updated += 1
                        corrected = [n for n in changed if n in _RESULT_FIELDS and was_finished]
                        if corrected:
                            stats.corrections += 1
                            self._log_changes(row, provider, old, corrected)
                self._sync_actual_result(row)
                stats.fixture_ids[fixture.provider_id] = row.id
        self.session.flush()
        return stats

    def _log_changes(
        self, row: db.Fixture, provider: str, old: dict[str, Any], fields: list[str]
    ) -> None:
        for name in fields:
            self.session.add(
                db.DataChangeLog(
                    entity="fixture",
                    entity_id=row.provider_id,
                    provider=provider,
                    field=name,
                    old_value=None if old[name] is None else str(old[name]),
                    new_value=None if getattr(row, name) is None else str(getattr(row, name)),
                )
            )
        logger.warning(
            "fixture_result_corrected",
            extra={"provider": provider, "fixture": row.provider_id, "fields": fields},
        )

    def _sync_actual_result(self, row: db.Fixture) -> None:
        """Maintain ``actual_results`` (90-minute result, used to settle 1X2 markets)."""
        if row.status_group != FixtureStatusGroup.FINISHED.value:
            return
        # score.fulltime = regulation time; fall back on goals (identical when no extra time)
        home = row.ft_home_goals if row.ft_home_goals is not None else row.home_goals
        away = row.ft_away_goals if row.ft_away_goals is not None else row.away_goals
        if home is None or away is None:
            return
        values = {
            "home_goals": home,
            "away_goals": away,
            "ht_home_goals": row.ht_home_goals,
            "ht_away_goals": row.ht_away_goals,
            "outcome": match_outcome(home, away),
            "status_short": row.status_short,
        }
        result = self.session.scalar(
            select(db.ActualResult).where(db.ActualResult.fixture_id == row.id)
        )
        if result is None:
            self.session.add(db.ActualResult(fixture_id=row.id, revision=1, **values))
        elif apply_changes(result, values):
            result.revision += 1
            result.updated_at = utcnow()
