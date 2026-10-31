"""Fixture details (statistics, lineups, players), standings, injuries and odds."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from football_ai.database import models as db
from football_ai.database.models import (
    PLAYER_STAT_COLUMNS,
    TEAM_STAT_COLUMNS,
    utcnow,
)
from football_ai.database.repositories._utils import apply_changes
from football_ai.database.repositories.fixtures import FixtureRepository, UpsertStats
from football_ai.database.repositories.reference import ReferenceRepository
from football_ai.domain import models as dm


class MatchDataRepository:
    def __init__(
        self,
        session: Session,
        refs: ReferenceRepository | None = None,
        fixtures: FixtureRepository | None = None,
    ) -> None:
        self.session = session
        self.refs = refs or ReferenceRepository(session)
        self.fixtures = fixtures or FixtureRepository(session, self.refs)

    # ------------------------------------------------------------ details
    def save_team_statistics(
        self, fixture_id: int, items: Sequence[dm.TeamFixtureStatistics]
    ) -> int:
        count = 0
        for stat in items:
            team = self.refs.ensure_team(stat.provider, stat.team)
            values: dict[str, Any] = {c: stat.values.get(c) for c in TEAM_STAT_COLUMNS}
            values["values"] = dict(stat.values)
            row = self.session.scalar(
                select(db.TeamStatistics).where(
                    db.TeamStatistics.fixture_id == fixture_id,
                    db.TeamStatistics.team_id == team.id,
                    db.TeamStatistics.provider == stat.provider,
                )
            )
            if row is None:
                self.session.add(
                    db.TeamStatistics(
                        fixture_id=fixture_id, team_id=team.id, provider=stat.provider, **values
                    )
                )
            else:
                apply_changes(row, values)
            count += 1
        return count

    def save_player_statistics(
        self, fixture_id: int, items: Sequence[dm.PlayerFixtureStatistics]
    ) -> int:
        existing = {
            (r.player_id, r.provider): r
            for r in self.session.scalars(
                select(db.PlayerStatistics).where(db.PlayerStatistics.fixture_id == fixture_id)
            )
        }
        for stat in items:
            team = self.refs.ensure_team(stat.provider, stat.team)
            player = self.refs.ensure_player(stat.provider, stat.player)
            values: dict[str, Any] = {c: getattr(stat, c) for c in PLAYER_STAT_COLUMNS}
            values["team_id"] = team.id
            row = existing.get((player.id, stat.provider))
            if row is None:
                row = db.PlayerStatistics(
                    fixture_id=fixture_id, player_id=player.id, provider=stat.provider, **values
                )
                self.session.add(row)
                existing[(player.id, stat.provider)] = row
            else:
                apply_changes(row, values)
        return len(items)

    def save_lineups(self, fixture_id: int, items: Sequence[dm.Lineup]) -> int:
        for lineup in items:
            team = self.refs.ensure_team(lineup.provider, lineup.team)
            row = self.session.scalar(
                select(db.Lineup).where(
                    db.Lineup.fixture_id == fixture_id,
                    db.Lineup.team_id == team.id,
                    db.Lineup.provider == lineup.provider,
                )
            )
            if row is None:
                row = db.Lineup(fixture_id=fixture_id, team_id=team.id, provider=lineup.provider)
                self.session.add(row)
            row.formation = lineup.formation
            row.coach_name = lineup.coach_name
            # lineup is replaced as a whole (the provider gives the complete list)
            row.players.clear()
            self.session.flush()
            seen: set[int] = set()
            for entry in lineup.players:
                player = self.refs.ensure_player(lineup.provider, entry.player)
                if player.id in seen:
                    continue
                seen.add(player.id)
                row.players.append(
                    db.LineupPlayer(
                        player_id=player.id,
                        number=entry.number,
                        position=entry.position,
                        grid=entry.grid,
                        is_starter=entry.is_starter,
                    )
                )
        return len(items)

    def save_fixture_details(self, details: Sequence[dm.FixtureDetails]) -> UpsertStats:
        stats = self.fixtures.upsert_fixtures([d.fixture for d in details])
        for item in details:
            fixture_id = stats.fixture_ids[item.fixture.provider_id]
            self.save_team_statistics(fixture_id, item.statistics)
            self.save_lineups(fixture_id, item.lineups)
            self.save_player_statistics(fixture_id, item.players)
            row = self.session.get(db.Fixture, fixture_id)
            if row is not None:
                row.details_fetched_at = utcnow()
        self.session.flush()
        return stats

    # ---------------------------------------------------------- standings
    def save_standings(
        self, rows: Sequence[dm.StandingRow], *, snapshot_date: date, league_key: str | None = None
    ) -> int:
        for item in rows:
            league = self.refs.ensure_league(
                item.provider,
                item.league_provider_id,
                league_key or str(item.league_provider_id),
                key=league_key,
            )
            team = self.refs.ensure_team(item.provider, item.team)
            values: dict[str, Any] = {
                "rank": item.rank,
                "points": item.points,
                "goals_diff": item.goals_diff,
                "form": item.form,
                "status": item.status,
                "description": item.description,
                "played": item.played,
                "win": item.win,
                "draw": item.draw,
                "lose": item.lose,
                "goals_for": item.goals_for,
                "goals_against": item.goals_against,
                "home": dict(item.home),
                "away": dict(item.away),
                "source_updated_at": item.source_updated_at,
            }
            group_filter = (
                db.Standing.group_name.is_(None)
                if item.group_name is None
                else db.Standing.group_name == item.group_name
            )
            row = self.session.scalar(
                select(db.Standing).where(
                    db.Standing.provider == item.provider,
                    db.Standing.league_id == league.id,
                    db.Standing.season_year == item.season,
                    db.Standing.team_id == team.id,
                    db.Standing.snapshot_date == snapshot_date,
                    group_filter,
                )
            )
            if row is None:
                self.session.add(
                    db.Standing(
                        provider=item.provider,
                        league_id=league.id,
                        season_year=item.season,
                        team_id=team.id,
                        snapshot_date=snapshot_date,
                        group_name=item.group_name,
                        **values,
                    )
                )
            else:
                apply_changes(row, values)
        self.session.flush()
        return len(rows)

    # ----------------------------------------------------------- injuries
    def save_injuries(self, items: Sequence[dm.Injury], *, league_key: str | None = None) -> int:
        for item in items:
            league = self.refs.ensure_league(
                item.provider,
                item.league_provider_id,
                league_key or str(item.league_provider_id),
                key=league_key,
            )
            team = self.refs.ensure_team(item.provider, item.team)
            player = self.refs.ensure_player(item.provider, item.player)
            fixture_id = None
            if item.fixture_provider_id is not None:
                fx = self.fixtures.by_provider_ids(item.provider, [item.fixture_provider_id])
                fixture_id = fx[item.fixture_provider_id].id if fx else None
            fixture_filter = (
                db.Injury.fixture_provider_id.is_(None)
                if item.fixture_provider_id is None
                else db.Injury.fixture_provider_id == item.fixture_provider_id
            )
            row = self.session.scalar(
                select(db.Injury).where(
                    db.Injury.provider == item.provider,
                    db.Injury.player_id == player.id,
                    db.Injury.league_id == league.id,
                    db.Injury.season_year == item.season,
                    fixture_filter,
                )
            )
            values: dict[str, Any] = {
                "team_id": team.id,
                "fixture_id": fixture_id,
                "fixture_date": item.fixture_date,
                "type": item.type,
                "reason": item.reason,
            }
            if row is None:
                self.session.add(
                    db.Injury(
                        provider=item.provider,
                        player_id=player.id,
                        league_id=league.id,
                        season_year=item.season,
                        fixture_provider_id=item.fixture_provider_id,
                        **values,
                    )
                )
            else:
                apply_changes(row, values)
        self.session.flush()
        return len(items)

    # --------------------------------------------------------------- odds
    def save_odds(self, quotes: Sequence[dm.OddsQuote]) -> int:
        """Insert odds snapshots; identical quotes (same unique key) are skipped."""
        if not quotes:
            return 0
        by_provider: dict[str, list[dm.OddsQuote]] = {}
        for q in quotes:
            by_provider.setdefault(q.provider, []).append(q)
        inserted = 0
        for provider, items in by_provider.items():
            fixtures = self.fixtures.by_provider_ids(
                provider, [q.fixture_provider_id for q in items]
            )
            fixture_ids = [f.id for f in fixtures.values()]
            existing = {
                (o.fixture_id, o.bookmaker, o.market, o.selection, o.odds_type, o.source_updated_at)
                for o in self.session.scalars(
                    select(db.Odds).where(
                        db.Odds.provider == provider, db.Odds.fixture_id.in_(fixture_ids)
                    )
                )
            }
            for q in items:
                fixture = fixtures.get(q.fixture_provider_id)
                if fixture is None:
                    continue  # odds for an unknown fixture: load the fixture first
                key = (
                    fixture.id,
                    q.bookmaker,
                    q.market,
                    q.selection,
                    q.odds_type.value,
                    q.source_updated_at,
                )
                if key in existing:
                    continue
                existing.add(key)
                self.session.add(
                    db.Odds(
                        fixture_id=fixture.id,
                        provider=provider,
                        bookmaker=q.bookmaker,
                        market=q.market,
                        selection=q.selection,
                        odd=q.odd,
                        odds_type=q.odds_type.value,
                        source_updated_at=q.source_updated_at,
                    )
                )
                inserted += 1
        self.session.flush()
        return inserted
