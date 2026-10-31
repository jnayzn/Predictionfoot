"""Leagues, seasons, teams and players."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from football_ai.database import models as db
from football_ai.database.repositories._utils import apply_changes
from football_ai.domain import models as dm


class ReferenceRepository:
    """Get-or-create helpers with a per-session identity cache."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self._leagues: dict[tuple[str, int], db.League] = {}
        self._teams: dict[tuple[str, int], db.Team] = {}
        self._players: dict[tuple[str, int], db.Player] = {}

    # ----------------------------------------------------------------- leagues
    def _find_league(self, provider: str, provider_id: int) -> db.League | None:
        cache_key = (provider, provider_id)
        if cache_key not in self._leagues:
            row = self.session.scalar(
                select(db.League).where(
                    db.League.provider == provider, db.League.provider_id == provider_id
                )
            )
            if row is None:
                return None
            self._leagues[cache_key] = row
        return self._leagues[cache_key]

    def ensure_league(
        self,
        provider: str,
        provider_id: int,
        name: str,
        *,
        key: str | None = None,
        country: str | None = None,
        type_: str | None = None,
    ) -> db.League:
        row = self._find_league(provider, provider_id)
        if row is None:
            row = db.League(
                provider=provider,
                provider_id=provider_id,
                name=name,
                key=key,
                country=country,
                type=type_,
            )
            self.session.add(row)
            self.session.flush()
            self._leagues[(provider, provider_id)] = row
        elif key and row.key != key:
            row.key = key
        return row

    def upsert_competition(self, competition: dm.Competition) -> db.League:
        row = self.ensure_league(
            competition.provider,
            competition.provider_id,
            competition.name,
            key=competition.key,
        )
        apply_changes(
            row,
            {
                "name": competition.name,
                "country": competition.country,
                "type": competition.type,
                "logo_url": competition.logo_url,
                "key": competition.key,
            },
        )
        existing = {
            s.year: s
            for s in self.session.scalars(select(db.Season).where(db.Season.league_id == row.id))
        }
        for season in competition.seasons:
            values = {
                "start_date": season.start,
                "end_date": season.end,
                "is_current": season.current,
                "coverage": dict(season.coverage),
            }
            current = existing.get(season.year)
            if current is None:
                self.session.add(db.Season(league_id=row.id, year=season.year, **values))
            else:
                apply_changes(current, values)
        self.session.flush()
        return row

    def league_by_key(self, key: str, provider: str) -> db.League | None:
        return self.session.scalar(
            select(db.League).where(db.League.key == key, db.League.provider == provider)
        )

    def season_coverage(self, league_id: int, year: int) -> dict[str, bool] | None:
        season = self.session.scalar(
            select(db.Season).where(db.Season.league_id == league_id, db.Season.year == year)
        )
        return dict(season.coverage) if season is not None else None

    # ------------------------------------------------------------------- teams
    def _find_team(self, provider: str, provider_id: int) -> db.Team | None:
        cache_key = (provider, provider_id)
        if cache_key not in self._teams:
            row = self.session.scalar(
                select(db.Team).where(
                    db.Team.provider == provider, db.Team.provider_id == provider_id
                )
            )
            if row is None:
                return None
            self._teams[cache_key] = row
        return self._teams[cache_key]

    def ensure_team(self, provider: str, ref: dm.TeamRef) -> db.Team:
        row = self._find_team(provider, ref.provider_id)
        if row is None:
            row = db.Team(
                provider=provider, provider_id=ref.provider_id, name=ref.name, logo_url=ref.logo_url
            )
            self.session.add(row)
            self.session.flush()
            self._teams[(provider, ref.provider_id)] = row
        else:
            updates: dict[str, object] = {"name": ref.name}
            if ref.logo_url:
                updates["logo_url"] = ref.logo_url
            apply_changes(row, updates)
        return row

    def upsert_team(self, team: dm.Team) -> db.Team:
        row = self.ensure_team(
            team.provider, dm.TeamRef(provider_id=team.provider_id, name=team.name)
        )
        apply_changes(
            row,
            {
                "code": team.code,
                "country": team.country,
                "founded": team.founded,
                "national": team.national,
                "logo_url": team.logo_url or row.logo_url,
                "venue_name": team.venue_name,
                "venue_city": team.venue_city,
                "venue_capacity": team.venue_capacity,
            },
        )
        return row

    # ----------------------------------------------------------------- players
    def ensure_player(self, provider: str, ref: dm.PlayerRef) -> db.Player:
        cache_key = (provider, ref.provider_id)
        row = self._players.get(cache_key)
        if row is None:
            row = self.session.scalar(
                select(db.Player).where(
                    db.Player.provider == provider, db.Player.provider_id == ref.provider_id
                )
            )
        if row is None:
            row = db.Player(provider=provider, provider_id=ref.provider_id, name=ref.name)
            self.session.add(row)
            self.session.flush()
        elif ref.name and row.name != ref.name:
            row.name = ref.name
        self._players[cache_key] = row
        return row
