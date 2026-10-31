"""Provider-independent domain models.

Every entity keeps the identifier of the provider it comes from
(``provider`` + ``provider_id``) so that data can be traced back to its
source and several providers can coexist.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FixtureStatusGroup(StrEnum):
    SCHEDULED = "scheduled"
    LIVE = "live"
    FINISHED = "finished"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"
    ABANDONED = "abandoned"
    AWARDED = "awarded"  # result decided off the pitch (technical loss / walkover)
    UNKNOWN = "unknown"


class Season(BaseModel):
    model_config = ConfigDict(frozen=True)

    year: int = Field(description="Season start year (e.g. 2025 for 2025/26).")
    start: date | None = None
    end: date | None = None
    current: bool = False
    coverage: dict[str, bool] = Field(
        default_factory=dict,
        description="Data availability flags reported by the provider (flattened).",
    )


class Competition(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str = Field(description="Internal provider-independent key (e.g. premier_league).")
    provider: str
    provider_id: int
    name: str
    country: str
    type: str
    logo_url: str | None = None
    seasons: tuple[Season, ...] = ()

    @property
    def current_season(self) -> Season | None:
        current = [s for s in self.seasons if s.current]
        return max(current, key=lambda s: s.year) if current else None


class TeamRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider_id: int
    name: str
    logo_url: str | None = None


class Score(BaseModel):
    model_config = ConfigDict(frozen=True)

    home: int | None = None
    away: int | None = None


class Fixture(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    provider_id: int
    competition_key: str | None = Field(
        default=None, description="Internal key when the league is one we cover."
    )
    league_provider_id: int
    league_name: str
    season: int
    round: str | None = None
    kickoff_utc: datetime = Field(description="Timezone-aware kick-off time in UTC.")
    provider_timezone: str | None = None
    status_short: str
    status_long: str | None = None
    status_group: FixtureStatusGroup
    elapsed: int | None = None
    venue_id: int | None = None
    venue_name: str | None = None
    venue_city: str | None = None
    referee: str | None = None
    home_team: TeamRef
    away_team: TeamRef
    goals: Score = Score()
    score_halftime: Score = Score()
    score_fulltime: Score = Score()
    score_extratime: Score = Score()
    score_penalty: Score = Score()

    @field_validator("kickoff_utc")
    @classmethod
    def _must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("kickoff_utc must be timezone-aware")
        return value
