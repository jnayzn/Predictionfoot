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
    kickoff_time_known: bool = Field(
        default=True,
        description="False when the source only gives the date (time component is then 00:00 "
        "local and must not be used as a real kick-off time).",
    )
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


# ---------------------------------------------------------------- reference data
class Team(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    provider_id: int
    name: str
    code: str | None = None
    country: str | None = None
    founded: int | None = None
    national: bool | None = None
    logo_url: str | None = None
    venue_name: str | None = None
    venue_city: str | None = None
    venue_capacity: int | None = None


class PlayerRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider_id: int
    name: str


# ------------------------------------------------------------------- standings
class StandingRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    league_provider_id: int
    season: int
    team: TeamRef
    rank: int
    points: int
    goals_diff: int | None = None
    group_name: str | None = None
    form: str | None = None
    status: str | None = None
    description: str | None = None
    played: int | None = None
    win: int | None = None
    draw: int | None = None
    lose: int | None = None
    goals_for: int | None = None
    goals_against: int | None = None
    home: dict[str, int | None] = Field(default_factory=dict)
    away: dict[str, int | None] = Field(default_factory=dict)
    source_updated_at: datetime | None = None


# ---------------------------------------------------------- match-level details
class TeamFixtureStatistics(BaseModel):
    """Per-team statistics of one fixture. ``values`` uses normalized snake_case keys."""

    model_config = ConfigDict(frozen=True)

    provider: str
    fixture_provider_id: int
    team: TeamRef
    values: dict[str, float | None] = Field(default_factory=dict)


class PlayerFixtureStatistics(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    fixture_provider_id: int
    team: TeamRef
    player: PlayerRef
    minutes: int | None = None
    rating: float | None = None
    position: str | None = None
    substitute: bool | None = None
    captain: bool | None = None
    goals: int | None = None
    assists: int | None = None
    shots_total: int | None = None
    shots_on: int | None = None
    passes_total: int | None = None
    passes_key: int | None = None
    pass_accuracy: float | None = None
    tackles: int | None = None
    yellow_cards: int | None = None
    red_cards: int | None = None


class LineupPlayer(BaseModel):
    model_config = ConfigDict(frozen=True)

    player: PlayerRef
    number: int | None = None
    position: str | None = None
    grid: str | None = None
    is_starter: bool


class Lineup(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    fixture_provider_id: int
    team: TeamRef
    formation: str | None = None
    coach_name: str | None = None
    players: tuple[LineupPlayer, ...] = ()


class FixtureDetails(BaseModel):
    """A fixture with everything the provider returned about it."""

    model_config = ConfigDict(frozen=True)

    fixture: Fixture
    statistics: tuple[TeamFixtureStatistics, ...] = ()
    lineups: tuple[Lineup, ...] = ()
    players: tuple[PlayerFixtureStatistics, ...] = ()


class Injury(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    player: PlayerRef
    team: TeamRef
    league_provider_id: int
    season: int
    fixture_provider_id: int | None = None
    fixture_date: datetime | None = None
    type: str | None = None
    reason: str | None = None


class OddsType(StrEnum):
    PRE_MATCH = "pre_match"  # snapshot taken before kick-off (collected_at known)
    OPENING = "opening"
    CLOSING = "closing"
    MARKET_AVERAGE = "market_average"
    MARKET_MAX = "market_max"


class OddsQuote(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    fixture_provider_id: int
    bookmaker: str
    market: str = Field(description="Normalized market, e.g. 1x2, over_under_2.5, btts.")
    selection: str = Field(description="Normalized selection, e.g. home/draw/away, over/under.")
    odd: float = Field(gt=1.0)
    odds_type: OddsType
    source_updated_at: datetime | None = None
