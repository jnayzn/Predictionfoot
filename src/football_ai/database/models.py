"""Relational schema (SQLAlchemy 2.0, PostgreSQL first).

Conventions:
* every table has a surrogate ``id`` primary key;
* provider-sourced entities carry ``provider`` + ``provider_id`` with a unique
  constraint, so the source identifier is never lost;
* all timestamps are timezone-aware UTC;
* ``predictions`` rows are immutable (enforced by a PostgreSQL trigger created
  in the initial migration) so past performance can never be rewritten.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Numeric,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

JsonType = JSON().with_variant(JSONB(), "postgresql")


class UTCDateTime(TypeDecorator[datetime]):
    """Timezone-aware datetime stored in UTC (also on SQLite, which drops tzinfo)."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetime are not allowed; use timezone-aware UTC values")
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime, default=utcnow, onupdate=utcnow, nullable=False
    )


# ============================================================ reference data
class League(TimestampMixin, Base):
    __tablename__ = "leagues"
    __table_args__ = (UniqueConstraint("provider", "provider_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    key: Mapped[str | None] = mapped_column(String(64), index=True)
    provider: Mapped[str] = mapped_column(String(32))
    provider_id: Mapped[int] = mapped_column(BigInteger)
    name: Mapped[str] = mapped_column(String(128))
    country: Mapped[str | None] = mapped_column(String(64))
    type: Mapped[str | None] = mapped_column(String(32))
    logo_url: Mapped[str | None] = mapped_column(String(512))

    seasons: Mapped[list[Season]] = relationship(back_populates="league")


class Season(TimestampMixin, Base):
    __tablename__ = "seasons"
    __table_args__ = (UniqueConstraint("league_id", "year"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id", ondelete="CASCADE"))
    year: Mapped[int] = mapped_column(Integer)
    start_date: Mapped[date | None] = mapped_column(Date)
    end_date: Mapped[date | None] = mapped_column(Date)
    is_current: Mapped[bool] = mapped_column(Boolean, default=False)
    coverage: Mapped[dict[str, Any]] = mapped_column(JsonType, default=dict)

    league: Mapped[League] = relationship(back_populates="seasons")


class Team(TimestampMixin, Base):
    __tablename__ = "teams"
    __table_args__ = (UniqueConstraint("provider", "provider_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(32))
    provider_id: Mapped[int] = mapped_column(BigInteger)
    name: Mapped[str] = mapped_column(String(128), index=True)
    code: Mapped[str | None] = mapped_column(String(16))
    country: Mapped[str | None] = mapped_column(String(64))
    founded: Mapped[int | None] = mapped_column(Integer)
    national: Mapped[bool | None] = mapped_column(Boolean)
    logo_url: Mapped[str | None] = mapped_column(String(512))
    venue_name: Mapped[str | None] = mapped_column(String(128))
    venue_city: Mapped[str | None] = mapped_column(String(128))
    venue_capacity: Mapped[int | None] = mapped_column(Integer)


class Player(TimestampMixin, Base):
    __tablename__ = "players"
    __table_args__ = (UniqueConstraint("provider", "provider_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(32))
    provider_id: Mapped[int] = mapped_column(BigInteger)
    name: Mapped[str] = mapped_column(String(128))
    birth_date: Mapped[date | None] = mapped_column(Date)
    nationality: Mapped[str | None] = mapped_column(String(64))
    position: Mapped[str | None] = mapped_column(String(32))


# ================================================================== fixtures
class Fixture(TimestampMixin, Base):
    __tablename__ = "fixtures"
    __table_args__ = (
        UniqueConstraint("provider", "provider_id"),
        CheckConstraint("home_team_id <> away_team_id", name="distinct_teams"),
        Index("ix_fixtures_league_season", "league_id", "season_year"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(32))
    provider_id: Mapped[int] = mapped_column(BigInteger)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id"))
    season_year: Mapped[int] = mapped_column(Integer)
    round: Mapped[str | None] = mapped_column(String(128))
    kickoff_utc: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    kickoff_time_known: Mapped[bool] = mapped_column(Boolean, default=True)
    provider_timezone: Mapped[str | None] = mapped_column(String(64))
    status_short: Mapped[str] = mapped_column(String(8))
    status_long: Mapped[str | None] = mapped_column(String(64))
    status_group: Mapped[str] = mapped_column(String(16), index=True)
    elapsed: Mapped[int | None] = mapped_column(Integer)
    venue_id: Mapped[int | None] = mapped_column(BigInteger)
    venue_name: Mapped[str | None] = mapped_column(String(128))
    venue_city: Mapped[str | None] = mapped_column(String(128))
    referee: Mapped[str | None] = mapped_column(String(128))
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    home_goals: Mapped[int | None] = mapped_column(Integer)
    away_goals: Mapped[int | None] = mapped_column(Integer)
    ht_home_goals: Mapped[int | None] = mapped_column(Integer)
    ht_away_goals: Mapped[int | None] = mapped_column(Integer)
    ft_home_goals: Mapped[int | None] = mapped_column(Integer)
    ft_away_goals: Mapped[int | None] = mapped_column(Integer)
    et_home_goals: Mapped[int | None] = mapped_column(Integer)
    et_away_goals: Mapped[int | None] = mapped_column(Integer)
    pen_home_goals: Mapped[int | None] = mapped_column(Integer)
    pen_away_goals: Mapped[int | None] = mapped_column(Integer)
    details_fetched_at: Mapped[datetime | None] = mapped_column(UTCDateTime)

    league: Mapped[League] = relationship()
    home_team: Mapped[Team] = relationship(foreign_keys=[home_team_id])
    away_team: Mapped[Team] = relationship(foreign_keys=[away_team_id])


class ActualResult(Base):
    """Final result of a finished fixture; ``revision`` > 1 means it was corrected."""

    __tablename__ = "actual_results"
    __table_args__ = (CheckConstraint("outcome IN ('H','D','A')", name="outcome"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    fixture_id: Mapped[int] = mapped_column(
        ForeignKey("fixtures.id", ondelete="CASCADE"), unique=True
    )
    home_goals: Mapped[int] = mapped_column(Integer)
    away_goals: Mapped[int] = mapped_column(Integer)
    ht_home_goals: Mapped[int | None] = mapped_column(Integer)
    ht_away_goals: Mapped[int | None] = mapped_column(Integer)
    outcome: Mapped[str] = mapped_column(String(1))
    status_short: Mapped[str] = mapped_column(String(8))
    revision: Mapped[int] = mapped_column(Integer, default=1)
    first_recorded_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)


class DataChangeLog(Base):
    """Audit trail of corrections detected on already-loaded data."""

    __tablename__ = "data_change_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    entity: Mapped[str] = mapped_column(String(32), index=True)
    entity_id: Mapped[int] = mapped_column(BigInteger)
    provider: Mapped[str] = mapped_column(String(32))
    field: Mapped[str] = mapped_column(String(64))
    old_value: Mapped[str | None] = mapped_column(Text)
    new_value: Mapped[str | None] = mapped_column(Text)
    detected_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)


# ======================================================== match-level details
class TeamStatistics(Base):
    __tablename__ = "team_statistics"
    __table_args__ = (UniqueConstraint("fixture_id", "team_id", "provider"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    fixture_id: Mapped[int] = mapped_column(ForeignKey("fixtures.id", ondelete="CASCADE"))
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    provider: Mapped[str] = mapped_column(String(32))
    shots_on_goal: Mapped[float | None] = mapped_column(Float)
    shots_off_goal: Mapped[float | None] = mapped_column(Float)
    shots_total: Mapped[float | None] = mapped_column(Float)
    shots_blocked: Mapped[float | None] = mapped_column(Float)
    shots_inside_box: Mapped[float | None] = mapped_column(Float)
    shots_outside_box: Mapped[float | None] = mapped_column(Float)
    corners: Mapped[float | None] = mapped_column(Float)
    fouls: Mapped[float | None] = mapped_column(Float)
    offsides: Mapped[float | None] = mapped_column(Float)
    possession_pct: Mapped[float | None] = mapped_column(Float)
    yellow_cards: Mapped[float | None] = mapped_column(Float)
    red_cards: Mapped[float | None] = mapped_column(Float)
    goalkeeper_saves: Mapped[float | None] = mapped_column(Float)
    passes_total: Mapped[float | None] = mapped_column(Float)
    passes_accurate: Mapped[float | None] = mapped_column(Float)
    passes_pct: Mapped[float | None] = mapped_column(Float)
    expected_goals: Mapped[float | None] = mapped_column(Float)
    values: Mapped[dict[str, Any]] = mapped_column(JsonType, default=dict)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)


TEAM_STAT_COLUMNS: tuple[str, ...] = (
    "shots_on_goal",
    "shots_off_goal",
    "shots_total",
    "shots_blocked",
    "shots_inside_box",
    "shots_outside_box",
    "corners",
    "fouls",
    "offsides",
    "possession_pct",
    "yellow_cards",
    "red_cards",
    "goalkeeper_saves",
    "passes_total",
    "passes_accurate",
    "passes_pct",
    "expected_goals",
)


class PlayerStatistics(Base):
    __tablename__ = "player_statistics"
    __table_args__ = (UniqueConstraint("fixture_id", "player_id", "provider"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    fixture_id: Mapped[int] = mapped_column(ForeignKey("fixtures.id", ondelete="CASCADE"))
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    provider: Mapped[str] = mapped_column(String(32))
    minutes: Mapped[int | None] = mapped_column(Integer)
    rating: Mapped[float | None] = mapped_column(Float)
    position: Mapped[str | None] = mapped_column(String(8))
    substitute: Mapped[bool | None] = mapped_column(Boolean)
    captain: Mapped[bool | None] = mapped_column(Boolean)
    goals: Mapped[int | None] = mapped_column(Integer)
    assists: Mapped[int | None] = mapped_column(Integer)
    shots_total: Mapped[int | None] = mapped_column(Integer)
    shots_on: Mapped[int | None] = mapped_column(Integer)
    passes_total: Mapped[int | None] = mapped_column(Integer)
    passes_key: Mapped[int | None] = mapped_column(Integer)
    pass_accuracy: Mapped[float | None] = mapped_column(Float)
    tackles: Mapped[int | None] = mapped_column(Integer)
    yellow_cards: Mapped[int | None] = mapped_column(Integer)
    red_cards: Mapped[int | None] = mapped_column(Integer)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)


PLAYER_STAT_COLUMNS: tuple[str, ...] = (
    "minutes",
    "rating",
    "position",
    "substitute",
    "captain",
    "goals",
    "assists",
    "shots_total",
    "shots_on",
    "passes_total",
    "passes_key",
    "pass_accuracy",
    "tackles",
    "yellow_cards",
    "red_cards",
)


class Lineup(Base):
    __tablename__ = "lineups"
    __table_args__ = (UniqueConstraint("fixture_id", "team_id", "provider"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    fixture_id: Mapped[int] = mapped_column(ForeignKey("fixtures.id", ondelete="CASCADE"))
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    provider: Mapped[str] = mapped_column(String(32))
    formation: Mapped[str | None] = mapped_column(String(16))
    coach_name: Mapped[str | None] = mapped_column(String(128))
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)

    players: Mapped[list[LineupPlayer]] = relationship(
        back_populates="lineup", cascade="all, delete-orphan"
    )


class LineupPlayer(Base):
    __tablename__ = "lineup_players"
    __table_args__ = (UniqueConstraint("lineup_id", "player_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    lineup_id: Mapped[int] = mapped_column(ForeignKey("lineups.id", ondelete="CASCADE"))
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"))
    number: Mapped[int | None] = mapped_column(Integer)
    position: Mapped[str | None] = mapped_column(String(8))
    grid: Mapped[str | None] = mapped_column(String(8))
    is_starter: Mapped[bool] = mapped_column(Boolean)

    lineup: Mapped[Lineup] = relationship(back_populates="players")


class Injury(Base):
    __tablename__ = "injuries"
    __table_args__ = (
        UniqueConstraint(
            "provider",
            "player_id",
            "league_id",
            "season_year",
            "fixture_provider_id",
            name="uq_injuries_identity",
            postgresql_nulls_not_distinct=True,
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(32))
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id"))
    season_year: Mapped[int] = mapped_column(Integer)
    fixture_provider_id: Mapped[int | None] = mapped_column(BigInteger)
    fixture_id: Mapped[int | None] = mapped_column(ForeignKey("fixtures.id", ondelete="SET NULL"))
    fixture_date: Mapped[datetime | None] = mapped_column(UTCDateTime)
    type: Mapped[str | None] = mapped_column(String(64))
    reason: Mapped[str | None] = mapped_column(String(256))
    collected_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class Standing(Base):
    """Standings snapshot. ``snapshot_date`` = when it was collected (leakage control)."""

    __tablename__ = "standings"
    __table_args__ = (
        UniqueConstraint(
            "provider",
            "league_id",
            "season_year",
            "team_id",
            "group_name",
            "snapshot_date",
            name="uq_standings_identity",
            postgresql_nulls_not_distinct=True,
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(32))
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id"))
    season_year: Mapped[int] = mapped_column(Integer)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    snapshot_date: Mapped[date] = mapped_column(Date)
    group_name: Mapped[str | None] = mapped_column(String(128))
    rank: Mapped[int] = mapped_column(Integer)
    points: Mapped[int] = mapped_column(Integer)
    goals_diff: Mapped[int | None] = mapped_column(Integer)
    form: Mapped[str | None] = mapped_column(String(16))
    status: Mapped[str | None] = mapped_column(String(16))
    description: Mapped[str | None] = mapped_column(String(256))
    played: Mapped[int | None] = mapped_column(Integer)
    win: Mapped[int | None] = mapped_column(Integer)
    draw: Mapped[int | None] = mapped_column(Integer)
    lose: Mapped[int | None] = mapped_column(Integer)
    goals_for: Mapped[int | None] = mapped_column(Integer)
    goals_against: Mapped[int | None] = mapped_column(Integer)
    home: Mapped[dict[str, Any]] = mapped_column(JsonType, default=dict)
    away: Mapped[dict[str, Any]] = mapped_column(JsonType, default=dict)
    source_updated_at: Mapped[datetime | None] = mapped_column(UTCDateTime)


class Odds(Base):
    __tablename__ = "odds"
    __table_args__ = (
        UniqueConstraint(
            "fixture_id",
            "provider",
            "bookmaker",
            "market",
            "selection",
            "odds_type",
            "source_updated_at",
            name="uq_odds_identity",
            postgresql_nulls_not_distinct=True,
        ),
        CheckConstraint("odd > 1", name="odd_gt_one"),
        Index("ix_odds_fixture_market", "fixture_id", "market"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    fixture_id: Mapped[int] = mapped_column(ForeignKey("fixtures.id", ondelete="CASCADE"))
    provider: Mapped[str] = mapped_column(String(32))
    bookmaker: Mapped[str] = mapped_column(String(64))
    market: Mapped[str] = mapped_column(String(32))
    selection: Mapped[str] = mapped_column(String(16))
    odd: Mapped[float] = mapped_column(Numeric(10, 3, asdecimal=False))
    odds_type: Mapped[str] = mapped_column(String(16))
    source_updated_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    collected_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


# ======================================================= modelling & evaluation
class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("name", "version"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    version: Mapped[str] = mapped_column(String(32))
    algorithm: Mapped[str] = mapped_column(String(64))
    params: Mapped[dict[str, Any]] = mapped_column(JsonType, default=dict)
    metrics: Mapped[dict[str, Any]] = mapped_column(JsonType, default=dict)
    training_data_start: Mapped[datetime | None] = mapped_column(UTCDateTime)
    training_data_end: Mapped[datetime | None] = mapped_column(UTCDateTime)
    trained_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    artifact_uri: Mapped[str | None] = mapped_column(String(512))
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="candidate")
    notes: Mapped[str | None] = mapped_column(Text)


class Prediction(Base):
    """Pre-match prediction. Immutable once written (DB trigger on PostgreSQL)."""

    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint(
            "fixture_id", "model_version_id", "created_at", name="uq_predictions_identity"
        ),
        CheckConstraint("created_at < kickoff_utc", name="before_kickoff"),
        CheckConstraint(
            "p_home >= 0 AND p_draw >= 0 AND p_away >= 0 "
            "AND abs(p_home + p_draw + p_away - 1) < 0.001",
            name="probabilities",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    fixture_id: Mapped[int] = mapped_column(ForeignKey("fixtures.id"), index=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    kickoff_utc: Mapped[datetime] = mapped_column(UTCDateTime)
    p_home: Mapped[float] = mapped_column(Float)
    p_draw: Mapped[float] = mapped_column(Float)
    p_away: Mapped[float] = mapped_column(Float)
    expected_home_goals: Mapped[float | None] = mapped_column(Float)
    expected_away_goals: Mapped[float | None] = mapped_column(Float)
    markets: Mapped[dict[str, Any]] = mapped_column(JsonType, default=dict)
    confidence: Mapped[float | None] = mapped_column(Float)
    data_quality: Mapped[dict[str, Any]] = mapped_column(JsonType, default=dict)
    feature_hash: Mapped[str | None] = mapped_column(String(64))


class PredictionEvaluation(Base):
    __tablename__ = "prediction_evaluations"

    id: Mapped[int] = mapped_column(primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), unique=True)
    fixture_id: Mapped[int] = mapped_column(ForeignKey("fixtures.id"), index=True)
    actual_outcome: Mapped[str] = mapped_column(String(1))
    actual_home_goals: Mapped[int] = mapped_column(Integer)
    actual_away_goals: Mapped[int] = mapped_column(Integer)
    log_loss: Mapped[float] = mapped_column(Float)
    brier_score: Mapped[float] = mapped_column(Float)
    correct: Mapped[bool] = mapped_column(Boolean)
    goals_abs_error: Mapped[float | None] = mapped_column(Float)
    evaluated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


# ================================================================ operations
class IngestionJob(Base):
    """Progress of resumable collection jobs (one row per provider/job/scope)."""

    __tablename__ = "ingestion_jobs"
    __table_args__ = (UniqueConstraint("provider", "job_type", "scope"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(32))
    job_type: Mapped[str] = mapped_column(String(32))
    scope: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    requests_used: Mapped[int] = mapped_column(Integer, default=0)
    items_loaded: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime, default=utcnow, onupdate=utcnow, server_default=func.now()
    )


# ============================================== canonical (cross-source) layer
class CanonicalTeam(TimestampMixin, Base):
    """One real-world club, whatever the number of sources describing it."""

    __tablename__ = "canonical_teams"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(128), index=True)
    country: Mapped[str | None] = mapped_column(String(64))


class TeamLink(Base):
    """Links a provider team row to its canonical team."""

    __tablename__ = "team_links"

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id", ondelete="CASCADE"), unique=True)
    canonical_team_id: Mapped[int] = mapped_column(
        ForeignKey("canonical_teams.id", ondelete="CASCADE"), index=True
    )
    method: Mapped[str] = mapped_column(String(32))  # primary | cooccurrence | alias | standalone
    confidence: Mapped[float] = mapped_column(Float)
    evidence: Mapped[dict[str, Any]] = mapped_column(JsonType, default=dict)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)


class Match(TimestampMixin, Base):
    """Canonical match built from one or several provider fixtures."""

    __tablename__ = "matches"
    __table_args__ = (
        CheckConstraint("home_team_id <> away_team_id", name="distinct_teams"),
        Index("ix_matches_competition_season", "competition_key", "season_year"),
        Index("ix_matches_home_kickoff", "home_team_id", "kickoff_utc"),
        Index("ix_matches_away_kickoff", "away_team_id", "kickoff_utc"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    competition_key: Mapped[str] = mapped_column(String(64))
    season_year: Mapped[int] = mapped_column(Integer)
    round: Mapped[str | None] = mapped_column(String(128))
    kickoff_utc: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    kickoff_time_known: Mapped[bool] = mapped_column(Boolean, default=True)
    home_team_id: Mapped[int] = mapped_column(ForeignKey("canonical_teams.id"))
    away_team_id: Mapped[int] = mapped_column(ForeignKey("canonical_teams.id"))
    status_group: Mapped[str] = mapped_column(String(16), index=True)
    home_goals: Mapped[int | None] = mapped_column(Integer)  # 90-minute score
    away_goals: Mapped[int | None] = mapped_column(Integer)
    ht_home_goals: Mapped[int | None] = mapped_column(Integer)
    ht_away_goals: Mapped[int | None] = mapped_column(Integer)
    primary_fixture_id: Mapped[int] = mapped_column(ForeignKey("fixtures.id"))
    sources: Mapped[list[str]] = mapped_column(JsonType, default=list)
    source_conflict: Mapped[bool] = mapped_column(Boolean, default=False)
    valid_for_training: Mapped[bool] = mapped_column(Boolean, default=True)
    exclusion_reason: Mapped[str | None] = mapped_column(String(128))


class MatchSource(Base):
    __tablename__ = "match_sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id", ondelete="CASCADE"), index=True)
    fixture_id: Mapped[int] = mapped_column(
        ForeignKey("fixtures.id", ondelete="CASCADE"), unique=True
    )
    provider: Mapped[str] = mapped_column(String(32))


class DataQualityReport(Base):
    __tablename__ = "data_quality_reports"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)
    scope: Mapped[str] = mapped_column(String(128))
    errors: Mapped[int] = mapped_column(Integer)
    warnings: Mapped[int] = mapped_column(Integer)
    summary: Mapped[dict[str, Any]] = mapped_column(JsonType, default=dict)


# ================================================================ telegram bot
class TelegramSubscriber(Base):
    """Chats that opted in for automatic reports (explicit /subscribe)."""

    __tablename__ = "telegram_subscribers"

    id: Mapped[int] = mapped_column(primary_key=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, unique=True)
    username: Mapped[str | None] = mapped_column(String(64))
    competitions: Mapped[list[str]] = mapped_column(JsonType, default=list)  # empty = all
    daily_report: Mapped[bool] = mapped_column(Boolean, default=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)


# ================================================================== automation
class ScheduledJobRun(Base):
    """One execution of a scheduled job (monitoring / audit)."""

    __tablename__ = "scheduled_job_runs"
    __table_args__ = (Index("ix_scheduled_job_runs_job_started", "job", "started_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    job: Mapped[str] = mapped_column(String(64))
    started_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    status: Mapped[str] = mapped_column(String(16))  # success | skipped | failed
    requests_used: Mapped[int] = mapped_column(Integer, default=0)
    details: Mapped[dict[str, Any]] = mapped_column(JsonType, default=dict)
    error: Mapped[str | None] = mapped_column(Text)


# ============================================================== commercial API
class ApiClient(Base):
    """A customer of the HTTP API. Only a SHA-256 hash of the key is stored."""

    __tablename__ = "api_clients"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    key_prefix: Mapped[str] = mapped_column(String(12), index=True)  # for identification only
    key_hash: Mapped[str] = mapped_column(String(64), unique=True)
    plan: Mapped[str] = mapped_column(String(32))
    daily_quota: Mapped[int] = mapped_column(Integer)
    rate_limit_per_minute: Mapped[int] = mapped_column(Integer)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    last_used_at: Mapped[datetime | None] = mapped_column(UTCDateTime)


class ApiUsage(Base):
    """Requests per client and UTC day (metering / billing)."""

    __tablename__ = "api_usage"
    __table_args__ = (UniqueConstraint("client_id", "day"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    client_id: Mapped[int] = mapped_column(ForeignKey("api_clients.id", ondelete="CASCADE"))
    day: Mapped[date] = mapped_column(Date)
    requests: Mapped[int] = mapped_column(Integer, default=0)
