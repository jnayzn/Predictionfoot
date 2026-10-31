"""Base frames feeding the feature pipeline (loaded from the canonical layer).

The feature code itself only manipulates these DataFrames, so it can be unit
tested without a database.

Contract
--------
``matches``    : match_id, competition_key, season_year, kickoff_utc (UTC), home_team_id,
                 away_team_id, status_group, home_goals, away_goals, valid_for_training
``team_stats`` : match_id, team_id, shots_total, shots_on_goal, expected_goals, corners
``odds``       : match_id, implied_home, implied_draw, implied_away, n_bookmakers
                 (pre-match odds only, overround removed - closing odds are NEVER used)
``injuries``   : match_id, team_id, missing_players (only where the source covers injuries)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from football_ai.database import models as db

MATCH_COLUMNS = [
    "match_id", "competition_key", "season_year", "kickoff_utc", "home_team_id", "away_team_id",
    "status_group", "home_goals", "away_goals", "valid_for_training",
]  # fmt: skip
STAT_COLUMNS = ["match_id", "team_id", "shots_total", "shots_on_goal", "expected_goals", "corners"]
ODDS_COLUMNS = ["match_id", "implied_home", "implied_draw", "implied_away", "n_bookmakers"]
INJURY_COLUMNS = ["match_id", "team_id", "missing_players"]
# odds usable as pre-match information
PRE_MATCH_ODDS_TYPES = ("pre_match", "market_average", "opening")
SOURCE_PRIORITY = {"api_football": 0, "football_data_uk": 1}


@dataclass(slots=True)
class BaseFrames:
    matches: pd.DataFrame
    team_stats: pd.DataFrame
    odds: pd.DataFrame
    injuries: pd.DataFrame

    @classmethod
    def empty(cls) -> BaseFrames:
        return cls(
            pd.DataFrame(columns=MATCH_COLUMNS),
            pd.DataFrame(columns=STAT_COLUMNS),
            pd.DataFrame(columns=ODDS_COLUMNS),
            pd.DataFrame(columns=INJURY_COLUMNS),
        )

    def normalized(self) -> BaseFrames:
        m = self.matches.copy()
        m["kickoff_utc"] = pd.to_datetime(m["kickoff_utc"], utc=True).astype("datetime64[ns, UTC]")
        for col in ("home_goals", "away_goals"):
            m[col] = pd.to_numeric(m[col], errors="coerce")
        m["valid_for_training"] = m["valid_for_training"].astype(bool)
        return BaseFrames(m, self.team_stats.copy(), self.odds.copy(), self.injuries.copy())


def _frame(session: Session, stmt: Any) -> pd.DataFrame:
    result = session.execute(stmt)
    return pd.DataFrame(result.fetchall(), columns=list(result.keys()))


def load_base_frames(session: Session) -> BaseFrames:
    """Load every canonical match (all competitions: team histories span competitions)."""
    m = db.Match
    matches = _frame(
        session,
        select(
            m.id.label("match_id"),
            m.competition_key,
            m.season_year,
            m.kickoff_utc,
            m.home_team_id,
            m.away_team_id,
            m.status_group,
            m.home_goals,
            m.away_goals,
            m.valid_for_training,
        ),
    )

    ts = db.TeamStatistics
    raw_stats = _frame(
        session,
        select(
            db.MatchSource.match_id,
            db.TeamLink.canonical_team_id.label("team_id"),
            ts.provider,
            ts.shots_total,
            ts.shots_on_goal,
            ts.expected_goals,
            ts.corners,
        )
        .join(db.MatchSource, db.MatchSource.fixture_id == ts.fixture_id)
        .join(db.TeamLink, db.TeamLink.team_id == ts.team_id),
    )
    if raw_stats.empty:
        team_stats = pd.DataFrame(columns=STAT_COLUMNS)
    else:
        # one row per (match, team): first non-null value by source priority
        raw_stats["prio"] = raw_stats["provider"].map(SOURCE_PRIORITY).fillna(9)
        team_stats = (
            raw_stats.sort_values("prio")
            .groupby(["match_id", "team_id"], as_index=False)[STAT_COLUMNS[2:]]
            .first()
        )

    o = db.Odds
    raw_odds = _frame(
        session,
        select(db.MatchSource.match_id, o.provider, o.bookmaker, o.selection, o.odd)
        .join(db.MatchSource, db.MatchSource.fixture_id == o.fixture_id)
        .where(o.market == "1x2", o.odds_type.in_(PRE_MATCH_ODDS_TYPES)),
    )
    odds = implied_probabilities(raw_odds)

    inj = db.Injury
    injury_rows = _frame(
        session,
        select(db.MatchSource.match_id, db.TeamLink.canonical_team_id.label("team_id"), inj.id)
        .join(db.MatchSource, db.MatchSource.fixture_id == inj.fixture_id)
        .join(db.TeamLink, db.TeamLink.team_id == inj.team_id),
    )
    coverage = _frame(  # matches whose source covers injuries (0 is then a real zero)
        session,
        select(db.MatchSource.match_id, db.Season.coverage)
        .join(db.Fixture, db.Fixture.id == db.MatchSource.fixture_id)
        .join(
            db.Season,
            (db.Season.league_id == db.Fixture.league_id)
            & (db.Season.year == db.Fixture.season_year),
        )
        .where(db.MatchSource.provider == "api_football"),
    )
    covered = {
        row.match_id
        for row in coverage.itertuples()
        if isinstance(row.coverage, dict) and row.coverage.get("injuries")
    }
    injuries = injury_counts(matches, injury_rows, covered)
    return BaseFrames(matches, team_stats, odds, injuries).normalized()


def implied_probabilities(raw_odds: pd.DataFrame) -> pd.DataFrame:
    """Average, overround-free implied 1X2 probabilities per match."""
    if raw_odds.empty:
        return pd.DataFrame(columns=ODDS_COLUMNS)
    wide = raw_odds.pivot_table(
        index=["match_id", "provider", "bookmaker"],
        columns="selection",
        values="odd",
        aggfunc="last",
    )
    wide = wide.dropna(subset=[c for c in ("home", "draw", "away") if c in wide.columns])
    if not {"home", "draw", "away"} <= set(wide.columns) or wide.empty:
        return pd.DataFrame(columns=ODDS_COLUMNS)
    inv = 1.0 / wide[["home", "draw", "away"]].astype(float)
    probs = inv.div(inv.sum(axis=1), axis=0)
    grouped = probs.groupby(level="match_id")
    out = grouped.mean().rename(
        columns={"home": "implied_home", "draw": "implied_draw", "away": "implied_away"}
    )
    out["n_bookmakers"] = grouped.size()
    return out.reset_index()[ODDS_COLUMNS]


def injury_counts(
    matches: pd.DataFrame, injury_rows: pd.DataFrame, covered_match_ids: set[Any]
) -> pd.DataFrame:
    if matches.empty or not covered_match_ids:
        return pd.DataFrame(columns=INJURY_COLUMNS)
    m = matches[matches["match_id"].isin(covered_match_ids)]
    base = pd.concat(
        [
            m[["match_id", "home_team_id"]].rename(columns={"home_team_id": "team_id"}),
            m[["match_id", "away_team_id"]].rename(columns={"away_team_id": "team_id"}),
        ]
    )
    counts = (
        injury_rows.groupby(["match_id", "team_id"]).size().rename("missing_players").reset_index()
        if not injury_rows.empty
        else pd.DataFrame(columns=INJURY_COLUMNS)
    )
    out = base.merge(counts, on=["match_id", "team_id"], how="left")
    out["missing_players"] = out["missing_players"].fillna(0).astype(float)
    return out[INJURY_COLUMNS]
