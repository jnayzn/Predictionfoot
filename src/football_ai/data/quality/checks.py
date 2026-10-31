"""Dataset-level data-quality checks on the canonical layer.

Record-level validation happens at ingestion (``data/validators``). These
checks look at the dataset as a whole: duplicates, scheduling anomalies,
incomplete seasons, stale results, cross-source conflicts, statistics and
odds consistency, and data coverage per competition-season.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from football_ai.database import models as db

ERROR = "error"
WARNING = "warning"
MAX_PLAUSIBLE_GOALS = 12
ODDS_OVERROUND_RANGE = (0.98, 1.35)
# cups / group stages do not follow the double round-robin rule
NON_ROUND_ROBIN = frozenset({"champions_league"})


@dataclass(slots=True)
class QualityFinding:
    check: str
    severity: str
    count: int
    message: str
    examples: list[str] = field(default_factory=list)


@dataclass(slots=True)
class QualityReport:
    scope: str
    created_at: datetime
    findings: list[QualityFinding] = field(default_factory=list)
    coverage: list[dict[str, Any]] = field(default_factory=list)

    @property
    def errors(self) -> int:
        return sum(f.count for f in self.findings if f.severity == ERROR)

    @property
    def warnings(self) -> int:
        return sum(f.count for f in self.findings if f.severity == WARNING)

    def add(
        self, check: str, severity: str, frame: pd.DataFrame, message: str, cols: list[str]
    ) -> None:
        if frame.empty:
            return
        examples = [", ".join(f"{c}={row[c]}" for c in cols) for _, row in frame.head(5).iterrows()]
        self.findings.append(QualityFinding(check, severity, len(frame), message, examples))

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "created_at": self.created_at.isoformat(),
            "errors": self.errors,
            "warnings": self.warnings,
            "findings": [asdict(f) for f in self.findings],
            "coverage": self.coverage,
        }


# ------------------------------------------------------------------- loading
def _frame(session: Session, stmt: Any) -> pd.DataFrame:
    result = session.execute(stmt)
    return pd.DataFrame(result.fetchall(), columns=list(result.keys()))


def load_frames(session: Session, competition_key: str | None = None) -> dict[str, pd.DataFrame]:
    m = db.Match
    match_stmt = select(
        m.id, m.competition_key, m.season_year, m.kickoff_utc, m.home_team_id, m.away_team_id,
        m.status_group, m.home_goals, m.away_goals, m.source_conflict, m.valid_for_training,
        m.exclusion_reason, m.sources,
    )  # fmt: skip
    if competition_key:
        match_stmt = match_stmt.where(m.competition_key == competition_key)
    matches = _frame(session, match_stmt)

    ts = db.TeamStatistics
    stats = _frame(
        session,
        select(
            db.MatchSource.match_id, ts.fixture_id, ts.team_id, ts.provider, ts.shots_total,
            ts.shots_on_goal, ts.possession_pct, ts.expected_goals,
        ).join(db.MatchSource, db.MatchSource.fixture_id == ts.fixture_id),
    )  # fmt: skip
    lineups = _frame(
        session,
        select(db.MatchSource.match_id).join(
            db.Lineup, db.Lineup.fixture_id == db.MatchSource.fixture_id
        ),
    )
    o = db.Odds
    odds = _frame(
        session,
        select(
            db.MatchSource.match_id, o.fixture_id, o.provider, o.bookmaker, o.market, o.selection,
            o.odds_type, o.odd,
        ).join(db.MatchSource, db.MatchSource.fixture_id == o.fixture_id),
    )  # fmt: skip
    if competition_key and not matches.empty:
        ids = set(matches["id"])
        stats = stats[stats["match_id"].isin(ids)]
        lineups = lineups[lineups["match_id"].isin(ids)]
        odds = odds[odds["match_id"].isin(ids)]
    return {"matches": matches, "stats": stats, "lineups": lineups, "odds": odds}


def _as_int(value: Any) -> int:
    return int(value)


# -------------------------------------------------------------------- checks
def run_checks(
    frames: dict[str, pd.DataFrame], *, now: datetime | None = None, scope: str = "all"
) -> QualityReport:
    now = now or datetime.now(UTC)
    report = QualityReport(scope=scope, created_at=now)
    matches = frames["matches"]
    if matches.empty:
        return report
    matches = matches.copy()
    matches["kickoff_utc"] = pd.to_datetime(matches["kickoff_utc"], utc=True)
    matches["day"] = matches["kickoff_utc"].dt.date

    _check_duplicates(report, matches)
    _check_double_booking(report, matches)
    _check_goals(report, matches)
    _check_stale(report, matches, now)
    _check_conflicts(report, matches)
    _check_season_completeness(report, matches, now)
    _check_stats(report, frames["stats"])
    _check_odds(report, frames["odds"])
    report.coverage = coverage_table(matches, frames)
    return report


def _check_duplicates(report: QualityReport, matches: pd.DataFrame) -> None:
    keys = ["competition_key", "home_team_id", "away_team_id", "day"]
    dup = matches[matches.duplicated(keys, keep=False)].sort_values(keys)
    report.add("duplicate_matches", ERROR, dup, "same teams on the same day twice", ["id", *keys])


def _check_double_booking(report: QualityReport, matches: pd.DataFrame) -> None:
    played = matches[~matches["status_group"].isin(["postponed", "cancelled"])]
    long = pd.concat(
        [
            played[["id", "kickoff_utc"]].assign(team=played["home_team_id"]),
            played[["id", "kickoff_utc"]].assign(team=played["away_team_id"]),
        ]
    ).sort_values(["team", "kickoff_utc"])
    gap = long.groupby("team")["kickoff_utc"].diff()
    clash = long[gap < pd.Timedelta(hours=36)]
    report.add(
        "team_double_booking", WARNING, clash,
        "a team plays twice within 36 hours (possible duplicate or wrong date)",
        ["id", "team", "kickoff_utc"],
    )  # fmt: skip


def _check_goals(report: QualityReport, matches: pd.DataFrame) -> None:
    goals = matches[["home_goals", "away_goals"]].astype("float")
    bad = matches[(goals > MAX_PLAUSIBLE_GOALS).any(axis=1) | (goals < 0).any(axis=1)]
    report.add(
        "implausible_goals", ERROR, bad, "implausible score", ["id", "home_goals", "away_goals"]
    )


def _check_stale(report: QualityReport, matches: pd.DataFrame, now: datetime) -> None:
    stale = matches[
        matches["status_group"].isin(["scheduled", "live"])
        & (matches["kickoff_utc"] < pd.Timestamp(now - timedelta(days=3)))
    ]
    report.add(
        "stale_results",
        WARNING,
        stale,
        "kick-off more than 3 days ago but result not updated",
        ["id", "kickoff_utc", "status_group"],
    )


def _check_conflicts(report: QualityReport, matches: pd.DataFrame) -> None:
    conflicts = matches[matches["source_conflict"].astype(bool)]
    report.add(
        "source_conflicts", WARNING, conflicts,
        "sources disagree on the score (primary source kept, excluded from training)",
        ["id", "home_goals", "away_goals", "sources"],
    )  # fmt: skip


def _check_season_completeness(report: QualityReport, matches: pd.DataFrame, now: datetime) -> None:
    rows = []
    for (comp, season), group in matches.groupby(["competition_key", "season_year"]):
        if comp in NON_ROUND_ROBIN or now.date() <= datetime(_as_int(season) + 1, 7, 31).date():
            continue
        teams = pd.unique(group[["home_team_id", "away_team_id"]].to_numpy().ravel())
        expected = len(teams) * (len(teams) - 1)
        finished = int((group["status_group"] == "finished").sum())
        if finished != expected:
            rows.append({"competition": comp, "season": season, "finished": finished,
                         "expected": expected})  # fmt: skip
    report.add(
        "incomplete_seasons", WARNING, pd.DataFrame(rows),
        "finished season without the expected n*(n-1) finished matches",
        ["competition", "season", "finished", "expected"],
    )  # fmt: skip


def _check_stats(report: QualityReport, stats: pd.DataFrame) -> None:
    if stats.empty:
        return
    shots = stats.dropna(subset=["shots_total", "shots_on_goal"])
    bad = shots[shots["shots_on_goal"] > shots["shots_total"]]
    report.add(
        "shots_inconsistent", ERROR, bad, "shots on goal > total shots",
        ["fixture_id", "team_id", "shots_on_goal", "shots_total"],
    )  # fmt: skip
    poss = stats.dropna(subset=["possession_pct"]).groupby("fixture_id")["possession_pct"]
    total = poss.sum()[poss.count() == 2]
    off = total[(total < 98) | (total > 102)].rename("possession_sum").reset_index()
    report.add(
        "possession_inconsistent", WARNING, off, "possession does not sum to ~100%",
        ["fixture_id", "possession_sum"],
    )  # fmt: skip


def _check_odds(report: QualityReport, odds: pd.DataFrame) -> None:
    if odds.empty:
        return
    book = odds[(odds["market"] == "1x2") & (odds["odds_type"] != "market_max")]
    keys = ["fixture_id", "provider", "bookmaker", "odds_type"]
    grouped = book.assign(inv=1.0 / book["odd"].astype(float)).groupby(keys)["inv"]
    complete = grouped.count() == 3
    overround = grouped.sum()[complete].rename("overround").reset_index()
    low, high = ODDS_OVERROUND_RANGE
    bad = overround[(overround["overround"] < low) | (overround["overround"] > high)]
    report.add(
        "odds_overround", WARNING, bad.round({"overround": 3}),
        f"1X2 implied probabilities sum outside [{low}, {high}]",
        [*keys, "overround"],
    )  # fmt: skip


# ------------------------------------------------------------------ coverage
def coverage_table(matches: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    finished = matches[matches["status_group"] == "finished"]
    if finished.empty:
        return []
    stats = frames["stats"]
    with_stats = set(stats["match_id"]) if not stats.empty else set()
    with_xg = set(stats.dropna(subset=["expected_goals"])["match_id"]) if not stats.empty else set()
    with_lineups = set(frames["lineups"]["match_id"]) if not frames["lineups"].empty else set()
    odds = frames["odds"]
    with_odds = set(odds["match_id"]) if not odds.empty else set()
    with_closing = (
        set(odds[odds["odds_type"] == "closing"]["match_id"]) if not odds.empty else set()
    )
    out = []
    for (comp, season), group in finished.groupby(["competition_key", "season_year"]):
        ids = set(group["id"])
        n = len(ids)

        def pct(subset: set[Any], ids: set[Any] = ids, n: int = n) -> float:
            return round(100.0 * len(ids & subset) / n, 1)

        out.append(
            {
                "competition": comp,
                "season": _as_int(season),
                "finished_matches": n,
                "valid_for_training": int(group["valid_for_training"].astype(bool).sum()),
                "pct_stats": pct(with_stats),
                "pct_xg": pct(with_xg),
                "pct_lineups": pct(with_lineups),
                "pct_odds": pct(with_odds),
                "pct_closing_odds": pct(with_closing),
                "multi_source_pct": round(
                    100.0 * float(np.mean([len(s) > 1 for s in group["sources"]])), 1
                ),
            }
        )
    return out
