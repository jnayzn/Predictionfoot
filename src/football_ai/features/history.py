"""Team-perspective history of played matches + as-of merge helper."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from football_ai.features.dataset import BaseFrames

KEY = "kickoff_utc"


def played_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Matches usable as history: finished with a known 90-minute score."""
    mask = (
        (matches["status_group"] == "finished")
        & matches["home_goals"].notna()
        & matches["away_goals"].notna()
    )
    return matches[mask]


def team_long(frames: BaseFrames) -> pd.DataFrame:
    """Two rows per played match (home and away perspective), sorted by team then time."""
    played = played_matches(frames.matches)
    common = ["match_id", "competition_key", "season_year", KEY]
    home = played[common].assign(
        team_id=played["home_team_id"],
        opponent_id=played["away_team_id"],
        is_home=True,
        gf=played["home_goals"].astype(float),
        ga=played["away_goals"].astype(float),
    )
    away = played[common].assign(
        team_id=played["away_team_id"],
        opponent_id=played["home_team_id"],
        is_home=False,
        gf=played["away_goals"].astype(float),
        ga=played["home_goals"].astype(float),
    )
    long = pd.concat([home, away], ignore_index=True)

    stats = frames.team_stats
    if not stats.empty:
        own = stats.rename(
            columns={
                "shots_total": "shots_for",
                "shots_on_goal": "sot_for",
                "expected_goals": "xg_for",
                "corners": "corners_for",
            }
        )
        opp = stats.rename(
            columns={
                "team_id": "opponent_id",
                "shots_total": "shots_against",
                "shots_on_goal": "sot_against",
                "expected_goals": "xg_against",
                "corners": "corners_against",
            }
        )
        long = long.merge(own, on=["match_id", "team_id"], how="left")
        long = long.merge(opp, on=["match_id", "opponent_id"], how="left")
    for col in (
        "shots_for", "sot_for", "xg_for", "corners_for",
        "shots_against", "sot_against", "xg_against", "corners_against",
    ):  # fmt: skip
        if col not in long.columns:
            long[col] = np.nan
        long[col] = pd.to_numeric(long[col], errors="coerce").astype(float)

    long["win"] = (long["gf"] > long["ga"]).astype(float)
    long["draw"] = (long["gf"] == long["ga"]).astype(float)
    long["loss"] = (long["gf"] < long["ga"]).astype(float)
    long["points"] = 3 * long["win"] + long["draw"]
    long["clean_sheet"] = (long["ga"] == 0).astype(float)
    long["failed_to_score"] = (long["gf"] == 0).astype(float)
    return long.sort_values(["team_id", KEY, "match_id"]).reset_index(drop=True)


def targets_long(matches: pd.DataFrame) -> pd.DataFrame:
    """Two rows per *target* match (any status) to receive team features."""
    cols = ["match_id", "competition_key", "season_year", KEY]
    home = matches[cols].assign(
        team_id=matches["home_team_id"], opponent_id=matches["away_team_id"], is_home=True
    )
    away = matches[cols].assign(
        team_id=matches["away_team_id"], opponent_id=matches["home_team_id"], is_home=False
    )
    return pd.concat([home, away], ignore_index=True)


def asof(
    targets: pd.DataFrame,
    states: pd.DataFrame,
    by: Sequence[str],
    columns: Sequence[str],
) -> pd.DataFrame:
    """For each target row, the last state row with the same ``by`` keys strictly before it.

    ``allow_exact_matches=False`` guarantees that a match never sees its own result.
    """
    left = targets.assign(_row=np.arange(len(targets))).sort_values(KEY, kind="stable")
    right = states[[*by, KEY, *columns]].dropna(subset=list(by)).sort_values(KEY, kind="stable")
    merged = pd.merge_asof(
        left, right, on=KEY, by=list(by), direction="backward", allow_exact_matches=False
    )
    # merge_asof returns rows in sorted order with a fresh index: restore the target order
    merged = merged.sort_values("_row").drop(columns="_row")
    merged.index = targets.index
    return merged


def rolling_mean(
    frame: pd.DataFrame, group: Sequence[str], column: str, window: int, min_periods: int = 1
) -> pd.Series:
    """Rolling mean over the group's rows *including* the current one (= state after match).

    NaN values are ignored (e.g. xG unavailable for some matches).
    """
    return (
        frame.groupby(list(group), sort=False)[column]
        .rolling(window, min_periods=min_periods)
        .mean()
        .reset_index(level=list(range(len(group))), drop=True)
    )
