"""Dynamic Elo rating (all competitions together, chronological).

Rules (World-Football-Elo style, parameters are hyperparameters tuned later
on validation data, never on the test set):

* expected home score ``E = 1 / (1 + 10 ** (-(R_home + home_advantage - R_away) / 400))``
* update ``R += k * G * (S - E)`` with ``G`` the goal-difference multiplier
  (1 for 0-1 goal, 1.5 for 2, (11 + d) / 8 for d >= 3);
* at the first match of a new season a team regresses towards the mean by
  ``season_regression``;
* a team never seen before starts at ``new_team_rating`` (e.g. promoted side).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class EloConfig:
    k: float = 20.0
    home_advantage: float = 60.0
    initial_rating: float = 1500.0
    new_team_rating: float = 1450.0
    season_regression: float = 0.2


def goal_multiplier(goal_diff: float) -> float:
    d = abs(goal_diff)
    if d <= 1:
        return 1.0
    if d == 2:
        return 1.5
    return (11.0 + d) / 8.0


def expected_home(r_home: float, r_away: float, home_advantage: float) -> float:
    return float(1.0 / (1.0 + 10 ** (-(r_home + home_advantage - r_away) / 400.0)))


def elo_states(played: pd.DataFrame, cfg: EloConfig) -> pd.DataFrame:
    """Post-match rating of both teams after every played match (chronological)."""
    ordered = played.sort_values(["kickoff_utc", "match_id"])
    home_ids = ordered["home_team_id"].to_numpy(dtype=np.int64)
    away_ids = ordered["away_team_id"].to_numpy(dtype=np.int64)
    seasons = ordered["season_year"].to_numpy(dtype=np.int64)
    home_goals = ordered["home_goals"].to_numpy(dtype=float)
    away_goals = ordered["away_goals"].to_numpy(dtype=float)
    kickoffs = ordered["kickoff_utc"].to_numpy()

    ratings: dict[int, float] = {}
    last_season: dict[int, int] = {}
    teams_out: list[int] = []
    kick_out: list[object] = []
    season_out: list[int] = []
    rating_out: list[float] = []
    for i in range(len(ordered)):
        season = int(seasons[i])
        home, away = int(home_ids[i]), int(away_ids[i])
        pre: dict[int, float] = {}
        for team in (home, away):
            rating = ratings.get(team, cfg.new_team_rating)
            if team in last_season and last_season[team] != season:
                rating = cfg.initial_rating + (rating - cfg.initial_rating) * (
                    1 - cfg.season_regression
                )
            pre[team] = rating
        e_home = expected_home(pre[home], pre[away], cfg.home_advantage)
        hg, ag = float(home_goals[i]), float(away_goals[i])
        score = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        delta = cfg.k * goal_multiplier(hg - ag) * (score - e_home)
        ratings[home] = pre[home] + delta
        ratings[away] = pre[away] - delta
        for team in (home, away):
            last_season[team] = season
            teams_out.append(team)
            kick_out.append(kickoffs[i])
            season_out.append(season)
            rating_out.append(ratings[team])
    out = pd.DataFrame(
        {
            "team_id": teams_out,
            "kickoff_utc": kick_out,
            "elo_season": season_out,
            "elo_post": rating_out,
        }
    )
    if out.empty:
        out["kickoff_utc"] = pd.Series(dtype="datetime64[ns, UTC]")
    else:
        out["kickoff_utc"] = pd.to_datetime(out["kickoff_utc"], utc=True)
    return out


def pre_match_elo(merged: pd.DataFrame, target_season: pd.Series, cfg: EloConfig) -> pd.Series:
    """Rating entering the target match (regression applied if the season changed)."""
    rating = merged["elo_post"].fillna(cfg.new_team_rating)
    new_season = merged["elo_season"].notna() & (merged["elo_season"] != target_season)
    regressed = cfg.initial_rating + (rating - cfg.initial_rating) * (1 - cfg.season_regression)
    return pd.Series(np.where(new_season, regressed, rating), index=merged.index)
