"""SYNTHETIC football data generator for tests (never real data).

Each team has latent attack/defence strengths; goals are Poisson with a home
advantage, so models and features can be checked against a KNOWN ground truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from football_ai.features.dataset import BaseFrames


@dataclass(frozen=True)
class SyntheticLeague:
    frames: BaseFrames
    attack: dict[int, float]
    defence: dict[int, float]
    home_advantage: float


def generate_league(
    *,
    n_teams: int = 10,
    seasons: tuple[int, ...] = (2021, 2022, 2023),
    seed: int = 0,
    competition_key: str = "premier_league",
    home_advantage: float = 0.25,
    base_rate: float = 0.25,
    upcoming_rounds: int = 1,
    with_stats: bool = True,
    with_odds: bool = True,
    team_offset: int = 1,
    match_offset: int = 1,
) -> SyntheticLeague:
    rng = np.random.default_rng(seed)
    teams = list(range(team_offset, team_offset + n_teams))
    attack = {t: float(rng.normal(0, 0.35)) for t in teams}
    defence = {t: float(rng.normal(0, 0.30)) for t in teams}
    rows, stats, odds = [], [], []
    match_id = match_offset
    last_season = seasons[-1]
    for season in seasons:
        order = teams[:]
        rounds = []
        for _ in range(n_teams - 1):
            rounds.append([(order[i], order[n_teams - 1 - i]) for i in range(n_teams // 2)])
            order = [order[0], order[-1], *order[1:-1]]
        rounds += [[(a, h) for h, a in r] for r in rounds]
        start = datetime(season, 8, 10, 14, tzinfo=UTC)
        for k, rnd in enumerate(rounds):
            upcoming = season == last_season and k >= len(rounds) - upcoming_rounds
            for i, (h, a) in enumerate(rnd):
                kick = start + timedelta(days=7 * k, hours=2 * i)
                lam_h = np.exp(base_rate + home_advantage + attack[h] - defence[a])
                lam_a = np.exp(base_rate + attack[a] - defence[h])
                hg, ag = int(rng.poisson(lam_h)), int(rng.poisson(lam_a))
                rows.append(
                    {
                        "match_id": match_id,
                        "competition_key": competition_key,
                        "season_year": season,
                        "kickoff_utc": kick,
                        "home_team_id": h,
                        "away_team_id": a,
                        "status_group": "scheduled" if upcoming else "finished",
                        "home_goals": None if upcoming else hg,
                        "away_goals": None if upcoming else ag,
                        "valid_for_training": not upcoming,
                    }
                )
                if with_stats and not upcoming:
                    for team, lam, goals in ((h, lam_h, hg), (a, lam_a, ag)):
                        shots = int(rng.poisson(lam * 9 + 2))
                        stats.append(
                            {
                                "match_id": match_id,
                                "team_id": team,
                                "shots_total": shots,
                                "shots_on_goal": min(shots, goals + int(rng.poisson(lam * 2))),
                                "expected_goals": round(float(lam * rng.uniform(0.8, 1.2)), 2),
                                "corners": int(rng.poisson(5)),
                            }
                        )
                if with_odds:
                    # crude market: independent Poisson probabilities + 5% margin
                    gh = np.arange(11)
                    ph = (
                        np.exp(-lam_h)
                        * lam_h**gh
                        / np.array([math.factorial(int(x)) for x in gh], dtype=float)
                    )
                    pa = (
                        np.exp(-lam_a)
                        * lam_a**gh
                        / np.array([math.factorial(int(x)) for x in gh], dtype=float)
                    )
                    grid = np.outer(ph, pa)
                    p = np.array([np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()])
                    p = p / p.sum()
                    odds.append(
                        {
                            "match_id": match_id,
                            "implied_home": p[0],
                            "implied_draw": p[1],
                            "implied_away": p[2],
                            "n_bookmakers": 1,
                        }
                    )
                match_id += 1
    frames = BaseFrames(
        pd.DataFrame(rows),
        pd.DataFrame(
            stats,
            columns=[
                "match_id",
                "team_id",
                "shots_total",
                "shots_on_goal",
                "expected_goals",
                "corners",
            ],
        ),
        pd.DataFrame(
            odds,
            columns=["match_id", "implied_home", "implied_draw", "implied_away", "n_bookmakers"],
        ),
        pd.DataFrame(columns=["match_id", "team_id", "missing_players"]),
    ).normalized()
    return SyntheticLeague(frames, attack, defence, home_advantage)
