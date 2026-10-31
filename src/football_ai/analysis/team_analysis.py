"""Team profile from the pre-match features (no information after kick-off)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from football_ai.features.history import played_matches


def _f(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


@dataclass(slots=True)
class TeamProfile:
    team_id: int
    name: str
    elo: float | None
    form_last5: str  # e.g. "VVNDV" (V = victoire, N = nul, D = défaite), most recent last
    ppg_5: float | None
    goals_for_avg: float | None
    goals_against_avg: float | None
    xg_for_avg: float | None
    xg_against_avg: float | None
    venue_ppg: float | None
    season_ppg: float | None
    rest_days: float | None
    missing_players: float | None
    attack_index: float | None  # vs league average (1.0 = average)
    defence_index: float | None  # goals conceded vs league average (lower = better)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)


def recent_form(matches: pd.DataFrame, team_id: int, before: datetime, n: int = 5) -> str:
    played = played_matches(matches)
    played = (
        played[
            ((played["home_team_id"] == team_id) | (played["away_team_id"] == team_id))
            & (played["kickoff_utc"] < pd.Timestamp(before))
        ]
        .sort_values("kickoff_utc")
        .tail(n)
    )
    is_home = played["home_team_id"].to_numpy() == team_id
    hg = played["home_goals"].to_numpy(dtype=float)
    ag = played["away_goals"].to_numpy(dtype=float)
    gf = np.where(is_home, hg, ag)
    ga = np.where(is_home, ag, hg)
    return "".join("V" if f > a else "N" if f == a else "D" for f, a in zip(gf, ga, strict=True))


def league_goal_average(
    matches: pd.DataFrame, competition_key: str, before: datetime, days: int = 365
) -> float | None:
    played = played_matches(matches)
    start = pd.Timestamp(before) - pd.Timedelta(days=days)
    window = played[
        (played["competition_key"] == competition_key)
        & (played["kickoff_utc"] < pd.Timestamp(before))
        & (played["kickoff_utc"] >= start)
    ]
    if window.empty:
        return None
    return float((window["home_goals"] + window["away_goals"]).mean() / 2)


def build_profile(
    features: Mapping[str, Any],
    side: str,
    name: str,
    matches: pd.DataFrame,
    league_avg: float | None,
) -> TeamProfile:
    def g(key: str) -> float | None:
        return _f(features, f"{side}_{key}")

    team_id = int(features[f"{side}_team_id"])
    kickoff = pd.Timestamp(features["kickoff_utc"]).to_pydatetime()
    gf, ga = g("gf_avg_10"), g("ga_avg_10")
    attack = gf / league_avg if gf is not None and league_avg else None
    defence = ga / league_avg if ga is not None and league_avg else None
    profile = TeamProfile(
        team_id=team_id,
        name=name,
        elo=g("elo"),
        form_last5=recent_form(matches, team_id, kickoff),
        ppg_5=g("ppg_5"),
        goals_for_avg=gf,
        goals_against_avg=ga,
        xg_for_avg=g("xg_for_avg_10"),
        xg_against_avg=g("xg_against_avg_10"),
        venue_ppg=g("venue_ppg_5"),
        season_ppg=g("season_ppg"),
        rest_days=g("rest_days"),
        missing_players=g("missing_players"),
        attack_index=attack,
        defence_index=defence,
    )
    if attack is not None:
        if attack >= 1.2:
            profile.strengths.append(
                f"attaque prolifique ({gf:.2f} buts/match, {attack:.0%} de la moyenne)"
            )
        elif attack <= 0.8:
            profile.weaknesses.append(f"attaque peu efficace ({gf:.2f} buts/match)")
    if defence is not None:
        if defence <= 0.8:
            profile.strengths.append(f"défense solide ({ga:.2f} buts encaissés/match)")
        elif defence >= 1.2:
            profile.weaknesses.append(f"défense fragile ({ga:.2f} buts encaissés/match)")
    if profile.ppg_5 is not None:
        if profile.ppg_5 >= 2.0:
            profile.strengths.append(
                f"excellente forme récente ({profile.ppg_5:.1f} pts/match sur 5)"
            )
        elif profile.ppg_5 <= 0.8:
            profile.weaknesses.append(
                f"forme récente difficile ({profile.ppg_5:.1f} pts/match sur 5)"
            )
    xgf, xga = profile.xg_for_avg, profile.xg_against_avg
    if xgf is not None and gf is not None and gf - xgf >= 0.4:
        profile.weaknesses.append(
            "marque nettement plus que ses xG (efficacité possiblement non durable)"
        )
    if xga is not None and ga is not None and xga - ga >= 0.4:
        profile.weaknesses.append(
            "encaisse moins que ses xG concédés (défense possiblement chanceuse)"
        )
    if profile.rest_days is not None and profile.rest_days <= 3:
        profile.weaknesses.append(f"peu de repos ({profile.rest_days:.0f} jours)")
    if profile.missing_players is not None and profile.missing_players >= 3:
        profile.weaknesses.append(f"{int(profile.missing_players)} joueurs indisponibles")
    return profile
