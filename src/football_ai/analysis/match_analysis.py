"""Statistical match analysis (French texts). Probabilities, never certainties."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from football_ai.analysis.team_analysis import TeamProfile, build_profile, league_goal_average
from football_ai.prediction.engine import MatchPrediction
from football_ai.prediction.probabilities import entropy

OUTCOME_LABELS = {"home": "victoire domicile", "draw": "match nul", "away": "victoire extérieur"}

# (feature, typical scale, French description) - positive diff favours the home team
FACTORS: tuple[tuple[str, float, str], ...] = (
    ("diff_elo", 100.0, "écart de niveau Elo"),
    ("diff_ppg_5", 0.8, "forme récente (points sur 5 matchs)"),
    ("diff_xg_for_avg_10", 0.5, "production offensive (xG)"),
    ("diff_gf_avg_10", 0.6, "buts marqués"),
    ("diff_ga_avg_10", -0.6, "buts encaissés"),
    ("diff_venue_ppg_5", 0.9, "performances à domicile / à l'extérieur"),
    ("diff_season_ppg", 0.7, "bilan de la saison"),
    ("diff_rest_days", 2.5, "jours de repos"),
    ("diff_missing_players", -2.0, "joueurs indisponibles"),
)


@dataclass(slots=True)
class Factor:
    feature: str
    description: str
    favours: str  # "home" | "away"
    strength: float  # standardized magnitude
    text: str


@dataclass(slots=True)
class MatchAnalysis:
    prediction: MatchPrediction
    home: TeamProfile
    away: TeamProfile
    favourite: str | None  # "home" | "draw" | "away" | None (balanced)
    summary: str
    key_factors: list[Factor] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    uncertainty_level: float = 0.0  # normalized entropy of the 1X2


def _value(features: dict[str, Any], key: str) -> float | None:
    v = features.get(key)
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def key_factors(prediction: MatchPrediction, top: int = 4) -> list[Factor]:
    out: list[Factor] = []
    for feature, scale, description in FACTORS:
        value = _value(prediction.features, feature)
        if value is None or value == 0:
            continue
        z = value / scale
        favours = "home" if z > 0 else "away"
        team = prediction.home_team if favours == "home" else prediction.away_team
        out.append(
            Factor(feature, description, favours, abs(z), f"{description} : avantage {team}")
        )
    out.sort(key=lambda f: f.strength, reverse=True)
    return [f for f in out[:top] if f.strength >= 0.25]


def analyse_match(prediction: MatchPrediction, matches: pd.DataFrame) -> MatchAnalysis:
    league_avg = league_goal_average(matches, prediction.competition_key, prediction.kickoff_utc)
    home = build_profile(prediction.features, "home", prediction.home_team, matches, league_avg)
    away = build_profile(prediction.features, "away", prediction.away_team, matches, league_avg)
    probs = prediction.probabilities
    ordered = sorted(probs, key=lambda k: probs[k], reverse=True)
    margin = probs[ordered[0]] - probs[ordered[1]]
    favourite = ordered[0] if margin >= 0.08 else None
    level = entropy(list(probs.values()))
    if favourite is None:
        summary = (
            f"Match équilibré : {OUTCOME_LABELS[ordered[0]]} {probs[ordered[0]]:.0%}, "
            f"{OUTCOME_LABELS[ordered[1]]} {probs[ordered[1]]:.0%}."
        )
    else:
        team = {"home": prediction.home_team, "away": prediction.away_team}.get(favourite, "nul")
        summary = (
            f"{team if favourite != 'draw' else 'Le nul'} est statistiquement favori "
            f"({probs[favourite]:.0%}), ce qui n'est pas une certitude : "
            f"probabilité de l'issue contraire {1 - probs[favourite]:.0%}."
        )
    if prediction.expected_goals:
        xg = prediction.expected_goals
        summary += f" Buts attendus : {xg['home']:.2f} - {xg['away']:.2f}."
    return MatchAnalysis(
        prediction=prediction,
        home=home,
        away=away,
        favourite=favourite,
        summary=summary,
        key_factors=key_factors(prediction),
        uncertainties=list(prediction.confidence.uncertainties),
        uncertainty_level=round(level, 3),
    )
