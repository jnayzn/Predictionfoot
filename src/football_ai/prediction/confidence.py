"""Confidence of a prediction = how much the inputs and models can be trusted.

It is NOT the probability of the outcome. A 45 % favourite can have a high
confidence (rich data, models agree) and a 70 % favourite a low one (promoted
team, no xG, models disagree).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from football_ai.prediction.probabilities import mean_pairwise_js_divergence


class ConfidenceLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class Confidence:
    score: float  # 0..1
    level: ConfidenceLevel
    uncertainties: list[str] = field(default_factory=list)
    components: dict[str, float] = field(default_factory=dict)


def _get(row: Mapping[str, object], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def assess_confidence(
    features: Mapping[str, object],
    member_probas: Mapping[str, np.ndarray] | None = None,
    known_teams: bool = True,
) -> Confidence:
    """Combine data-quality and model-agreement signals into a 0..1 score."""
    uncertainties: list[str] = []
    components: dict[str, float] = {}

    history = min(
        _get(features, "home_matches_played") or 0, _get(features, "away_matches_played") or 0
    )
    components["history"] = min(history / 20.0, 1.0)
    if history < 10:
        uncertainties.append(f"historique court pour au moins une équipe ({int(history)} matchs)")
    if not known_teams:
        components["history"] = min(components["history"], 0.2)
        uncertainties.append("au moins une équipe absente des données d'entraînement du modèle")

    season_played = min(
        _get(features, "home_season_played") or 0, _get(features, "away_season_played") or 0
    )
    components["season"] = min(season_played / 6.0, 1.0)
    if season_played < 4:
        uncertainties.append("début de saison : forme de la saison encore peu informative")

    xg = _get(features, "home_xg_for_avg_10")
    components["xg"] = 1.0 if xg is not None else 0.5
    if xg is None:
        uncertainties.append("xG non disponibles pour ce match")

    if _get(features, "home_missing_players") is None:
        components["availability"] = 0.6
        uncertainties.append("blessures et suspensions non couvertes par les sources")
    else:
        components["availability"] = 1.0

    if member_probas and len(member_probas) >= 2:
        disagreement = mean_pairwise_js_divergence([p.tolist() for p in member_probas.values()])
        components["agreement"] = float(max(0.0, 1.0 - disagreement * 20))
        if components["agreement"] < 0.6:
            uncertainties.append("désaccord notable entre les modèles")
    else:
        components["agreement"] = 0.7

    weights = {"history": 0.3, "season": 0.15, "xg": 0.15, "availability": 0.15, "agreement": 0.25}
    score = float(sum(weights[k] * components[k] for k in weights))
    level = (
        ConfidenceLevel.HIGH
        if score >= 0.75
        else ConfidenceLevel.MEDIUM
        if score >= 0.5
        else ConfidenceLevel.LOW
    )
    return Confidence(
        round(score, 3), level, uncertainties, {k: round(v, 3) for k, v in components.items()}
    )
