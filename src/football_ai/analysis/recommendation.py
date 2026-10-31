"""Recommendation engine: when (and why) a selection has statistical value.

A selection is NEVER recommended just because it is the most likely outcome.
It must offer positive expected value against available, sane market odds,
after shrinking the model probability towards the market according to the
confidence of the prediction:

    p_adj = lambda * p_model + (1 - lambda) * p_market      (lambda: 0 / 0.4 / 0.6)
    EV    = p_adj * odds - 1

Without odds, or when no selection passes every filter, the engine says so
explicitly ("no reliable recommendation") and lists the reasons.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from football_ai.prediction.confidence import ConfidenceLevel
from football_ai.prediction.engine import MatchPrediction
from football_ai.prediction.probabilities import implied_from_odds, overround

DISCLAIMER = (
    "Analyse statistique probabiliste, pas un conseil financier ni une garantie de résultat. "
    "Les paris comportent un risque de perte ; jouez de façon responsable."
)
SHRINKAGE = {ConfidenceLevel.LOW: 0.0, ConfidenceLevel.MEDIUM: 0.4, ConfidenceLevel.HIGH: 0.6}


class RecommendationStatus(StrEnum):
    VALUE = "value"  # at least one selection passes every filter
    NONE = "none"  # odds available, nothing reliable
    NO_ODDS = "no_odds"  # cannot assess value without odds


@dataclass(frozen=True, slots=True)
class RecommendationConfig:
    min_expected_value: float = 0.03
    min_probability: float = 0.25
    max_odds: float = 6.0
    min_overround: float = 1.0  # below 1: stale / inconsistent prices (arbitrage)
    max_overround: float = 1.12
    max_selections: int = 2


@dataclass(slots=True)
class Selection:
    market: str
    selection: str
    model_probability: float
    market_probability: float
    adjusted_probability: float
    odds: float
    expected_value: float
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Recommendation:
    status: RecommendationStatus
    selections: list[Selection]
    most_likely: list[dict[str, Any]]
    reasons: list[str]
    disclaimer: str = DISCLAIMER


def model_market_probabilities(prediction: MatchPrediction) -> dict[str, dict[str, float]]:
    """Model probabilities per (market, selection) - same keys as the odds table."""
    out: dict[str, dict[str, float]] = {"1x2": dict(prediction.probabilities)}
    m = prediction.markets
    if m:
        out["double_chance"] = dict(m["double_chance"])
        out["btts"] = dict(m["btts"])
        for line, probs in m["over_under"].items():
            out[f"over_under_{line}"] = dict(probs)
    else:
        p = prediction.probabilities
        out["double_chance"] = {"1x": p["home"] + p["draw"], "12": p["home"] + p["away"],
                                "x2": p["draw"] + p["away"]}  # fmt: skip
    return out


LABELS = {
    ("1x2", "home"): "victoire {home}",
    ("1x2", "draw"): "match nul",
    ("1x2", "away"): "victoire {away}",
    ("double_chance", "1x"): "{home} ou nul",
    ("double_chance", "x2"): "{away} ou nul",
    ("double_chance", "12"): "pas de match nul",
    ("btts", "yes"): "les deux équipes marquent",
    ("btts", "no"): "au moins une équipe ne marque pas",
}


def selection_label(market: str, selection: str, prediction: MatchPrediction) -> str:
    if market.startswith("over_under_"):
        line = market.removeprefix("over_under_")
        return f"{'plus' if selection == 'over' else 'moins'} de {line} buts"
    template = LABELS.get((market, selection), f"{market} {selection}")
    return template.format(home=prediction.home_team, away=prediction.away_team)


def most_likely_scenarios(prediction: MatchPrediction, k: int = 3) -> list[dict[str, Any]]:
    """Most probable outcomes (information only - NOT recommendations)."""
    items: list[dict[str, Any]] = []
    for market, probs in model_market_probabilities(prediction).items():
        if market == "double_chance":
            continue  # trivially high, not informative
        for sel, p in probs.items():
            items.append({"market": market, "selection": sel, "probability": round(p, 4),
                          "label": selection_label(market, sel, prediction)})  # fmt: skip
    if prediction.markets:
        top = prediction.markets["top_scores"][0]
        items.append({"market": "exact_score", "selection": top["score"],
                      "probability": round(top["probability"], 4),
                      "label": f"score exact {top['score']}"})  # fmt: skip
    items.sort(key=lambda x: float(x["probability"]), reverse=True)
    return items[:k]


def recommend(
    prediction: MatchPrediction,
    odds: Mapping[str, Mapping[str, float]] | None,
    config: RecommendationConfig | None = None,
) -> Recommendation:
    cfg = config or RecommendationConfig()
    likely = most_likely_scenarios(prediction)
    if not odds:
        return Recommendation(
            RecommendationStatus.NO_ODDS, [], likely,
            ["aucune cote fiable disponible : la valeur d'une sélection ne peut pas être évaluée"],
        )  # fmt: skip
    level = prediction.confidence.level
    lam = SHRINKAGE[level]
    reasons: list[str] = []
    if level is ConfidenceLevel.LOW:
        return Recommendation(
            RecommendationStatus.NONE,
            [],
            likely,
            [
                "confiance faible dans la prédiction (données ou modèles insuffisants)",
                "aucune recommandation suffisamment fiable pour ce match",
            ],
        )
    model_probs = model_market_probabilities(prediction)
    candidates: list[Selection] = []
    for market, market_odds in odds.items():
        if market not in model_probs:
            continue
        complete = set(market_odds) >= set(model_probs[market]) or market == "double_chance"
        if not complete:
            continue
        if market != "double_chance":
            book = overround(market_odds)
            if book > cfg.max_overround:
                reasons.append(f"marge du bookmaker trop élevée sur {market}")
                continue
            if book < cfg.min_overround:
                reasons.append(f"cotes incohérentes sur {market} (marge < 0) : ignorées")
                continue
        implied = (
            {k: min(1.0, 1.0 / v) for k, v in market_odds.items()}
            if market == "double_chance"
            else implied_from_odds(market_odds)
        )
        for sel, odd in market_odds.items():
            if sel not in model_probs[market] or sel not in implied:
                continue
            p_model, p_market = model_probs[market][sel], implied[sel]
            p_adj = lam * p_model + (1 - lam) * p_market
            ev = p_adj * odd - 1
            candidates.append(Selection(market, sel, p_model, p_market, p_adj, odd, ev))
    valid = []
    for c in candidates:
        if c.expected_value < cfg.min_expected_value:
            continue
        if c.adjusted_probability < cfg.min_probability or c.odds > cfg.max_odds:
            continue
        if c.model_probability * c.odds - 1 < cfg.min_expected_value:
            continue  # the model itself must see the value, not only the normalization
        c.reasons = [
            f"probabilité du modèle {c.model_probability:.1%} contre {c.market_probability:.1%} "
            f"implicite du marché",
            f"probabilité ajustée (prudence) {c.adjusted_probability:.1%}, cote {c.odds:.2f}, "
            f"valeur attendue {c.expected_value:+.1%}",
            f"confiance {prediction.confidence.level.value}",
        ]
        valid.append(c)
    valid.sort(key=lambda c: c.expected_value, reverse=True)
    if valid:
        return Recommendation(
            RecommendationStatus.VALUE, valid[: cfg.max_selections], likely, reasons
        )
    if lam > 0 and candidates:
        best = max(candidates, key=lambda c: c.expected_value)
        reasons.append(
            f"meilleure valeur attendue {best.expected_value:+.1%} "
            f"({selection_label(best.market, best.selection, prediction)}), sous le seuil de "
            f"{cfg.min_expected_value:.0%} ou hors des limites de prudence"
        )
    reasons.append("aucune recommandation suffisamment fiable pour ce match")
    return Recommendation(RecommendationStatus.NONE, [], likely, reasons)
