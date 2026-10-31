"""Derived markets from a score matrix (all probabilities, never certainties)."""

from __future__ import annotations

from typing import Any

import numpy as np

from football_ai.prediction.score_prediction import (
    expected_goals,
    outcome_probabilities,
    top_scores,
)

GOAL_LINES = (1.5, 2.5, 3.5)


def over_probability(matrix: np.ndarray, line: float) -> np.ndarray:
    g = matrix.shape[1]
    total = np.add.outer(np.arange(g), np.arange(g))
    return (matrix * (total > line)[None]).sum(axis=(1, 2))


def btts_probability(matrix: np.ndarray) -> np.ndarray:
    return matrix[:, 1:, 1:].sum(axis=(1, 2))


def team_scores_at_least(matrix: np.ndarray, side: str, goals: int) -> np.ndarray:
    if side == "home":
        return matrix[:, goals:, :].sum(axis=(1, 2))
    return matrix[:, :, goals:].sum(axis=(1, 2))


def markets_from_matrix(matrix: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
    """Full market dictionary per match."""
    outcome = outcome_probabilities(matrix)
    xg = expected_goals(matrix)
    btts = btts_probability(matrix)
    overs = {line: over_probability(matrix, line) for line in GOAL_LINES}
    scores = top_scores(matrix, top_k)
    out: list[dict[str, Any]] = []
    for i in range(matrix.shape[0]):
        p_home, p_draw, p_away = (float(x) for x in outcome[i])
        out.append(
            {
                "1x2": {"home": p_home, "draw": p_draw, "away": p_away},
                "double_chance": {
                    "1x": p_home + p_draw,
                    "12": p_home + p_away,
                    "x2": p_draw + p_away,
                },
                "expected_goals": {"home": float(xg[i, 0]), "away": float(xg[i, 1])},
                "over_under": {
                    str(line): {"over": float(p[i]), "under": float(1 - p[i])}
                    for line, p in overs.items()
                },
                "btts": {"yes": float(btts[i]), "no": float(1 - btts[i])},
                "team_goals": {
                    side: {
                        f"{k}+": float(team_scores_at_least(matrix[i : i + 1], side, k)[0])
                        for k in (1, 2, 3)
                    }
                    for side in ("home", "away")
                },
                "top_scores": [{"score": f"{h}-{a}", "probability": p} for h, a, p in scores[i]],
                "most_likely_score": f"{scores[i][0][0]}-{scores[i][0][1]}",
            }
        )
    return out
