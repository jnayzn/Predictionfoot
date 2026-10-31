"""Make score-matrix markets consistent with the (calibrated) ensemble 1X2."""

from __future__ import annotations

import numpy as np

from football_ai.prediction.score_prediction import normalize, outcome_probabilities


def reconcile_matrix(matrix: np.ndarray, target_1x2: np.ndarray) -> np.ndarray:
    """Rescale each region (home win / draw / away win) to the target probabilities.

    The relative shape inside each region (which home-win score is likelier)
    comes from the goal model; the region totals come from the ensemble.
    """
    g = matrix.shape[1]
    current = outcome_probabilities(matrix)
    factors = target_1x2 / np.clip(current, 1e-12, None)
    i, j = np.indices((g, g))
    region = np.where(i > j, 0, np.where(i == j, 1, 2))
    scaled = matrix * factors[:, region]
    return normalize(scaled)


def combine_matrices(matrices: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    total = sum(weights[n] for n in matrices if weights.get(n, 0) > 0)
    if total <= 0:
        raise ValueError("no positive weight for the goal models")
    out = sum((weights[n] / total) * m for n, m in matrices.items() if weights.get(n, 0) > 0)
    return normalize(np.asarray(out, dtype=float))


def score_log_likelihood(
    matrix: np.ndarray, home_goals: np.ndarray, away_goals: np.ndarray
) -> float:
    """Mean log probability of the exact observed scores (scores beyond the grid are capped)."""
    g = matrix.shape[1] - 1
    h = np.clip(home_goals.astype(int), 0, g)
    a = np.clip(away_goals.astype(int), 0, g)
    p = matrix[np.arange(len(h)), h, a]
    return float(np.mean(np.log(np.clip(p, 1e-12, 1))))


def fit_matrix_weights(
    matrices: dict[str, np.ndarray], home_goals: np.ndarray, away_goals: np.ndarray
) -> dict[str, float]:
    """Simplex weights of goal models maximizing the exact-score log-likelihood."""
    from scipy.optimize import minimize

    names = list(matrices)
    if len(names) == 1:
        return {names[0]: 1.0}

    def objective(z: np.ndarray) -> float:
        w = np.exp(z - z.max())
        w = w / w.sum()
        mix = combine_matrices(matrices, dict(zip(names, w, strict=True)))
        return -score_log_likelihood(mix, home_goals, away_goals)

    result = minimize(objective, np.zeros(len(names)), method="Nelder-Mead")
    w = np.exp(result.x - result.x.max())
    w = w / w.sum()
    return {n: float(v) for n, v in zip(names, w, strict=True)}
