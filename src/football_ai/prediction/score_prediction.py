"""Score matrices: P(home = i, away = j), vectorized over matches (shape n x G x G)."""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

MAX_GOALS = 10  # goals 0..10 per team; the tail mass is renormalized


def poisson_score_matrix(
    lambda_home: np.ndarray, lambda_away: np.ndarray, max_goals: int = MAX_GOALS
) -> np.ndarray:
    goals = np.arange(max_goals + 1)
    ph = poisson.pmf(goals[None, :], np.asarray(lambda_home, dtype=float)[:, None])
    pa = poisson.pmf(goals[None, :], np.asarray(lambda_away, dtype=float)[:, None])
    matrix = ph[:, :, None] * pa[:, None, :]
    return normalize(matrix)


def normalize(matrix: np.ndarray) -> np.ndarray:
    total = matrix.sum(axis=(1, 2), keepdims=True)
    return matrix / np.where(total > 0, total, 1.0)


def outcome_probabilities(matrix: np.ndarray) -> np.ndarray:
    """(n, 3): home win, draw, away win."""
    home = np.tril(np.ones(matrix.shape[1:]), -1)[None] * matrix
    away = np.triu(np.ones(matrix.shape[1:]), 1)[None] * matrix
    draw = np.einsum("nii->n", matrix)
    return np.stack([home.sum(axis=(1, 2)), draw, away.sum(axis=(1, 2))], axis=1)


def expected_goals(matrix: np.ndarray) -> np.ndarray:
    """(n, 2) expectation of home and away goals under the matrix."""
    goals = np.arange(matrix.shape[1])
    return np.stack(
        [np.einsum("nij,i->n", matrix, goals), np.einsum("nij,j->n", matrix, goals)], axis=1
    )


def top_scores(matrix: np.ndarray, k: int = 5) -> list[list[tuple[int, int, float]]]:
    """For each match, the k most likely exact scores as (home, away, probability)."""
    n, g, _ = matrix.shape
    flat = matrix.reshape(n, -1)
    idx = np.argsort(-flat, axis=1)[:, :k]
    return [[(int(i // g), int(i % g), float(flat[row, i])) for i in idx[row]] for row in range(n)]
