"""Common model interfaces.

Every model consumes rows of the feature frame (one row per match, see
``features/builder.py``) and returns calibrated-or-not probabilities in the
fixed order ``[home win, draw, away win]``.
"""

from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

import numpy as np
import pandas as pd

OUTCOMES = ("home", "draw", "away")


@runtime_checkable
class OutcomeModel(Protocol):
    name: str

    def fit(self, train: pd.DataFrame) -> Self: ...

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        """(n, 3) probabilities: home win, draw, away win."""
        ...


@runtime_checkable
class GoalModel(OutcomeModel, Protocol):
    def score_matrix(self, rows: pd.DataFrame) -> np.ndarray:
        """(n, G, G) joint probability of the exact score."""
        ...


def played_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for training: played, valid, with a known outcome."""
    mask = frame["outcome"].notna() & frame["valid_for_training"].astype(bool)
    return frame[mask]


def check_probabilities(p: np.ndarray) -> np.ndarray:
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"expected (n, 3) probabilities, got {p.shape}")
    if not np.all(np.isfinite(p)) or (p < -1e-9).any():
        raise ValueError("invalid probabilities")
    clipped = np.clip(p, 1e-9, 1.0)
    normalized: np.ndarray = clipped / clipped.sum(axis=1, keepdims=True)
    return normalized
