"""Reference models every serious model must beat."""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from football_ai.models.base import check_probabilities, played_rows


class FrequencyModel:
    """Historical outcome frequencies per competition (Laplace-smoothed)."""

    name = "frequency"

    def __init__(self, smoothing: float = 1.0) -> None:
        self.smoothing = smoothing
        self.by_competition: dict[str, np.ndarray] = {}
        self.overall = np.full(3, 1 / 3)

    def fit(self, train: pd.DataFrame) -> Self:
        data = played_rows(train)
        counts = np.bincount(data["outcome"].astype(int), minlength=3) + self.smoothing
        self.overall = counts / counts.sum()
        for comp, grp in data.groupby("competition_key"):
            c = np.bincount(grp["outcome"].astype(int), minlength=3) + self.smoothing
            self.by_competition[str(comp)] = c / c.sum()
        return self

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        p = np.array(
            [self.by_competition.get(str(c), self.overall) for c in rows["competition_key"]]
        )
        return check_probabilities(p.reshape(-1, 3))


class EloModel:
    """Multinomial logistic regression of the outcome on the pre-match Elo difference."""

    name = "elo"
    features = ("diff_elo",)

    def __init__(self, c: float = 1.0) -> None:
        self.model = LogisticRegression(C=c, max_iter=1000)

    def _x(self, rows: pd.DataFrame) -> np.ndarray:
        return rows[list(self.features)].to_numpy(dtype=float) / 100.0

    def fit(self, train: pd.DataFrame) -> Self:
        data = played_rows(train)
        self.model.fit(self._x(data), data["outcome"].astype(int))
        return self

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        proba = np.zeros((len(rows), 3))
        proba[:, self.model.classes_.astype(int)] = self.model.predict_proba(self._x(rows))
        return check_probabilities(proba)
