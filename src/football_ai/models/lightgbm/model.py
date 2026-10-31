"""LightGBM models: 1X2 classifier and a tabular Poisson goals model."""

from __future__ import annotations

import inspect
from typing import Any, Self

import lightgbm as lgb
import numpy as np
import pandas as pd

from football_ai.models.base import check_probabilities
from football_ai.models.tabular.common import (
    chronological_holdout,
    importance_table,
    matrix,
    select_features,
    training_rows,
)
from football_ai.prediction.score_prediction import (
    MAX_GOALS,
    outcome_probabilities,
    poisson_score_matrix,
)

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 800,
    "learning_rate": 0.02,
    "num_leaves": 15,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_lambda": 5.0,
}


def _eval_kwargs(estimator: Any, x: np.ndarray | None, y: np.ndarray | None) -> dict[str, Any]:
    """Validation data in the API of the installed LightGBM (eval_X/eval_y since 4.6)."""
    if x is None or y is None:
        return {}
    if "eval_X" in inspect.signature(estimator.fit).parameters:
        return {"eval_X": (x,), "eval_y": (y,)}
    return {"eval_set": [(x, y)]}


def _callbacks(holdout: pd.DataFrame | None, rounds: int) -> list[Any]:
    cbs: list[Any] = [lgb.log_evaluation(period=0)]
    if holdout is not None:
        cbs.append(lgb.early_stopping(rounds, verbose=False))
    return cbs


class LightGBMModel:
    name = "lightgbm"

    def __init__(
        self,
        *,
        use_market: bool = False,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
        **params: Any,
    ) -> None:
        self.use_market = use_market
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.params = {**DEFAULT_PARAMS, **params}
        self.columns: list[str] = []
        self.model: lgb.LGBMClassifier | None = None
        if use_market:
            self.name = "lightgbm_market"

    def fit(self, train: pd.DataFrame) -> Self:
        data = training_rows(train)
        self.columns = select_features(data, use_market=self.use_market)
        fit_part, holdout = chronological_holdout(data)
        self.model = lgb.LGBMClassifier(
            objective="multiclass",
            random_state=self.random_state,
            verbose=-1,
            n_jobs=1,
            **self.params,
        )
        hold_x = matrix(holdout, self.columns) if holdout is not None else None
        hold_y = holdout["outcome"].to_numpy(dtype=int) if holdout is not None else None
        self.model.fit(
            matrix(fit_part, self.columns),
            fit_part["outcome"].to_numpy(dtype=int),
            callbacks=_callbacks(holdout, self.early_stopping_rounds),
            **_eval_kwargs(self.model, hold_x, hold_y),
        )
        return self

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        proba = np.zeros((len(rows), 3))
        proba[:, self.model.classes_.astype(int)] = self.model.predict_proba(
            matrix(rows, self.columns)
        )
        return check_probabilities(proba)

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        gain = self.model.booster_.feature_importance(importance_type="gain")
        return importance_table(self.columns, gain)


class LightGBMGoalsModel:
    """Two Poisson regressors (home and away goals) -> independent Poisson score matrix."""

    name = "lightgbm_goals"

    def __init__(
        self,
        *,
        use_market: bool = False,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
        max_goals: int = MAX_GOALS,
        **params: Any,
    ) -> None:
        self.use_market = use_market
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.max_goals = max_goals
        self.params = {**DEFAULT_PARAMS, "n_estimators": 600, **params}
        self.columns: list[str] = []
        self.models: dict[str, lgb.LGBMRegressor] = {}

    def fit(self, train: pd.DataFrame) -> Self:
        data = training_rows(train)
        data = data[data["home_goals"].notna() & data["away_goals"].notna()]
        self.columns = select_features(data, use_market=self.use_market)
        fit_part, holdout = chronological_holdout(data)
        for side in ("home", "away"):
            target = f"{side}_goals"
            model = lgb.LGBMRegressor(
                objective="poisson",
                random_state=self.random_state,
                verbose=-1,
                n_jobs=1,
                **self.params,
            )
            hold_x = matrix(holdout, self.columns) if holdout is not None else None
            hold_y = holdout[target].to_numpy(dtype=float) if holdout is not None else None
            model.fit(
                matrix(fit_part, self.columns),
                fit_part[target].to_numpy(dtype=float),
                callbacks=_callbacks(holdout, self.early_stopping_rounds),
                **_eval_kwargs(model, hold_x, hold_y),
            )
            self.models[side] = model
        return self

    def expected_goals(self, rows: pd.DataFrame) -> np.ndarray:
        if not self.models:
            raise RuntimeError("model is not fitted")
        x = matrix(rows, self.columns)
        lam = np.stack([self.models[s].predict(x) for s in ("home", "away")], axis=1)
        clipped: np.ndarray = np.clip(lam, 0.05, 8.0)
        return clipped

    def score_matrix(self, rows: pd.DataFrame) -> np.ndarray:
        lam = self.expected_goals(rows)
        return poisson_score_matrix(lam[:, 0], lam[:, 1], self.max_goals)

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        return check_probabilities(outcome_probabilities(self.score_matrix(rows)))
