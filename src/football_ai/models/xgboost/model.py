"""XGBoost 1X2 classifier (multi:softprob). NaN handled natively."""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd
import xgboost as xgb

from football_ai.models.base import check_probabilities
from football_ai.models.tabular.common import (
    chronological_holdout,
    importance_table,
    matrix,
    select_features,
    training_rows,
)

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 600,
    "learning_rate": 0.03,
    "max_depth": 3,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_lambda": 5.0,
}


class XGBoostModel:
    name = "xgboost"

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
        self.model: xgb.XGBClassifier | None = None
        if use_market:
            self.name = "xgboost_market"

    def fit(self, train: pd.DataFrame) -> Self:
        data = training_rows(train)
        self.columns = select_features(data, use_market=self.use_market)
        fit_part, holdout = chronological_holdout(data)
        self.model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=self.random_state,
            n_jobs=0,
            early_stopping_rounds=self.early_stopping_rounds if holdout is not None else None,
            **self.params,
        )
        eval_set = (
            [(matrix(holdout, self.columns), holdout["outcome"].astype(int))]
            if holdout is not None
            else None
        )
        self.model.fit(
            matrix(fit_part, self.columns),
            fit_part["outcome"].astype(int),
            eval_set=eval_set,
            verbose=False,
        )
        return self

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        return check_probabilities(self.model.predict_proba(matrix(rows, self.columns)))

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        return importance_table(self.columns, self.model.feature_importances_)

    @property
    def best_iteration(self) -> int | None:
        if self.model is None:
            return None
        try:
            return int(self.model.best_iteration)
        except AttributeError:
            return None
