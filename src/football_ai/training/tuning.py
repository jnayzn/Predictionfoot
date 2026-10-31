"""Random-search hyperparameter tuning on the VALIDATION set (never the test set).

Each trial fits on ``split.train`` and is scored by log loss on
``split.validation``. Models use their own chronological holdout for early
stopping, so the validation set stays an honest selection criterion.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from football_ai.models.base import OutcomeModel
from football_ai.training.datasets import TemporalSplit
from football_ai.training.evaluate import log_loss

ParamSampler = Callable[[np.random.Generator], dict[str, Any]]

SEARCH_SPACES: dict[str, ParamSampler] = {
    "xgboost": lambda r: {
        "max_depth": int(r.integers(2, 6)),
        "learning_rate": float(10 ** r.uniform(-2.2, -1.2)),
        "min_child_weight": float(r.choice([1, 3, 5, 10, 20])),
        "subsample": float(r.uniform(0.6, 1.0)),
        "colsample_bytree": float(r.uniform(0.4, 1.0)),
        "reg_lambda": float(10 ** r.uniform(-1, 1.5)),
    },
    "lightgbm": lambda r: {
        "num_leaves": int(r.choice([7, 15, 31])),
        "learning_rate": float(10 ** r.uniform(-2.2, -1.2)),
        "min_child_samples": int(r.choice([20, 40, 80, 150])),
        "subsample": float(r.uniform(0.6, 1.0)),
        "colsample_bytree": float(r.uniform(0.4, 1.0)),
        "reg_lambda": float(10 ** r.uniform(-1, 1.5)),
    },
    "neural": lambda r: {
        "hidden": int(r.choice([32, 64, 128])),
        "embedding_dim": int(r.choice([0, 4, 8, 16])),
        "dropout": float(r.uniform(0.0, 0.4)),
        "learning_rate": float(10 ** r.uniform(-3.5, -2.3)),
        "weight_decay": float(10 ** r.uniform(-4, -1.5)),
        "goal_loss_weight": float(r.choice([0.0, 0.25, 0.5, 1.0])),
    },
    "dixon_coles": lambda r: {
        "time_decay": float(r.choice([0.0005, 0.001, 0.0019, 0.003, 0.005])),
        "l2": float(r.choice([0.1, 0.5, 1.0, 3.0, 10.0])),
    },
}


@dataclass(slots=True)
class TuningResult:
    best_params: dict[str, Any]
    best_log_loss: float
    trials: pd.DataFrame


def random_search(
    factory: Callable[..., OutcomeModel],
    sampler: ParamSampler,
    split: TemporalSplit,
    *,
    n_trials: int = 20,
    seed: int = 0,
    fixed: Mapping[str, Any] | None = None,
) -> TuningResult:
    if split.validation.empty:
        raise ValueError("tuning requires a validation set")
    rng = np.random.default_rng(seed)
    y = split.validation["outcome"].to_numpy(dtype=int)
    rows = []
    sampled: list[dict[str, Any]] = []
    for trial in range(n_trials):
        params = sampler(rng)
        sampled.append(params)
        model = factory(**{**(fixed or {}), **params}).fit(split.train)
        loss = log_loss(y, model.predict_proba(split.validation))
        rows.append({"trial": trial, "log_loss": loss, **params})
    trials = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    best_trial = int(trials.iloc[0]["trial"])
    # original dict (a DataFrame row would upcast ints to floats)
    return TuningResult(sampled[best_trial], float(trials.iloc[0]["log_loss"]), trials)
