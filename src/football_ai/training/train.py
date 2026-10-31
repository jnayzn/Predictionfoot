"""Model comparison on a temporal split (walk-forward on the evaluation period)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from football_ai.models.base import OutcomeModel
from football_ai.models.baselines import EloModel, FrequencyModel
from football_ai.models.deep_learning import TORCH_AVAILABLE
from football_ai.models.dixon_coles import DixonColesModel
from football_ai.models.lightgbm import LightGBMGoalsModel, LightGBMModel
from football_ai.models.poisson import PoissonModel
from football_ai.models.xgboost import XGBoostModel
from football_ai.training.backtesting import walk_forward
from football_ai.training.datasets import TemporalSplit
from football_ai.training.evaluate import evaluate_goals, evaluate_probabilities

logger = logging.getLogger(__name__)

ModelFactory = Callable[[], OutcomeModel]

BASELINE_FACTORIES: dict[str, ModelFactory] = {
    "frequency": FrequencyModel,
    "elo": EloModel,
    "poisson": PoissonModel,
    "dixon_coles": DixonColesModel,
}

TABULAR_FACTORIES: dict[str, ModelFactory] = {
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
    "lightgbm_goals": LightGBMGoalsModel,
    "xgboost_market": lambda: XGBoostModel(use_market=True),
    "lightgbm_market": lambda: LightGBMModel(use_market=True),
}

ALL_FACTORIES: dict[str, ModelFactory] = {**BASELINE_FACTORIES, **TABULAR_FACTORIES}

if TORCH_AVAILABLE:  # deep learning is optional (heavy dependency)
    from football_ai.models.deep_learning.model import NeuralModel

    ALL_FACTORIES["neural"] = NeuralModel
    ALL_FACTORIES["neural_market"] = lambda: NeuralModel(use_market=True)
# classes accepting hyperparameters (used by tuning / final fits)
MODEL_CLASSES: dict[str, Callable[..., OutcomeModel]] = {
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
    "lightgbm_goals": LightGBMGoalsModel,
    "dixon_coles": DixonColesModel,
    "poisson": PoissonModel,
}
if TORCH_AVAILABLE:
    MODEL_CLASSES["neural"] = NeuralModel


@dataclass(slots=True)
class ModelEvaluation:
    name: str
    metrics: dict[str, Any]
    proba: np.ndarray
    rows: pd.DataFrame = field(repr=False)
    score_matrices: np.ndarray | None = field(default=None, repr=False)


def evaluate_models(
    frame: pd.DataFrame,
    split: TemporalSplit,
    factories: Mapping[str, ModelFactory] = BASELINE_FACTORIES,
    *,
    on: str = "validation",
    refit_every_days: int = 30,
) -> list[ModelEvaluation]:
    """Walk-forward evaluation of each model on ``split.<on>`` (train history + earlier windows)."""
    targets = getattr(split, on)
    if targets.empty:
        raise ValueError(f"empty {on} set")
    # history available to the models: everything played before each window
    history = pd.concat([split.train, split.validation, split.test]).drop_duplicates("match_id")
    results = []
    for name, factory in factories.items():
        wf = walk_forward(factory, history, targets, refit_every_days=refit_every_days)
        y = wf.rows["outcome"].to_numpy(dtype=int)
        metrics = evaluate_probabilities(y, wf.proba)
        metrics["refits"] = wf.refits
        if wf.expected_goals is not None:
            actual = wf.rows[["home_goals", "away_goals"]].to_numpy(dtype=float)
            metrics.update(evaluate_goals(actual, wf.expected_goals))
        logger.info("model_evaluated", extra={"model": name, "log_loss": metrics["log_loss"]})
        results.append(ModelEvaluation(name, metrics, wf.proba, wf.rows, wf.score_matrices))
    return results


def comparison_table(results: list[ModelEvaluation]) -> pd.DataFrame:
    cols = ["n", "log_loss", "brier", "rps", "accuracy", "f1_macro", "ece", "mae_total_goals"]
    table = pd.DataFrame(
        [{"model": r.name, **{c: r.metrics.get(c) for c in cols}} for r in results]
    ).set_index("model")
    return table.sort_values("log_loss")
