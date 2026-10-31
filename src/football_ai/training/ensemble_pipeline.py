"""Ensemble learning + calibration pipeline.

1. Walk-forward out-of-sample predictions of every candidate model on the
   VALIDATION period.
2. Honest estimate: chronological cross-fitting inside the validation period
   (weights + calibrator fitted on one half, scored on the other, both ways).
3. Final ensemble weights and calibrator fitted on the whole validation period.
4. Goal-model weights for the score matrix (exact-score likelihood) and
   reconciliation of the matrix with the calibrated 1X2.
5. Optional, explicitly requested: final evaluation on the TEST period with
   everything frozen (weights, calibrator chosen on validation).
6. Production members re-fitted on train + validation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from football_ai.models.base import OutcomeModel, check_probabilities
from football_ai.models.calibration import Calibrator, select_calibrator
from football_ai.models.calibration.calibrators import chronological_folds
from football_ai.models.ensemble import (
    EnsembleWeights,
    combine_matrices,
    fit_matrix_weights,
    fit_weights,
    reconcile_matrix,
)
from football_ai.models.ensemble.weights import POOLS
from football_ai.training.datasets import TemporalSplit
from football_ai.training.evaluate import evaluate_probabilities
from football_ai.training.train import ModelEvaluation, ModelFactory, evaluate_models


@dataclass(slots=True)
class EnsembleModel:
    """Production ensemble: fitted members + frozen weights + calibrator."""

    weights: EnsembleWeights
    calibrator: Calibrator
    goal_weights: dict[str, float]
    members: dict[str, OutcomeModel] = field(default_factory=dict)
    name: str = "ensemble"

    def member_probas(self, rows: pd.DataFrame) -> dict[str, np.ndarray]:
        return {
            n: m.predict_proba(rows)
            for n, m in self.members.items()
            if self.weights.weights.get(n, 0) > 0
        }

    def raw_proba(self, rows: pd.DataFrame) -> np.ndarray:
        return self.weights.predict(self.member_probas(rows))

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        return check_probabilities(self.calibrator.transform(self.raw_proba(rows)))

    def score_matrix(self, rows: pd.DataFrame) -> np.ndarray:
        matrices = {
            n: self.members[n].score_matrix(rows)  # type: ignore[attr-defined]
            for n, w in self.goal_weights.items()
            if w > 0
        }
        return reconcile_matrix(
            combine_matrices(matrices, self.goal_weights), self.predict_proba(rows)
        )

    def fit(self, train: pd.DataFrame) -> EnsembleModel:  # pragma: no cover - not used
        raise NotImplementedError("the ensemble is built by build_ensemble()")


@dataclass(slots=True)
class EnsembleReport:
    pool: str
    weights: dict[str, float]
    pruned: list[str]
    calibration_method: str
    calibration_scores: dict[str, float]
    goal_weights: dict[str, float]
    validation_members: dict[str, dict[str, Any]]
    validation_cv: dict[str, dict[str, Any]]
    test: dict[str, dict[str, Any]] | None = None
    # per-match test predictions (meta + one probability triple per model); not serialized
    test_predictions: pd.DataFrame | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pool": self.pool, "weights": self.weights, "pruned": self.pruned,
            "calibration_method": self.calibration_method,
            "calibration_scores": self.calibration_scores, "goal_weights": self.goal_weights,
            "validation_members": self.validation_members, "validation_cv": self.validation_cv,
            "test": self.test,
        }  # fmt: skip


def predictions_frame(rows: pd.DataFrame, probas: Mapping[str, np.ndarray]) -> pd.DataFrame:
    meta = ["match_id", "competition_key", "season_year", "kickoff_utc", "outcome"]
    out = rows[meta].reset_index(drop=True).copy()
    columns = {
        f"{name}__{k}": p[:, i]
        for name, p in probas.items()
        for i, k in enumerate(("home", "draw", "away"))
    }
    return pd.concat([out, pd.DataFrame(columns)], axis=1)


def _aligned(results: list[ModelEvaluation]) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    reference = results[0].rows["match_id"].to_numpy()
    for r in results[1:]:
        if not np.array_equal(r.rows["match_id"].to_numpy(), reference):
            raise RuntimeError("model predictions are not aligned on the same matches")
    return {r.name: r.proba for r in results}, results[0].rows


def _goal_matrices(results: list[ModelEvaluation]) -> dict[str, np.ndarray]:
    return {r.name: r.score_matrices for r in results if r.score_matrices is not None}


def _fit_stack(
    probas: Mapping[str, np.ndarray], y: np.ndarray, pool: str
) -> tuple[EnsembleWeights, Calibrator, dict[str, float]]:
    weights = fit_weights(probas, y, pool=pool)
    calibrator, scores = select_calibrator(weights.predict(probas), y)
    return weights, calibrator, scores


def cross_fitted_metrics(
    probas: Mapping[str, np.ndarray], y: np.ndarray, pool: str
) -> dict[str, Any]:
    """Out-of-fold ensemble predictions (time-ordered halves) -> honest metrics."""
    oof = np.zeros((len(y), 3))
    for fit_idx, eval_idx in chronological_folds(len(y)):
        sub = {n: p[fit_idx] for n, p in probas.items()}
        weights, calibrator, _ = _fit_stack(sub, y[fit_idx], pool)
        oof[eval_idx] = calibrator.transform(
            weights.predict({n: p[eval_idx] for n, p in probas.items()})
        )
    return evaluate_probabilities(y, oof)


def build_ensemble(
    frame: pd.DataFrame,
    split: TemporalSplit,
    factories: Mapping[str, ModelFactory],
    *,
    pool: str = "auto",
    refit_every_days: int = 30,
    evaluate_test: bool = False,
    fit_production_members: bool = True,
) -> tuple[EnsembleModel, EnsembleReport]:
    validation = evaluate_models(frame, split, factories, refit_every_days=refit_every_days)
    probas, rows = _aligned(validation)
    y = rows["outcome"].to_numpy(dtype=int)

    pools = list(POOLS) if pool == "auto" else [pool]
    cv = {p: cross_fitted_metrics(probas, y, p) for p in pools}
    chosen = min(cv, key=lambda p: cv[p]["log_loss"])
    weights, calibrator, cal_scores = _fit_stack(probas, y, chosen)

    matrices = _goal_matrices(validation)
    goal_weights: dict[str, float] = {}
    if matrices:
        goal_weights = fit_matrix_weights(
            matrices,
            rows["home_goals"].to_numpy(dtype=float),
            rows["away_goals"].to_numpy(dtype=float),
        )

    report = EnsembleReport(
        pool=chosen,
        weights=weights.weights,
        pruned=weights.pruned,
        calibration_method=calibrator.name,
        calibration_scores=cal_scores,
        goal_weights=goal_weights,
        validation_members={r.name: r.metrics for r in validation},
        validation_cv={f"ensemble_{p}": m for p, m in cv.items()},
    )

    if evaluate_test:
        test_results = evaluate_models(
            frame, split, factories, on="test", refit_every_days=refit_every_days
        )
        test_probas, test_rows = _aligned(test_results)
        y_test = test_rows["outcome"].to_numpy(dtype=int)
        ensemble_test = calibrator.transform(weights.predict(test_probas))
        report.test = {r.name: r.metrics for r in test_results}
        report.test["ensemble"] = evaluate_probabilities(y_test, ensemble_test)
        report.test_predictions = predictions_frame(
            test_rows, {**test_probas, "ensemble": ensemble_test}
        )

    members: dict[str, OutcomeModel] = {}
    if fit_production_members:
        needed = {n for n, w in weights.weights.items() if w > 0} | {
            n for n, w in goal_weights.items() if w > 0
        }
        production_data = pd.concat([split.train, split.validation])
        members = {name: factories[name]().fit(production_data) for name in sorted(needed)}
    return EnsembleModel(weights, calibrator, goal_weights, members), report
