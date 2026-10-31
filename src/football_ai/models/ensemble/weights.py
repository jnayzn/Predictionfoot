"""Ensemble weights learned on validation (out-of-sample) predictions.

* ``linear`` pool:     p = sum_m w_m p_m
* ``log_linear`` pool: p ∝ prod_m p_m^w_m  (geometric; sharper when models agree)

Weights live on the simplex (softmax parameterization) and minimize the log
loss of the validation predictions. Models whose weight is below
``prune_below`` are removed and the remaining weights re-fitted.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from football_ai.models.base import check_probabilities
from football_ai.training.evaluate import log_loss

EPS = 1e-12
POOLS = ("linear", "log_linear")


def combine(
    probas: Mapping[str, np.ndarray], weights: Mapping[str, float], pool: str
) -> np.ndarray:
    names = [n for n in weights if weights[n] > 0]
    if pool == "linear":
        p = sum(weights[n] * probas[n] for n in names)
    elif pool == "log_linear":
        logp = sum(weights[n] * np.log(np.clip(probas[n], EPS, 1)) for n in names)
        logp = logp - np.max(logp, axis=1, keepdims=True)
        p = np.exp(logp)
    else:
        raise ValueError(f"unknown pool {pool!r}")
    return check_probabilities(np.asarray(p, dtype=float))


@dataclass(slots=True)
class EnsembleWeights:
    pool: str
    weights: dict[str, float]
    validation_log_loss: float
    pruned: list[str] = field(default_factory=list)

    def predict(self, probas: Mapping[str, np.ndarray]) -> np.ndarray:
        return combine(probas, self.weights, self.pool)


def _fit_simplex(probas: Mapping[str, np.ndarray], y: np.ndarray, pool: str) -> dict[str, float]:
    names = list(probas)
    if len(names) == 1:
        return {names[0]: 1.0}

    def objective(z: np.ndarray) -> float:
        w = np.exp(z - z.max())
        w = w / w.sum()
        return log_loss(y, combine(probas, dict(zip(names, w, strict=True)), pool))

    result = minimize(objective, np.zeros(len(names)), method="Nelder-Mead",
                      options={"maxiter": 4000, "xatol": 1e-5, "fatol": 1e-7})  # fmt: skip
    w = np.exp(result.x - result.x.max())
    w = w / w.sum()
    return {n: float(v) for n, v in zip(names, w, strict=True)}


def fit_weights(
    probas: Mapping[str, np.ndarray],
    y: np.ndarray,
    *,
    pool: str = "linear",
    prune_below: float = 0.02,
) -> EnsembleWeights:
    if pool not in POOLS:
        raise ValueError(f"unknown pool {pool!r}")
    y = y.astype(int)
    current = dict(probas)
    pruned: list[str] = []
    weights = _fit_simplex(current, y, pool)
    while len(current) > 1:
        weak = [n for n, w in weights.items() if w < prune_below]
        if not weak:
            break
        for name in weak:
            current.pop(name)
            pruned.append(name)
        weights = _fit_simplex(current, y, pool)
    weights = {n: weights.get(n, 0.0) for n in probas}
    loss = log_loss(y, combine(probas, weights, pool))
    return EnsembleWeights(pool, weights, loss, pruned)
