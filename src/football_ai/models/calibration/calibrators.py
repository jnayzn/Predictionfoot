"""Multiclass probability calibration, fitted on OUT-OF-SAMPLE predictions only.

* ``temperature`` - one scalar T: p ∝ p_raw^(1/T) (keeps the ranking, fixes
  over/under-confidence; very data-efficient);
* ``platt`` - vector scaling: multinomial logistic regression on log p with
  one scale and one bias per class (Platt scaling generalized to 3 classes);
* ``isotonic`` - one isotonic regression per class (one-vs-rest), then
  renormalization; flexible, needs more data;
* ``none`` - identity.

:func:`select_calibrator` picks the method by chronological cross-validation
(earlier half -> later half and the reverse), never on the test set.
"""

from __future__ import annotations

from typing import Protocol, Self

import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

from football_ai.models.base import check_probabilities
from football_ai.training.evaluate import log_loss

EPS = 1e-12


class Calibrator(Protocol):
    name: str

    def fit(self, proba: np.ndarray, y: np.ndarray) -> Self: ...

    def transform(self, proba: np.ndarray) -> np.ndarray: ...


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    out: np.ndarray = e / e.sum(axis=1, keepdims=True)
    return out


class IdentityCalibrator:
    name = "none"

    def fit(self, proba: np.ndarray, y: np.ndarray) -> Self:
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        return check_probabilities(proba)


class TemperatureCalibrator:
    name = "temperature"

    def __init__(self) -> None:
        self.temperature = 1.0

    def fit(self, proba: np.ndarray, y: np.ndarray) -> Self:
        logp = np.log(np.clip(proba, EPS, 1))

        def nll(log_t: np.ndarray) -> float:
            p = _softmax(logp / np.exp(log_t[0]))
            return log_loss(y, p)

        result = minimize(nll, np.zeros(1), method="L-BFGS-B", bounds=[(-2.0, 2.0)])
        self.temperature = float(np.exp(result.x[0]))
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        return check_probabilities(_softmax(np.log(np.clip(proba, EPS, 1)) / self.temperature))


class PlattCalibrator:
    """Vector scaling: z_k = a_k * log p_k + b_k (b_0 fixed to 0), softmax."""

    name = "platt"

    def __init__(self, l2: float = 1e-3) -> None:
        self.l2 = l2
        self.scale = np.ones(3)
        self.bias = np.zeros(3)

    def fit(self, proba: np.ndarray, y: np.ndarray) -> Self:
        logp = np.log(np.clip(proba, EPS, 1))
        onehot = np.eye(3)[y.astype(int)]

        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            a, b = theta[:3], np.concatenate([[0.0], theta[3:]])
            p = _softmax(logp * a + b)
            n = len(y)
            loss = -np.sum(onehot * np.log(np.clip(p, EPS, 1))) / n
            loss += 0.5 * self.l2 * np.sum((a - 1) ** 2 + b**2)
            diff = (p - onehot) / n
            grad_a = np.sum(diff * logp, axis=0) + self.l2 * (a - 1)
            grad_b = np.sum(diff, axis=0)[1:] + self.l2 * b[1:]
            return float(loss), np.concatenate([grad_a, grad_b])

        theta0 = np.concatenate([np.ones(3), np.zeros(2)])
        result = minimize(objective, theta0, jac=True, method="L-BFGS-B")
        self.scale = result.x[:3]
        self.bias = np.concatenate([[0.0], result.x[3:]])
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        logp = np.log(np.clip(proba, EPS, 1))
        return check_probabilities(_softmax(logp * self.scale + self.bias))


class IsotonicCalibrator:
    name = "isotonic"

    def __init__(self) -> None:
        self.models: list[IsotonicRegression] = []

    def fit(self, proba: np.ndarray, y: np.ndarray) -> Self:
        self.models = []
        for k in range(3):
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(proba[:, k], (y == k).astype(float))
            self.models.append(iso)
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        cal = np.column_stack([m.predict(proba[:, k]) for k, m in enumerate(self.models)])
        cal = np.clip(cal, 1e-4, 1.0)  # an isotonic step at 0 would give log loss = inf
        return check_probabilities(cal)


CALIBRATORS: dict[
    str, type[IdentityCalibrator | TemperatureCalibrator | PlattCalibrator | IsotonicCalibrator]
] = {
    "none": IdentityCalibrator,
    "temperature": TemperatureCalibrator,
    "platt": PlattCalibrator,
    "isotonic": IsotonicCalibrator,
}


def chronological_folds(n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Two folds on time-ordered rows: first half -> second half, second -> first."""
    half = n // 2
    first, second = np.arange(half), np.arange(half, n)
    return [(first, second), (second, first)]


def select_calibrator(
    proba: np.ndarray,
    y: np.ndarray,
    methods: tuple[str, ...] = ("none", "temperature", "platt", "isotonic"),
    min_rows: int = 200,
) -> tuple[Calibrator, dict[str, float]]:
    """Choose the method with the best cross-validated log loss, refit it on all rows.

    ``proba``/``y`` must be time-ordered out-of-sample predictions (validation).
    With fewer than ``min_rows`` rows only the low-variance methods are tried.
    """
    y = y.astype(int)
    if len(y) < min_rows:
        methods = tuple(m for m in methods if m in ("none", "temperature"))
    scores: dict[str, float] = {}
    for method in methods:
        losses = []
        for fit_idx, eval_idx in chronological_folds(len(y)):
            cal = CALIBRATORS[method]().fit(proba[fit_idx], y[fit_idx])
            losses.append(log_loss(y[eval_idx], cal.transform(proba[eval_idx])))
        scores[method] = float(np.mean(losses))
    best = min(scores, key=lambda m: scores[m])
    return CALIBRATORS[best]().fit(proba, y), scores
