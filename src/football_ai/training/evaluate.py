"""Probabilistic evaluation metrics for 1X2 forecasts.

* **log loss** - proper scoring rule, heavily penalizes confident mistakes;
* **Brier score** (multi-class, sum over outcomes);
* **RPS** (ranked probability score) - the standard for football since the
  outcomes are ordered (home > draw > away);
* accuracy / precision / recall / F1 (macro) - reported for completeness,
  they ignore probability quality;
* calibration table (reliability) and expected calibration error.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

EPS = 1e-15


def one_hot(y: np.ndarray, k: int = 3) -> np.ndarray:
    out = np.zeros((len(y), k))
    out[np.arange(len(y)), y.astype(int)] = 1.0
    return out


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1.0)
    return float(-np.mean(np.log(p[np.arange(len(y)), y.astype(int)])))


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.sum((p - one_hot(y, p.shape[1])) ** 2, axis=1)))


def ranked_probability_score(y: np.ndarray, p: np.ndarray) -> float:
    cum_p = np.cumsum(p, axis=1)[:, :-1]
    cum_y = np.cumsum(one_hot(y, p.shape[1]), axis=1)[:, :-1]
    return float(np.mean(np.sum((cum_p - cum_y) ** 2, axis=1) / (p.shape[1] - 1)))


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    """Reliability of every (match, outcome) probability, pooled over the 3 outcomes."""
    probs = p.ravel()
    hits = one_hot(y, p.shape[1]).ravel()
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in itertools.pairwise(edges):
        mask = (probs >= lo) & (probs < hi if hi < 1 else probs <= hi)
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bin_low": float(lo),
                "bin_high": float(hi),
                "count": int(mask.sum()),
                "mean_predicted": float(probs[mask].mean()),
                "observed_frequency": float(hits[mask].mean()),
            }
        )
    return rows


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    table = calibration_table(y, p, bins)
    total = sum(r["count"] for r in table)
    return float(
        sum(r["count"] * abs(r["mean_predicted"] - r["observed_frequency"]) for r in table)
        / max(total, 1)
    )


def evaluate_probabilities(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y).astype(int)
    pred = p.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, labels=[0, 1, 2], average="macro", zero_division=0
    )
    return {
        "n": len(y),
        "log_loss": log_loss(y, p),
        "brier": brier_score(y, p),
        "rps": ranked_probability_score(y, p),
        "accuracy": float(accuracy_score(y, pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "ece": expected_calibration_error(y, p),
        "mean_p_draw": float(p[:, 1].mean()),
        "observed_draw_rate": float((y == 1).mean()),
    }


def evaluate_goals(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    """MAE of home / away / total goals given (n, 2) expected goals."""
    actual = np.asarray(actual, dtype=float)
    return {
        "mae_home_goals": float(np.mean(np.abs(actual[:, 0] - expected[:, 0]))),
        "mae_away_goals": float(np.mean(np.abs(actual[:, 1] - expected[:, 1]))),
        "mae_total_goals": float(np.mean(np.abs(actual.sum(axis=1) - expected.sum(axis=1)))),
    }
