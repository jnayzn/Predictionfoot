"""Probability helpers shared by the engine, analysis and recommendation modules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

EPS = 1e-12


def entropy(p: Sequence[float]) -> float:
    """Normalized Shannon entropy in [0, 1] (1 = maximally uncertain)."""
    arr = np.clip(np.asarray(p, dtype=float), EPS, 1.0)
    return float(-(arr * np.log(arr)).sum() / np.log(len(arr)))


def implied_from_odds(odds: Mapping[str, float]) -> dict[str, float]:
    """Overround-free implied probabilities of a complete market (decimal odds)."""
    inv = {k: 1.0 / v for k, v in odds.items() if v and v > 1.0}
    total = sum(inv.values())
    return {k: v / total for k, v in inv.items()} if total > 0 else {}


def overround(odds: Mapping[str, float]) -> float:
    return float(sum(1.0 / v for v in odds.values() if v and v > 1.0))


def mean_pairwise_js_divergence(probas: Sequence[Sequence[float]]) -> float:
    """Average Jensen-Shannon divergence between model predictions (disagreement, 0..1)."""
    arrays = [np.clip(np.asarray(p, dtype=float), EPS, 1.0) for p in probas]
    if len(arrays) < 2:
        return 0.0

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(a * np.log(a / b)))

    values = []
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            m = 0.5 * (arrays[i] + arrays[j])
            values.append(0.5 * kl(arrays[i], m) + 0.5 * kl(arrays[j], m))
    return float(np.mean(values) / np.log(2))
