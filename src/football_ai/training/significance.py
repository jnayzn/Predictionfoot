"""Statistical comparison of two forecasters on the same matches.

Per-match losses are paired. Matches played the same day share conditions
(and errors), so the bootstrap resamples whole **match days** (block
bootstrap) instead of individual matches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

EPS = 1e-15


def per_match_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    losses: np.ndarray = -np.log(np.clip(p[np.arange(len(y)), y.astype(int)], EPS, 1.0))
    return losses


def per_match_rps(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    onehot = np.eye(p.shape[1])[y.astype(int)]
    diff = np.cumsum(p, axis=1)[:, :-1] - np.cumsum(onehot, axis=1)[:, :-1]
    out: np.ndarray = np.sum(diff**2, axis=1) / (p.shape[1] - 1)
    return out


@dataclass(frozen=True, slots=True)
class PairedComparison:
    model: str
    reference: str
    metric: str
    n: int
    mean_difference: float  # model - reference (negative = model better)
    ci_low: float
    ci_high: float
    p_value: float  # two-sided paired t-test on day-level mean differences

    @property
    def significantly_better(self) -> bool:
        return self.ci_high < 0

    @property
    def significantly_worse(self) -> bool:
        return self.ci_low > 0


def compare_losses(
    loss_model: np.ndarray,
    loss_reference: np.ndarray,
    blocks: np.ndarray,
    *,
    model: str,
    reference: str,
    metric: str = "log_loss",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PairedComparison:
    diff = np.asarray(loss_model, dtype=float) - np.asarray(loss_reference, dtype=float)
    unique, inverse = np.unique(blocks, return_inverse=True)
    sums = np.bincount(inverse, weights=diff, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(unique), size=(n_boot, len(unique)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    day_means = sums / counts
    if len(day_means) > 1 and np.std(day_means) > 0:
        p_value = float(stats.ttest_1samp(day_means, 0.0).pvalue)
    else:
        p_value = 1.0
    return PairedComparison(
        model, reference, metric, len(diff), float(diff.mean()), float(lo), float(hi), p_value
    )
