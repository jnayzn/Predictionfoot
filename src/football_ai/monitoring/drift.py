"""Data drift and performance drift detection.

* **PSI** (population stability index) per feature between a reference period
  (what the model was trained on) and a recent period. Rule of thumb:
  < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 significant shift.
* **KS** two-sample test p-value as a complementary signal.
* **Performance drift**: rolling live log loss of published predictions vs the
  backtest reference, with a tolerance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from football_ai.features.builder import feature_columns, feature_group

PSI_MODERATE = 0.1
PSI_SIGNIFICANT = 0.25


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI with quantile bins of the reference; NaN-free inputs."""
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) < 20 or len(current) < 20:
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0 if np.allclose(np.median(reference), np.median(current)) else float("nan")
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(reference, edges)[0] / len(reference)
    cur_share = np.histogram(current, edges)[0] / len(current)
    ref_share = np.clip(ref_share, 1e-4, None)
    cur_share = np.clip(cur_share, 1e-4, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


@dataclass(slots=True)
class FeatureDrift:
    feature: str
    group: str
    psi: float
    ks_pvalue: float
    reference_mean: float
    current_mean: float
    missing_rate_reference: float
    missing_rate_current: float


@dataclass(slots=True)
class DriftReport:
    created_at: str
    reference_rows: int
    current_rows: int
    features: list[FeatureDrift] = field(default_factory=list)
    performance: dict[str, Any] | None = None

    @property
    def significant(self) -> list[FeatureDrift]:
        return [f for f in self.features if np.isfinite(f.psi) and f.psi > PSI_SIGNIFICANT]

    @property
    def max_psi(self) -> float:
        values = [f.psi for f in self.features if np.isfinite(f.psi)]
        return max(values) if values else 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["max_psi"] = self.max_psi
        data["significant"] = [f.feature for f in self.significant]
        return data


def feature_drift(
    reference: pd.DataFrame, current: pd.DataFrame, *, now: datetime, top_missing: float = 0.5
) -> DriftReport:
    report = DriftReport(now.isoformat(), len(reference), len(current))
    for col in feature_columns(reference):
        if col not in current:
            continue
        ref, cur = reference[col].astype(float), current[col].astype(float)
        if ref.isna().mean() > top_missing:  # mostly unavailable in the reference: skip
            continue
        ref_v, cur_v = ref.dropna().to_numpy(), cur.dropna().to_numpy()
        ks = (
            float(stats.ks_2samp(ref_v, cur_v).pvalue)
            if len(ref_v) >= 20 and len(cur_v) >= 20
            else float("nan")
        )
        report.features.append(
            FeatureDrift(
                feature=col,
                group=feature_group(col),
                psi=psi(ref_v, cur_v),
                ks_pvalue=ks,
                reference_mean=float(ref.mean()),
                current_mean=float(cur.mean()) if cur.notna().any() else float("nan"),
                missing_rate_reference=float(ref.isna().mean()),
                missing_rate_current=float(cur.isna().mean()),
            )
        )
    report.features.sort(key=lambda f: -f.psi if np.isfinite(f.psi) else 0.0)
    return report


def performance_drift(
    live_log_losses: list[float],
    reference_log_loss: float | None,
    *,
    tolerance: float = 0.05,
    min_matches: int = 50,
) -> dict[str, Any]:
    """Degradation when the rolling live log loss exceeds the reference by > tolerance."""
    n = len(live_log_losses)
    live = float(np.mean(live_log_losses)) if n else None
    degraded = (
        live is not None
        and reference_log_loss is not None
        and n >= min_matches
        and live > reference_log_loss + tolerance
    )
    return {
        "n": n,
        "live_log_loss": live,
        "reference_log_loss": reference_log_loss,
        "tolerance": tolerance,
        "degraded": bool(degraded),
        "enough_data": n >= min_matches,
    }
