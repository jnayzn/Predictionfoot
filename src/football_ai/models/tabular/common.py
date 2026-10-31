"""Feature selection and chronological early-stopping split for tabular models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from football_ai.features.builder import feature_columns, feature_group
from football_ai.models.base import played_rows

MARKET_GROUP = "market"


def select_features(frame: pd.DataFrame, *, use_market: bool) -> list[str]:
    cols = feature_columns(frame)
    if not use_market:
        cols = [c for c in cols if feature_group(c) != MARKET_GROUP]
    return cols


def chronological_holdout(
    data: pd.DataFrame, fraction: float = 0.15, min_rows: int = 50
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Last ``fraction`` of the *training* period, for early stopping only."""
    data = data.sort_values("kickoff_utc")
    n_hold = int(len(data) * fraction)
    if n_hold < min_rows or len(data) - n_hold < min_rows:
        return data, None
    return data.iloc[:-n_hold], data.iloc[-n_hold:]


def training_rows(train: pd.DataFrame) -> pd.DataFrame:
    data = played_rows(train)
    if data.empty:
        raise ValueError("no played match to fit on")
    return data


def matrix(rows: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Feature matrix in the trained column order; missing columns become NaN."""
    return rows.reindex(columns=list(columns)).to_numpy(dtype=float)


def importance_table(columns: Sequence[str], values: np.ndarray) -> pd.DataFrame:
    total = float(np.sum(values)) or 1.0
    table = pd.DataFrame({"feature": list(columns), "importance": np.asarray(values) / total})
    table["group"] = table["feature"].map(feature_group)
    return table.sort_values("importance", ascending=False).reset_index(drop=True)
