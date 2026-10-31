"""Strictly chronological train / validation / test splits."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from football_ai.models.base import played_rows


@dataclass(frozen=True, slots=True)
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def describe(self) -> dict[str, dict[str, object]]:
        def span(df: pd.DataFrame) -> dict[str, object]:
            if df.empty:
                return {"rows": 0}
            return {
                "rows": len(df),
                "from": df["kickoff_utc"].min().isoformat(),
                "to": df["kickoff_utc"].max().isoformat(),
            }

        return {
            "train": span(self.train),
            "validation": span(self.validation),
            "test": span(self.test),
        }


def _check_order(*parts: pd.DataFrame) -> None:
    non_empty = [p for p in parts if not p.empty]
    for before, after in itertools.pairwise(non_empty):
        if before["kickoff_utc"].max() >= after["kickoff_utc"].min():
            raise ValueError("temporal split violates chronology (overlapping periods)")


def split_by_seasons(
    frame: pd.DataFrame, validation_seasons: Sequence[int], test_seasons: Sequence[int] = ()
) -> TemporalSplit:
    """Train = every season before the validation seasons; test = the latest seasons.

    Only played, valid rows are kept. Raises if periods overlap in time
    (e.g. a validation season listed after a test season).
    """
    data = played_rows(frame).sort_values("kickoff_utc")
    valid_set, test_set = set(validation_seasons), set(test_seasons)
    if valid_set & test_set:
        raise ValueError("a season cannot be both validation and test")
    first_eval = min(valid_set | test_set) if valid_set | test_set else None
    train = data[data["season_year"] < first_eval] if first_eval is not None else data
    validation = data[data["season_year"].isin(valid_set)]
    test = data[data["season_year"].isin(test_set)]
    _check_order(train, validation, test)
    return TemporalSplit(train, validation, test)


def split_by_dates(
    frame: pd.DataFrame, validation_start: pd.Timestamp, test_start: pd.Timestamp | None = None
) -> TemporalSplit:
    data = played_rows(frame).sort_values("kickoff_utc")
    train = data[data["kickoff_utc"] < validation_start]
    if test_start is None:
        return TemporalSplit(train, data[data["kickoff_utc"] >= validation_start], data.iloc[0:0])
    validation = data[
        (data["kickoff_utc"] >= validation_start) & (data["kickoff_utc"] < test_start)
    ]
    test = data[data["kickoff_utc"] >= test_start]
    _check_order(train, validation, test)
    return TemporalSplit(train, validation, test)
