"""Automated leakage check.

For sampled target matches, features are recomputed from data *truncated at
the match's kick-off* (every match starting at or after kick-off is turned
into an unplayed fixture: no score, no statistics, no injuries collected
later). Any difference with the full computation means some feature used
information from the future.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from football_ai.features.builder import FeatureBuilder, feature_columns
from football_ai.features.dataset import BaseFrames


@dataclass(slots=True)
class LeakageResult:
    checked: int = 0
    leaking_features: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.leaking_features


def truncate_at(frames: BaseFrames, kickoff: pd.Timestamp) -> BaseFrames:
    m = frames.matches.copy()
    future = m["kickoff_utc"] >= kickoff
    m.loc[future, ["home_goals", "away_goals"]] = np.nan
    m.loc[future, "status_group"] = "scheduled"
    future_ids = set(m.loc[future, "match_id"])
    stats = frames.team_stats[~frames.team_stats["match_id"].isin(future_ids)]
    # pre-match odds / injuries of the target are legitimately known before kick-off
    return BaseFrames(m, stats, frames.odds, frames.injuries)


def check_leakage(
    frames: BaseFrames,
    builder: FeatureBuilder,
    *,
    sample: int = 25,
    seed: int = 0,
    full: pd.DataFrame | None = None,
) -> LeakageResult:
    frames = frames.normalized()
    full = full if full is not None else builder.build(frames)
    played = full[full["outcome"].notna()]
    if played.empty:
        return LeakageResult()
    picked = played.sample(n=min(sample, len(played)), random_state=seed)
    cols = feature_columns(full)
    result = LeakageResult()
    for _, row in picked.iterrows():
        truncated = builder.build(
            truncate_at(frames, row["kickoff_utc"]), match_ids=[row["match_id"]]
        )
        recomputed = truncated.iloc[0]
        for col in cols:
            a, b = row[col], recomputed[col]
            same = (pd.isna(a) and pd.isna(b)) or (
                not pd.isna(a) and not pd.isna(b) and np.isclose(float(a), float(b))
            )
            if not same:
                result.leaking_features[col] = result.leaking_features.get(col, 0) + 1
        result.checked += 1
    return result
