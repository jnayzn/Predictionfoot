"""Head-to-head record between the two teams (any venue, any competition)."""

from __future__ import annotations

import pandas as pd

from football_ai.features.history import asof, rolling_mean


def h2h_features(targets: pd.DataFrame, long: pd.DataFrame, window: int) -> pd.DataFrame:
    pair = ["team_id", "opponent_id"]
    ordered = long.sort_values([*pair, "kickoff_utc"]).reset_index(drop=True)
    states = ordered[[*pair, "kickoff_utc"]].copy()
    states["h2h_ppg"] = rolling_mean(ordered, pair, "points", window)
    states["h2h_gd_avg"] = rolling_mean(
        ordered.assign(_gd=ordered["gf"] - ordered["ga"]), pair, "_gd", window
    )
    states["h2h_count"] = (ordered.groupby(pair).cumcount() + 1).clip(upper=window).astype(float)
    merged = asof(targets, states, pair, ["h2h_ppg", "h2h_gd_avg", "h2h_count"])
    merged["h2h_count"] = merged["h2h_count"].fillna(0.0)
    return merged[["h2h_ppg", "h2h_gd_avg", "h2h_count"]]
