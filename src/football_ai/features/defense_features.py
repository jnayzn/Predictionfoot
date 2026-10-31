"""Defensive solidity: goals/xG/shots conceded, clean sheets."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from football_ai.features.history import rolling_mean


def defense_states(long: pd.DataFrame, windows: Sequence[int]) -> pd.DataFrame:
    out = long[["team_id", "kickoff_utc"]].copy()
    team = ["team_id"]
    for w in windows:
        out[f"ga_avg_{w}"] = rolling_mean(long, team, "ga", w)
        out[f"xg_against_avg_{w}"] = rolling_mean(long, team, "xg_against", w)
        out[f"shots_against_avg_{w}"] = rolling_mean(long, team, "shots_against", w)
        out[f"sot_against_avg_{w}"] = rolling_mean(long, team, "sot_against", w)
        out[f"clean_sheet_rate_{w}"] = rolling_mean(long, team, "clean_sheet", w)
    return out
