"""Attacking strength: goals, xG, shots, efficiency."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from football_ai.features.history import rolling_mean


def attack_states(long: pd.DataFrame, windows: Sequence[int]) -> pd.DataFrame:
    out = long[["team_id", "kickoff_utc"]].copy()
    team = ["team_id"]
    for w in windows:
        out[f"gf_avg_{w}"] = rolling_mean(long, team, "gf", w)
        out[f"xg_for_avg_{w}"] = rolling_mean(long, team, "xg_for", w)
        out[f"shots_for_avg_{w}"] = rolling_mean(long, team, "shots_for", w)
        out[f"sot_for_avg_{w}"] = rolling_mean(long, team, "sot_for", w)
        out[f"failed_to_score_rate_{w}"] = rolling_mean(long, team, "failed_to_score", w)
        # efficiency: goals per shot on target over the window (NaN without shot data)
        sot = long["sot_for"].where(long["sot_for"] > 0)
        conv = (long["gf"] / sot).replace([np.inf, -np.inf], np.nan)
        out[f"conversion_{w}"] = rolling_mean(long.assign(_c=conv), team, "_c", w)
        out[f"xg_overperf_{w}"] = rolling_mean(
            long.assign(_d=long["gf"] - long["xg_for"]), team, "_d", w
        )
    return out
