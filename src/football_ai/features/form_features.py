"""Recent form: points, results, streaks (overall and venue-specific)."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from football_ai.features.history import rolling_mean


def _streak(flags: pd.Series, group: pd.Series) -> pd.Series:
    """Length of the current run of 1s (state after each row)."""
    # a new block starts every time the flag is 0
    blocks = (flags == 0).groupby(group).cumsum()
    return flags.groupby([group, blocks]).cumsum()


def form_states(long: pd.DataFrame, windows: Sequence[int]) -> pd.DataFrame:
    out = long[["team_id", "kickoff_utc"]].copy()
    for w in windows:
        out[f"ppg_{w}"] = rolling_mean(long, ["team_id"], "points", w)
        out[f"win_rate_{w}"] = rolling_mean(long, ["team_id"], "win", w)
        out[f"draw_rate_{w}"] = rolling_mean(long, ["team_id"], "draw", w)
        out[f"loss_rate_{w}"] = rolling_mean(long, ["team_id"], "loss", w)
    team = long["team_id"]
    out["win_streak"] = _streak(long["win"], team)
    out["unbeaten_streak"] = _streak(1 - long["loss"], team)
    out["loss_streak"] = _streak(long["loss"], team)
    out["winless_streak"] = _streak(1 - long["win"], team)
    out["matches_played"] = long.groupby("team_id").cumcount() + 1
    return out


def venue_states(long: pd.DataFrame, window: int) -> pd.DataFrame:
    """Form computed only on home matches (for home rows) or away matches (for away rows)."""
    out = long[["team_id", "is_home", "kickoff_utc"]].copy()
    group = ["team_id", "is_home"]
    out[f"venue_ppg_{window}"] = rolling_mean(long, group, "points", window)
    out[f"venue_gf_{window}"] = rolling_mean(long, group, "gf", window)
    out[f"venue_ga_{window}"] = rolling_mean(long, group, "ga", window)
    return out
