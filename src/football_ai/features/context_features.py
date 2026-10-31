"""Match context: rest, fixture congestion, season-to-date record."""

from __future__ import annotations

import numpy as np
import pandas as pd

from football_ai.features.history import asof

COMPETITION_IMPORTANCE = {"champions_league": 2.0}  # domestic leagues default to 1.0


def schedule_features(targets: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    """rest_days, games_last_14d, games_last_30d - counted strictly before kick-off."""
    out = pd.DataFrame(index=targets.index)
    out["rest_days"] = np.nan
    out["games_last_14d"] = 0.0
    out["games_last_30d"] = 0.0
    history = {
        team: grp["kickoff_utc"].to_numpy(dtype="datetime64[ns]")
        for team, grp in long.groupby("team_id")
    }
    kick = (
        targets["kickoff_utc"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[ns]")
    )
    teams = targets["team_id"].to_numpy()
    rest = np.full(len(targets), np.nan)
    g14 = np.zeros(len(targets))
    g30 = np.zeros(len(targets))
    d14, d30 = np.timedelta64(14, "D"), np.timedelta64(30, "D")
    for i, (team, t) in enumerate(zip(teams, kick, strict=True)):
        times = history.get(team)
        if times is None or len(times) == 0:
            continue
        idx = np.searchsorted(times, t, side="left")  # matches strictly before t
        if idx == 0:
            continue
        rest[i] = (t - times[idx - 1]) / np.timedelta64(1, "D")
        g14[i] = idx - np.searchsorted(times, t - d14, side="left")
        g30[i] = idx - np.searchsorted(times, t - d30, side="left")
    out["rest_days"], out["games_last_14d"], out["games_last_30d"] = rest, g14, g30
    return out


def season_states(long: pd.DataFrame) -> pd.DataFrame:
    """Cumulative record in the current competition-season (state after each match)."""
    keys = ["team_id", "competition_key", "season_year"]
    grp = long.groupby(keys, sort=False)
    out = long[[*keys, "kickoff_utc"]].copy()
    played = grp.cumcount() + 1
    out["season_played"] = played.astype(float)
    out["season_ppg"] = grp["points"].cumsum() / played
    out["season_gd_pg"] = (grp["gf"].cumsum() - grp["ga"].cumsum()) / played
    out["season_gf_pg"] = grp["gf"].cumsum() / played
    out["season_ga_pg"] = grp["ga"].cumsum() / played
    return out


def season_features(targets: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    keys = ["team_id", "competition_key", "season_year"]
    cols = ["season_played", "season_ppg", "season_gd_pg", "season_gf_pg", "season_ga_pg"]
    merged = asof(targets, season_states(long), keys, cols)
    merged["season_played"] = merged["season_played"].fillna(0.0)
    return merged[cols]
