"""Pre-match betting market features (closing odds are never used as inputs)."""

from __future__ import annotations

import pandas as pd

MARKET_COLUMNS = ["market_p_home", "market_p_draw", "market_p_away", "market_available"]


def market_features(matches: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    merged = matches[["match_id"]].merge(odds, on="match_id", how="left")
    out = pd.DataFrame(index=matches.index)
    out["market_p_home"] = merged["implied_home"].astype(float).to_numpy()
    out["market_p_draw"] = merged["implied_draw"].astype(float).to_numpy()
    out["market_p_away"] = merged["implied_away"].astype(float).to_numpy()
    out["market_available"] = merged["implied_home"].notna().astype(float).to_numpy()
    return out
