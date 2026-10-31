"""Player availability (injuries / suspensions known for the fixture).

NaN means "not covered by any source" - never confused with 0 missing players.
Lineups are deliberately NOT used: they are published ~1 h before kick-off and
would not be available at typical prediction time.
"""

from __future__ import annotations

import pandas as pd


def player_features(matches: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=matches.index)
    for side in ("home", "away"):
        merged = matches[["match_id", f"{side}_team_id"]].merge(
            injuries.rename(columns={"team_id": f"{side}_team_id"}),
            on=["match_id", f"{side}_team_id"],
            how="left",
        )
        out[f"{side}_missing_players"] = merged["missing_players"].astype(float).to_numpy()
    return out
