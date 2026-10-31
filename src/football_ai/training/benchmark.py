"""Market benchmark: overround-free implied probabilities of CLOSING odds.

Closing odds summarize all public information at kick-off; beating them is
the hardest test for a forecaster. They are used for evaluation only.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from football_ai.database import models as db
from football_ai.features.dataset import implied_probabilities

MARKET_COLUMNS = ["match_id", "market__home", "market__draw", "market__away"]


def load_market_benchmark(
    session: Session, odds_types: Sequence[str] = ("closing",)
) -> pd.DataFrame:
    o = db.Odds
    result = session.execute(
        select(db.MatchSource.match_id, o.provider, o.bookmaker, o.selection, o.odd)
        .join(db.MatchSource, db.MatchSource.fixture_id == o.fixture_id)
        .where(o.market == "1x2", o.odds_type.in_(list(odds_types)))
    )
    raw = pd.DataFrame(result.fetchall(), columns=list(result.keys()))
    probs = implied_probabilities(raw)
    if probs.empty:
        return pd.DataFrame(columns=MARKET_COLUMNS)
    return probs.rename(
        columns={
            "implied_home": "market__home",
            "implied_draw": "market__draw",
            "implied_away": "market__away",
        }
    )[MARKET_COLUMNS]
