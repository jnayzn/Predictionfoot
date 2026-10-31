"""Pre-match odds of a canonical match, averaged over bookmakers.

Only odds knowable before kick-off are used (``pre_match``, ``market_average``,
``opening``); closing odds and per-market maxima (``market_max``) are
excluded. The average - not the best price - is used, so the expected value
is not inflated by cherry-picking.
"""

from __future__ import annotations

from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session

from football_ai.database import models as db

USABLE_TYPES = ("pre_match", "market_average", "opening")


def load_match_odds(session: Session, match_id: int) -> dict[str, dict[str, float]]:
    rows = session.execute(
        select(
            db.Odds.bookmaker, db.Odds.market, db.Odds.selection, db.Odds.odd, db.Odds.collected_at
        )
        .join(db.MatchSource, db.MatchSource.fixture_id == db.Odds.fixture_id)
        .where(db.MatchSource.match_id == match_id, db.Odds.odds_type.in_(USABLE_TYPES))
    ).all()
    latest: dict[tuple[str, str, str], tuple[object, float]] = {}
    for bookmaker, market, selection, odd, collected in rows:
        key = (bookmaker, market, selection)
        if key not in latest or collected > latest[key][0]:  # type: ignore[operator]
            latest[key] = (collected, float(odd))
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (_, market, selection), (_, odd) in latest.items():
        grouped[market][selection].append(odd)
    return {
        market: {sel: sum(values) / len(values) for sel, values in sels.items()}
        for market, sels in grouped.items()
    }
