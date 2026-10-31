"""Features computed from the real database layers (canonical matches, stats, odds)."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from football_ai.config.competitions import CompetitionConfig
from football_ai.data.cleaning.canonical import CanonicalBuilder
from football_ai.database import models as db
from football_ai.database.connection import session_scope
from football_ai.features.builder import FeatureBuilder
from football_ai.features.dataset import load_base_frames
from football_ai.features.leakage import check_leakage
from tests.integration.test_canonical import setup_two_sources


def test_load_frames_and_build_features(
    session_factory: sessionmaker[Session], competitions: list[CompetitionConfig]
) -> None:
    setup_two_sources(session_factory, competitions)
    with session_scope(session_factory) as s:
        CanonicalBuilder(s).build("premier_league")
        # a closing odd that must never reach the features
        fixture = s.scalars(
            select(db.Fixture).where(db.Fixture.provider == "football_data_uk")
        ).first()
        assert fixture is not None
        s.add(db.Odds(fixture_id=fixture.id, provider="football_data_uk", bookmaker="Leak",
                      market="1x2", selection="home", odd=1.01, odds_type="closing"))  # fmt: skip
    with session_scope(session_factory) as s:
        frames = load_base_frames(s)
        canonical_ids = set(s.scalars(select(db.CanonicalTeam.id)))

    assert len(frames.matches) == 60
    assert set(frames.team_stats["team_id"]) <= canonical_ids  # canonical ids, not source ids
    # stats exist for both seasons (CSV) and one row per (match, team)
    assert not frames.team_stats.duplicated(["match_id", "team_id"]).any()
    assert len(frames.odds) == 60
    assert (
        frames.odds[["implied_home", "implied_draw", "implied_away"]]
        .sum(axis=1)
        .round(6)
        .eq(1)
        .all()
    )
    assert (frames.odds["implied_home"] < 0.9).all()  # the 1.01 closing odd was ignored

    builder = FeatureBuilder()
    features = builder.build(frames, competition_keys=["premier_league"])
    assert len(features) == 60
    assert features["outcome"].notna().sum() == 60
    assert pd.api.types.is_float_dtype(features["home_elo"])
    assert check_leakage(frames, builder, sample=10, full=features).ok
