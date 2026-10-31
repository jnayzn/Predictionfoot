"""Seed a database with a SYNTHETIC league + a registered model (tests only).

Kick-off times are shifted so that the upcoming round starts two days after
the real current time: tests stay valid whatever the date they run on.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from football_ai.data.cleaning.canonical import CanonicalBuilder
from football_ai.database import models as db
from football_ai.database.connection import session_scope
from football_ai.database.repositories import FixtureRepository, MatchDataRepository
from football_ai.domain.models import (
    Fixture,
    FixtureStatusGroup,
    Score,
    TeamFixtureStatistics,
    TeamRef,
)
from football_ai.features.builder import FeatureBuilder
from football_ai.features.dataset import load_base_frames
from football_ai.training.datasets import split_by_seasons
from football_ai.training.ensemble_pipeline import build_ensemble
from football_ai.training.registry import save_model
from football_ai.training.train import ALL_FACTORIES
from tests.synthetic import generate_league

PROVIDER = "api_football"
TEAM_NAMES = {i: f"Synthetic FC {i}" for i in range(1, 40)}


@dataclass(slots=True)
class Seeded:
    upcoming_match_ids: list[int]
    played_match_ids: list[int]
    team_ids: list[int]  # canonical ids
    model_version: str


def seed_database(
    session_factory: sessionmaker[Session], root: Path, *, n_teams: int = 8, seed: int = 4
) -> Seeded:
    league = generate_league(n_teams=n_teams, seasons=(2022, 2023, 2024, 2025), seed=seed)
    m = league.frames.matches.copy()
    upcoming = m["status_group"] == "scheduled"
    shift = (datetime.now(UTC) + timedelta(days=2)) - m.loc[upcoming, "kickoff_utc"].min()
    m["kickoff_utc"] = m["kickoff_utc"] + shift

    fixtures = [
        Fixture(
            provider=PROVIDER,
            provider_id=int(r.match_id),
            competition_key="premier_league",
            league_provider_id=39,
            league_name="Premier League",
            season=int(r.season_year),
            round=None,
            kickoff_utc=r.kickoff_utc.to_pydatetime(),
            status_short="FT" if r.status_group == "finished" else "NS",
            status_group=FixtureStatusGroup(r.status_group),
            home_team=TeamRef(
                provider_id=int(r.home_team_id), name=TEAM_NAMES[int(r.home_team_id)]
            ),
            away_team=TeamRef(
                provider_id=int(r.away_team_id), name=TEAM_NAMES[int(r.away_team_id)]
            ),
            goals=Score(
                home=None if pd.isna(r.home_goals) else int(r.home_goals),
                away=None if pd.isna(r.away_goals) else int(r.away_goals),
            ),
            score_fulltime=Score(
                home=None if pd.isna(r.home_goals) else int(r.home_goals),
                away=None if pd.isna(r.away_goals) else int(r.away_goals),
            ),
        )
        for r in m.itertuples()
    ]
    with session_scope(session_factory) as s:
        repo = FixtureRepository(s)
        stats = repo.upsert_fixtures(fixtures)
        league_row = s.scalar(select(db.League))
        assert league_row is not None
        s.add(
            db.Season(
                league_id=league_row.id, year=2025, is_current=True, coverage={"injuries": False}
            )
        )
        md = MatchDataRepository(s, repo.refs, repo)
        for match_id, group in league.frames.team_stats.groupby("match_id"):
            md.save_team_statistics(
                stats.fixture_ids[int(match_id)],
                [
                    TeamFixtureStatistics(
                        provider=PROVIDER,
                        fixture_provider_id=int(match_id),
                        team=TeamRef(
                            provider_id=int(row.team_id), name=TEAM_NAMES[int(row.team_id)]
                        ),
                        values={
                            "shots_total": float(row.shots_total),
                            "shots_on_goal": float(row.shots_on_goal),
                            "expected_goals": float(row.expected_goals),
                        },
                    )
                    for row in group.itertuples()
                ],
            )
        # pre-match odds for the upcoming round (synthetic market + 5% margin)
        for row in league.frames.odds.itertuples():
            if not upcoming[m["match_id"] == row.match_id].any():
                continue
            for sel, p in (
                ("home", row.implied_home),
                ("draw", row.implied_draw),
                ("away", row.implied_away),
            ):
                s.add(
                    db.Odds(
                        fixture_id=stats.fixture_ids[int(row.match_id)],
                        provider=PROVIDER,
                        bookmaker="SyntheticBook",
                        market="1x2",
                        selection=sel,
                        odd=round(1 / (p * 1.05), 3),
                        odds_type="pre_match",
                    )
                )
    with session_scope(session_factory) as s:
        CanonicalBuilder(s).build("premier_league")
    with session_scope(session_factory) as s:
        frames = load_base_frames(s)
    features = FeatureBuilder().build(frames)
    split = split_by_seasons(features, [2024])
    factories = {k: ALL_FACTORIES[k] for k in ("frequency", "elo", "dixon_coles")}
    ensemble, _ = build_ensemble(features, split, factories, refit_every_days=200)
    with session_scope(session_factory) as s:
        registered = save_model(
            ensemble,
            name="ensemble",
            algorithm="EnsembleModel",
            params={},
            metrics={},
            data_fingerprint="synthetic",
            training_start=None,
            training_end=None,
            root=root,
            session=s,
        )
        matches = list(s.scalars(select(db.Match)))
        team_ids = sorted({t for x in matches for t in (x.home_team_id, x.away_team_id)})
        return Seeded(
            upcoming_match_ids=sorted(x.id for x in matches if x.status_group == "scheduled"),
            played_match_ids=sorted(x.id for x in matches if x.status_group == "finished"),
            team_ids=team_ids,
            model_version=registered.version,
        )
