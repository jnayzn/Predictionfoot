"""Repository tests - run on SQLite and on PostgreSQL when TEST_DATABASE_URL is set."""

from __future__ import annotations

from datetime import date

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from football_ai.data.collectors.api_football.parsers import (
    parse_fixture,
    parse_fixture_details,
    parse_injury,
    parse_league,
    parse_odds,
    parse_standings,
    parse_team,
)
from football_ai.database import models as db
from football_ai.database.connection import session_scope
from football_ai.database.repositories import (
    FixtureRepository,
    JobRepository,
    MatchDataRepository,
    ReferenceRepository,
)
from tests.conftest import fixture_item, league_item
from tests.payloads import details_item, injury_item, odds_item, standings_item, team_item


def count(session: Session, model: type[db.Base]) -> int:
    return session.scalar(select(func.count()).select_from(model)) or 0


def test_competition_upsert_with_seasons(session_factory: sessionmaker[Session]) -> None:
    comp = parse_league(
        league_item(39, "Premier League", "England", years=range(2020, 2023)), key="premier_league"
    )
    for _ in range(2):  # idempotent
        with session_scope(session_factory) as s:
            ReferenceRepository(s).upsert_competition(comp)
    with session_scope(session_factory) as s:
        assert count(s, db.League) == 1 and count(s, db.Season) == 3
        league = ReferenceRepository(s).league_by_key("premier_league", "api_football")
        assert league is not None and league.provider_id == 39
        coverage = ReferenceRepository(s).season_coverage(league.id, 2022)
        assert coverage is not None and coverage["injuries"] is True


def test_fixture_insert_update_unchanged(session_factory: sessionmaker[Session]) -> None:
    scheduled = parse_fixture(
        fixture_item(1, 39, (33, "A"), (34, "B")), competition_key="premier_league"
    )
    with session_scope(session_factory) as s:
        stats = FixtureRepository(s).upsert_fixtures([scheduled])
        assert (stats.inserted, stats.updated) == (1, 0)
    with session_scope(session_factory) as s:
        stats = FixtureRepository(s).upsert_fixtures([scheduled])
        assert stats.unchanged == 1
        assert count(s, db.ActualResult) == 0  # not finished yet

    finished = parse_fixture(
        fixture_item(
            1, 39, (33, "A"), (34, "B"), status="FT", goals=(2, 0), date="2026-10-30T15:00:00+00:00"
        ),
        competition_key="premier_league",
    )
    with session_scope(session_factory) as s:
        stats = FixtureRepository(s).upsert_fixtures([finished])
        assert (stats.updated, stats.corrections) == (1, 0)  # NS -> FT is not a correction
        result = s.scalar(select(db.ActualResult))
        assert result is not None and (result.outcome, result.revision) == ("H", 1)
        assert count(s, db.Team) == 2 and count(s, db.League) == 1


def test_result_correction_is_audited(session_factory: sessionmaker[Session]) -> None:
    def ft(goals: tuple[int, int]) -> object:
        return parse_fixture(
            fixture_item(
                7,
                39,
                (33, "A"),
                (34, "B"),
                status="FT",
                goals=goals,
                date="2026-10-30T15:00:00+00:00",
            )
        )

    with session_scope(session_factory) as s:
        FixtureRepository(s).upsert_fixtures([ft((1, 1))])  # type: ignore[list-item]
    with session_scope(session_factory) as s:
        stats = FixtureRepository(s).upsert_fixtures([ft((1, 2))])  # type: ignore[list-item]
        assert stats.corrections == 1
    with session_scope(session_factory) as s:
        result = s.scalar(select(db.ActualResult))
        assert result is not None and (result.outcome, result.revision) == ("A", 2)
        changes = {(c.field, c.old_value, c.new_value) for c in s.scalars(select(db.DataChangeLog))}
        assert ("away_goals", "1", "2") in changes


def test_teams_standings_injuries_odds_idempotent(session_factory: sessionmaker[Session]) -> None:
    fixture = parse_fixture(fixture_item(900, 39, (33, "Alpha"), (34, "Beta")))
    for _ in range(2):
        with session_scope(session_factory) as s:
            refs = ReferenceRepository(s)
            refs.upsert_team(parse_team(team_item(33, "Alpha")))
            FixtureRepository(s, refs).upsert_fixtures([fixture])
            md = MatchDataRepository(s, refs)
            md.save_standings(
                parse_standings(
                    standings_item(39, 2026, [(33, "Alpha", 1, 20), (34, "Beta", 2, 18)])
                ),
                snapshot_date=date(2026, 10, 31),
                league_key="premier_league",
            )
            md.save_injuries(
                [
                    parse_injury(injury_item(77, (33, "Alpha"), 39, 2026, 900)),
                    parse_injury(injury_item(78, (34, "Beta"), 39, 2026, None)),
                ]
            )
            md.save_odds(parse_odds(odds_item(900)))
    with session_scope(session_factory) as s:
        assert count(s, db.Team) == 2
        team = s.scalar(select(db.Team).where(db.Team.provider_id == 33))
        assert team is not None and team.venue_capacity == 40000
        assert count(s, db.Standing) == 2
        assert count(s, db.Injury) == 2
        injury = s.scalar(select(db.Injury).where(db.Injury.fixture_provider_id == 900))
        assert injury is not None and injury.fixture_id is not None
        assert count(s, db.Odds) == 8
    # a new standings snapshot on another day is kept separately (no leakage overwrite)
    with session_scope(session_factory) as s:
        MatchDataRepository(s).save_standings(
            parse_standings(standings_item(39, 2026, [(33, "Alpha", 1, 23)])),
            snapshot_date=date(2026, 11, 7),
        )
        assert count(s, db.Standing) == 3


def test_fixture_details_saved_and_replaced(session_factory: sessionmaker[Session]) -> None:
    details = parse_fixture_details(
        details_item(500, 39, (33, "Alpha"), (34, "Beta")), competition_key="premier_league"
    )
    for _ in range(2):
        with session_scope(session_factory) as s:
            MatchDataRepository(s).save_fixture_details([details])
    with session_scope(session_factory) as s:
        assert count(s, db.TeamStatistics) == 2
        home = s.scalar(select(db.TeamStatistics).order_by(db.TeamStatistics.id))
        assert home is not None and home.expected_goals == 1.85 and home.values["shots_total"] == 12
        assert count(s, db.Lineup) == 2
        assert count(s, db.LineupPlayer) == 24
        assert count(s, db.PlayerStatistics) == 1
        fixture = s.scalar(select(db.Fixture))
        assert fixture is not None and fixture.details_fetched_at is not None
        league = s.scalar(select(db.League))
        assert league is not None
        assert FixtureRepository(s).finished_without_details("api_football", league.id, 2026) == []


def test_jobs_lifecycle(session_factory: sessionmaker[Session]) -> None:
    with session_scope(session_factory) as s:
        jobs = JobRepository(s)
        job = jobs.get_or_create("api_football", "fixtures", "premier_league:2023")
        jobs.start(job)
        jobs.succeed(job, requests=1, items=380)
    with session_scope(session_factory) as s:
        jobs = JobRepository(s)
        assert jobs.finished_at("api_football", "fixtures", "premier_league:2023") is not None
        assert jobs.finished_at("api_football", "teams", "premier_league:2023") is None
        job = jobs.get_or_create("api_football", "fixtures", "premier_league:2023")
        assert (job.attempts, job.items_loaded) == (1, 380)
