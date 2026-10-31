from __future__ import annotations

from datetime import UTC, datetime

import pytest

from football_ai.data.collectors.api_football.parsers import ParseError, parse_fixture, parse_league
from football_ai.data.validators.fixtures import validate_fixture
from football_ai.data.validators.issues import Severity
from football_ai.domain.models import FixtureStatusGroup
from tests.conftest import fixture_item, league_item

NOW = datetime(2026, 10, 31, 12, 0, tzinfo=UTC)


def test_parse_fixture_full() -> None:
    raw = fixture_item(
        42,
        39,
        (33, "Home"),
        (34, "Away"),
        date="2026-10-31T17:30:00+02:00",
        status="FT",
        goals=(3, 0),
    )
    f = parse_fixture(raw, competition_key="premier_league")
    assert f.provider == "api_football" and f.provider_id == 42
    assert f.kickoff_utc == datetime(2026, 10, 31, 15, 30, tzinfo=UTC)
    assert f.status_group is FixtureStatusGroup.FINISHED
    assert f.score_fulltime.home == 3
    assert f.venue_name == "Synthetic Stadium"
    assert f.referee is None  # missing data stays missing
    assert f.season == 2026 and f.round == "Regular Season - 10"


@pytest.mark.parametrize(
    ("status", "group"),
    [
        ("NS", FixtureStatusGroup.SCHEDULED),
        ("HT", FixtureStatusGroup.LIVE),
        ("PEN", FixtureStatusGroup.FINISHED),
        ("PST", FixtureStatusGroup.POSTPONED),
        ("CANC", FixtureStatusGroup.CANCELLED),
        ("ABD", FixtureStatusGroup.ABANDONED),
        ("AWD", FixtureStatusGroup.AWARDED),
        ("XYZ", FixtureStatusGroup.UNKNOWN),
    ],
)
def test_status_groups(status: str, group: FixtureStatusGroup) -> None:
    raw = fixture_item(1, 39, (1, "a"), (2, "b"), status=status)
    assert parse_fixture(raw).status_group is group


def test_parse_fixture_naive_date_rejected() -> None:
    raw = fixture_item(1, 39, (1, "a"), (2, "b"), date="2026-10-31T15:00:00")
    with pytest.raises(ParseError):
        parse_fixture(raw)


def test_parse_fixture_missing_team_id_rejected() -> None:
    raw = fixture_item(1, 39, (1, "a"), (2, "b"))
    raw["teams"]["home"]["id"] = None
    with pytest.raises(ParseError):
        parse_fixture(raw)


def test_parse_fixture_timestamp_fallback() -> None:
    raw = fixture_item(1, 39, (1, "a"), (2, "b"))
    raw["fixture"]["date"] = None
    raw["fixture"]["timestamp"] = 1_793_448_000
    assert parse_fixture(raw).kickoff_utc == NOW


def test_parse_league_sorts_seasons_and_flattens_coverage() -> None:
    raw = league_item(39, "Premier League", "England", years=range(2020, 2023))
    raw["seasons"].reverse()
    comp = parse_league(raw, key="premier_league")
    assert [s.year for s in comp.seasons] == [2020, 2021, 2022]
    assert comp.seasons[0].coverage["fixtures.events"] is True
    assert comp.current_season is not None and comp.current_season.year == 2022


def test_validator_accepts_clean_fixture() -> None:
    f = parse_fixture(fixture_item(1, 39, (1, "a"), (2, "b")))
    assert validate_fixture(f, now=NOW) == []


def test_validator_flags_finished_without_goals() -> None:
    f = parse_fixture(
        fixture_item(1, 39, (1, "a"), (2, "b"), status="FT", date="2026-10-30T15:00:00+00:00")
    )
    issues = validate_fixture(f, now=NOW)
    assert [(i.field, i.severity) for i in issues] == [("goals", Severity.ERROR)]


def test_validator_flags_finished_in_future() -> None:
    f = parse_fixture(
        fixture_item(
            1, 39, (1, "a"), (2, "b"), status="FT", goals=(1, 1), date="2026-12-01T15:00:00+00:00"
        )
    )
    assert any(i.field == "kickoff_utc" for i in validate_fixture(f, now=NOW))


def test_validator_warns_scheduled_with_goals() -> None:
    f = parse_fixture(fixture_item(1, 39, (1, "a"), (2, "b"), status="NS", goals=(0, 0)))
    issues = validate_fixture(f, now=NOW)
    assert [i.severity for i in issues] == [Severity.WARNING]


def test_validator_negative_goals() -> None:
    f = parse_fixture(fixture_item(1, 39, (1, "a"), (2, "b"), status="FT", goals=(-1, 0)))
    assert any(i.field == "goals.home" for i in validate_fixture(f, now=NOW))
