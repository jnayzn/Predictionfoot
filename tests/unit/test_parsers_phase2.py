from __future__ import annotations

from football_ai.cli.common import parse_seasons
from football_ai.data.collectors.api_football.parsers import (
    parse_fixture_details,
    parse_injury,
    parse_odds,
    parse_standings,
    parse_team,
    statistic_key,
)
from football_ai.domain.models import OddsType
from tests.payloads import details_item, injury_item, odds_item, standings_item, team_item


def test_parse_team() -> None:
    team = parse_team(team_item(33, "Alpha"))
    assert (team.provider_id, team.name, team.code, team.venue_capacity) == (
        33,
        "Alpha",
        "ALP",
        40000,
    )


def test_parse_standings() -> None:
    rows = parse_standings(standings_item(39, 2026, [(33, "Alpha", 1, 20), (34, "Beta", 2, 18)]))
    assert [(r.team.provider_id, r.rank, r.points) for r in rows] == [(33, 1, 20), (34, 2, 18)]
    assert rows[0].home["win"] == 4 and rows[0].goals_for == 18
    assert rows[0].source_updated_at is not None


def test_parse_details_statistics_lineups_players() -> None:
    d = parse_fixture_details(details_item(500, 39, (33, "Alpha"), (34, "Beta")))
    assert d.fixture.provider_id == 500
    home_stats = d.statistics[0].values
    assert home_stats["expected_goals"] == 1.85
    assert home_stats["possession_pct"] == 55.0
    assert home_stats["passes_pct"] == 81.0
    assert home_stats["corners"] is None  # null stays null
    assert len(d.lineups) == 2
    assert d.lineups[0].formation == "4-3-3"
    # 11 starters + 1 identified substitute; the player without id is skipped
    assert len(d.lineups[0].players) == 12
    assert sum(p.is_starter for p in d.lineups[0].players) == 11
    scorer = d.players[0]
    assert (scorer.goals, scorer.rating, scorer.captain, scorer.assists) == (2, 7.8, True, None)


def test_statistic_key_fallback() -> None:
    assert statistic_key("Shots insidebox") == "shots_inside_box"
    assert statistic_key("Some New Stat %") == "some_new_stat"


def test_parse_injury() -> None:
    injury = parse_injury(injury_item(77, (33, "Alpha"), 39, 2026, 500))
    assert (injury.player.provider_id, injury.fixture_provider_id, injury.reason) == (
        77,
        500,
        "Knee Injury",
    )


def test_parse_odds_normalizes_supported_markets_only() -> None:
    quotes = parse_odds(odds_item(900))
    keys = {(q.market, q.selection, q.odd) for q in quotes}
    assert ("1x2", "home", 2.10) in keys
    assert ("over_under_2.5", "under", 1.95) in keys
    assert ("btts", "yes", 1.75) in keys
    assert ("double_chance", "1x", 1.30) in keys
    assert len(quotes) == 8  # "Corners 1x2" ignored
    assert all(q.odds_type is OddsType.PRE_MATCH and q.source_updated_at for q in quotes)


def test_parse_seasons() -> None:
    assert parse_seasons("2021") == [2021]
    assert parse_seasons("2019-2021,2015") == [2015, 2019, 2020, 2021]
