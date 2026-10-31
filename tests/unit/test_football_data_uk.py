"""football-data.co.uk CSV parser. The CSV below is SYNTHETIC (format per notes.txt)."""

from __future__ import annotations

from datetime import UTC, datetime

from football_ai.data.collectors.football_data_uk.parser import (
    decode,
    parse_season_csv,
    season_code,
    stable_id,
)
from football_ai.domain.models import FixtureStatusGroup, OddsType

CSV_WITH_TIME = (
    "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,Referee,HS,AS,HST,AST,HF,AF,HC,AC,"
    "HY,AY,HR,AR,B365H,B365D,B365A,PSH,PSD,PSA,AvgH,AvgD,AvgA,B365>2.5,B365<2.5,B365CH,B365CD,B365CA,"
    "PSCH,PSCD,PSCA,AvgC>2.5,AvgC<2.5\n"
    "E0,16/08/2024,20:00,Alpha FC,Beta United,1,0,H,0,0,D,A Ref,14,8,5,2,10,12,7,3,1,2,0,0,"
    "1.50,4.50,6.00,1.52,4.60,6.20,1.49,4.40,5.90,1.70,2.10,1.45,4.75,6.50,1.47,4.80,6.80,1.80,2.00\n"
    "E0,17/08/2024,15:00,Gamma,Delta,2,2,D,1,1,D,B Ref,9,11,3,4,9,8,4,6,2,1,0,1,"
    "2.80,3.30,2.60,,,,2.75,3.25,2.55,1.90,1.90,,,,,,,,\n"
    "E0,17/08/2024,15:00,Gamma,Delta,2,2,D,1,1,D,B Ref,9,11,3,4,9,8,4,6,2,1,0,1,"
    "2.80,3.30,2.60,,,,2.75,3.25,2.55,1.90,1.90,,,,,,,,\n"
    "E0,18/08/2024,15:00,,Delta,1,0,H,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,\n"
    ",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,\n"
)

CSV_OLD_FORMAT = (
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,BbAvH,BbAvD,BbAvA,BbMx>2.5\n"
    "D1,12/08/06,Alpha,Beta,3,1,H,1.80,3.40,4.20,2.05\n"
)


def test_season_code_and_stable_ids() -> None:
    assert season_code(2024) == "2425"
    assert season_code(1999) == "9900"
    assert stable_id("a", "B") == stable_id("A", "b ")
    assert 0 < stable_id("x") < 2**60


def test_parse_recent_csv() -> None:
    parsed = parse_season_csv(
        CSV_WITH_TIME,
        division="E0",
        season=2024,
        competition_key="premier_league",
        league_name="Premier League",
    )
    assert parsed.rows == 4
    assert len(parsed.fixtures) == 2
    first = parsed.fixtures[0]
    # 20:00 London in August = 19:00 UTC (BST)
    assert first.kickoff_utc == datetime(2024, 8, 16, 19, 0, tzinfo=UTC)
    assert first.kickoff_time_known is True
    assert first.status_group is FixtureStatusGroup.FINISHED
    assert (first.goals.home, first.goals.away, first.score_halftime.home) == (1, 0, 0)
    assert first.referee == "A Ref"
    # duplicate row + empty row reported
    assert any("duplicate" in i.message for i in parsed.issues)
    assert any(i.field == "row" and "missing" in i.message for i in parsed.issues)

    stats = [s for s in parsed.statistics if s.fixture_provider_id == first.provider_id]
    assert stats[0].values["shots_total"] == 14 and stats[1].values["corners"] == 3

    odds = [q for q in parsed.odds if q.fixture_provider_id == first.provider_id]
    by_key = {(q.bookmaker, q.market, q.selection, q.odds_type): q.odd for q in odds}
    assert by_key[("Bet365", "1x2", "home", OddsType.PRE_MATCH)] == 1.50
    assert by_key[("Bet365", "1x2", "home", OddsType.CLOSING)] == 1.45
    assert by_key[("Pinnacle", "1x2", "away", OddsType.CLOSING)] == 6.80
    assert by_key[("market_average", "1x2", "draw", OddsType.MARKET_AVERAGE)] == 4.40
    assert by_key[("market_average_closing", "over_under_2.5", "over", OddsType.CLOSING)] == 1.80
    assert by_key[("Bet365", "over_under_2.5", "under", OddsType.PRE_MATCH)] == 2.10


def test_parse_old_csv_without_time() -> None:
    parsed = parse_season_csv(
        CSV_OLD_FORMAT,
        division="D1",
        season=2006,
        competition_key="bundesliga",
        league_name="Bundesliga",
    )
    fixture = parsed.fixtures[0]
    assert fixture.kickoff_time_known is False
    assert fixture.kickoff_utc.date().isoformat() in {"2006-08-11", "2006-08-12"}
    types = {(q.odds_type, q.market) for q in parsed.odds}
    assert (OddsType.MARKET_AVERAGE, "1x2") in types
    assert (OddsType.MARKET_MAX, "over_under_2.5") in types


def test_decode_handles_bom_and_latin1() -> None:
    assert decode("﻿Div".encode()) == "Div"
    assert decode("Mönchengladbach".encode("cp1252")) == "Mönchengladbach"
