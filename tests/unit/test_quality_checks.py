from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from football_ai.data.quality.checks import run_checks

NOW = datetime(2026, 10, 31, tzinfo=UTC)


def frames(
    matches: list[dict], stats: list[dict] | None = None, odds: list[dict] | None = None
) -> dict:  # type: ignore[type-arg]
    base = {
        "competition_key": "premier_league",
        "season_year": 2026,
        "status_group": "finished",
        "home_goals": 1,
        "away_goals": 0,
        "source_conflict": False,
        "valid_for_training": True,
        "exclusion_reason": None,
        "sources": ["api_football"],
    }
    m = pd.DataFrame([{**base, **row} for row in matches])
    st = pd.DataFrame(
        stats or [],
        columns=[
            "match_id",
            "fixture_id",
            "team_id",
            "provider",
            "shots_total",
            "shots_on_goal",
            "possession_pct",
            "expected_goals",
        ],
    )
    od = pd.DataFrame(
        odds or [],
        columns=[
            "match_id",
            "fixture_id",
            "provider",
            "bookmaker",
            "market",
            "selection",
            "odds_type",
            "odd",
        ],
    )
    return {"matches": m, "stats": st, "lineups": pd.DataFrame(columns=["match_id"]), "odds": od}


def checks(report) -> dict[str, int]:  # type: ignore[no-untyped-def]
    return {f.check: f.count for f in report.findings}


def test_duplicates_double_booking_goals_stale() -> None:
    k = "2026-10-01T15:00:00+00:00"
    report = run_checks(
        frames(
            [
                {"id": 1, "kickoff_utc": k, "home_team_id": 1, "away_team_id": 2},
                {"id": 2, "kickoff_utc": k, "home_team_id": 1, "away_team_id": 2},
                {
                    "id": 3,
                    "kickoff_utc": "2026-10-10T15:00:00+00:00",
                    "home_team_id": 3,
                    "away_team_id": 4,
                    "home_goals": 15,
                },
                {
                    "id": 4,
                    "kickoff_utc": "2026-10-20T15:00:00+00:00",
                    "home_team_id": 5,
                    "away_team_id": 6,
                    "status_group": "scheduled",
                    "home_goals": None,
                    "away_goals": None,
                },
            ]
        ),
        now=NOW,
    )
    found = checks(report)
    assert found["duplicate_matches"] == 2
    assert found["team_double_booking"] == 2  # both teams of the duplicated match
    assert found["implausible_goals"] == 1
    assert found["stale_results"] == 1
    assert report.errors == 3


def test_stats_and_odds_consistency() -> None:
    report = run_checks(
        frames(
            [
                {
                    "id": 1,
                    "kickoff_utc": "2026-10-01T15:00:00+00:00",
                    "home_team_id": 1,
                    "away_team_id": 2,
                }
            ],
            stats=[
                {
                    "match_id": 1,
                    "fixture_id": 10,
                    "team_id": 1,
                    "provider": "p",
                    "shots_total": 5,
                    "shots_on_goal": 7,
                    "possession_pct": 60,
                    "expected_goals": 1.1,
                },
                {
                    "match_id": 1,
                    "fixture_id": 10,
                    "team_id": 2,
                    "provider": "p",
                    "shots_total": 9,
                    "shots_on_goal": 3,
                    "possession_pct": 60,
                    "expected_goals": 0.4,
                },
            ],
            odds=[
                {
                    "match_id": 1,
                    "fixture_id": 10,
                    "provider": "p",
                    "bookmaker": "B",
                    "market": "1x2",
                    "selection": s,
                    "odds_type": "pre_match",
                    "odd": o,
                }
                for s, o in (("home", 1.2), ("draw", 1.5), ("away", 2.0))
            ]
            + [
                {
                    "match_id": 1,
                    "fixture_id": 10,
                    "provider": "p",
                    "bookmaker": "M",
                    "market": "1x2",
                    "selection": s,
                    "odds_type": "market_max",
                    "odd": o,
                }
                for s, o in (("home", 5.0), ("draw", 5.0), ("away", 5.0))
            ],
        ),
        now=NOW,
    )
    found = checks(report)
    assert found["shots_inconsistent"] == 1
    assert found["possession_inconsistent"] == 1
    assert found["odds_overround"] == 1  # market_max is excluded from the check
    assert report.coverage[0]["pct_xg"] == 100.0


def test_incomplete_finished_season() -> None:
    rows = [
        {
            "id": 1,
            "kickoff_utc": "2024-10-01T15:00:00+00:00",
            "season_year": 2024,
            "home_team_id": 1,
            "away_team_id": 2,
        }
    ]
    found = checks(run_checks(frames(rows), now=NOW))
    assert found["incomplete_seasons"] == 1
