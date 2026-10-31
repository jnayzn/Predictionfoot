"""SYNTHETIC API-Football payload items for Phase 2 endpoints (structure per the v3 docs,
values invented for tests - never real football data)."""

from __future__ import annotations

from typing import Any

from tests.conftest import fixture_item


def team_item(team_id: int, name: str) -> dict[str, Any]:
    return {
        "team": {
            "id": team_id,
            "name": name,
            "code": name[:3].upper(),
            "country": "Testland",
            "founded": 1900,
            "national": False,
            "logo": f"https://x/t{team_id}.png",
        },
        "venue": {
            "id": team_id * 10,
            "name": f"{name} Park",
            "city": "Testville",
            "capacity": 40000,
        },
    }


def standings_item(
    league_id: int, season: int, rows: list[tuple[int, str, int, int]]
) -> dict[str, Any]:
    """rows: (team_id, name, rank, points)."""
    return {
        "league": {
            "id": league_id,
            "name": "League",
            "country": "Testland",
            "season": season,
            "standings": [
                [
                    {
                        "rank": rank,
                        "team": {"id": tid, "name": name, "logo": None},
                        "points": pts,
                        "goalsDiff": 3,
                        "group": "Regular Season",
                        "form": "WWDLW",
                        "status": "same",
                        "description": None,
                        "all": {
                            "played": 10,
                            "win": 6,
                            "draw": 2,
                            "lose": 2,
                            "goals": {"for": 18, "against": 15},
                        },
                        "home": {
                            "played": 5,
                            "win": 4,
                            "draw": 1,
                            "lose": 0,
                            "goals": {"for": 10, "against": 4},
                        },
                        "away": {
                            "played": 5,
                            "win": 2,
                            "draw": 1,
                            "lose": 2,
                            "goals": {"for": 8, "against": 11},
                        },
                        "update": "2026-10-30T00:00:00+00:00",
                    }
                    for tid, name, rank, pts in rows
                ]
            ],
        }
    }


def details_item(
    fixture_id: int,
    league_id: int,
    home: tuple[int, str],
    away: tuple[int, str],
    goals: tuple[int, int] = (2, 1),
    date: str = "2026-10-20T15:00:00+00:00",
) -> dict[str, Any]:
    item = fixture_item(fixture_id, league_id, home, away, status="FT", goals=goals, date=date)
    return add_details(item, home, away)


def add_details(
    item: dict[str, Any], home: tuple[int, str], away: tuple[int, str]
) -> dict[str, Any]:
    """Add statistics / lineups / players blocks to a fixture item (``/fixtures?ids=``)."""
    item = dict(item)

    def stats(xg: str, poss: str) -> list[dict[str, Any]]:
        return [
            {"type": "Shots on Goal", "value": 5},
            {"type": "Total Shots", "value": 12},
            {"type": "Ball Possession", "value": poss},
            {"type": "Corner Kicks", "value": None},
            {"type": "expected_goals", "value": xg},
            {"type": "Passes %", "value": "81%"},
        ]

    item["statistics"] = [
        {
            "team": {"id": home[0], "name": home[1], "logo": None},
            "statistics": stats("1.85", "55%"),
        },
        {
            "team": {"id": away[0], "name": away[1], "logo": None},
            "statistics": stats("0.92", "45%"),
        },
    ]

    def lineup(team: tuple[int, str], base: int) -> dict[str, Any]:
        return {
            "team": {"id": team[0], "name": team[1], "logo": None},
            "coach": {"id": base, "name": f"Coach {team[1]}"},
            "formation": "4-3-3",
            "startXI": [
                {
                    "player": {
                        "id": base + i,
                        "name": f"P{base + i}",
                        "number": i,
                        "pos": "M",
                        "grid": f"2:{i}",
                    }
                }
                for i in range(1, 12)
            ],
            "substitutes": [
                {
                    "player": {
                        "id": base + 50,
                        "name": f"P{base + 50}",
                        "number": 20,
                        "pos": "F",
                        "grid": None,
                    }
                },
                {"player": {"id": None, "name": "Unknown", "number": 30, "pos": "G", "grid": None}},
            ],
        }

    item["lineups"] = [lineup(home, home[0] * 1000), lineup(away, away[0] * 1000)]
    item["players"] = [
        {
            "team": {"id": home[0], "name": home[1], "logo": None},
            "players": [
                {
                    "player": {"id": home[0] * 1000 + 1, "name": "Scorer"},
                    "statistics": [
                        {
                            "games": {
                                "minutes": 90,
                                "number": 9,
                                "position": "F",
                                "rating": "7.8",
                                "captain": True,
                                "substitute": False,
                            },
                            "shots": {"total": 4, "on": 3},
                            "goals": {"total": 2, "conceded": 0, "assists": None, "saves": None},
                            "passes": {"total": 30, "key": 2, "accuracy": "24"},
                            "tackles": {"total": None},
                            "cards": {"yellow": 0, "red": 0},
                        }
                    ],
                }
            ],
        }
    ]
    return item


def injury_item(
    player_id: int, team: tuple[int, str], league_id: int, season: int, fixture_id: int | None
) -> dict[str, Any]:
    return {
        "player": {
            "id": player_id,
            "name": f"Injured {player_id}",
            "type": "Missing Fixture",
            "reason": "Knee Injury",
        },
        "team": {"id": team[0], "name": team[1], "logo": None},
        "fixture": {
            "id": fixture_id,
            "timezone": "UTC",
            "date": "2026-10-20T15:00:00+00:00",
            "timestamp": 0,
        },
        "league": {"id": league_id, "season": season, "name": "League"},
    }


def odds_item(fixture_id: int) -> dict[str, Any]:
    return {
        "fixture": {"id": fixture_id, "timezone": "UTC", "date": "2026-11-01T15:00:00+00:00"},
        "update": "2026-10-31T09:00:00+00:00",
        "bookmakers": [
            {
                "id": 8,
                "name": "Bet365",
                "bets": [
                    {
                        "id": 1,
                        "name": "Match Winner",
                        "values": [
                            {"value": "Home", "odd": "2.10"},
                            {"value": "Draw", "odd": "3.40"},
                            {"value": "Away", "odd": "3.60"},
                        ],
                    },
                    {
                        "id": 5,
                        "name": "Goals Over/Under",
                        "values": [
                            {"value": "Over 2.5", "odd": "1.90"},
                            {"value": "Under 2.5", "odd": "1.95"},
                        ],
                    },
                    {
                        "id": 8,
                        "name": "Both Teams Score",
                        "values": [{"value": "Yes", "odd": "1.75"}, {"value": "No", "odd": "2.05"}],
                    },
                    {
                        "id": 12,
                        "name": "Double Chance",
                        "values": [{"value": "Home/Draw", "odd": "1.30"}],
                    },
                    {"id": 99, "name": "Corners 1x2", "values": [{"value": "Home", "odd": "1.5"}]},
                ],
            }
        ],
    }
