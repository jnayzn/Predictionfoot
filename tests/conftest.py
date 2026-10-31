"""Shared test fixtures.

IMPORTANT: every payload below is SYNTHETIC test data. It reproduces the JSON
*structure* documented at https://www.api-football.com/documentation-v3 but
the values (scores, dates, team ids...) are invented for the tests and must
never be used as real football data.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from football_ai.config.competitions import CompetitionConfig, load_competitions
from football_ai.config.settings import ApiFootballSettings
from football_ai.data.collectors.api_football.client import ApiFootballClient
from football_ai.data.collectors.api_football.quota import QuotaManager
from football_ai.data.storage.raw_archive import RawArchive
from football_ai.data.storage.response_cache import ResponseCache

BASE_URL = "https://api-football.test"
TEST_KEY = "test-key-0123456789"
ROOT = Path(__file__).resolve().parents[1]


class FakeClock:
    """Controllable wall clock + monotonic clock + async sleep."""

    def __init__(self, start: float = 1_793_448_000.0) -> None:  # 2026-10-31T12:00:00Z
        self.now = start
        self.mono = 1000.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.mono

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        self.mono += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def api_settings() -> ApiFootballSettings:
    return ApiFootballSettings(
        _env_file=None,  # type: ignore[call-arg]
        key=TEST_KEY,
        base_url=BASE_URL,
        daily_limit=100,
        per_minute_limit=10,
        quota_reserve=5,
        max_retries=3,
        backoff_base_seconds=1.0,
        backoff_max_seconds=30.0,
    )


@pytest.fixture
def competitions() -> list[CompetitionConfig]:
    return load_competitions(ROOT / "configs" / "competitions.toml")


@pytest.fixture
def make_client(
    tmp_path: Path, api_settings: ApiFootballSettings, clock: FakeClock
) -> Callable[..., ApiFootballClient]:
    def _make(
        settings: ApiFootballSettings | None = None, *, with_cache: bool = True
    ) -> ApiFootballClient:
        cfg = settings or api_settings
        quota = QuotaManager(
            daily_limit=cfg.daily_limit,
            per_minute_limit=cfg.per_minute_limit,
            reserve=cfg.quota_reserve,
            state_file=tmp_path / "state" / "quota.json",
            clock=clock.time,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
        return ApiFootballClient(
            cfg,
            quota,
            cache=ResponseCache(tmp_path / "cache", clock=clock.time) if with_cache else None,
            archive=RawArchive(tmp_path / "raw", clock=clock.time),
            sleep=clock.sleep,
            rng=lambda: 0.0,
            clock=clock.time,
        )

    return _make


# ----------------------------------------------------------- payload builders
def envelope(
    endpoint: str,
    response: Any,
    *,
    params: dict[str, Any] | None = None,
    errors: Any = None,
    paging: tuple[int, int] = (1, 1),
) -> dict[str, Any]:
    return {
        "get": endpoint,
        "parameters": params or {},
        "errors": errors if errors is not None else [],
        "results": len(response) if isinstance(response, list) else 1,
        "paging": {"current": paging[0], "total": paging[1]},
        "response": response,
    }


def league_item(
    league_id: int,
    name: str,
    country: str,
    league_type: str = "League",
    years: range = range(2010, 2027),
) -> dict[str, Any]:
    return {
        "league": {
            "id": league_id,
            "name": name,
            "type": league_type,
            "logo": f"https://x/{league_id}.png",
        },
        "country": {"name": country, "code": None, "flag": None},
        "seasons": [
            {
                "year": y,
                "start": f"{y}-08-01",
                "end": f"{y + 1}-05-31",
                "current": y == years[-1],
                "coverage": {
                    "fixtures": {
                        "events": True,
                        "lineups": True,
                        "statistics_fixtures": y >= 2015,
                        "statistics_players": y >= 2016,
                    },
                    "standings": True,
                    "players": True,
                    "injuries": y >= 2020,
                    "predictions": True,
                    "odds": y >= 2022,
                },
            }
            for y in years
        ],
    }


def all_leagues_payload() -> dict[str, Any]:
    items = [
        league_item(39, "Premier League", "England"),
        league_item(140, "La Liga", "Spain"),
        league_item(78, "Bundesliga", "Germany"),
        league_item(135, "Serie A", "Italy"),
        league_item(61, "Ligue 1", "France"),
        league_item(2, "UEFA Champions League", "World", "Cup"),
        league_item(40, "Championship", "England"),
        league_item(71, "Serie A", "Brazil"),
    ]
    return envelope("leagues", items)


def fixture_item(
    fixture_id: int,
    league_id: int,
    home: tuple[int, str],
    away: tuple[int, str],
    *,
    date: str = "2026-10-31T15:00:00+00:00",
    status: str = "NS",
    goals: tuple[int | None, int | None] = (None, None),
    season: int = 2026,
    league_name: str = "League",
) -> dict[str, Any]:
    return {
        "fixture": {
            "id": fixture_id,
            "referee": None,
            "timezone": "UTC",
            "date": date,
            "timestamp": 0,
            "venue": {"id": 1, "name": "Synthetic Stadium", "city": "Testville"},
            "status": {"long": "status", "short": status, "elapsed": None},
        },
        "league": {
            "id": league_id,
            "name": league_name,
            "country": "Testland",
            "season": season,
            "round": "Regular Season - 10",
        },
        "teams": {
            "home": {"id": home[0], "name": home[1], "logo": None, "winner": None},
            "away": {"id": away[0], "name": away[1], "logo": None, "winner": None},
        },
        "goals": {"home": goals[0], "away": goals[1]},
        "score": {
            "halftime": {"home": None, "away": None},
            "fulltime": {"home": goals[0], "away": goals[1]},
            "extratime": {"home": None, "away": None},
            "penalty": {"home": None, "away": None},
        },
    }


def quota_headers(remaining: int, limit: int = 100) -> dict[str, str]:
    return {
        "x-ratelimit-requests-limit": str(limit),
        "x-ratelimit-requests-remaining": str(remaining),
        "X-RateLimit-Limit": "10",
        "X-RateLimit-Remaining": "9",
    }
