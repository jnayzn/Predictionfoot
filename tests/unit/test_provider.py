"""Tests of the API-Football provider: competition discovery and fixtures by date."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any

import httpx
import pytest
import respx

from football_ai.config.competitions import CompetitionConfig
from football_ai.config.settings import ApiFootballSettings
from football_ai.core.exceptions import ConfigurationError
from football_ai.data.collectors.api_football.client import ApiFootballClient
from football_ai.data.collectors.api_football.provider import ApiFootballProvider
from football_ai.data.collectors.base import FootballDataProvider
from football_ai.domain.models import FixtureStatusGroup
from tests.conftest import BASE_URL, all_leagues_payload, envelope, fixture_item, league_item

NOW = datetime(2026, 10, 31, 12, 0, tzinfo=UTC)
EXPECTED_KEYS = {
    "premier_league",
    "la_liga",
    "bundesliga",
    "serie_a",
    "ligue_1",
    "champions_league",
}


def make_provider(
    client: ApiFootballClient,
    competitions: list[CompetitionConfig],
    settings: ApiFootballSettings,
) -> ApiFootballProvider:
    return ApiFootballProvider(client, competitions, settings, now=lambda: NOW)


def test_provider_implements_protocol(
    make_client: Callable[..., ApiFootballClient],
    competitions: list[CompetitionConfig],
    api_settings: ApiFootballSettings,
) -> None:
    assert isinstance(
        make_provider(make_client(), competitions, api_settings), FootballDataProvider
    )


# ------------------------------------------------------------------ discovery
@respx.mock
async def test_discovers_six_competitions_with_one_request(
    make_client: Callable[..., ApiFootballClient],
    competitions: list[CompetitionConfig],
    api_settings: ApiFootballSettings,
) -> None:
    route = respx.get(f"{BASE_URL}/leagues").mock(
        return_value=httpx.Response(200, json=all_leagues_payload())
    )
    async with make_client() as client:
        provider = make_provider(client, competitions, api_settings)
        result = await provider.discover_competitions()
        again = await provider.discover_competitions()

    assert route.call_count == 1  # second discovery served from cache
    assert again.from_cache is True
    assert {c.key for c in result.competitions} == EXPECTED_KEYS
    assert result.missing == []
    assert result.issues == []
    pl = next(c for c in result.competitions if c.key == "premier_league")
    assert pl.provider_id == 39 and pl.country == "England"
    assert pl.current_season is not None and pl.current_season.year == 2026
    assert [s.year for s in pl.seasons] == list(range(2010, 2027))
    season_2023 = next(s for s in pl.seasons if s.year == 2023)
    assert season_2023.coverage["fixtures.statistics_fixtures"] is True
    assert season_2023.coverage["injuries"] is True
    ucl = next(c for c in result.competitions if c.key == "champions_league")
    assert ucl.type == "Cup"


@respx.mock
async def test_missing_competition_is_reported(
    make_client: Callable[..., ApiFootballClient],
    competitions: list[CompetitionConfig],
    api_settings: ApiFootballSettings,
) -> None:
    payload = all_leagues_payload()
    payload["response"] = [i for i in payload["response"] if i["league"]["id"] != 61]
    respx.get(f"{BASE_URL}/leagues").mock(return_value=httpx.Response(200, json=payload))
    async with make_client() as client:
        result = await make_provider(client, competitions, api_settings).discover_competitions()
    assert result.missing == ["ligue_1"]
    assert len(result.competitions) == 5


@respx.mock
async def test_id_change_falls_back_to_name_and_country(
    make_client: Callable[..., ApiFootballClient],
    competitions: list[CompetitionConfig],
    api_settings: ApiFootballSettings,
) -> None:
    payload = all_leagues_payload()
    payload["response"] = [i for i in payload["response"] if i["league"]["id"] != 78]
    payload["response"].append(league_item(9078, "Bundesliga", "Germany"))
    respx.get(f"{BASE_URL}/leagues").mock(return_value=httpx.Response(200, json=payload))
    async with make_client() as client:
        result = await make_provider(client, competitions, api_settings).discover_competitions()
    bundesliga = next(c for c in result.competitions if c.key == "bundesliga")
    assert bundesliga.provider_id == 9078
    assert any(i.field == "provider_id" for i in result.issues)


@respx.mock
async def test_country_mismatch_is_rejected(
    make_client: Callable[..., ApiFootballClient],
    competitions: list[CompetitionConfig],
    api_settings: ApiFootballSettings,
) -> None:
    payload = all_leagues_payload()
    payload["response"] = [i for i in payload["response"] if i["league"]["id"] != 135]
    payload["response"].append(league_item(135, "Serie A", "Brazil"))  # wrong mapping
    respx.get(f"{BASE_URL}/leagues").mock(return_value=httpx.Response(200, json=payload))
    async with make_client() as client:
        result = await make_provider(client, competitions, api_settings).discover_competitions()
    assert "serie_a" in result.missing
    assert any(i.field == "country" for i in result.issues)


# ------------------------------------------------------------------- fixtures
def _day_payload() -> dict[str, Any]:
    items = [
        fixture_item(1001, 39, (33, "Team A"), (34, "Team B"), date="2026-10-31T15:00:00+00:00"),
        fixture_item(
            1002,
            61,
            (85, "Team C"),
            (91, "Team D"),
            date="2026-10-31T10:00:00+00:00",
            status="FT",
            goals=(2, 1),
        ),
        fixture_item(1003, 40, (60, "Other 1"), (61, "Other 2")),  # Championship: not covered
        fixture_item(1004, 71, (120, "Br 1"), (121, "Br 2")),  # Brazilian Serie A: not covered
        fixture_item(1001, 39, (33, "Team A"), (34, "Team B"), date="2026-10-31T15:00:00+00:00"),
        fixture_item(1005, 135, (489, "Same"), (489, "Same")),  # invalid: same team
        fixture_item(1006, 140, (529, "Team E"), (530, "Team F"), status="FT"),  # FT w/o goals
        {"fixture": {"id": 1007}, "league": {"id": 78}},  # malformed
        fixture_item(
            1008,
            2,
            (50, "Team G"),
            (541, "Team H"),
            date="2026-10-31T21:00:00+01:00",
            status="1H",
            goals=(0, 0),
        ),
    ]
    return envelope("fixtures", items, params={"date": "2026-10-31"})


@respx.mock
async def test_fixtures_by_date_single_request_filtered_and_validated(
    make_client: Callable[..., ApiFootballClient],
    competitions: list[CompetitionConfig],
    api_settings: ApiFootballSettings,
) -> None:
    route = respx.get(f"{BASE_URL}/fixtures").mock(
        return_value=httpx.Response(200, json=_day_payload())
    )
    async with make_client() as client:
        batch = await make_provider(client, competitions, api_settings).get_fixtures_by_date(
            date(2026, 10, 31)
        )

    assert route.call_count == 1
    params = route.calls.last.request.url.params
    assert params["date"] == "2026-10-31" and params["timezone"] == "UTC"
    assert [f.provider_id for f in batch.fixtures] == [1002, 1001, 1008]  # sorted by kick-off
    assert batch.total_received == 9
    assert batch.other_competitions_skipped == 2
    assert batch.duplicates_removed == 1
    assert batch.rejected == 3

    ft = next(f for f in batch.fixtures if f.provider_id == 1002)
    assert ft.competition_key == "ligue_1"
    assert ft.status_group is FixtureStatusGroup.FINISHED
    assert (ft.goals.home, ft.goals.away) == (2, 1)
    live = next(f for f in batch.fixtures if f.provider_id == 1008)
    assert live.kickoff_utc == datetime(2026, 10, 31, 20, 0, tzinfo=UTC)  # converted to UTC
    assert live.status_group is FixtureStatusGroup.LIVE


@respx.mock
async def test_fixtures_league_filter(
    make_client: Callable[..., ApiFootballClient],
    competitions: list[CompetitionConfig],
    api_settings: ApiFootballSettings,
) -> None:
    respx.get(f"{BASE_URL}/fixtures").mock(return_value=httpx.Response(200, json=_day_payload()))
    async with make_client() as client:
        batch = await make_provider(client, competitions, api_settings).get_fixtures_by_date(
            date(2026, 10, 31), ["premier_league"]
        )
    assert [f.provider_id for f in batch.fixtures] == [1001]


async def test_unknown_league_key_rejected(
    make_client: Callable[..., ApiFootballClient],
    competitions: list[CompetitionConfig],
    api_settings: ApiFootballSettings,
) -> None:
    async with make_client() as client:
        provider = make_provider(client, competitions, api_settings)
        with pytest.raises(ConfigurationError):
            await provider.get_fixtures_by_date(date(2026, 10, 31), ["eredivisie"])


@respx.mock
async def test_past_dates_use_long_cache_and_recent_dates_short_cache(
    make_client: Callable[..., ApiFootballClient],
    competitions: list[CompetitionConfig],
    api_settings: ApiFootballSettings,
) -> None:
    respx.get(f"{BASE_URL}/fixtures").mock(
        return_value=httpx.Response(200, json=envelope("fixtures", []))
    )
    async with make_client() as client:
        provider = make_provider(client, competitions, api_settings)
        assert provider._fixtures_ttl(date(2026, 9, 1)) == api_settings.cache_ttl_fixtures_past
        assert provider._fixtures_ttl(date(2026, 10, 31)) == api_settings.cache_ttl_fixtures_recent
        assert provider._fixtures_ttl(date(2026, 11, 2)) == api_settings.cache_ttl_fixtures_recent
