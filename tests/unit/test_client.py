"""Tests of the low-level API-Football client (HTTP fully mocked with respx)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
import respx

from football_ai.config.settings import ApiFootballSettings
from football_ai.core.exceptions import (
    ApiNetworkError,
    ApiRequestError,
    ApiResponseFormatError,
    ApiServerError,
    AuthenticationError,
    MissingApiKeyError,
    PlanRestrictionError,
    QuotaExceededError,
    RateLimitError,
)
from football_ai.data.collectors.api_football.client import AUTH_HEADER, ApiFootballClient
from tests.conftest import BASE_URL, TEST_KEY, FakeClock, envelope, quota_headers

ClientFactory = Callable[..., ApiFootballClient]


@respx.mock
async def test_success_sends_auth_header_and_parses(make_client: ClientFactory) -> None:
    route = respx.get(f"{BASE_URL}/fixtures").mock(
        return_value=httpx.Response(
            200, json=envelope("fixtures", [{"a": 1}]), headers=quota_headers(90)
        )
    )
    async with make_client() as client:
        resp = await client.get("fixtures", {"date": "2026-10-31", "timezone": "UTC"})
    assert route.call_count == 1
    request = route.calls.last.request
    assert request.headers[AUTH_HEADER] == TEST_KEY
    assert request.url.params["date"] == "2026-10-31"
    assert resp.response == [{"a": 1}]
    assert resp.results == 1
    assert resp.from_cache is False
    assert client.quota.snapshot().used == 10  # resynced from server headers (100 - 90)


@respx.mock
async def test_api_key_never_in_url(make_client: ClientFactory) -> None:
    route = respx.get(f"{BASE_URL}/leagues").mock(
        return_value=httpx.Response(200, json=envelope("leagues", []))
    )
    async with make_client() as client:
        await client.get("leagues")
    assert TEST_KEY not in str(route.calls.last.request.url)


async def test_missing_key_raises_before_any_network_call(
    make_client: ClientFactory, api_settings: ApiFootballSettings
) -> None:
    settings = api_settings.model_copy(update={"key": None})
    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(f"{BASE_URL}/leagues")
        async with make_client(settings) as client:
            with pytest.raises(MissingApiKeyError):
                await client.get("leagues")
        assert route.call_count == 0
        assert client.quota.snapshot().used == 0


@respx.mock
async def test_cache_hit_avoids_second_call_and_quota(
    make_client: ClientFactory, clock: FakeClock
) -> None:
    route = respx.get(f"{BASE_URL}/leagues").mock(
        return_value=httpx.Response(200, json=envelope("leagues", [{"x": 1}]))
    )
    async with make_client() as client:
        first = await client.get("leagues", cache_ttl=3600)
        second = await client.get("leagues", cache_ttl=3600)
        assert route.call_count == 1
        assert first.from_cache is False and second.from_cache is True
        assert second.response == [{"x": 1}]
        assert client.quota.snapshot().used == 1

        clock.now += 3601  # TTL expired -> new call
        third = await client.get("leagues", cache_ttl=3600)
        assert third.from_cache is False
        assert route.call_count == 2


@respx.mock
async def test_param_order_does_not_change_cache_key(make_client: ClientFactory) -> None:
    route = respx.get(f"{BASE_URL}/fixtures").mock(
        return_value=httpx.Response(200, json=envelope("fixtures", []))
    )
    async with make_client() as client:
        await client.get("fixtures", {"date": "2026-10-31", "timezone": "UTC"}, cache_ttl=60)
        await client.get("fixtures", {"timezone": "UTC", "date": "2026-10-31"}, cache_ttl=60)
    assert route.call_count == 1


@respx.mock
async def test_raw_response_is_archived(make_client: ClientFactory, tmp_path: Path) -> None:
    respx.get(f"{BASE_URL}/fixtures").mock(
        return_value=httpx.Response(200, json=envelope("fixtures", [{"id": 7}]))
    )
    async with make_client() as client:
        await client.get("fixtures", {"date": "2026-10-31"})
    files = list((tmp_path / "raw" / "api_football" / "fixtures").rglob("*.json"))
    assert len(files) == 1
    stored = json.loads(files[0].read_text())
    assert stored["payload"]["response"] == [{"id": 7}]
    assert stored["meta"]["params"] == {"date": "2026-10-31"}
    assert stored["meta"]["status_code"] == 200
    assert TEST_KEY not in files[0].read_text()


@respx.mock
async def test_retries_on_5xx_then_succeeds(make_client: ClientFactory, clock: FakeClock) -> None:
    route = respx.get(f"{BASE_URL}/fixtures").mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(500),
            httpx.Response(200, json=envelope("fixtures", [])),
        ]
    )
    async with make_client() as client:
        resp = await client.get("fixtures")
    assert route.call_count == 3
    assert resp.response == []
    assert clock.sleeps == [1.0, 2.0]  # exponential backoff (jitter disabled in tests)


@respx.mock
async def test_5xx_exhausts_retries(make_client: ClientFactory) -> None:
    route = respx.get(f"{BASE_URL}/fixtures").mock(return_value=httpx.Response(502))
    async with make_client() as client:
        with pytest.raises(ApiServerError):
            await client.get("fixtures")
    assert route.call_count == 4  # 1 + max_retries(3)


@respx.mock
async def test_429_honours_retry_after(make_client: ClientFactory, clock: FakeClock) -> None:
    respx.get(f"{BASE_URL}/fixtures").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "7"}),
            httpx.Response(200, json=envelope("fixtures", [])),
        ]
    )
    async with make_client() as client:
        await client.get("fixtures")
    assert clock.sleeps == [7.0]


@respx.mock
async def test_429_exhausted_raises_rate_limit(make_client: ClientFactory) -> None:
    respx.get(f"{BASE_URL}/fixtures").mock(return_value=httpx.Response(429))
    async with make_client() as client:
        with pytest.raises(RateLimitError):
            await client.get("fixtures")


@respx.mock
async def test_network_error_retried_then_raised(make_client: ClientFactory) -> None:
    route = respx.get(f"{BASE_URL}/fixtures").mock(side_effect=httpx.ConnectTimeout("boom"))
    async with make_client() as client:
        with pytest.raises(ApiNetworkError):
            await client.get("fixtures")
    assert route.call_count == 4


@respx.mock
async def test_network_error_then_success(make_client: ClientFactory) -> None:
    respx.get(f"{BASE_URL}/fixtures").mock(
        side_effect=[httpx.ConnectError("down"), httpx.Response(200, json=envelope("fixtures", []))]
    )
    async with make_client() as client:
        resp = await client.get("fixtures")
    assert resp.response == []


@pytest.mark.parametrize("status", [401, 403])
@respx.mock
async def test_http_auth_error_not_retried(make_client: ClientFactory, status: int) -> None:
    route = respx.get(f"{BASE_URL}/fixtures").mock(return_value=httpx.Response(status))
    async with make_client() as client:
        with pytest.raises(AuthenticationError):
            await client.get("fixtures")
    assert route.call_count == 1


@respx.mock
async def test_http_404_not_retried(make_client: ClientFactory) -> None:
    route = respx.get(f"{BASE_URL}/nope").mock(return_value=httpx.Response(404))
    async with make_client() as client:
        with pytest.raises(ApiRequestError) as excinfo:
            await client.get("nope")
    assert excinfo.value.status_code == 404
    assert route.call_count == 1


@pytest.mark.parametrize(
    ("errors", "expected"),
    [
        ({"token": "Error/Missing application key."}, AuthenticationError),
        ({"requests": "You have reached the request limit for the day"}, QuotaExceededError),
        ({"plan": "Free plans do not have access to this season"}, PlanRestrictionError),
        ({"date": "The Date field must contain a valid date."}, ApiRequestError),
    ],
)
@respx.mock
async def test_api_level_errors_in_http_200(
    make_client: ClientFactory, errors: dict[str, str], expected: type[Exception]
) -> None:
    route = respx.get(f"{BASE_URL}/fixtures").mock(
        return_value=httpx.Response(200, json=envelope("fixtures", [], errors=errors))
    )
    async with make_client() as client:
        with pytest.raises(expected):
            await client.get("fixtures", cache_ttl=60)
        assert route.call_count == 1
        # error responses must never be cached
        with pytest.raises(expected):
            await client.get("fixtures", cache_ttl=60)
    assert route.call_count == 2 or expected is QuotaExceededError


@respx.mock
async def test_daily_limit_error_marks_quota_exhausted(make_client: ClientFactory) -> None:
    route = respx.get(f"{BASE_URL}/fixtures").mock(
        return_value=httpx.Response(
            200, json=envelope("fixtures", [], errors={"requests": "limit reached"})
        )
    )
    async with make_client() as client:
        with pytest.raises(QuotaExceededError):
            await client.get("fixtures")
        # Next call is refused locally, without hitting the network.
        with pytest.raises(QuotaExceededError):
            await client.get("leagues")
    assert route.call_count == 1


@respx.mock
async def test_api_rate_limit_error_is_retried(make_client: ClientFactory) -> None:
    route = respx.get(f"{BASE_URL}/fixtures").mock(
        side_effect=[
            httpx.Response(
                200, json=envelope("fixtures", [], errors={"rateLimit": "Too many requests"})
            ),
            httpx.Response(200, json=envelope("fixtures", [{"ok": True}])),
        ]
    )
    async with make_client() as client:
        resp = await client.get("fixtures")
    assert route.call_count == 2
    assert resp.response == [{"ok": True}]


@respx.mock
async def test_invalid_json_raises_format_error(make_client: ClientFactory) -> None:
    respx.get(f"{BASE_URL}/fixtures").mock(return_value=httpx.Response(200, text="<html>"))
    async with make_client() as client:
        with pytest.raises(ApiResponseFormatError):
            await client.get("fixtures")


@respx.mock
async def test_missing_response_field_raises_format_error(make_client: ClientFactory) -> None:
    respx.get(f"{BASE_URL}/fixtures").mock(return_value=httpx.Response(200, json={"errors": []}))
    async with make_client() as client:
        with pytest.raises(ApiResponseFormatError):
            await client.get("fixtures")


@respx.mock
async def test_concurrent_identical_requests_are_deduplicated(make_client: ClientFactory) -> None:
    async def slow_response(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.01)
        return httpx.Response(200, json=envelope("leagues", [{"id": 1}]))

    route = respx.get(f"{BASE_URL}/leagues").mock(side_effect=slow_response)
    async with make_client() as client:
        results = await asyncio.gather(*(client.get("leagues") for _ in range(5)))
    assert route.call_count == 1
    assert all(r.response == [{"id": 1}] for r in results)
    assert client.quota.snapshot().used == 1


@respx.mock
async def test_pagination_follows_all_pages(make_client: ClientFactory) -> None:
    def pager(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params.get("page", "1"))
        return httpx.Response(200, json=envelope("players", [{"page": page}], paging=(page, 3)))

    route = respx.get(f"{BASE_URL}/players").mock(side_effect=pager)
    async with make_client() as client:
        pages = await client.get_all_pages("players", {"league": 39, "season": 2023})
    assert [p.response for p in pages] == [[{"page": 1}], [{"page": 2}], [{"page": 3}]]
    assert route.call_count == 3


@respx.mock
async def test_status_does_not_consume_daily_quota(make_client: ClientFactory) -> None:
    respx.get(f"{BASE_URL}/status").mock(
        return_value=httpx.Response(200, json=envelope("status", {"account": {}}))
    )
    async with make_client() as client:
        resp = await client.status()
    assert isinstance(resp.response, dict)
    assert client.quota.snapshot().used == 0


@respx.mock
async def test_local_quota_prevents_network_calls(
    make_client: ClientFactory, api_settings: ApiFootballSettings
) -> None:
    settings = api_settings.model_copy(update={"daily_limit": 3, "quota_reserve": 1})
    route = respx.get(f"{BASE_URL}/fixtures").mock(
        return_value=httpx.Response(200, json=envelope("fixtures", []))
    )
    async with make_client(settings) as client:
        await client.get("fixtures", {"date": "2026-10-01"})
        await client.get("fixtures", {"date": "2026-10-02"})
        with pytest.raises(QuotaExceededError):
            await client.get("fixtures", {"date": "2026-10-03"})
    assert route.call_count == 2
