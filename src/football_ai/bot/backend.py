"""The bot's only door to the system: a small async backend interface.

* :class:`ServiceBackend` - same process, calls :class:`FootballService`
  (blocking database work runs in a thread);
* :class:`HttpBackend` - separate process/container, calls the HTTP API with
  an API key.

Both return the API's Pydantic schemas, so handlers never depend on how the
backend is deployed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import date
from typing import Any, Protocol, TypeVar

import httpx
from pydantic import TypeAdapter

from football_ai.backend.schemas.api import (
    AnalysisOut,
    CompetitionOut,
    MatchOut,
    PerformanceOut,
    PredictionOut,
    ResultOut,
    SubscriberOut,
    TeamRefOut,
)
from football_ai.backend.services.football_service import FootballService, NotFoundError
from football_ai.core.exceptions import FootballAIError

T = TypeVar("T")


class BackendNotFoundError(FootballAIError):
    """Unknown match / team / competition."""


class BackendUnavailableError(FootballAIError):
    """Backend temporarily unavailable (no model, network error...)."""


class BotBackend(Protocol):
    async def competitions(self) -> list[CompetitionOut]: ...
    async def fixtures(self, day: date, competition: str | None = None) -> list[MatchOut]: ...
    async def match(self, match_id: int) -> MatchOut: ...
    async def prediction(self, match_id: int) -> PredictionOut: ...
    async def analysis(self, match_id: int) -> AnalysisOut: ...
    async def search_teams(self, query: str) -> list[TeamRefOut]: ...
    async def team_matches(self, team_id: int, limit: int = 5) -> list[MatchOut]: ...
    async def compare(self, home_team_id: int, away_team_id: int) -> PredictionOut: ...
    async def results(self, day: date, competition: str | None = None) -> list[ResultOut]: ...
    async def performance(self) -> PerformanceOut: ...
    async def subscribe(
        self, chat_id: int, username: str | None, competitions: list[str]
    ) -> SubscriberOut: ...
    async def unsubscribe(self, chat_id: int) -> bool: ...
    async def erase(self, chat_id: int) -> bool: ...
    async def subscribers(self) -> list[SubscriberOut]: ...


class ServiceBackend:
    def __init__(self, service: FootballService) -> None:
        self.service = service

    async def _call(self, fn: Callable[..., T], *args: Any) -> T:
        try:
            return await asyncio.to_thread(fn, *args)
        except NotFoundError as exc:
            raise BackendNotFoundError(str(exc)) from exc
        except FootballAIError as exc:
            raise BackendUnavailableError(str(exc)) from exc

    async def competitions(self) -> list[CompetitionOut]:
        return await self._call(self.service.list_competitions)

    async def fixtures(self, day: date, competition: str | None = None) -> list[MatchOut]:
        return await self._call(self.service.fixtures, day, competition)

    async def match(self, match_id: int) -> MatchOut:
        return await self._call(self.service.match, match_id)

    async def prediction(self, match_id: int) -> PredictionOut:
        return await self._call(self.service.predict, match_id)

    async def analysis(self, match_id: int) -> AnalysisOut:
        return await self._call(self.service.analyse, match_id)

    async def search_teams(self, query: str) -> list[TeamRefOut]:
        return await self._call(self.service.search_teams, query)

    async def team_matches(self, team_id: int, limit: int = 5) -> list[MatchOut]:
        return await self._call(self.service.team_matches, team_id, limit)

    async def compare(self, home_team_id: int, away_team_id: int) -> PredictionOut:
        return await self._call(self.service.compare, home_team_id, away_team_id)

    async def results(self, day: date, competition: str | None = None) -> list[ResultOut]:
        return await self._call(self.service.results, day, competition)

    async def performance(self) -> PerformanceOut:
        return await self._call(self.service.performance)

    async def subscribe(
        self, chat_id: int, username: str | None, competitions: list[str]
    ) -> SubscriberOut:
        return await self._call(self.service.subscribe, chat_id, username, competitions)

    async def unsubscribe(self, chat_id: int) -> bool:
        return await self._call(self.service.unsubscribe, chat_id)

    async def erase(self, chat_id: int) -> bool:
        return await self._call(self.service.erase_subscriber, chat_id)

    async def subscribers(self) -> list[SubscriberOut]:
        return await self._call(self.service.subscribers)


class HttpBackend:
    def __init__(self, base_url: str, api_key: str | None, *, timeout: float = 30.0,
                 client: httpx.AsyncClient | None = None) -> None:  # fmt: skip
        headers = {"X-API-Key": api_key} if api_key else {}
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, model: Any, **kwargs: Any) -> Any:
        try:
            resp = await self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise BackendUnavailableError(f"API unreachable: {exc}") from exc
        if resp.status_code == 404:
            raise BackendNotFoundError(resp.json().get("error", "not found"))
        if resp.status_code >= 400:
            raise BackendUnavailableError(f"API error {resp.status_code}")
        return TypeAdapter(model).validate_python(resp.json())

    async def competitions(self) -> list[CompetitionOut]:
        result: list[CompetitionOut] = await self._request(
            "GET", "/v1/competitions", list[CompetitionOut]
        )
        return result

    async def fixtures(self, day: date, competition: str | None = None) -> list[MatchOut]:
        params = {"day": day.isoformat(), **({"competition": competition} if competition else {})}
        result: list[MatchOut] = await self._request(
            "GET", "/v1/fixtures", list[MatchOut], params=params
        )
        return result

    async def match(self, match_id: int) -> MatchOut:
        result: MatchOut = await self._request("GET", f"/v1/matches/{match_id}", MatchOut)
        return result

    async def prediction(self, match_id: int) -> PredictionOut:
        result: PredictionOut = await self._request(
            "GET", f"/v1/matches/{match_id}/prediction", PredictionOut
        )
        return result

    async def analysis(self, match_id: int) -> AnalysisOut:
        result: AnalysisOut = await self._request(
            "GET", f"/v1/matches/{match_id}/analysis", AnalysisOut
        )
        return result

    async def search_teams(self, query: str) -> list[TeamRefOut]:
        result: list[TeamRefOut] = await self._request(
            "GET", "/v1/teams", list[TeamRefOut], params={"search": query}
        )
        return result

    async def team_matches(self, team_id: int, limit: int = 5) -> list[MatchOut]:
        result: list[MatchOut] = await self._request(
            "GET", f"/v1/teams/{team_id}/matches", list[MatchOut], params={"limit": limit}
        )
        return result

    async def compare(self, home_team_id: int, away_team_id: int) -> PredictionOut:
        result: PredictionOut = await self._request(
            "GET", "/v1/compare", PredictionOut, params={"home": home_team_id, "away": away_team_id}
        )
        return result

    async def results(self, day: date, competition: str | None = None) -> list[ResultOut]:
        params = {"day": day.isoformat(), **({"competition": competition} if competition else {})}
        result: list[ResultOut] = await self._request(
            "GET", "/v1/results", list[ResultOut], params=params
        )
        return result

    async def performance(self) -> PerformanceOut:
        result: PerformanceOut = await self._request("GET", "/v1/performance", PerformanceOut)
        return result

    async def subscribe(
        self, chat_id: int, username: str | None, competitions: list[str]
    ) -> SubscriberOut:
        body = {"chat_id": chat_id, "username": username, "competitions": competitions}
        result: SubscriberOut = await self._request(
            "POST", "/v1/subscribers", SubscriberOut, json=body
        )
        return result

    async def unsubscribe(self, chat_id: int) -> bool:
        data: dict[str, bool] = await self._request(
            "DELETE", f"/v1/subscribers/{chat_id}", dict[str, bool]
        )
        return bool(data.get("unsubscribed"))

    async def erase(self, chat_id: int) -> bool:
        data: dict[str, bool] = await self._request(
            "DELETE", f"/v1/subscribers/{chat_id}", dict[str, bool], params={"erase": "true"}
        )
        return bool(data.get("erased"))

    async def subscribers(self) -> list[SubscriberOut]:
        result: list[SubscriberOut] = await self._request(
            "GET", "/v1/subscribers", list[SubscriberOut]
        )
        return result
