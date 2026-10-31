"""HTTP routes. Input validation through typed path/query parameters."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request

from football_ai.backend.schemas.api import (
    AnalysisOut,
    CompetitionOut,
    Health,
    MatchOut,
    PerformanceOut,
    PredictionOut,
    ResultOut,
    TeamRefOut,
)
from football_ai.backend.security import authenticate
from football_ai.backend.services.football_service import FootballService

public = APIRouter()
router = APIRouter(prefix="/v1", dependencies=[Depends(authenticate)])

CompetitionQuery = Annotated[str | None, Query(pattern=r"^[a-z0-9_]{2,64}$")]
MatchId = Annotated[int, Query(gt=0)]


def service(request: Request) -> FootballService:
    svc: FootballService = request.app.state.service
    return svc


Svc = Annotated[FootballService, Depends(service)]


@public.get("/health", response_model=Health, tags=["system"])
def health(svc: Svc) -> Health:
    return Health(**svc.health())


@router.get("/competitions", response_model=list[CompetitionOut], tags=["reference"])
def competitions(svc: Svc) -> list[CompetitionOut]:
    return svc.list_competitions()


@router.get("/fixtures", response_model=list[MatchOut], tags=["matches"])
def fixtures(
    svc: Svc, day: date | None = None, competition: CompetitionQuery = None
) -> list[MatchOut]:
    return svc.fixtures(day or datetime.now(UTC).date(), competition)


@router.get("/matches/{match_id}", response_model=MatchOut, tags=["matches"])
def match(svc: Svc, match_id: int) -> MatchOut:
    return svc.match(match_id)


@router.get("/matches/{match_id}/prediction", response_model=PredictionOut, tags=["predictions"])
def prediction(svc: Svc, match_id: int, store: bool = False) -> PredictionOut:
    return svc.predict(match_id, store=store)


@router.get("/matches/{match_id}/analysis", response_model=AnalysisOut, tags=["predictions"])
def analysis(svc: Svc, match_id: int) -> AnalysisOut:
    return svc.analyse(match_id)


@router.get("/teams", response_model=list[TeamRefOut], tags=["teams"])
def teams(svc: Svc, search: Annotated[str, Query(min_length=2, max_length=64)]) -> list[TeamRefOut]:
    return svc.search_teams(search)


@router.get("/teams/{team_id}/matches", response_model=list[MatchOut], tags=["teams"])
def team_matches(
    svc: Svc, team_id: int, limit: Annotated[int, Query(ge=1, le=50)] = 10
) -> list[MatchOut]:
    return svc.team_matches(team_id, limit)


@router.get("/compare", response_model=PredictionOut, tags=["teams"])
def compare(
    svc: Svc,
    home: Annotated[int, Query(gt=0)],
    away: Annotated[int, Query(gt=0)],
) -> PredictionOut:
    return svc.compare(home, away)


@router.get("/results", response_model=list[ResultOut], tags=["matches"])
def results(
    svc: Svc, day: date | None = None, competition: CompetitionQuery = None
) -> list[ResultOut]:
    return svc.results(day or datetime.now(UTC).date(), competition)


@router.get("/performance", response_model=PerformanceOut, tags=["predictions"])
def performance(svc: Svc) -> PerformanceOut:
    return svc.performance()
