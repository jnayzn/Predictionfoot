"""Public API response models (validated, documented in OpenAPI)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

DISCLAIMER = (
    "Probabilités statistiques, jamais des certitudes. Ceci n'est pas un conseil de pari. "
    "Les paris comportent un risque de perte."
)


class Health(BaseModel):
    status: str
    database: bool
    model: str | None
    model_version: str | None


class CompetitionOut(BaseModel):
    key: str
    name: str
    country: str
    type: str


class TeamRefOut(BaseModel):
    id: int
    name: str


class MatchOut(BaseModel):
    id: int
    competition: str
    season: int
    kickoff_utc: datetime
    kickoff_time_known: bool
    status: str
    home: TeamRefOut
    away: TeamRefOut
    score: dict[str, int] | None = None


class ConfidenceOut(BaseModel):
    score: float = Field(ge=0, le=1)
    level: str
    uncertainties: list[str]
    components: dict[str, float]


class PredictionOut(BaseModel):
    match: MatchOut
    model: str
    model_version: str
    probabilities: dict[str, float]
    expected_goals: dict[str, float] | None
    markets: dict[str, Any] | None
    confidence: ConfidenceOut
    member_probabilities: dict[str, dict[str, float]]
    stored: bool = False
    disclaimer: str = DISCLAIMER


class FactorOut(BaseModel):
    description: str
    favours: str
    strength: float
    text: str


class TeamProfileOut(BaseModel):
    team: TeamRefOut
    elo: float | None
    form_last5: str
    ppg_5: float | None
    goals_for_avg: float | None
    goals_against_avg: float | None
    xg_for_avg: float | None
    xg_against_avg: float | None
    venue_ppg: float | None
    season_ppg: float | None
    rest_days: float | None
    missing_players: float | None
    attack_index: float | None
    defence_index: float | None
    strengths: list[str]
    weaknesses: list[str]


class SelectionOut(BaseModel):
    market: str
    selection: str
    label: str
    model_probability: float
    market_probability: float
    adjusted_probability: float
    odds: float
    expected_value: float
    reasons: list[str]


class RecommendationOut(BaseModel):
    status: str
    selections: list[SelectionOut]
    most_likely: list[dict[str, Any]]
    reasons: list[str]
    disclaimer: str


class AnalysisOut(BaseModel):
    prediction: PredictionOut
    summary: str
    favourite: str | None
    home: TeamProfileOut
    away: TeamProfileOut
    key_factors: list[FactorOut]
    uncertainties: list[str]
    uncertainty_level: float
    recommendation: RecommendationOut


class ResultOut(BaseModel):
    match: MatchOut
    predicted: dict[str, float] | None = None
    prediction_correct: bool | None = None


class PerformanceOut(BaseModel):
    live: dict[str, Any]
    backtest: dict[str, Any] | None
    note: str


class SubscriberIn(BaseModel):
    chat_id: int
    username: str | None = Field(default=None, max_length=64)
    competitions: list[str] = Field(default_factory=list, max_length=20)


class SubscriberOut(BaseModel):
    chat_id: int
    username: str | None
    competitions: list[str]
    active: bool
