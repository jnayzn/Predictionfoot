"""Prediction engine: production model + features -> probabilities, markets, confidence.

The engine is the single entry point used by the API, the bot and the
scheduler. It never trains anything: it loads the latest registered
production model (default priority: ensemble > dixon_coles > elo).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from football_ai.core.exceptions import FootballAIError
from football_ai.database import models as db
from football_ai.database.connection import session_scope
from football_ai.database.repositories import PredictionRepository
from football_ai.features.builder import FeatureBuilder
from football_ai.features.dataset import BaseFrames, load_base_frames
from football_ai.models.base import OutcomeModel
from football_ai.prediction.confidence import Confidence, assess_confidence
from football_ai.prediction.market_prediction import markets_from_matrix
from football_ai.prediction.score_prediction import expected_goals as matrix_expected_goals
from football_ai.training.registry import load_model

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PRIORITY = ("ensemble", "dixon_coles", "elo")


class NoProductionModelError(FootballAIError):
    """No registered model is available for predictions."""


class UnknownMatchError(FootballAIError):
    """The requested match does not exist in the canonical layer."""


@dataclass(slots=True)
class ProductionModel:
    model: OutcomeModel
    name: str
    version: str
    model_version_id: int | None


@dataclass(slots=True)
class MatchPrediction:
    match_id: int
    competition_key: str
    season_year: int
    kickoff_utc: datetime
    status_group: str
    home_team_id: int
    away_team_id: int
    home_team: str
    away_team: str
    probabilities: dict[str, float]
    confidence: Confidence
    model_name: str
    model_version: str
    markets: dict[str, Any] | None = None
    expected_goals: dict[str, float] | None = None
    member_probabilities: dict[str, dict[str, float]] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    result: dict[str, int] | None = None

    @property
    def is_upcoming(self) -> bool:
        return self.status_group == "scheduled" and self.kickoff_utc > datetime.now(UTC)


def load_production_model(
    session: Session, priority: Sequence[str] = DEFAULT_MODEL_PRIORITY
) -> ProductionModel:
    for name in priority:
        row = session.scalar(
            select(db.ModelVersion)
            .where(db.ModelVersion.name == name, db.ModelVersion.artifact_uri.is_not(None))
            .order_by(db.ModelVersion.trained_at.desc())
            .limit(1)
        )
        if row is not None and row.artifact_uri and Path(row.artifact_uri).is_file():
            return ProductionModel(load_model(Path(row.artifact_uri)), name, row.version, row.id)
    raise NoProductionModelError(
        f"no registered model among {list(priority)} - run `football-ai train-ensemble --register`"
    )


def feature_hash(features: dict[str, Any]) -> str:
    """Fingerprint of the information used for a prediction (rounded feature values).

    A new prediction version is stored only when this changes (new result,
    odds, injuries...) or when the model changes.
    """
    from football_ai.features.builder import feature_columns

    cols = sorted(c for c in feature_columns(pd.DataFrame([features])) if c in features)
    payload = {}
    for c in cols:
        v = features[c]
        payload[c] = (
            None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 4)
        )
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:32]


def _to_dict(p: np.ndarray) -> dict[str, float]:
    return {"home": float(p[0]), "draw": float(p[1]), "away": float(p[2])}


def _known_teams(model: Any, rows: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "known_teams"):
        known: np.ndarray = model.known_teams(rows)
        return known
    members = getattr(model, "members", {})
    for member in members.values():
        if hasattr(member, "known_teams"):
            known = member.known_teams(rows)
            return known
    return np.ones(len(rows), dtype=bool)


class PredictionEngine:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        production: ProductionModel | None = None,
        model_priority: Sequence[str] = DEFAULT_MODEL_PRIORITY,
        builder: FeatureBuilder | None = None,
        frames_ttl_seconds: int = 300,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.session_factory = session_factory
        self.model_priority = tuple(model_priority)
        self.builder = builder or FeatureBuilder()
        self._production = production
        self._frames: BaseFrames | None = None
        self._frames_loaded_at = 0.0
        self._ttl = frames_ttl_seconds
        self._clock = clock

    # ------------------------------------------------------------ resources
    @property
    def production(self) -> ProductionModel:
        if self._production is None:
            with session_scope(self.session_factory) as session:
                self._production = load_production_model(session, self.model_priority)
            logger.info(
                "production_model_loaded",
                extra={"model": self._production.name, "version": self._production.version},
            )
        return self._production

    def frames(self) -> BaseFrames:
        if self._frames is None or self._clock() - self._frames_loaded_at > self._ttl:
            with session_scope(self.session_factory) as session:
                self._frames = load_base_frames(session)
            self._frames_loaded_at = self._clock()
        return self._frames

    def invalidate(self) -> None:
        """Drop caches (new data or new model available)."""
        self._frames = None
        self._production = None

    def _team_names(self, ids: set[int]) -> dict[int, str]:
        with session_scope(self.session_factory) as session:
            rows = session.execute(
                select(db.CanonicalTeam.id, db.CanonicalTeam.name).where(
                    db.CanonicalTeam.id.in_(list(ids))
                )
            ).all()
        return {int(i): str(n) for i, n in rows}

    # -------------------------------------------------------------- predict
    def predict(
        self, match_ids: Sequence[int], *, frames: BaseFrames | None = None
    ) -> list[MatchPrediction]:
        if not match_ids:
            return []
        frames = frames if frames is not None else self.frames()
        known_ids = set(frames.matches["match_id"].astype(int))
        missing = [m for m in match_ids if int(m) not in known_ids]
        if missing:
            raise UnknownMatchError(f"unknown match ids: {missing}")
        features = self.builder.build(frames, match_ids=list(match_ids))
        production = self.production
        model: Any = production.model
        proba = model.predict_proba(features)
        matrices = model.score_matrix(features) if hasattr(model, "score_matrix") else None
        members = model.member_probas(features) if hasattr(model, "member_probas") else {}
        known = _known_teams(model, features)
        names = self._team_names(set(features["home_team_id"]) | set(features["away_team_id"]))

        out: list[MatchPrediction] = []
        records: list[dict[str, Any]] = [
            {str(k): v for k, v in r.items()} for r in features.to_dict(orient="records")
        ]
        for i, row in enumerate(records):
            member_i = {n: p[i] for n, p in members.items()}
            markets = expected = None
            if matrices is not None:
                markets = markets_from_matrix(matrices[i : i + 1])[0]
                xg = matrix_expected_goals(matrices[i : i + 1])[0]
                expected = {"home": float(xg[0]), "away": float(xg[1])}
            result = None
            if not pd.isna(row.get("home_goals")) and not pd.isna(row.get("away_goals")):
                result = {"home": int(row["home_goals"]), "away": int(row["away_goals"])}
            out.append(
                MatchPrediction(
                    match_id=int(row["match_id"]),
                    competition_key=str(row["competition_key"]),
                    season_year=int(row["season_year"]),
                    kickoff_utc=pd.Timestamp(row["kickoff_utc"]).to_pydatetime(),
                    status_group=str(row["status_group"]),
                    home_team_id=int(row["home_team_id"]),
                    away_team_id=int(row["away_team_id"]),
                    home_team=names.get(int(row["home_team_id"]), str(row["home_team_id"])),
                    away_team=names.get(int(row["away_team_id"]), str(row["away_team_id"])),
                    probabilities=_to_dict(proba[i]),
                    confidence=assess_confidence(row, member_i, bool(known[i])),
                    model_name=production.name,
                    model_version=production.version,
                    markets=markets,
                    expected_goals=expected,
                    member_probabilities={n: _to_dict(p) for n, p in member_i.items()},
                    features=row,
                    result=result,
                )
            )
        return out

    def predict_hypothetical(
        self,
        home_team_id: int,
        away_team_id: int,
        *,
        competition_key: str | None = None,
        kickoff_utc: datetime | None = None,
    ) -> MatchPrediction:
        """Prediction for a match that is not scheduled (team comparison).

        Uses the same features as a real fixture, computed from every match
        played before ``kickoff_utc`` (default: now).
        """
        if home_team_id == away_team_id:
            raise ValueError("a team cannot play itself")
        frames = self.frames()
        m = frames.matches
        known_teams = set(m["home_team_id"]) | set(m["away_team_id"])
        unknown = {home_team_id, away_team_id} - {int(t) for t in known_teams}
        if unknown:
            raise UnknownMatchError(f"unknown team ids: {sorted(unknown)}")
        kickoff = pd.Timestamp(kickoff_utc or datetime.now(UTC))
        if competition_key is None:
            home_games = m[m["home_team_id"] == home_team_id].sort_values("kickoff_utc")
            competition_key = (
                str(home_games["competition_key"].iloc[-1]) if len(home_games) else "friendly"
            )
        season_rows = m[m["competition_key"] == competition_key]
        season = int(season_rows["season_year"].max()) if len(season_rows) else kickoff.year
        virtual_id = int(m["match_id"].min()) - 1 if len(m) else -1
        virtual_id = min(virtual_id, -1)
        row = {
            "match_id": virtual_id, "competition_key": competition_key, "season_year": season,
            "kickoff_utc": kickoff, "home_team_id": home_team_id, "away_team_id": away_team_id,
            "status_group": "scheduled", "home_goals": None, "away_goals": None,
            "valid_for_training": False,
        }  # fmt: skip
        extended = BaseFrames(
            pd.concat([m, pd.DataFrame([row])], ignore_index=True),
            frames.team_stats,
            frames.odds,
            frames.injuries,
        )
        return self.predict([virtual_id], frames=extended)[0]

    def predict_and_store(self, match_ids: Sequence[int]) -> tuple[list[MatchPrediction], int]:
        """Predict and persist (immutable) predictions of matches not started yet.

        A new version is stored only when the information (feature hash) or the
        model version changed since the latest stored prediction of the match.
        """
        predictions = self.predict(match_ids)
        production = self.production
        if production.model_version_id is None:
            return predictions, 0
        stored = 0
        now = datetime.now(UTC)
        with session_scope(self.session_factory) as session:
            repo = PredictionRepository(session)
            for p in predictions:
                if p.kickoff_utc <= now or p.status_group != "scheduled":
                    continue
                match = session.get(db.Match, p.match_id)
                if match is None:
                    continue
                fingerprint = feature_hash(p.features)
                latest = repo.latest_for_fixture(
                    match.primary_fixture_id, production.model_version_id
                )
                if latest is not None and latest.feature_hash == fingerprint:
                    continue  # nothing new since the last published prediction
                repo.record(
                    fixture_id=match.primary_fixture_id,
                    model_version_id=production.model_version_id,
                    kickoff_utc=p.kickoff_utc,
                    probabilities=(
                        p.probabilities["home"],
                        p.probabilities["draw"],
                        p.probabilities["away"],
                    ),
                    expected_goals=(p.expected_goals["home"], p.expected_goals["away"])
                    if p.expected_goals
                    else None,
                    markets=p.markets,
                    confidence=p.confidence.score,
                    data_quality={
                        "uncertainties": p.confidence.uncertainties,
                        **p.confidence.components,
                    },
                    feature_hash=fingerprint,
                    created_at=now,
                )
                stored += 1
        return predictions, stored
