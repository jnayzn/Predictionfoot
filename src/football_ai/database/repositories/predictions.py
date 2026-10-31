"""Stored pre-match predictions (insert-only) and their evaluation after the match.

The track record of the live system is computed ONLY from predictions stored
before kick-off; rows can never be modified afterwards (PostgreSQL trigger),
so past performance cannot be rewritten.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from football_ai.database import models as db
from football_ai.database.models import utcnow

EPS = 1e-15
OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}


class PredictionTooLateError(ValueError):
    """A prediction must be recorded strictly before kick-off."""


@dataclass(frozen=True, slots=True)
class PerformanceSummary:
    n: int
    log_loss: float | None
    brier: float | None
    accuracy: float | None
    mae_total_goals: float | None


class PredictionRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def record(
        self,
        *,
        fixture_id: int,
        model_version_id: int,
        kickoff_utc: datetime,
        probabilities: Sequence[float],
        expected_goals: tuple[float, float] | None = None,
        markets: dict[str, Any] | None = None,
        confidence: float | None = None,
        data_quality: dict[str, Any] | None = None,
        feature_hash: str | None = None,
        created_at: datetime | None = None,
    ) -> db.Prediction:
        created = created_at or utcnow()
        if created >= kickoff_utc:
            raise PredictionTooLateError(
                f"prediction for fixture {fixture_id} after kick-off ({created} >= {kickoff_utc})"
            )
        p_home, p_draw, p_away = (float(x) for x in probabilities)
        total = p_home + p_draw + p_away
        row = db.Prediction(
            fixture_id=fixture_id,
            model_version_id=model_version_id,
            created_at=created,
            kickoff_utc=kickoff_utc,
            p_home=p_home / total,
            p_draw=p_draw / total,
            p_away=p_away / total,
            expected_home_goals=expected_goals[0] if expected_goals else None,
            expected_away_goals=expected_goals[1] if expected_goals else None,
            markets=markets or {},
            confidence=confidence,
            data_quality=data_quality or {},
            feature_hash=feature_hash,
        )
        self.session.add(row)
        self.session.flush()
        return row

    def latest_for_fixture(
        self, fixture_id: int, model_version_id: int | None = None
    ) -> db.Prediction | None:
        stmt = select(db.Prediction).where(db.Prediction.fixture_id == fixture_id)
        if model_version_id is not None:
            stmt = stmt.where(db.Prediction.model_version_id == model_version_id)
        return self.session.scalar(stmt.order_by(db.Prediction.created_at.desc()).limit(1))

    def evaluate_pending(self) -> int:
        """Create evaluations for predictions whose match now has a final result."""
        rows = self.session.execute(
            select(db.Prediction, db.ActualResult)
            .join(db.ActualResult, db.ActualResult.fixture_id == db.Prediction.fixture_id)
            .outerjoin(
                db.PredictionEvaluation,
                db.PredictionEvaluation.prediction_id == db.Prediction.id,
            )
            .where(db.PredictionEvaluation.id.is_(None))
        ).all()
        for prediction, result in rows:
            probs = (prediction.p_home, prediction.p_draw, prediction.p_away)
            k = OUTCOME_INDEX[result.outcome]
            brier = sum((p - (1.0 if i == k else 0.0)) ** 2 for i, p in enumerate(probs))
            goals_error = None
            if (
                prediction.expected_home_goals is not None
                and prediction.expected_away_goals is not None
            ):
                goals_error = abs(
                    (prediction.expected_home_goals + prediction.expected_away_goals)
                    - (result.home_goals + result.away_goals)
                )
            self.session.add(
                db.PredictionEvaluation(
                    prediction_id=prediction.id,
                    fixture_id=prediction.fixture_id,
                    actual_outcome=result.outcome,
                    actual_home_goals=result.home_goals,
                    actual_away_goals=result.away_goals,
                    log_loss=-math.log(max(probs[k], EPS)),
                    brier_score=brier,
                    correct=max(range(3), key=lambda i: probs[i]) == k,
                    goals_abs_error=goals_error,
                )
            )
        self.session.flush()
        return len(rows)

    def performance(
        self, *, model_version_id: int | None = None, since: datetime | None = None
    ) -> PerformanceSummary:
        stmt = select(db.PredictionEvaluation).join(
            db.Prediction, db.Prediction.id == db.PredictionEvaluation.prediction_id
        )
        if model_version_id is not None:
            stmt = stmt.where(db.Prediction.model_version_id == model_version_id)
        if since is not None:
            stmt = stmt.where(db.Prediction.kickoff_utc >= since)
        evals = list(self.session.scalars(stmt))
        if not evals:
            return PerformanceSummary(0, None, None, None, None)
        n = len(evals)
        goals = [e.goals_abs_error for e in evals if e.goals_abs_error is not None]
        return PerformanceSummary(
            n=n,
            log_loss=sum(e.log_loss for e in evals) / n,
            brier=sum(e.brier_score for e in evals) / n,
            accuracy=sum(e.correct for e in evals) / n,
            mae_total_goals=sum(goals) / len(goals) if goals else None,
        )
