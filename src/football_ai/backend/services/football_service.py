"""Internal service layer shared by the API and the Telegram bot.

Every consumer (HTTP API, bot, scheduler) goes through this class, so the
business rules live in one place.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, sessionmaker

from football_ai.analysis.match_analysis import MatchAnalysis, analyse_match
from football_ai.analysis.recommendation import Recommendation, recommend, selection_label
from football_ai.analysis.team_analysis import TeamProfile
from football_ai.backend.schemas.api import (
    AnalysisOut,
    CompetitionOut,
    ConfidenceOut,
    FactorOut,
    MatchOut,
    PerformanceOut,
    PredictionOut,
    RecommendationOut,
    ResultOut,
    SelectionOut,
    TeamProfileOut,
    TeamRefOut,
)
from football_ai.config.competitions import CompetitionConfig
from football_ai.core.exceptions import FootballAIError
from football_ai.database import models as db
from football_ai.database.connection import session_scope
from football_ai.database.repositories import PredictionRepository
from football_ai.prediction.engine import MatchPrediction, PredictionEngine, UnknownMatchError
from football_ai.prediction.odds import load_match_odds


class NotFoundError(FootballAIError):
    """Requested resource does not exist."""


def _match_out(m: db.Match, names: dict[int, str]) -> MatchOut:
    score = None
    if m.home_goals is not None and m.away_goals is not None and m.status_group == "finished":
        score = {"home": m.home_goals, "away": m.away_goals}
    return MatchOut(
        id=m.id,
        competition=m.competition_key,
        season=m.season_year,
        kickoff_utc=m.kickoff_utc,
        kickoff_time_known=m.kickoff_time_known,
        status=m.status_group,
        home=TeamRefOut(id=m.home_team_id, name=names.get(m.home_team_id, "?")),
        away=TeamRefOut(id=m.away_team_id, name=names.get(m.away_team_id, "?")),
        score=score,
    )


def _profile_out(p: TeamProfile) -> TeamProfileOut:
    data = asdict(p)
    data["team"] = TeamRefOut(id=data.pop("team_id"), name=data.pop("name"))
    return TeamProfileOut(**data)


class FootballService:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        engine: PredictionEngine,
        competitions: list[CompetitionConfig],
        reports_dir: Path | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.engine = engine
        self.competitions = {c.key: c for c in competitions}
        self.reports_dir = reports_dir

    # ----------------------------------------------------------- helpers
    def _names(self, session: Session, ids: set[int]) -> dict[int, str]:
        if not ids:
            return {}
        rows = session.execute(
            select(db.CanonicalTeam.id, db.CanonicalTeam.name).where(db.CanonicalTeam.id.in_(ids))
        ).all()
        return {int(i): str(n) for i, n in rows}

    def _matches_out(self, session: Session, matches: list[db.Match]) -> list[MatchOut]:
        ids = {m.home_team_id for m in matches} | {m.away_team_id for m in matches}
        names = self._names(session, ids)
        return [_match_out(m, names) for m in matches]

    def _get_match(self, session: Session, match_id: int) -> db.Match:
        match = session.get(db.Match, match_id)
        if match is None:
            raise NotFoundError(f"match {match_id} not found")
        return match

    # ---------------------------------------------------------- queries
    def list_competitions(self) -> list[CompetitionOut]:
        return [
            CompetitionOut(key=c.key, name=c.name, country=c.country, type=c.type)
            for c in self.competitions.values()
        ]

    def fixtures(self, day: date, competition: str | None = None) -> list[MatchOut]:
        if competition and competition not in self.competitions:
            raise NotFoundError(f"unknown competition {competition!r}")
        start = datetime.combine(day, time(0), tzinfo=UTC)
        with session_scope(self.session_factory) as session:
            stmt = select(db.Match).where(
                db.Match.kickoff_utc >= start, db.Match.kickoff_utc < start + timedelta(days=1)
            )
            if competition:
                stmt = stmt.where(db.Match.competition_key == competition)
            matches = list(session.scalars(stmt.order_by(db.Match.kickoff_utc)))
            return self._matches_out(session, matches)

    def match(self, match_id: int) -> MatchOut:
        with session_scope(self.session_factory) as session:
            return self._matches_out(session, [self._get_match(session, match_id)])[0]

    # ------------------------------------------------------- predictions
    def _prediction_out(self, p: MatchPrediction, match: MatchOut, stored: bool) -> PredictionOut:
        return PredictionOut(
            match=match,
            model=p.model_name,
            model_version=p.model_version,
            probabilities=p.probabilities,
            expected_goals=p.expected_goals,
            markets=p.markets,
            confidence=ConfidenceOut(
                score=p.confidence.score,
                level=p.confidence.level.value,
                uncertainties=p.confidence.uncertainties,
                components=p.confidence.components,
            ),
            member_probabilities=p.member_probabilities,
            stored=stored,
        )

    def predict(self, match_id: int, *, store: bool = False) -> PredictionOut:
        match = self.match(match_id)
        try:
            if store:
                predictions, n_stored = self.engine.predict_and_store([match_id])
            else:
                predictions, n_stored = self.engine.predict([match_id]), 0
        except UnknownMatchError as exc:
            raise NotFoundError(str(exc)) from exc
        return self._prediction_out(predictions[0], match, n_stored > 0)

    def _recommendation_out(self, rec: Recommendation, p: MatchPrediction) -> RecommendationOut:
        return RecommendationOut(
            status=rec.status.value,
            selections=[
                SelectionOut(
                    market=s.market,
                    selection=s.selection,
                    label=selection_label(s.market, s.selection, p),
                    model_probability=s.model_probability,
                    market_probability=s.market_probability,
                    adjusted_probability=s.adjusted_probability,
                    odds=s.odds,
                    expected_value=s.expected_value,
                    reasons=s.reasons,
                )
                for s in rec.selections
            ],
            most_likely=rec.most_likely,
            reasons=rec.reasons,
            disclaimer=rec.disclaimer,
        )

    def analyse(self, match_id: int) -> AnalysisOut:
        match = self.match(match_id)
        try:
            prediction = self.engine.predict([match_id])[0]
        except UnknownMatchError as exc:
            raise NotFoundError(str(exc)) from exc
        analysis: MatchAnalysis = analyse_match(prediction, self.engine.frames().matches)
        with session_scope(self.session_factory) as session:
            odds = load_match_odds(session, match_id)
        rec = recommend(prediction, odds)
        return AnalysisOut(
            prediction=self._prediction_out(prediction, match, False),
            summary=analysis.summary,
            favourite=analysis.favourite,
            home=_profile_out(analysis.home),
            away=_profile_out(analysis.away),
            key_factors=[
                FactorOut(
                    description=f.description,
                    favours=f.favours,
                    strength=round(f.strength, 3),
                    text=f.text,
                )
                for f in analysis.key_factors
            ],
            uncertainties=analysis.uncertainties,
            uncertainty_level=analysis.uncertainty_level,
            recommendation=self._recommendation_out(rec, prediction),
        )

    def compare(self, home_team_id: int, away_team_id: int) -> PredictionOut:
        try:
            p = self.engine.predict_hypothetical(home_team_id, away_team_id)
        except (UnknownMatchError, ValueError) as exc:
            raise NotFoundError(str(exc)) from exc
        match = MatchOut(
            id=p.match_id, competition=p.competition_key, season=p.season_year,
            kickoff_utc=p.kickoff_utc, kickoff_time_known=True, status="hypothetical",
            home=TeamRefOut(id=p.home_team_id, name=p.home_team),
            away=TeamRefOut(id=p.away_team_id, name=p.away_team),
        )  # fmt: skip
        return self._prediction_out(p, match, False)

    # -------------------------------------------------------------- teams
    def search_teams(self, query: str, limit: int = 10) -> list[TeamRefOut]:
        with session_scope(self.session_factory) as session:
            rows = session.execute(
                select(db.CanonicalTeam.id, db.CanonicalTeam.name)
                .where(func.lower(db.CanonicalTeam.name).contains(query.lower()))
                .order_by(db.CanonicalTeam.name)
                .limit(limit)
            ).all()
        return [TeamRefOut(id=int(i), name=str(n)) for i, n in rows]

    def team_matches(self, team_id: int, limit: int = 10) -> list[MatchOut]:
        with session_scope(self.session_factory) as session:
            if session.get(db.CanonicalTeam, team_id) is None:
                raise NotFoundError(f"team {team_id} not found")
            matches = list(
                session.scalars(
                    select(db.Match)
                    .where(or_(db.Match.home_team_id == team_id, db.Match.away_team_id == team_id))
                    .where(db.Match.status_group == "finished")
                    .order_by(db.Match.kickoff_utc.desc())
                    .limit(limit)
                )
            )
            return self._matches_out(session, matches)

    # ---------------------------------------------------- results / perf
    def results(self, day: date, competition: str | None = None) -> list[ResultOut]:
        matches = [m for m in self.fixtures(day, competition) if m.status == "finished"]
        out: list[ResultOut] = []
        with session_scope(self.session_factory) as session:
            repo = PredictionRepository(session)
            for m in matches:
                match_row = self._get_match(session, m.id)
                pred = repo.latest_for_fixture(match_row.primary_fixture_id)
                predicted = correct = None
                if pred is not None and m.score is not None:
                    predicted = {"home": pred.p_home, "draw": pred.p_draw, "away": pred.p_away}
                    actual = (
                        "home" if m.score["home"] > m.score["away"]
                        else "away" if m.score["away"] > m.score["home"] else "draw"
                    )  # fmt: skip
                    correct = max(predicted, key=lambda k: predicted[k]) == actual
                out.append(ResultOut(match=m, predicted=predicted, prediction_correct=correct))
        return out

    def performance(self) -> PerformanceOut:
        with session_scope(self.session_factory) as session:
            summary = PredictionRepository(session).performance()
        backtest = None
        if self.reports_dir and self.reports_dir.is_dir():
            reports = sorted(self.reports_dir.glob("backtest_*.json"))
            if reports:
                data = json.loads(reports[-1].read_text(encoding="utf-8"))
                backtest = {"period": data.get("period"), "overall": data.get("overall")}
        return PerformanceOut(
            live=asdict(summary),
            backtest=backtest,
            note=(
                "live = predictions published before kick-off (immutable); "
                "backtest = walk-forward simulation on past seasons"
            ),
        )

    def health(self) -> dict[str, Any]:
        database = True
        try:
            with session_scope(self.session_factory) as session:
                session.execute(select(1))
        except Exception:  # pragma: no cover - depends on infrastructure
            database = False
        model: str | None = None
        version: str | None = None
        try:
            production = self.engine.production
            model, version = production.name, production.version
        except FootballAIError:
            pass
        status = "ok" if database and model else "degraded"
        return {"status": status, "database": database, "model": model, "model_version": version}
