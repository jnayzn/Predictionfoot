"""Prometheus metrics: HTTP traffic + application state collected at scrape time."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Iterator
from datetime import UTC, datetime

from fastapi import Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from football_ai.data.collectors.api_football.quota import QuotaManager
from football_ai.database import models as db
from football_ai.database.connection import session_scope


class AppStateCollector(Collector):
    """Reads quota, jobs, predictions and model state on every scrape."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        quota_factory: Callable[[], QuotaManager] | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.quota_factory = quota_factory

    def collect(self) -> Iterator[GaugeMetricFamily]:
        if self.quota_factory is not None:
            snap = self.quota_factory().snapshot()
            used = GaugeMetricFamily("football_ai_api_quota_used", "API requests used today")
            used.add_metric([], snap.used)
            limit = GaugeMetricFamily("football_ai_api_quota_limit", "Daily API request limit")
            limit.add_metric([], snap.daily_limit)
            yield used
            yield limit
        now = datetime.now(UTC)
        with session_scope(self.session_factory) as session:
            last_success = GaugeMetricFamily(
                "football_ai_job_last_success_age_seconds",
                "Seconds since the last successful run of each scheduled job",
                labels=["job"],
            )
            rows = session.execute(
                select(db.ScheduledJobRun.job, func.max(db.ScheduledJobRun.finished_at))
                .where(db.ScheduledJobRun.status.in_(["success", "partial", "skipped"]))
                .group_by(db.ScheduledJobRun.job)
            ).all()
            for job, finished in rows:
                if finished is not None:
                    finished = finished if finished.tzinfo else finished.replace(tzinfo=UTC)
                    last_success.add_metric([job], (now - finished).total_seconds())
            yield last_success

            failures = GaugeMetricFamily(
                "football_ai_job_failures_24h",
                "Failed job runs in the last 24 hours",
                labels=["job"],
            )
            since = datetime.fromtimestamp(time.time() - 86400, tz=UTC)
            for job, count in session.execute(
                select(db.ScheduledJobRun.job, func.count())
                .where(
                    db.ScheduledJobRun.status == "failed", db.ScheduledJobRun.started_at >= since
                )
                .group_by(db.ScheduledJobRun.job)
            ).all():
                failures.add_metric([job], count)
            yield failures

            predictions = GaugeMetricFamily("football_ai_predictions_stored", "Stored predictions")
            predictions.add_metric(
                [], session.scalar(select(func.count()).select_from(db.Prediction)) or 0
            )
            yield predictions

            evals = session.execute(
                select(
                    func.count(),
                    func.avg(db.PredictionEvaluation.log_loss),
                    func.avg(db.PredictionEvaluation.brier_score),
                )
            ).one()
            evaluated = GaugeMetricFamily(
                "football_ai_predictions_evaluated", "Evaluated predictions"
            )
            evaluated.add_metric([], evals[0] or 0)
            yield evaluated
            if evals[0]:
                ll = GaugeMetricFamily(
                    "football_ai_live_log_loss", "Mean log loss of published predictions"
                )
                ll.add_metric([], float(evals[1]))
                yield ll
                brier = GaugeMetricFamily(
                    "football_ai_live_brier", "Mean Brier score of published predictions"
                )
                brier.add_metric([], float(evals[2]))
                yield brier

            latest_fixture = session.scalar(select(func.max(db.Fixture.updated_at)))
            if latest_fixture is not None:
                latest_fixture = (
                    latest_fixture if latest_fixture.tzinfo else latest_fixture.replace(tzinfo=UTC)
                )
                fresh = GaugeMetricFamily(
                    "football_ai_data_age_seconds", "Seconds since the last fixture update"
                )
                fresh.add_metric([], (now - latest_fixture).total_seconds())
                yield fresh

            model = session.scalar(
                select(db.ModelVersion).order_by(db.ModelVersion.trained_at.desc()).limit(1)
            )
            if model is not None:
                info = GaugeMetricFamily(
                    "football_ai_model_info", "Latest registered model", labels=["name", "version"]
                )
                info.add_metric([model.name, model.version], 1)
                yield info


class HttpMetrics:
    def __init__(self, registry: CollectorRegistry) -> None:
        self.requests = Counter(
            "football_ai_http_requests_total",
            "HTTP requests",
            ["method", "route", "status"],
            registry=registry,
        )
        self.latency = Histogram(
            "football_ai_http_request_seconds",
            "HTTP latency",
            ["method", "route"],
            registry=registry,
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
        )

    async def middleware(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        started = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        template = getattr(route, "path", "unmatched")  # template, not raw path (cardinality)
        self.requests.labels(request.method, template, str(response.status_code)).inc()
        self.latency.labels(request.method, template).observe(time.perf_counter() - started)
        return response


def metrics_response(registry: CollectorRegistry) -> Response:
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
