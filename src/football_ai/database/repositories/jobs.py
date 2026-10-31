"""Resumable ingestion job bookkeeping."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from football_ai.database import models as db
from football_ai.database.models import utcnow

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"


class JobRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, provider: str, job_type: str, scope: str) -> db.IngestionJob | None:
        return self.session.scalar(
            select(db.IngestionJob).where(
                db.IngestionJob.provider == provider,
                db.IngestionJob.job_type == job_type,
                db.IngestionJob.scope == scope,
            )
        )

    def get_or_create(self, provider: str, job_type: str, scope: str) -> db.IngestionJob:
        job = self.get(provider, job_type, scope)
        if job is None:
            job = db.IngestionJob(
                provider=provider,
                job_type=job_type,
                scope=scope,
                status=STATUS_PENDING,
                attempts=0,
                requests_used=0,
                items_loaded=0,
            )
            self.session.add(job)
            self.session.flush()
        return job

    def finished_at(self, provider: str, job_type: str, scope: str) -> datetime | None:
        job = self.get(provider, job_type, scope)
        if job is None or job.status != STATUS_DONE:
            return None
        return job.finished_at

    def start(self, job: db.IngestionJob) -> None:
        job.status = STATUS_RUNNING
        job.attempts += 1
        job.started_at = utcnow()
        job.last_error = None
        self.session.flush()

    def succeed(self, job: db.IngestionJob, *, requests: int, items: int) -> None:
        job.status = STATUS_DONE
        job.requests_used += requests
        job.items_loaded = items
        job.finished_at = utcnow()
        self.session.flush()

    def fail(self, job: db.IngestionJob, error: str, *, requests: int = 0) -> None:
        job.status = STATUS_FAILED
        job.requests_used += requests
        job.last_error = error[:2000]
        job.finished_at = utcnow()
        self.session.flush()

    def list(self, provider: str | None = None) -> list[db.IngestionJob]:
        stmt = select(db.IngestionJob).order_by(db.IngestionJob.job_type, db.IngestionJob.scope)
        if provider:
            stmt = stmt.where(db.IngestionJob.provider == provider)
        return list(self.session.scalars(stmt))
