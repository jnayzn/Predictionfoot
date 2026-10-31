"""API clients (hashed keys, plans) and usage metering."""

from __future__ import annotations

import hashlib
import secrets
import tomllib
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from football_ai.core.exceptions import ConfigurationError
from football_ai.database import models as db

KEY_PREFIX = "fai_"


@dataclass(frozen=True, slots=True)
class Plan:
    name: str
    daily_quota: int
    rate_limit_per_minute: int
    description: str = ""


def load_plans(path: Path) -> dict[str, Plan]:
    if not path.is_file():
        raise ConfigurationError(f"plans file not found: {path}")
    raw = tomllib.loads(path.read_text(encoding="utf-8")).get("plans", {})
    return {
        name: Plan(
            name, int(v["daily_quota"]), int(v["rate_limit_per_minute"]), v.get("description", "")
        )
        for name, v in raw.items()
    }


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ClientIdentity:
    id: int
    name: str
    plan: str
    daily_quota: int
    rate_limit_per_minute: int


class ApiClientRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, name: str, plan: Plan) -> tuple[db.ApiClient, str]:
        """Create a client; the clear-text key is returned ONCE and never stored."""
        key = KEY_PREFIX + secrets.token_urlsafe(32)
        row = db.ApiClient(
            name=name,
            key_prefix=key[:12],
            key_hash=hash_key(key),
            plan=plan.name,
            daily_quota=plan.daily_quota,
            rate_limit_per_minute=plan.rate_limit_per_minute,
            active=True,
        )
        self.session.add(row)
        self.session.flush()
        return row, key

    def authenticate(self, key: str) -> ClientIdentity | None:
        row = self.session.scalar(
            select(db.ApiClient).where(
                db.ApiClient.key_hash == hash_key(key), db.ApiClient.active.is_(True)
            )
        )
        if row is None:
            return None
        return ClientIdentity(
            row.id, row.name, row.plan, row.daily_quota, row.rate_limit_per_minute
        )

    def record_request(self, client_id: int, day: date | None = None) -> int:
        """Increment today's counter and return the new value."""
        day = day or datetime.now(UTC).date()
        usage = self.session.scalar(
            select(db.ApiUsage).where(db.ApiUsage.client_id == client_id, db.ApiUsage.day == day)
        )
        if usage is None:
            usage = db.ApiUsage(client_id=client_id, day=day, requests=0)
            self.session.add(usage)
        usage.requests += 1
        client = self.session.get(db.ApiClient, client_id)
        if client is not None:
            client.last_used_at = datetime.now(UTC)
        self.session.flush()
        return usage.requests

    def usage(
        self, client_id: int | None = None, since: date | None = None
    ) -> list[tuple[str, date, int]]:
        stmt = select(db.ApiClient.name, db.ApiUsage.day, db.ApiUsage.requests).join(
            db.ApiClient, db.ApiClient.id == db.ApiUsage.client_id
        )
        if client_id is not None:
            stmt = stmt.where(db.ApiUsage.client_id == client_id)
        if since is not None:
            stmt = stmt.where(db.ApiUsage.day >= since)
        return [(n, d, int(r)) for n, d, r in self.session.execute(stmt.order_by(db.ApiUsage.day))]

    def list(self) -> list[db.ApiClient]:
        return list(self.session.scalars(select(db.ApiClient).order_by(db.ApiClient.created_at)))

    def revoke(self, key_prefix: str) -> int:
        rows = list(
            self.session.scalars(select(db.ApiClient).where(db.ApiClient.key_prefix == key_prefix))
        )
        for row in rows:
            row.active = False
        self.session.flush()
        return len(rows)
