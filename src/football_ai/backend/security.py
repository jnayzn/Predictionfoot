"""API key authentication and per-client rate limiting."""

from __future__ import annotations

import hmac
import time
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException, Request, status

API_KEY_HEADER = "X-API-Key"


class RateLimiter:
    """Sliding 60-second window per client (in-process; Redis-backed in production)."""

    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.per_minute = per_minute
        self.clock = clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, client: str, per_minute: int | None = None) -> None:
        limit = per_minute or self.per_minute
        now = self.clock()
        window = self._hits[client]
        while window and now - window[0] >= 60:
            window.popleft()
        if len(window) >= limit:
            retry = int(60 - (now - window[0])) + 1
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                detail="rate limit exceeded",
                headers={"Retry-After": str(retry)},
            )
        window.append(now)


class RedisRateLimiter:
    """Fixed one-minute window shared by every API worker (INCR + EXPIRE)."""

    def __init__(
        self, client: Any, per_minute: int, clock: Callable[[], float] = time.time
    ) -> None:
        self.client = client
        self.per_minute = per_minute
        self.clock = clock

    def check(self, client_id: str, per_minute: int | None = None) -> None:
        now = self.clock()
        window = int(now // 60)
        key = f"football_ai:ratelimit:{client_id}:{window}"
        count = int(self.client.incr(key))
        if count == 1:
            self.client.expire(key, 120)
        if count > (per_minute or self.per_minute):
            retry = int(60 - now % 60) + 1
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                detail="rate limit exceeded",
                headers={"Retry-After": str(retry)},
            )


def build_rate_limiter(redis_url: str | None, per_minute: int) -> RateLimiter | RedisRateLimiter:
    if redis_url:
        import redis

        return RedisRateLimiter(redis.Redis.from_url(redis_url), per_minute)
    return RateLimiter(per_minute)


class ClientDirectory:
    """Resolves database API clients (hashed keys) with a short in-memory cache,
    and meters their daily usage."""

    def __init__(self, session_factory: Any, ttl_seconds: float = 60.0) -> None:
        self.session_factory = session_factory
        self.ttl = ttl_seconds
        self._cache: dict[str, tuple[float, Any]] = {}

    def lookup(self, key: str) -> Any:
        from football_ai.database.connection import session_scope
        from football_ai.database.repositories import ApiClientRepository
        from football_ai.database.repositories.api_clients import hash_key

        digest = hash_key(key)
        cached = self._cache.get(digest)
        if cached is not None and time.monotonic() - cached[0] < self.ttl:
            return cached[1]
        with session_scope(self.session_factory) as session:
            identity = ApiClientRepository(session).authenticate(key)
        self._cache[digest] = (time.monotonic(), identity)
        return identity

    def record(self, client_id: int) -> int:
        from football_ai.database.connection import session_scope
        from football_ai.database.repositories import ApiClientRepository

        with session_scope(self.session_factory) as session:
            return ApiClientRepository(session).record_request(client_id)


def authenticate(request: Request) -> str:
    """Return the client identity; raise 401/429 according to keys, plans and quotas.

    * admin keys (``API_KEYS``) - unlimited daily usage, default rate limit;
    * database clients (``football-ai api-keys create``) - plan rate limit and
      daily quota, usage metered per day;
    * no key configured at all - open access, development only.
    """
    state = request.app.state
    provided = request.headers.get(API_KEY_HEADER, "")
    admin_keys: set[str] = state.api_keys
    request.state.client = None
    if provided and any(hmac.compare_digest(provided, key) for key in admin_keys):
        client = f"admin:{provided[:6]}"
        state.rate_limiter.check(client)
        return client
    directory: ClientDirectory | None = getattr(state, "clients", None)
    identity = directory.lookup(provided) if provided and directory is not None else None
    if identity is not None:
        client = f"client:{identity.id}"
        state.rate_limiter.check(client, identity.rate_limit_per_minute)
        used = directory.record(identity.id) if directory is not None else 0
        if used > identity.daily_quota:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"daily quota of the {identity.plan} plan exceeded ({identity.daily_quota})",
            )
        request.state.client = identity
        return client
    if provided or not state.open_access:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="invalid or missing API key")
    client = f"ip:{request.client.host if request.client else 'unknown'}"
    state.rate_limiter.check(client)
    return client
