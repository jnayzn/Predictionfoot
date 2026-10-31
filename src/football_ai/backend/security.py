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

    def check(self, client: str) -> None:
        now = self.clock()
        window = self._hits[client]
        while window and now - window[0] >= 60:
            window.popleft()
        if len(window) >= self.per_minute:
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

    def check(self, client_id: str) -> None:
        now = self.clock()
        window = int(now // 60)
        key = f"football_ai:ratelimit:{client_id}:{window}"
        count = int(self.client.incr(key))
        if count == 1:
            self.client.expire(key, 120)
        if count > self.per_minute:
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


def authenticate(request: Request) -> str:
    """Return the client identity; raise 401 when keys are configured and none matches."""
    keys: set[str] = request.app.state.api_keys
    provided = request.headers.get(API_KEY_HEADER, "")
    if keys:
        if not any(hmac.compare_digest(provided, key) for key in keys):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="invalid or missing API key")
        client = f"key:{provided[:6]}"
    else:  # development mode without keys: identify by address
        client = f"ip:{request.client.host if request.client else 'unknown'}"
    request.app.state.rate_limiter.check(client)
    return client
