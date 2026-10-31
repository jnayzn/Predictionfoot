"""Quota management for API-Football.

Two limits are enforced *before* each request:

* **Daily quota** - persisted on disk so it survives restarts. API-Football
  resets the daily counter at 00:00 UTC. A configurable ``reserve`` is never
  consumed automatically (kept for manual / emergency calls).
* **Per-minute rate limit** - sliding 60 s window; the caller waits for a free
  slot instead of being rejected by the server.

The local counter is re-synchronised with the server's own figures whenever
the response carries the ``x-ratelimit-requests-*`` headers, which makes the
server the source of truth.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from football_ai.core.exceptions import QuotaExceededError
from football_ai.core.io import atomic_write_json, read_json

logger = logging.getLogger(__name__)

HEADER_DAILY_LIMIT = "x-ratelimit-requests-limit"
HEADER_DAILY_REMAINING = "x-ratelimit-requests-remaining"
HEADER_MINUTE_LIMIT = "x-ratelimit-limit"
HEADER_MINUTE_REMAINING = "x-ratelimit-remaining"

_WINDOW_SECONDS = 60.0


@dataclass(frozen=True, slots=True)
class QuotaSnapshot:
    day: str
    used: int
    daily_limit: int
    reserve: int
    server_remaining: int | None
    minute_limit: int

    @property
    def remaining(self) -> int:
        return max(0, self.daily_limit - self.used)

    @property
    def available(self) -> int:
        """Requests the collector may still spend today (reserve excluded)."""
        return max(0, self.daily_limit - self.reserve - self.used)


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value.strip())
    except ValueError:
        return None


class QuotaManager:
    def __init__(
        self,
        *,
        daily_limit: int,
        per_minute_limit: int,
        reserve: int = 0,
        state_file: Path | None = None,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if reserve >= daily_limit:
            raise ValueError("quota reserve must be lower than the daily limit")
        self._configured_daily_limit = daily_limit
        self._server_daily_limit: int | None = None
        self._minute_limit = per_minute_limit
        self._reserve = reserve
        self._state_file = state_file
        self._clock = clock
        self._monotonic = monotonic
        self._sleep = sleep
        self._lock = asyncio.Lock()
        self._window: deque[float] = deque()
        self._day = self._today()
        self._used = 0
        self._server_remaining: int | None = None
        self._load()

    # ------------------------------------------------------------------ state
    def _today(self) -> str:
        return datetime.fromtimestamp(self._clock(), tz=UTC).date().isoformat()

    @property
    def daily_limit(self) -> int:
        if self._server_daily_limit is not None:
            return min(self._configured_daily_limit, self._server_daily_limit)
        return self._configured_daily_limit

    def _rollover(self) -> None:
        today = self._today()
        if today != self._day:
            logger.info("quota_daily_reset", extra={"previous_day": self._day, "day": today})
            self._day = today
            self._used = 0
            self._server_remaining = None
            self._persist()

    def _load(self) -> None:
        if not self._state_file or not self._state_file.is_file():
            return
        try:
            state = read_json(self._state_file)
            if state.get("day") == self._day:
                self._used = int(state.get("used", 0))
                remaining = state.get("server_remaining")
                self._server_remaining = int(remaining) if remaining is not None else None
                server_limit = state.get("server_daily_limit")
                self._server_daily_limit = int(server_limit) if server_limit is not None else None
        except (OSError, ValueError, TypeError, AttributeError):
            logger.warning("quota_state_unreadable", extra={"path": str(self._state_file)})

    def _persist(self) -> None:
        if not self._state_file:
            return
        atomic_write_json(
            self._state_file,
            {
                "day": self._day,
                "used": self._used,
                "server_remaining": self._server_remaining,
                "server_daily_limit": self._server_daily_limit,
                "updated_at": datetime.fromtimestamp(self._clock(), tz=UTC).isoformat(),
            },
        )

    def snapshot(self) -> QuotaSnapshot:
        self._rollover()
        return QuotaSnapshot(
            day=self._day,
            used=self._used,
            daily_limit=self.daily_limit,
            reserve=self._reserve,
            server_remaining=self._server_remaining,
            minute_limit=self._minute_limit,
        )

    # ---------------------------------------------------------------- acquire
    async def acquire(self, *, counts_against_daily: bool = True) -> None:
        """Reserve one request slot, waiting for the per-minute window if needed.

        Raises:
            QuotaExceededError: if the daily quota (minus the reserve) is exhausted.
        """
        async with self._lock:
            self._rollover()
            if counts_against_daily and self._used + 1 > self.daily_limit - self._reserve:
                raise QuotaExceededError(
                    f"Daily API quota reached: {self._used}/{self.daily_limit} used "
                    f"({self._reserve} kept in reserve). Resets at 00:00 UTC."
                )
            await self._wait_for_minute_slot()
            self._window.append(self._monotonic())
            if counts_against_daily:
                self._used += 1
                self._persist()

    async def _wait_for_minute_slot(self) -> None:
        while True:
            now = self._monotonic()
            while self._window and now - self._window[0] >= _WINDOW_SECONDS:
                self._window.popleft()
            if len(self._window) < self._minute_limit:
                return
            wait = _WINDOW_SECONDS - (now - self._window[0]) + 0.05
            logger.info("quota_minute_window_wait", extra={"wait_seconds": round(wait, 2)})
            await self._sleep(wait)

    # ------------------------------------------------------- server feedback
    def update_from_headers(self, headers: Mapping[str, str]) -> None:
        lowered = {k.lower(): v for k, v in headers.items()}
        limit = _parse_int(lowered.get(HEADER_DAILY_LIMIT))
        remaining = _parse_int(lowered.get(HEADER_DAILY_REMAINING))
        minute_limit = _parse_int(lowered.get(HEADER_MINUTE_LIMIT))
        if minute_limit is not None and minute_limit > 0:
            self._minute_limit = min(self._minute_limit, minute_limit)
        if limit is not None and limit > 0:
            self._server_daily_limit = limit
        if remaining is not None:
            self._server_remaining = remaining
            reference = limit if limit is not None else self.daily_limit
            server_used = max(0, reference - remaining)
            if server_used != self._used:
                logger.debug(
                    "quota_resync", extra={"local_used": self._used, "server_used": server_used}
                )
            self._used = server_used
        if limit is not None or remaining is not None:
            self._persist()

    def mark_exhausted(self) -> None:
        """Called when the server says the daily quota is reached."""
        self._used = self.daily_limit
        self._server_remaining = 0
        self._persist()
