"""Low-level asynchronous client for API-Football v3.

Responsibilities (and nothing more - no business logic here):

* authentication header (``x-apisports-key``), never logged;
* response cache lookup (a hit costs no quota);
* de-duplication of identical concurrent requests (one network call);
* quota / rate limit enforcement via :class:`QuotaManager`;
* retries with exponential backoff + jitter on network errors, 429 and 5xx;
* mapping of HTTP errors *and* of API-level errors (API-Football returns
  HTTP 200 with a non-empty ``errors`` field for many failures);
* archiving of every raw JSON response;
* pagination helper.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from football_ai.config.settings import ApiFootballSettings
from football_ai.core.exceptions import (
    ApiNetworkError,
    ApiRequestError,
    ApiResponseFormatError,
    ApiServerError,
    AuthenticationError,
    MissingApiKeyError,
    PlanRestrictionError,
    QuotaExceededError,
    RateLimitError,
)
from football_ai.data.collectors.api_football.quota import QuotaManager
from football_ai.data.storage.raw_archive import RawArchive
from football_ai.data.storage.response_cache import ResponseCache, request_fingerprint

logger = logging.getLogger(__name__)

PROVIDER = "api_football"
AUTH_HEADER = "x-apisports-key"
# Endpoints that the provider documents as not counting against the daily quota.
_QUOTA_FREE_ENDPOINTS = frozenset({"status"})
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True, slots=True)
class ApiResponse:
    endpoint: str
    params: dict[str, str]
    payload: dict[str, Any]
    from_cache: bool
    fetched_at: float
    paging_current: int = 1
    paging_total: int = 1
    results: int = 0
    response: list[Any] | dict[str, Any] = field(default_factory=list)


class _RetryableApiError(Exception):
    def __init__(self, reason: str, retry_after: float | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.retry_after = retry_after


def _normalize_params(params: Mapping[str, Any] | None) -> dict[str, str]:
    if not params:
        return {}
    out: dict[str, str] = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, bool):
            out[key] = "true" if value else "false"
        else:
            out[key] = str(value)
    return dict(sorted(out.items()))


def _parse_retry_after(value: str | None, now: float) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        return max(0.0, parsedate_to_datetime(value).timestamp() - now)
    except (TypeError, ValueError):
        return None


def _errors_as_dict(errors: Any) -> dict[str, str]:
    """API-Football returns ``errors`` as ``[]`` when empty, or a dict / list otherwise."""
    if not errors:
        return {}
    if isinstance(errors, dict):
        return {str(k): str(v) for k, v in errors.items()}
    if isinstance(errors, list):
        return {str(i): str(v) for i, v in enumerate(errors)}
    return {"error": str(errors)}


class ApiFootballClient:
    def __init__(
        self,
        settings: ApiFootballSettings,
        quota: QuotaManager,
        *,
        cache: ResponseCache | None = None,
        archive: RawArchive | None = None,
        http_client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: Callable[[], float] = random.random,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._settings = settings
        self._quota = quota
        self._cache = cache
        self._archive = archive
        self._owns_http = http_client is None
        self._http = http_client or httpx.AsyncClient(
            base_url=settings.base_url.rstrip("/"),
            timeout=httpx.Timeout(settings.timeout_seconds),
            headers={"Accept": "application/json", "User-Agent": "football-ai-predictor/0.1"},
        )
        self._sleep = sleep
        self._rng = rng
        self._clock = clock
        self._inflight: dict[str, asyncio.Future[ApiResponse]] = {}

    # --------------------------------------------------------------- lifecycle
    async def __aenter__(self) -> ApiFootballClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    @property
    def quota(self) -> QuotaManager:
        return self._quota

    # -------------------------------------------------------------- public API
    async def get(
        self,
        endpoint: str,
        params: Mapping[str, Any] | None = None,
        *,
        cache_ttl: int | None = None,
    ) -> ApiResponse:
        """GET an endpoint. ``cache_ttl=None`` disables the cache for this call."""
        endpoint = endpoint.strip("/")
        norm = _normalize_params(params)
        fingerprint = request_fingerprint(PROVIDER, endpoint, norm)

        if self._cache is not None and cache_ttl is not None and cache_ttl > 0:
            hit = self._cache.get(PROVIDER, endpoint, fingerprint, cache_ttl)
            if hit is not None:
                logger.debug("api_cache_hit", extra={"endpoint": endpoint, "params": norm})
                return self._build_response(endpoint, norm, hit.payload, True, hit.cached_at)

        existing = self._inflight.get(fingerprint)
        if existing is not None:
            logger.debug("api_request_deduplicated", extra={"endpoint": endpoint})
            return await asyncio.shield(existing)

        future: asyncio.Future[ApiResponse] = asyncio.get_running_loop().create_future()
        self._inflight[fingerprint] = future
        try:
            result = await self._fetch(endpoint, norm, fingerprint, cache_ttl)
        except asyncio.CancelledError:
            future.cancel()
            raise
        except BaseException as exc:
            future.set_exception(exc)
            future.exception()  # mark retrieved: avoid "never retrieved" warnings
            raise
        else:
            future.set_result(result)
            return result
        finally:
            self._inflight.pop(fingerprint, None)

    async def get_all_pages(
        self,
        endpoint: str,
        params: Mapping[str, Any] | None = None,
        *,
        cache_ttl: int | None = None,
        max_pages: int = 50,
    ) -> list[ApiResponse]:
        """Follow API-Football ``paging`` (``page`` parameter) until the last page."""
        base = dict(params or {})
        first = await self.get(endpoint, base, cache_ttl=cache_ttl)
        pages = [first]
        total = min(first.paging_total, max_pages)
        if first.paging_total > max_pages:
            logger.warning(
                "api_pagination_truncated",
                extra={"endpoint": endpoint, "total": first.paging_total, "max_pages": max_pages},
            )
        for page in range(2, total + 1):
            pages.append(await self.get(endpoint, {**base, "page": page}, cache_ttl=cache_ttl))
        return pages

    async def status(self) -> ApiResponse:
        """Account / subscription status (does not count against the daily quota)."""
        return await self.get("status")

    # ---------------------------------------------------------------- internals
    def _build_response(
        self,
        endpoint: str,
        params: dict[str, str],
        payload: dict[str, Any],
        from_cache: bool,
        fetched_at: float,
    ) -> ApiResponse:
        paging = payload.get("paging") or {}
        try:
            current = int(paging.get("current", 1))
            total = int(paging.get("total", 1))
        except (TypeError, ValueError):
            current, total = 1, 1
        response = payload.get("response", [])
        if not isinstance(response, list | dict):
            raise ApiResponseFormatError(f"'response' has unexpected type {type(response)}")
        results = payload.get("results")
        return ApiResponse(
            endpoint=endpoint,
            params=params,
            payload=payload,
            from_cache=from_cache,
            fetched_at=fetched_at,
            paging_current=current,
            paging_total=max(total, 1),
            results=int(results) if isinstance(results, int) else len(response),
            response=response,
        )

    def _backoff(self, attempt: int) -> float:
        base = self._settings.backoff_base_seconds * (2**attempt)
        jitter = base * 0.25 * self._rng()
        return float(min(self._settings.backoff_max_seconds, base + jitter))

    async def _fetch(
        self, endpoint: str, params: dict[str, str], fingerprint: str, cache_ttl: int | None
    ) -> ApiResponse:
        if not self._settings.is_configured or self._settings.key is None:
            raise MissingApiKeyError(
                "API_FOOTBALL_KEY is not set. Add it to your .env file (see .env.example)."
            )
        key = self._settings.key.get_secret_value()
        counts = endpoint not in _QUOTA_FREE_ENDPOINTS
        max_attempts = self._settings.max_retries + 1

        for attempt in range(max_attempts):
            await self._quota.acquire(counts_against_daily=counts)
            started = time.perf_counter()
            try:
                payload, status_code = await self._send(endpoint, params, key)
            except _RetryableApiError as exc:
                last_attempt = attempt + 1 >= max_attempts
                logger.warning(
                    "api_request_retryable_failure",
                    extra={
                        "endpoint": endpoint,
                        "reason": exc.reason,
                        "attempt": attempt + 1,
                        "max_attempts": max_attempts,
                    },
                )
                if last_attempt:
                    raise self._exhausted_error(exc) from exc
                delay = max(exc.retry_after or 0.0, self._backoff(attempt))
                await self._sleep(min(delay, max(self._settings.backoff_max_seconds, 1.0)))
                continue

            snapshot = self._quota.snapshot()
            logger.info(
                "api_request_ok",
                extra={
                    "endpoint": endpoint,
                    "params": params,
                    "status": status_code,
                    "results": payload.get("results"),
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                    "quota_used": snapshot.used,
                    "quota_limit": snapshot.daily_limit,
                },
            )
            fetched_at = self._clock()
            response = self._build_response(endpoint, params, payload, False, fetched_at)
            if self._archive is not None:
                self._archive.save(
                    provider=PROVIDER,
                    endpoint=endpoint,
                    params=params,
                    fingerprint=fingerprint,
                    payload=payload,
                    status_code=status_code,
                    meta={"quota_used": snapshot.used, "quota_limit": snapshot.daily_limit},
                )
            if self._cache is not None and cache_ttl is not None and cache_ttl > 0:
                self._cache.set(PROVIDER, endpoint, fingerprint, params, payload)
            return response

        raise AssertionError("unreachable")  # pragma: no cover

    @staticmethod
    def _exhausted_error(exc: _RetryableApiError) -> Exception:
        if exc.reason.startswith("rate_limit"):
            return RateLimitError(f"Rate limit still hit after retries ({exc.reason})")
        if exc.reason.startswith("network"):
            return ApiNetworkError(f"Network error after retries ({exc.reason})")
        return ApiServerError(f"Server error after retries ({exc.reason})")

    async def _send(
        self, endpoint: str, params: dict[str, str], key: str
    ) -> tuple[dict[str, Any], int]:
        try:
            resp = await self._http.get(f"/{endpoint}", params=params, headers={AUTH_HEADER: key})
        except httpx.TimeoutException as exc:
            raise _RetryableApiError(f"network_timeout:{type(exc).__name__}") from exc
        except httpx.TransportError as exc:
            raise _RetryableApiError(f"network_error:{type(exc).__name__}") from exc

        self._quota.update_from_headers(resp.headers)
        status = resp.status_code

        if status in (401, 403):
            raise AuthenticationError(f"API-Football rejected the API key (HTTP {status}).")
        if status in _RETRYABLE_STATUS:
            retry_after = _parse_retry_after(resp.headers.get("retry-after"), self._clock())
            reason = "rate_limit_http_429" if status == 429 else f"server_http_{status}"
            raise _RetryableApiError(reason, retry_after)
        if status >= 400:
            raise ApiRequestError(f"API-Football HTTP {status} for /{endpoint}", status_code=status)

        try:
            payload = resp.json()
        except ValueError as exc:
            raise ApiResponseFormatError(f"Invalid JSON from /{endpoint}") from exc
        if not isinstance(payload, dict):
            raise ApiResponseFormatError(f"Unexpected JSON root type from /{endpoint}")

        self._raise_for_api_errors(endpoint, _errors_as_dict(payload.get("errors")), status)
        if "response" not in payload:
            raise ApiResponseFormatError(f"Missing 'response' field in /{endpoint} payload")
        return payload, status

    def _raise_for_api_errors(self, endpoint: str, errors: dict[str, str], status: int) -> None:
        if not errors:
            return
        keys = {k.lower() for k in errors}
        text = "; ".join(f"{k}: {v}" for k, v in errors.items())
        if "token" in keys or "access" in keys:
            raise AuthenticationError(f"API-Football authentication error: {text}")
        if "requests" in keys:
            self._quota.mark_exhausted()
            raise QuotaExceededError(f"API-Football daily quota reached: {text}")
        if "ratelimit" in keys:
            # Per-minute limit: wait roughly one slot of the minute window.
            per_minute = self._quota.snapshot().minute_limit
            raise _RetryableApiError("rate_limit_api_error", retry_after=60.0 / per_minute)
        if "plan" in keys:
            raise PlanRestrictionError(f"Not available on the current plan: {text}")
        raise ApiRequestError(
            f"API-Football error on /{endpoint}: {text}", errors=errors, status_code=status
        )
