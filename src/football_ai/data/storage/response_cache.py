"""Disk cache of provider responses with per-call TTL.

A cache hit never consumes API quota. Keys are a stable hash of
``(provider, endpoint, sorted params)`` - the API key is never part of it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from football_ai.core.io import atomic_write_json, read_json

logger = logging.getLogger(__name__)


def request_fingerprint(provider: str, endpoint: str, params: Mapping[str, str]) -> str:
    canonical = json.dumps(
        {"provider": provider, "endpoint": endpoint, "params": sorted(params.items())},
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def endpoint_slug(endpoint: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", endpoint.strip("/")) or "root"


@dataclass(frozen=True, slots=True)
class CachedResponse:
    payload: dict[str, Any]
    cached_at: float


class ResponseCache:
    def __init__(self, root: Path, clock: Callable[[], float] = time.time) -> None:
        self._root = root
        self._clock = clock

    def _path(self, provider: str, endpoint: str, fingerprint: str) -> Path:
        return self._root / provider / endpoint_slug(endpoint) / f"{fingerprint}.json"

    def get(
        self, provider: str, endpoint: str, fingerprint: str, ttl_seconds: int
    ) -> CachedResponse | None:
        path = self._path(provider, endpoint, fingerprint)
        if not path.is_file():
            return None
        try:
            entry = read_json(path)
            cached_at = float(entry["cached_at"])
            payload = entry["payload"]
        except (OSError, ValueError, KeyError, TypeError):
            logger.warning("cache_entry_corrupted", extra={"path": str(path)})
            path.unlink(missing_ok=True)
            return None
        if self._clock() - cached_at > ttl_seconds:
            return None
        return CachedResponse(payload=payload, cached_at=cached_at)

    def set(
        self,
        provider: str,
        endpoint: str,
        fingerprint: str,
        params: Mapping[str, str],
        payload: dict[str, Any],
    ) -> None:
        atomic_write_json(
            self._path(provider, endpoint, fingerprint),
            {
                "cached_at": self._clock(),
                "provider": provider,
                "endpoint": endpoint,
                "params": dict(params),
                "payload": payload,
            },
        )
