"""Append-only archive of every raw JSON response fetched from a provider.

Layout::

    <root>/<provider>/<endpoint>/<YYYY-MM-DD>/<HHMMSS.ffffff>_<fingerprint[:12]>.json

Files are never overwritten: they are the audit trail used by the ETL
(Phase 2) and allow re-processing without spending API quota.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_ai.core.io import atomic_write_json
from football_ai.data.storage.response_cache import endpoint_slug


class RawArchive:
    def __init__(self, root: Path, clock: Callable[[], float] = time.time) -> None:
        self._root = root
        self._clock = clock

    def save(
        self,
        *,
        provider: str,
        endpoint: str,
        params: Mapping[str, str],
        fingerprint: str,
        payload: dict[str, Any],
        status_code: int,
        meta: Mapping[str, Any] | None = None,
    ) -> Path:
        fetched = datetime.fromtimestamp(self._clock(), tz=UTC)
        directory = self._root / provider / endpoint_slug(endpoint) / fetched.strftime("%Y-%m-%d")
        path = directory / f"{fetched.strftime('%H%M%S.%f')}_{fingerprint[:12]}.json"
        suffix = 1
        while path.exists():  # extremely unlikely, but never overwrite
            path = directory / f"{fetched.strftime('%H%M%S.%f')}_{fingerprint[:12]}_{suffix}.json"
            suffix += 1
        atomic_write_json(
            path,
            {
                "meta": {
                    "provider": provider,
                    "endpoint": endpoint,
                    "params": dict(params),
                    "fingerprint": fingerprint,
                    "fetched_at": fetched.isoformat(),
                    "status_code": status_code,
                    **(dict(meta) if meta else {}),
                },
                "payload": payload,
            },
        )
        return path
