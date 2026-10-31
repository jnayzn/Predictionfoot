"""Download of football-data.co.uk season CSV files (free, no key).

Each downloaded file is archived under ``<raw>/football_data_uk/<div>/<season>.csv``.
A finished season already archived is never downloaded again.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from pathlib import Path

import httpx

from football_ai.core.exceptions import ApiNetworkError, ApiRequestError
from football_ai.data.collectors.football_data_uk.parser import season_code

logger = logging.getLogger(__name__)

BASE_URL = "https://www.football-data.co.uk/mmz4281"


class FootballDataUkCollector:
    def __init__(
        self, raw_root: Path, http_client: httpx.AsyncClient | None = None, timeout: float = 30.0
    ) -> None:
        self.root = raw_root / "football_data_uk"
        self._owns_http = http_client is None
        self._http = http_client or httpx.AsyncClient(timeout=timeout, follow_redirects=True)

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    def archive_path(self, division: str, season: int) -> Path:
        return self.root / division / f"{season}.csv"

    async def fetch_season(
        self, division: str, season: int, *, today: date | None = None, force: bool = False
    ) -> bytes:
        path = self.archive_path(division, season)
        today = today or datetime.now(UTC).date()
        finished = today > date(season + 1, 7, 31)
        if path.is_file() and (finished or not force):
            return path.read_bytes()
        url = f"{BASE_URL}/{season_code(season)}/{division}.csv"
        try:
            resp = await self._http.get(url)
        except httpx.HTTPError as exc:
            raise ApiNetworkError(f"football-data.co.uk unreachable: {exc}") from exc
        if resp.status_code != 200:
            raise ApiRequestError(
                f"football-data.co.uk HTTP {resp.status_code} for {url}",
                status_code=resp.status_code,
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(resp.content)
        tmp.replace(path)
        logger.info("csv_downloaded", extra={"url": url, "bytes": len(resp.content)})
        return resp.content
