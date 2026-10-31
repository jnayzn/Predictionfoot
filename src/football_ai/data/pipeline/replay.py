"""Rebuild the database from the raw JSON archive - zero API request.

Every archived response is re-parsed with exactly the same provider logic as
a live call, then loaded through the repositories (idempotent upserts).
Files are processed in chronological order so the latest data wins.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from football_ai.core.io import read_json
from football_ai.data.collectors.api_football.parsers import PROVIDER, ParseError, parse_league
from football_ai.data.collectors.api_football.provider import ApiFootballProvider
from football_ai.database.connection import session_scope
from football_ai.database.repositories import (
    FixtureRepository,
    MatchDataRepository,
    ReferenceRepository,
)

logger = logging.getLogger(__name__)

REPLAYABLE = ("leagues", "teams", "fixtures", "standings", "injuries", "odds")


@dataclass(slots=True)
class ReplayReport:
    files: int = 0
    skipped_files: int = 0
    items: dict[str, int] = field(default_factory=dict)

    def add(self, kind: str, count: int) -> None:
        self.items[kind] = self.items.get(kind, 0) + count


class RawReplayer:
    def __init__(
        self,
        raw_root: Path,
        provider: ApiFootballProvider,
        session_factory: sessionmaker[Session],
    ) -> None:
        self.root = raw_root / PROVIDER
        self.provider = provider
        self.session_factory = session_factory

    def _files(self, endpoints: tuple[str, ...]) -> list[tuple[str, Path]]:
        files: list[tuple[str, Path]] = []
        for endpoint in endpoints:
            directory = self.root / endpoint
            if directory.is_dir():
                files.extend((endpoint, p) for p in directory.rglob("*.json"))
        # chronological: day directory then HHMMSS.micro filename
        return sorted(files, key=lambda item: (item[1].parent.name, item[1].name))

    def run(self, endpoints: tuple[str, ...] = REPLAYABLE) -> ReplayReport:
        report = ReplayReport()
        for endpoint, path in self._files(endpoints):
            try:
                envelope = read_json(path)
                meta: dict[str, Any] = envelope["meta"]
                response = envelope["payload"]["response"]
            except (OSError, ValueError, KeyError, TypeError):
                logger.warning("replay_unreadable_file", extra={"path": str(path)})
                report.skipped_files += 1
                continue
            fetched = datetime.fromisoformat(meta["fetched_at"]).date()
            with session_scope(self.session_factory) as session:
                self._load(session, endpoint, meta.get("params") or {}, response, fetched, report)
            report.files += 1
        logger.info("replay_finished", extra={"files": report.files, "items": report.items})
        return report

    def _load(
        self,
        session: Session,
        endpoint: str,
        params: dict[str, str],
        response: Any,
        fetched: date,
        report: ReplayReport,
    ) -> None:
        items = response if isinstance(response, list) else []
        refs = ReferenceRepository(session)
        match_data = MatchDataRepository(session, refs)
        if endpoint == "leagues":
            by_id = {int(c.provider_ids[PROVIDER]): c for c in self.provider.competitions}
            for item in items:
                try:
                    league_id = int(item["league"]["id"])
                except (KeyError, TypeError, ValueError):
                    continue
                if league_id in by_id:
                    try:
                        refs.upsert_competition(parse_league(item, key=by_id[league_id].key))
                    except ParseError:
                        continue
                    report.add("competitions", 1)
        elif endpoint == "teams":
            for team in self.provider.parse_teams_payload(items):
                refs.upsert_team(team)
            report.add("teams", len(items))
        elif endpoint == "fixtures" and "ids" in params:
            details = self.provider.parse_details_payload(items)
            match_data.save_fixture_details(details.details)
            report.add("details", len(details.details))
        elif endpoint == "fixtures":
            batch = self.provider.build_fixture_batch(items, self.provider.covered_league_ids)
            FixtureRepository(session, refs).upsert_fixtures(batch.fixtures)
            report.add("fixtures", len(batch.fixtures))
        elif endpoint == "standings":
            rows = self.provider.parse_standings_payload(items)
            match_data.save_standings(rows, snapshot_date=fetched)
            report.add("standings", len(rows))
        elif endpoint == "injuries":
            injuries = self.provider.parse_injuries_payload(items)
            match_data.save_injuries(injuries)
            report.add("injuries", len(injuries))
        elif endpoint == "odds":
            report.add("odds", match_data.save_odds(self.provider.parse_odds_payload(items)))
