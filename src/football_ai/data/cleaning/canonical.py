"""Build the canonical layer: canonical teams and one ``matches`` row per real match.

Source priority (configurable): API-Football first (richest, has kick-off
times and stable ids), then football-data.co.uk. The primary source of a
match provides its fields; other sources are linked through ``match_sources``
and any score disagreement sets ``source_conflict``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from football_ai.config.aliases import TeamAlias
from football_ai.data.cleaning.names import name_similarity
from football_ai.data.cleaning.team_resolution import FixtureKey, TeamMatch, resolve_teams
from football_ai.database import models as db
from football_ai.database.repositories._utils import apply_changes, chunks
from football_ai.domain.models import FixtureStatusGroup

logger = logging.getLogger(__name__)

DEFAULT_PRIORITY: tuple[str, ...] = ("api_football", "football_data_uk")
METHOD_PRIMARY = "primary"
METHOD_ALIAS = "alias"
METHOD_COOCCURRENCE = "cooccurrence"
METHOD_STANDALONE = "standalone"

_EXCLUDED_STATUS = {
    FixtureStatusGroup.AWARDED.value: "awarded",
    FixtureStatusGroup.ABANDONED.value: "abandoned",
    FixtureStatusGroup.CANCELLED.value: "cancelled",
    FixtureStatusGroup.POSTPONED.value: "postponed",
}


@dataclass(slots=True)
class BuildReport:
    competition_key: str
    links: dict[str, int] = field(default_factory=dict)
    low_confidence: list[str] = field(default_factory=list)
    matches_inserted: int = 0
    matches_updated: int = 0
    conflicts: int = 0
    excluded: int = 0

    def summary(self) -> str:
        return (
            f"{self.competition_key}: links={self.links} matches(+{self.matches_inserted} "
            f"~{self.matches_updated}) conflicts={self.conflicts} excluded={self.excluded} "
            f"low_confidence={len(self.low_confidence)}"
        )


def regulation_score(f: db.Fixture) -> tuple[int | None, int | None]:
    home = f.ft_home_goals if f.ft_home_goals is not None else f.home_goals
    away = f.ft_away_goals if f.ft_away_goals is not None else f.away_goals
    return home, away


class CanonicalBuilder:
    def __init__(
        self,
        session: Session,
        *,
        aliases: Sequence[TeamAlias] = (),
        priority: Sequence[str] = DEFAULT_PRIORITY,
        min_confidence: float = 0.6,
        min_appearances: int = 3,
        day_tolerance: int = 1,
    ) -> None:
        self.session = session
        self.aliases = {(a.provider, a.name.casefold()): a.canonical for a in aliases}
        self.priority = list(priority)
        self.min_confidence = min_confidence
        self.min_appearances = min_appearances
        self.day_tolerance = day_tolerance

    def _rank(self, provider: str) -> int:
        return self.priority.index(provider) if provider in self.priority else len(self.priority)

    # ------------------------------------------------------------------ load
    def _fixtures_by_provider(self, competition_key: str) -> dict[str, list[db.Fixture]]:
        league_ids = list(
            self.session.scalars(select(db.League.id).where(db.League.key == competition_key))
        )
        out: dict[str, list[db.Fixture]] = defaultdict(list)
        if not league_ids:
            return out
        for fx in self.session.scalars(
            select(db.Fixture).where(db.Fixture.league_id.in_(league_ids))
        ):
            out[fx.provider].append(fx)
        return out

    def _links(self, team_ids: Sequence[int]) -> dict[int, db.TeamLink]:
        links: dict[int, db.TeamLink] = {}
        for chunk in chunks(list(team_ids)):
            for link in self.session.scalars(
                select(db.TeamLink).where(db.TeamLink.team_id.in_(chunk))
            ):
                links[link.team_id] = link
        return links

    # ----------------------------------------------------------------- teams
    def _canonical_by_name(self, name: str, country: str | None) -> db.CanonicalTeam:
        row = self.session.scalar(select(db.CanonicalTeam).where(db.CanonicalTeam.name == name))
        if row is None:
            row = db.CanonicalTeam(name=name, country=country)
            self.session.add(row)
            self.session.flush()
        return row

    def _own_canonical(
        self, link: db.TeamLink | None, name: str, country: str | None
    ) -> db.CanonicalTeam:
        """Canonical team owned by one provider team (never merged by name)."""
        if link is not None and link.method in (METHOD_PRIMARY, METHOD_STANDALONE):
            row = self.session.get(db.CanonicalTeam, link.canonical_team_id)
            if row is not None:
                apply_changes(row, {"name": name, "country": country})
                return row
        row = db.CanonicalTeam(name=name, country=country)
        self.session.add(row)
        self.session.flush()
        return row

    def _set_link(
        self,
        links: dict[int, db.TeamLink],
        team: db.Team,
        canonical_id: int,
        method: str,
        confidence: float,
        evidence: dict[str, object],
    ) -> None:
        link = links.get(team.id)
        if link is None:
            link = db.TeamLink(
                team_id=team.id,
                canonical_team_id=canonical_id,
                method=method,
                confidence=confidence,
                evidence=evidence,
            )
            self.session.add(link)
            links[team.id] = link
        elif link.method != METHOD_ALIAS or method == METHOD_ALIAS:
            apply_changes(
                link,
                {
                    "canonical_team_id": canonical_id,
                    "method": method,
                    "confidence": confidence,
                    "evidence": evidence,
                },
            )

    def _link_teams(
        self,
        by_provider: dict[str, list[db.Fixture]],
        report: BuildReport,
        country: str | None,
    ) -> dict[int, int]:
        """Return team_id -> canonical_team_id for every team of the competition."""
        providers = sorted(by_provider, key=self._rank)
        team_ids = {
            tid for fixtures in by_provider.values() for f in fixtures
            for tid in (f.home_team_id, f.away_team_id)
        }  # fmt: skip
        teams = {
            t.id: t for t in self.session.scalars(select(db.Team).where(db.Team.id.in_(team_ids)))
        }
        links = self._links(list(team_ids))
        primary = providers[0]
        counts: dict[str, int] = defaultdict(int)

        primary_teams = {
            tid for f in by_provider[primary] for tid in (f.home_team_id, f.away_team_id)
        }
        for tid in primary_teams:
            team = teams[tid]
            alias = self.aliases.get((team.provider, team.name.casefold()))
            if tid in links and links[tid].method == METHOD_ALIAS:
                counts[METHOD_ALIAS] += 1
                continue
            if alias:
                canonical = self._canonical_by_name(alias, team.country or country)
                method = METHOD_ALIAS
            else:
                canonical = self._own_canonical(links.get(tid), team.name, team.country or country)
                method = METHOD_PRIMARY
            self._set_link(links, team, canonical.id, method, 1.0, {})
            counts[method] += 1
        self.session.flush()

        def keys(fixtures: list[db.Fixture]) -> list[FixtureKey]:
            out = []
            for f in fixtures:
                h, a = regulation_score(f)
                out.append(FixtureKey(f.kickoff_utc.date(), f.home_team_id, f.away_team_id, h, a))
            return out

        primary_keys = keys(by_provider[primary])
        for provider in providers[1:]:
            matches: dict[int, TeamMatch] = resolve_teams(
                keys(by_provider[provider]), primary_keys, day_tolerance=self.day_tolerance
            )
            provider_teams = {
                tid for f in by_provider[provider] for tid in (f.home_team_id, f.away_team_id)
            }
            for tid in sorted(provider_teams):
                team = teams[tid]
                existing = links.get(tid)
                if existing is not None and existing.method == METHOD_ALIAS:
                    counts[METHOD_ALIAS] += 1
                    continue
                alias = self.aliases.get((team.provider, team.name.casefold()))
                if alias:
                    canonical = self._canonical_by_name(alias, team.country or country)
                    self._set_link(links, team, canonical.id, METHOD_ALIAS, 1.0, {})
                    counts[METHOD_ALIAS] += 1
                    continue
                match = matches.get(tid)
                if (
                    match is not None
                    and match.appearances >= self.min_appearances
                    and match.confidence >= self.min_confidence
                ):
                    target = teams[match.target_team]
                    evidence = {
                        "target_team": target.name,
                        "votes": match.votes,
                        "appearances": match.appearances,
                        "verified": match.verified,
                        "name_similarity": round(name_similarity(team.name, target.name), 3),
                    }
                    self._set_link(
                        links,
                        team,
                        links[target.id].canonical_team_id,
                        METHOD_COOCCURRENCE,
                        match.confidence,
                        evidence,
                    )
                    counts[METHOD_COOCCURRENCE] += 1
                    continue
                if match is not None and match.appearances >= self.min_appearances:
                    report.low_confidence.append(
                        f"{provider}:{team.name} -> {teams[match.target_team].name} "
                        f"(confidence {match.confidence:.2f})"
                    )
                if existing is not None and existing.method == METHOD_COOCCURRENCE:
                    counts[METHOD_COOCCURRENCE] += 1  # keep a previous confident link
                    continue
                previous = existing if existing and existing.method == METHOD_STANDALONE else None
                canonical = self._own_canonical(previous, team.name, team.country)
                self._set_link(links, team, canonical.id, METHOD_STANDALONE, 0.0, {})
                counts[METHOD_STANDALONE] += 1
        self.session.flush()
        report.links = dict(counts)
        return {tid: link.canonical_team_id for tid, link in links.items()}

    # --------------------------------------------------------------- matches
    def _cluster(
        self, fixtures: list[db.Fixture], team_map: dict[int, int]
    ) -> list[list[db.Fixture]]:
        """Group fixtures describing the same match (same canonical teams, close dates)."""
        groups: dict[tuple[int, int], list[db.Fixture]] = defaultdict(list)
        for f in fixtures:
            groups[(team_map[f.home_team_id], team_map[f.away_team_id])].append(f)
        clusters: list[list[db.Fixture]] = []
        tolerance = timedelta(days=self.day_tolerance, hours=12)
        for items in groups.values():
            items.sort(key=lambda f: f.kickoff_utc)
            current: list[db.Fixture] = []
            for f in items:
                same_source = any(c.provider == f.provider for c in current)
                if current and (f.kickoff_utc - current[0].kickoff_utc > tolerance or same_source):
                    clusters.append(current)
                    current = []
                current.append(f)
            if current:
                clusters.append(current)
        return clusters

    def build(self, competition_key: str) -> BuildReport:
        report = BuildReport(competition_key)
        by_provider = self._fixtures_by_provider(competition_key)
        if not by_provider:
            return report
        country = None
        league = self.session.scalar(
            select(db.League).where(db.League.key == competition_key).order_by(db.League.id)
        )
        if league is not None:
            country = league.country
        team_map = self._link_teams(by_provider, report, country)

        all_fixtures = [f for items in by_provider.values() for f in items]
        existing_sources = {
            ms.fixture_id: ms
            for chunk in chunks([f.id for f in all_fixtures])
            for ms in self.session.scalars(
                select(db.MatchSource).where(db.MatchSource.fixture_id.in_(chunk))
            )
        }
        for cluster in self._cluster(all_fixtures, team_map):
            self._upsert_match(competition_key, cluster, team_map, existing_sources, report)
        self.session.flush()
        logger.info("canonical_build", extra={"summary": report.summary()})
        return report

    def _upsert_match(
        self,
        competition_key: str,
        cluster: list[db.Fixture],
        team_map: dict[int, int],
        existing_sources: dict[int, db.MatchSource],
        report: BuildReport,
    ) -> None:
        cluster.sort(key=lambda f: self._rank(f.provider))
        primary = cluster[0]
        timed = next((f for f in cluster if f.kickoff_time_known), primary)
        home, away = regulation_score(primary)
        finished = [f for f in cluster if f.status_group == FixtureStatusGroup.FINISHED.value]
        scores = {regulation_score(f) for f in finished if None not in regulation_score(f)}
        conflict = len(scores) > 1

        reason: str | None = _EXCLUDED_STATUS.get(primary.status_group)
        if reason is None and primary.status_group != FixtureStatusGroup.FINISHED.value:
            reason = "not_finished"
        if reason is None and (home is None or away is None):
            reason = "missing_score"
        if reason is None and conflict:
            reason = "source_conflict"
        values = {
            "competition_key": competition_key,
            "season_year": primary.season_year,
            "round": primary.round,
            "kickoff_utc": timed.kickoff_utc,
            "kickoff_time_known": timed.kickoff_time_known,
            "home_team_id": team_map[primary.home_team_id],
            "away_team_id": team_map[primary.away_team_id],
            "status_group": primary.status_group,
            "home_goals": home,
            "away_goals": away,
            "ht_home_goals": primary.ht_home_goals,
            "ht_away_goals": primary.ht_away_goals,
            "primary_fixture_id": primary.id,
            "sources": sorted({f.provider for f in cluster}, key=self._rank),
            "source_conflict": conflict,
            "valid_for_training": reason is None,
            "exclusion_reason": reason,
        }
        report.conflicts += int(conflict)
        report.excluded += int(reason not in (None, "not_finished"))

        match_id = next(
            (existing_sources[f.id].match_id for f in cluster if f.id in existing_sources), None
        )
        match = self.session.get(db.Match, match_id) if match_id is not None else None
        if match is None:
            match = db.Match(**values)
            self.session.add(match)
            self.session.flush()
            report.matches_inserted += 1
        elif apply_changes(match, values):
            report.matches_updated += 1
        for f in cluster:
            source = existing_sources.get(f.id)
            if source is None:
                source = db.MatchSource(match_id=match.id, fixture_id=f.id, provider=f.provider)
                self.session.add(source)
                existing_sources[f.id] = source
            elif source.match_id != match.id:
                source.match_id = match.id
