"""Canonical layer: cross-source team linking, match merging, quality report."""

from __future__ import annotations

import random
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from football_ai.config.aliases import TeamAlias
from football_ai.config.competitions import CompetitionConfig
from football_ai.data.cleaning.canonical import CanonicalBuilder
from football_ai.data.collectors.api_football.parsers import parse_fixture
from football_ai.data.pipeline.football_data_uk import load_csv_season
from football_ai.data.quality.checks import load_frames, run_checks
from football_ai.data.quality.report import save_report
from football_ai.database import models as db
from football_ai.database.connection import session_scope
from football_ai.database.repositories import FixtureRepository
from tests.conftest import fixture_item

# Synthetic clubs: (api id, api name, csv name)
CLUBS = [
    (33, "Manchester United", "Man United"),
    (40, "Liverpool", "Liverpool"),
    (42, "Arsenal", "Arsenal"),
    (49, "Chelsea", "Chelsea"),
    (50, "Manchester City", "Man City"),
    (47, "Tottenham", "Tottenham"),
]
NOW = datetime(2026, 10, 31, 12, tzinfo=UTC)


def schedule(season: int, seed: int) -> list[tuple[datetime, int, int, int, int]]:
    rng = random.Random(seed)
    n = len(CLUBS)
    order = list(range(n))
    rounds = []
    for _ in range(n - 1):
        rounds.append([(order[i], order[n - 1 - i]) for i in range(n // 2)])
        order = [order[0], order[-1], *order[1:-1]]
    rounds += [[(a, h) for h, a in r] for r in rounds]
    out = []
    for k, rnd in enumerate(rounds):
        for i, (h, a) in enumerate(rnd):
            kick = datetime(season, 8, 15, 14, tzinfo=UTC) + timedelta(days=7 * k, hours=3 * i)
            out.append((kick, h, a, rng.randint(0, 4), rng.randint(0, 3)))
    return out


def load_api(session_factory: sessionmaker[Session], season: int, games: list) -> None:  # type: ignore[type-arg]
    fixtures = [
        parse_fixture(
            fixture_item(
                season * 1000 + idx,
                39,
                (CLUBS[h][0], CLUBS[h][1]),
                (CLUBS[a][0], CLUBS[a][1]),
                date=kick.isoformat(),
                status="FT",
                goals=(hg, ag),
                season=season,
                league_name="Premier League",
            ),
            competition_key="premier_league",
        )
        for idx, (kick, h, a, hg, ag) in enumerate(games)
    ]
    with session_scope(session_factory) as s:
        FixtureRepository(s).upsert_fixtures(fixtures)


def csv_text(games: list) -> str:  # type: ignore[type-arg]
    lines = ["Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST,B365H,B365D,B365A"]
    for kick, h, a, hg, ag in games:
        local = kick + timedelta(hours=1)  # BST
        res = "H" if hg > ag else "A" if ag > hg else "D"
        lines.append(
            f"E0,{local:%d/%m/%Y},{local:%H:%M},{CLUBS[h][2]},{CLUBS[a][2]},{hg},{ag},{res},"
            f"10,8,4,3,2.0,3.4,3.9"
        )
    return "\n".join(lines) + "\n"


def count(session_factory: sessionmaker[Session], model: type[db.Base], *where: object) -> int:
    with session_scope(session_factory) as s:
        stmt = select(func.count()).select_from(model)
        for w in where:
            stmt = stmt.where(w)  # type: ignore[arg-type]
        return s.scalar(stmt) or 0


def setup_two_sources(
    session_factory: sessionmaker[Session], competitions: list[CompetitionConfig]
) -> tuple[list, list]:  # type: ignore[type-arg]
    pl = next(c for c in competitions if c.key == "premier_league")
    games_2024, games_2018 = schedule(2024, 1), schedule(2018, 2)
    load_api(session_factory, 2024, games_2024)
    # CSV covers the API season and an older season the API does not have
    games_csv_2024 = list(games_2024)
    kick, h, a, hg, ag = games_csv_2024[5]
    games_csv_2024[5] = (kick, h, a, hg + 1, ag)  # deliberate score conflict
    load_csv_season(csv_text(games_csv_2024).encode(), pl, 2024, session_factory)
    load_csv_season(csv_text(games_2018).encode(), pl, 2018, session_factory)
    return games_2024, games_2018


def test_build_links_teams_and_merges_matches(
    session_factory: sessionmaker[Session], competitions: list[CompetitionConfig]
) -> None:
    games_2024, games_2018 = setup_two_sources(session_factory, competitions)
    with session_scope(session_factory) as s:
        report = CanonicalBuilder(s).build("premier_league")

    assert report.links == {"primary": 6, "cooccurrence": 6}
    assert count(session_factory, db.CanonicalTeam) == 6
    assert report.matches_inserted == len(games_2024) + len(games_2018)  # 30 + 30
    assert count(session_factory, db.MatchSource) == 90  # 30 api + 60 csv
    assert report.conflicts == 1
    with session_scope(session_factory) as s:
        links = {
            (t.name, link.method, link.confidence)
            for link, t in s.execute(select(db.TeamLink, db.Team).join(db.Team))
        }
        assert ("Man United", "cooccurrence", 1.0) in links
        canonical_names = set(s.scalars(select(db.CanonicalTeam.name)))
        assert "Manchester United" in canonical_names and "Man United" not in canonical_names
        merged = s.scalars(select(db.Match).where(db.Match.season_year == 2024)).all()
        assert all(m.sources == ["api_football", "football_data_uk"] for m in merged)
        conflict = [m for m in merged if m.source_conflict]
        assert len(conflict) == 1 and conflict[0].exclusion_reason == "source_conflict"
        assert conflict[0].valid_for_training is False
        old = s.scalars(select(db.Match).where(db.Match.season_year == 2018)).all()
        assert len(old) == 30 and all(m.sources == ["football_data_uk"] for m in old)
        # older CSV-only matches use the canonical (API) identities
        assert all(m.valid_for_training for m in old)

    # rebuild is idempotent
    with session_scope(session_factory) as s:
        again = CanonicalBuilder(s).build("premier_league")
    assert again.matches_inserted == 0 and again.matches_updated == 0
    assert count(session_factory, db.Match) == 60


def test_alias_override_and_standalone(
    session_factory: sessionmaker[Session], competitions: list[CompetitionConfig]
) -> None:
    pl = next(c for c in competitions if c.key == "premier_league")
    # CSV only: no overlap -> standalone canonical teams, except the alias
    load_csv_season(csv_text(schedule(2018, 2)).encode(), pl, 2018, session_factory)
    aliases = [
        TeamAlias(provider="football_data_uk", name="Man United", canonical="Manchester United")
    ]
    with session_scope(session_factory) as s:
        report = CanonicalBuilder(s, aliases=aliases).build("premier_league")
    assert report.links == {"alias": 1, "primary": 5}  # csv is the only (hence primary) source
    with session_scope(session_factory) as s:
        assert "Manchester United" in set(s.scalars(select(db.CanonicalTeam.name)))


def test_quality_report_on_canonical_data(
    session_factory: sessionmaker[Session], competitions: list[CompetitionConfig], tmp_path: Path
) -> None:
    setup_two_sources(session_factory, competitions)
    with session_scope(session_factory) as s:
        CanonicalBuilder(s).build("premier_league")
    with session_scope(session_factory) as s:
        report = run_checks(load_frames(s, "premier_league"), now=NOW, scope="premier_league")
        path = save_report(report, s, tmp_path)
    checks = {f.check: f.count for f in report.findings}
    assert checks.get("source_conflicts") == 1
    assert "duplicate_matches" not in checks and "incomplete_seasons" not in checks
    cov = {(c["season"]): c for c in report.coverage}
    assert cov[2024]["finished_matches"] == 30 and cov[2024]["valid_for_training"] == 29
    assert cov[2024]["pct_stats"] == 100.0 and cov[2024]["multi_source_pct"] == 100.0
    assert cov[2018]["pct_odds"] == 100.0 and cov[2018]["pct_xg"] == 0.0
    assert path.is_file()
    assert count(session_factory, db.DataQualityReport) == 1
    assert date.today()  # keep import used
