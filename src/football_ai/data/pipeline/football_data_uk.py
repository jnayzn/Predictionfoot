"""ETL for football-data.co.uk CSV seasons."""

from __future__ import annotations

from sqlalchemy.orm import Session, sessionmaker

from football_ai.config.competitions import CompetitionConfig
from football_ai.data.collectors.football_data_uk.parser import (
    PROVIDER,
    decode,
    parse_season_csv,
)
from football_ai.data.pipeline.etl import LoadReport
from football_ai.data.validators.fixtures import validate_fixture
from football_ai.data.validators.issues import has_errors
from football_ai.database.connection import session_scope
from football_ai.database.repositories import FixtureRepository, MatchDataRepository
from football_ai.domain.models import TeamFixtureStatistics


def division_of(config: CompetitionConfig) -> str:
    division = config.provider_id(PROVIDER)
    if division is None:
        raise ValueError(f"{config.key} has no football_data_uk division code")
    return str(division)


def load_csv_season(
    content: bytes,
    config: CompetitionConfig,
    season: int,
    session_factory: sessionmaker[Session],
) -> LoadReport:
    parsed = parse_season_csv(
        decode(content),
        division=division_of(config),
        season=season,
        competition_key=config.key,
        league_name=config.name,
    )
    report = LoadReport("csv", f"{config.key}:{season}", issues=list(parsed.issues))
    valid = []
    for fixture in parsed.fixtures:
        issues = validate_fixture(fixture)
        report.issues.extend(issues)
        if not has_errors(issues):
            valid.append(fixture)
    valid_ids = {f.provider_id for f in valid}
    with session_scope(session_factory) as session:
        fixtures_repo = FixtureRepository(session)
        report.fixtures = fixtures_repo.upsert_fixtures(valid)
        match_data = MatchDataRepository(session, fixtures_repo.refs, fixtures_repo)
        by_fixture: dict[int, list[TeamFixtureStatistics]] = {}
        for stat in parsed.statistics:
            if stat.fixture_provider_id in valid_ids:
                by_fixture.setdefault(stat.fixture_provider_id, []).append(stat)
        for provider_id, stats in by_fixture.items():
            match_data.save_team_statistics(report.fixtures.fixture_ids[provider_id], stats)
        report.items = match_data.save_odds(
            [q for q in parsed.odds if q.fixture_provider_id in valid_ids]
        )
    return report
