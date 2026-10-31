"""Data-quality checks on normalized fixtures and competitions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from football_ai.config.competitions import CompetitionConfig
from football_ai.data.validators.issues import DataQualityIssue, Severity
from football_ai.domain.models import Competition, Fixture, FixtureStatusGroup

# Earliest plausible date for the competitions we cover (sanity bound).
_MIN_KICKOFF = datetime(1888, 1, 1, tzinfo=UTC)


def validate_fixture(fixture: Fixture, *, now: datetime | None = None) -> list[DataQualityIssue]:
    now = now or datetime.now(UTC)
    fid = str(fixture.provider_id)
    issues: list[DataQualityIssue] = []

    def add(field: str, severity: Severity, message: str) -> None:
        issues.append(DataQualityIssue("fixture", fid, field, severity, message))

    if fixture.home_team.provider_id == fixture.away_team.provider_id:
        add("teams", Severity.ERROR, "home and away team are identical")

    for side, value in (("home", fixture.goals.home), ("away", fixture.goals.away)):
        if value is not None and value < 0:
            add(f"goals.{side}", Severity.ERROR, f"negative goal count: {value}")

    has_both = fixture.goals.home is not None and fixture.goals.away is not None
    if fixture.status_group is FixtureStatusGroup.FINISHED and not has_both:
        add("goals", Severity.ERROR, f"finished ({fixture.status_short}) but goals missing")
    if fixture.status_group is FixtureStatusGroup.SCHEDULED and (
        fixture.goals.home is not None or fixture.goals.away is not None
    ):
        add("goals", Severity.WARNING, "scheduled fixture already has goals")
    if fixture.status_group is FixtureStatusGroup.UNKNOWN:
        add("status_short", Severity.WARNING, f"unknown status code {fixture.status_short!r}")

    if fixture.kickoff_utc < _MIN_KICKOFF:
        add("kickoff_utc", Severity.ERROR, "implausible kick-off date")
    if (
        fixture.status_group is FixtureStatusGroup.FINISHED
        and fixture.kickoff_utc > now + timedelta(hours=1)
    ):
        add("kickoff_utc", Severity.ERROR, "finished fixture has a kick-off in the future")

    if fixture.season < 1888 or fixture.season > now.year + 1:
        add("season", Severity.ERROR, f"implausible season {fixture.season}")

    return issues


def validate_competition_mapping(
    competition: Competition, expected: CompetitionConfig
) -> list[DataQualityIssue]:
    """Check that the provider's league matches what we expect from the config."""
    issues: list[DataQualityIssue] = []
    cid = f"{competition.provider}:{competition.provider_id}"
    if competition.name.casefold() != expected.name.casefold():
        issues.append(
            DataQualityIssue(
                "competition",
                cid,
                "name",
                Severity.WARNING,
                f"provider name {competition.name!r} != expected {expected.name!r}",
            )
        )
    if competition.country.casefold() != expected.country.casefold():
        issues.append(
            DataQualityIssue(
                "competition",
                cid,
                "country",
                Severity.ERROR,
                f"provider country {competition.country!r} != expected {expected.country!r}",
            )
        )
    if not competition.seasons:
        issues.append(
            DataQualityIssue("competition", cid, "seasons", Severity.WARNING, "no season listed")
        )
    return issues
