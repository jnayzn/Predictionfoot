"""Translate raw API-Football JSON into provider-independent domain models.

Parsers are pure functions: no I/O, no defaults invented for missing data.
A missing *required* field raises :class:`ParseError`; optional fields stay ``None``.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import Any

from football_ai.domain.models import (
    Competition,
    Fixture,
    FixtureStatusGroup,
    Score,
    Season,
    TeamRef,
)

PROVIDER = "api_football"

# https://www.api-football.com/documentation-v3#tag/Fixtures (status codes)
STATUS_GROUPS: dict[str, FixtureStatusGroup] = {
    "TBD": FixtureStatusGroup.SCHEDULED,
    "NS": FixtureStatusGroup.SCHEDULED,
    "1H": FixtureStatusGroup.LIVE,
    "HT": FixtureStatusGroup.LIVE,
    "2H": FixtureStatusGroup.LIVE,
    "ET": FixtureStatusGroup.LIVE,
    "BT": FixtureStatusGroup.LIVE,
    "P": FixtureStatusGroup.LIVE,
    "SUSP": FixtureStatusGroup.LIVE,
    "INT": FixtureStatusGroup.LIVE,
    "LIVE": FixtureStatusGroup.LIVE,
    "FT": FixtureStatusGroup.FINISHED,
    "AET": FixtureStatusGroup.FINISHED,
    "PEN": FixtureStatusGroup.FINISHED,
    "PST": FixtureStatusGroup.POSTPONED,
    "CANC": FixtureStatusGroup.CANCELLED,
    "ABD": FixtureStatusGroup.ABANDONED,
    "AWD": FixtureStatusGroup.AWARDED,
    "WO": FixtureStatusGroup.AWARDED,
}


class ParseError(ValueError):
    """A required field is missing or malformed."""


def _require(obj: Mapping[str, Any] | None, key: str, path: str) -> Any:
    if not isinstance(obj, Mapping) or obj.get(key) is None:
        raise ParseError(f"missing required field '{path}.{key}'")
    return obj[key]


def _as_int(value: Any, path: str) -> int:
    if isinstance(value, bool):
        raise ParseError(f"'{path}' is not an integer: {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ParseError(f"'{path}' is not an integer: {value!r}") from exc


def _opt_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _opt_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _opt_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _flatten_coverage(coverage: Any, prefix: str = "") -> dict[str, bool]:
    flat: dict[str, bool] = {}
    if not isinstance(coverage, Mapping):
        return flat
    for key, value in coverage.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(_flatten_coverage(value, f"{name}."))
        elif isinstance(value, bool):
            flat[name] = value
    return flat


def _score(obj: Any) -> Score:
    if not isinstance(obj, Mapping):
        return Score()
    return Score(home=_opt_int(obj.get("home")), away=_opt_int(obj.get("away")))


def parse_season(raw: Mapping[str, Any]) -> Season:
    return Season(
        year=_as_int(_require(raw, "year", "season"), "season.year"),
        start=_opt_date(raw.get("start")),
        end=_opt_date(raw.get("end")),
        current=bool(raw.get("current", False)),
        coverage=_flatten_coverage(raw.get("coverage")),
    )


def parse_league(raw: Mapping[str, Any], *, key: str) -> Competition:
    league = _require(raw, "league", "item")
    country = raw.get("country") or {}
    seasons = tuple(
        sorted(
            (parse_season(s) for s in raw.get("seasons") or [] if isinstance(s, Mapping)),
            key=lambda s: s.year,
        )
    )
    return Competition(
        key=key,
        provider=PROVIDER,
        provider_id=_as_int(_require(league, "id", "league"), "league.id"),
        name=str(_require(league, "name", "league")),
        country=str(country.get("name") or ""),
        type=str(league.get("type") or ""),
        logo_url=_opt_str(league.get("logo")),
        seasons=seasons,
    )


def _parse_kickoff(fixture: Mapping[str, Any]) -> datetime:
    raw_date = fixture.get("date")
    if isinstance(raw_date, str):
        try:
            parsed = datetime.fromisoformat(raw_date)
        except ValueError as exc:
            raise ParseError(f"invalid fixture.date {raw_date!r}") from exc
        if parsed.tzinfo is None:
            raise ParseError(f"fixture.date has no timezone offset: {raw_date!r}")
        return parsed.astimezone(UTC)
    timestamp = _opt_int(fixture.get("timestamp"))
    if timestamp is not None:
        return datetime.fromtimestamp(timestamp, tz=UTC)
    raise ParseError("missing required field 'fixture.date'")


def _team(raw: Any, path: str) -> TeamRef:
    return TeamRef(
        provider_id=_as_int(_require(raw, "id", path), f"{path}.id"),
        name=str(_require(raw, "name", path)),
        logo_url=_opt_str(raw.get("logo")),
    )


def parse_fixture(raw: Mapping[str, Any], *, competition_key: str | None = None) -> Fixture:
    fixture = _require(raw, "fixture", "item")
    league = _require(raw, "league", "item")
    teams = _require(raw, "teams", "item")
    status = fixture.get("status") or {}
    status_short = str(status.get("short") or "").upper()
    if not status_short:
        raise ParseError("missing required field 'fixture.status.short'")
    venue = fixture.get("venue") or {}
    score = raw.get("score") or {}

    return Fixture(
        provider=PROVIDER,
        provider_id=_as_int(_require(fixture, "id", "fixture"), "fixture.id"),
        competition_key=competition_key,
        league_provider_id=_as_int(_require(league, "id", "league"), "league.id"),
        league_name=str(league.get("name") or ""),
        season=_as_int(_require(league, "season", "league"), "league.season"),
        round=_opt_str(league.get("round")),
        kickoff_utc=_parse_kickoff(fixture),
        provider_timezone=_opt_str(fixture.get("timezone")),
        status_short=status_short,
        status_long=_opt_str(status.get("long")),
        status_group=STATUS_GROUPS.get(status_short, FixtureStatusGroup.UNKNOWN),
        elapsed=_opt_int(status.get("elapsed")),
        venue_id=_opt_int(venue.get("id")),
        venue_name=_opt_str(venue.get("name")),
        venue_city=_opt_str(venue.get("city")),
        referee=_opt_str(fixture.get("referee")),
        home_team=_team(teams.get("home"), "teams.home"),
        away_team=_team(teams.get("away"), "teams.away"),
        goals=_score(raw.get("goals")),
        score_halftime=_score(score.get("halftime")),
        score_fulltime=_score(score.get("fulltime")),
        score_extratime=_score(score.get("extratime")),
        score_penalty=_score(score.get("penalty")),
    )
