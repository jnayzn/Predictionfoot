"""Translate raw API-Football JSON into provider-independent domain models.

Parsers are pure functions: no I/O, no defaults invented for missing data.
A missing *required* field raises :class:`ParseError`; optional fields stay ``None``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import Any

from football_ai.domain.models import (
    Competition,
    Fixture,
    FixtureDetails,
    FixtureStatusGroup,
    Injury,
    Lineup,
    LineupPlayer,
    OddsQuote,
    OddsType,
    PlayerFixtureStatistics,
    PlayerRef,
    Score,
    Season,
    StandingRow,
    Team,
    TeamFixtureStatistics,
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


# ----------------------------------------------------------------------- helpers
def _opt_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        text = value.strip().rstrip("%").strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _opt_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _opt_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else None


def _player(raw: Any, path: str) -> PlayerRef:
    return PlayerRef(
        provider_id=_as_int(_require(raw, "id", path), f"{path}.id"),
        name=str(raw.get("name") or ""),
    )


# ------------------------------------------------------------------------- teams
def parse_team(raw: Mapping[str, Any]) -> Team:
    team = _require(raw, "team", "item")
    venue = raw.get("venue") or {}
    return Team(
        provider=PROVIDER,
        provider_id=_as_int(_require(team, "id", "team"), "team.id"),
        name=str(_require(team, "name", "team")),
        code=_opt_str(team.get("code")),
        country=_opt_str(team.get("country")),
        founded=_opt_int(team.get("founded")),
        national=_opt_bool(team.get("national")),
        logo_url=_opt_str(team.get("logo")),
        venue_name=_opt_str(venue.get("name")),
        venue_city=_opt_str(venue.get("city")),
        venue_capacity=_opt_int(venue.get("capacity")),
    )


# --------------------------------------------------------------------- standings
def _split(raw: Any) -> dict[str, int | None]:
    if not isinstance(raw, Mapping):
        return {}
    goals = raw.get("goals") or {}
    return {
        "played": _opt_int(raw.get("played")),
        "win": _opt_int(raw.get("win")),
        "draw": _opt_int(raw.get("draw")),
        "lose": _opt_int(raw.get("lose")),
        "goals_for": _opt_int(goals.get("for")),
        "goals_against": _opt_int(goals.get("against")),
    }


def parse_standings(raw: Mapping[str, Any]) -> list[StandingRow]:
    """Parse one item of ``/standings`` (a league with one or several groups)."""
    league = _require(raw, "league", "item")
    league_id = _as_int(_require(league, "id", "league"), "league.id")
    season = _as_int(_require(league, "season", "league"), "league.season")
    rows: list[StandingRow] = []
    for group in league.get("standings") or []:
        for entry in group or []:
            overall = _split(entry.get("all"))
            rows.append(
                StandingRow(
                    provider=PROVIDER,
                    league_provider_id=league_id,
                    season=season,
                    team=_team(entry.get("team"), "standings.team"),
                    rank=_as_int(_require(entry, "rank", "standings"), "standings.rank"),
                    points=_as_int(_require(entry, "points", "standings"), "standings.points"),
                    goals_diff=_opt_int(entry.get("goalsDiff")),
                    group_name=_opt_str(entry.get("group")),
                    form=_opt_str(entry.get("form")),
                    status=_opt_str(entry.get("status")),
                    description=_opt_str(entry.get("description")),
                    played=overall.get("played"),
                    win=overall.get("win"),
                    draw=overall.get("draw"),
                    lose=overall.get("lose"),
                    goals_for=overall.get("goals_for"),
                    goals_against=overall.get("goals_against"),
                    home=_split(entry.get("home")),
                    away=_split(entry.get("away")),
                    source_updated_at=_opt_datetime(entry.get("update")),
                )
            )
    return rows


# ---------------------------------------------------------- fixture statistics
STATISTIC_KEYS: dict[str, str] = {
    "shots on goal": "shots_on_goal",
    "shots off goal": "shots_off_goal",
    "total shots": "shots_total",
    "blocked shots": "shots_blocked",
    "shots insidebox": "shots_inside_box",
    "shots outsidebox": "shots_outside_box",
    "fouls": "fouls",
    "corner kicks": "corners",
    "offsides": "offsides",
    "ball possession": "possession_pct",
    "yellow cards": "yellow_cards",
    "red cards": "red_cards",
    "goalkeeper saves": "goalkeeper_saves",
    "total passes": "passes_total",
    "passes accurate": "passes_accurate",
    "passes %": "passes_pct",
    "expected_goals": "expected_goals",
    "goals_prevented": "goals_prevented",
}


def statistic_key(label: str) -> str:
    known = STATISTIC_KEYS.get(label.strip().lower())
    if known:
        return known
    return re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")


def parse_team_statistics(raw: Mapping[str, Any], fixture_id: int) -> TeamFixtureStatistics:
    values: dict[str, float | None] = {}
    for stat in raw.get("statistics") or []:
        if isinstance(stat, Mapping) and isinstance(stat.get("type"), str):
            values[statistic_key(stat["type"])] = _opt_float(stat.get("value"))
    return TeamFixtureStatistics(
        provider=PROVIDER,
        fixture_provider_id=fixture_id,
        team=_team(raw.get("team"), "statistics.team"),
        values=values,
    )


def parse_lineup(raw: Mapping[str, Any], fixture_id: int) -> Lineup:
    players: list[LineupPlayer] = []
    for key, starter in (("startXI", True), ("substitutes", False)):
        for entry in raw.get(key) or []:
            p = (entry or {}).get("player")
            if not isinstance(p, Mapping) or p.get("id") is None:
                continue  # unidentified player: skipped rather than invented
            players.append(
                LineupPlayer(
                    player=_player(p, "lineup.player"),
                    number=_opt_int(p.get("number")),
                    position=_opt_str(p.get("pos")),
                    grid=_opt_str(p.get("grid")),
                    is_starter=starter,
                )
            )
    coach = raw.get("coach") or {}
    return Lineup(
        provider=PROVIDER,
        fixture_provider_id=fixture_id,
        team=_team(raw.get("team"), "lineup.team"),
        formation=_opt_str(raw.get("formation")),
        coach_name=_opt_str(coach.get("name")),
        players=tuple(players),
    )


def parse_player_statistics(
    raw: Mapping[str, Any], fixture_id: int
) -> list[PlayerFixtureStatistics]:
    team = _team(raw.get("team"), "players.team")
    out: list[PlayerFixtureStatistics] = []
    for entry in raw.get("players") or []:
        p = (entry or {}).get("player")
        if not isinstance(p, Mapping) or p.get("id") is None:
            continue
        stats_list = entry.get("statistics") or [{}]
        st = stats_list[0] if isinstance(stats_list[0], Mapping) else {}
        games = st.get("games") or {}
        shots = st.get("shots") or {}
        goals = st.get("goals") or {}
        passes = st.get("passes") or {}
        tackles = st.get("tackles") or {}
        cards = st.get("cards") or {}
        out.append(
            PlayerFixtureStatistics(
                provider=PROVIDER,
                fixture_provider_id=fixture_id,
                team=team,
                player=_player(p, "players.player"),
                minutes=_opt_int(games.get("minutes")),
                rating=_opt_float(games.get("rating")),
                position=_opt_str(games.get("position")),
                substitute=_opt_bool(games.get("substitute")),
                captain=_opt_bool(games.get("captain")),
                goals=_opt_int(goals.get("total")),
                assists=_opt_int(goals.get("assists")),
                shots_total=_opt_int(shots.get("total")),
                shots_on=_opt_int(shots.get("on")),
                passes_total=_opt_int(passes.get("total")),
                passes_key=_opt_int(passes.get("key")),
                pass_accuracy=_opt_float(passes.get("accuracy")),
                tackles=_opt_int(tackles.get("total")),
                yellow_cards=_opt_int(cards.get("yellow")),
                red_cards=_opt_int(cards.get("red")),
            )
        )
    return out


def parse_fixture_details(
    raw: Mapping[str, Any], *, competition_key: str | None = None
) -> FixtureDetails:
    """Parse an item of ``/fixtures?ids=...`` (fixture + statistics, lineups, players)."""
    fixture = parse_fixture(raw, competition_key=competition_key)
    fid = fixture.provider_id
    return FixtureDetails(
        fixture=fixture,
        statistics=tuple(parse_team_statistics(s, fid) for s in raw.get("statistics") or [] if s),
        lineups=tuple(parse_lineup(lu, fid) for lu in raw.get("lineups") or [] if lu),
        players=tuple(
            p
            for block in raw.get("players") or []
            if block
            for p in parse_player_statistics(block, fid)
        ),
    )


# ---------------------------------------------------------------------- injuries
def parse_injury(raw: Mapping[str, Any]) -> Injury:
    player = _require(raw, "player", "item")
    league = _require(raw, "league", "item")
    fixture = raw.get("fixture") or {}
    return Injury(
        provider=PROVIDER,
        player=_player(player, "player"),
        team=_team(raw.get("team"), "team"),
        league_provider_id=_as_int(_require(league, "id", "league"), "league.id"),
        season=_as_int(_require(league, "season", "league"), "league.season"),
        fixture_provider_id=_opt_int(fixture.get("id")),
        fixture_date=_opt_datetime(fixture.get("date")),
        type=_opt_str(player.get("type")),
        reason=_opt_str(player.get("reason")),
    )


# -------------------------------------------------------------------------- odds
_DOUBLE_CHANCE = {"home/draw": "1x", "home/away": "12", "draw/away": "x2"}


def _normalize_bet(bet_name: str, value: str) -> tuple[str, str] | None:
    bet = bet_name.strip().lower()
    val = value.strip().lower()
    if bet == "match winner" and val in {"home", "draw", "away"}:
        return "1x2", val
    if bet == "goals over/under":
        match = re.fullmatch(r"(over|under)\s+(\d+(?:\.\d+)?)", val)
        if match:
            return f"over_under_{match.group(2)}", match.group(1)
    if bet == "both teams score" and val in {"yes", "no"}:
        return "btts", val
    if bet == "double chance" and val in _DOUBLE_CHANCE:
        return "double_chance", _DOUBLE_CHANCE[val]
    return None


def parse_odds(raw: Mapping[str, Any]) -> list[OddsQuote]:
    """Parse one item of ``/odds``. Unsupported markets are ignored (raw JSON is archived)."""
    fixture = _require(raw, "fixture", "item")
    fid = _as_int(_require(fixture, "id", "fixture"), "fixture.id")
    updated = _opt_datetime(raw.get("update"))
    quotes: list[OddsQuote] = []
    for bookmaker in raw.get("bookmakers") or []:
        name = _opt_str((bookmaker or {}).get("name"))
        if not name:
            continue
        for bet in bookmaker.get("bets") or []:
            bet_name = (bet or {}).get("name")
            if not isinstance(bet_name, str):
                continue
            for value in bet.get("values") or []:
                label = (value or {}).get("value")
                odd = _opt_float((value or {}).get("odd"))
                if not isinstance(label, str) or odd is None or odd <= 1.0:
                    continue
                normalized = _normalize_bet(bet_name, label)
                if normalized is None:
                    continue
                quotes.append(
                    OddsQuote(
                        provider=PROVIDER,
                        fixture_provider_id=fid,
                        bookmaker=name,
                        market=normalized[0],
                        selection=normalized[1],
                        odd=odd,
                        odds_type=OddsType.PRE_MATCH,
                        source_updated_at=updated,
                    )
                )
    return quotes
