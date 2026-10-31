"""Parser of football-data.co.uk season CSV files.

Column reference: https://www.football-data.co.uk/notes.txt

Notes / assumptions (documented in docs/data_sources.md):
* the source has no match identifier: a deterministic id is derived from
  (division, date, home team, away team) - stable across re-imports;
* teams are identified by name within this source (cross-source mapping is
  done in Phase 3);
* kick-off ``Time`` (available from 2019/20) is interpreted as UK local time
  (Europe/London). Without a Time column, ``kickoff_time_known`` is False;
* "pre-closing" odds (e.g. B365H) are stored as ``pre_match`` without a
  timestamp, closing odds (e.g. B365CH) as ``closing``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, time
from zoneinfo import ZoneInfo

from football_ai.data.validators.issues import DataQualityIssue, Severity
from football_ai.domain.models import (
    Fixture,
    FixtureStatusGroup,
    OddsQuote,
    OddsType,
    Score,
    TeamFixtureStatistics,
    TeamRef,
)

PROVIDER = "football_data_uk"
SOURCE_TZ = ZoneInfo("Europe/London")

BOOKMAKERS: dict[str, str] = {
    "B365": "Bet365",
    "BW": "Bet&Win",
    "IW": "Interwetten",
    "PS": "Pinnacle",
    "P": "Pinnacle",
    "WH": "William Hill",
    "VC": "VC Bet",
    "LB": "Ladbrokes",
    "Avg": "market_average",
    "Max": "market_max",
    "BbAv": "market_average",
    "BbMx": "market_max",
}
_AGGREGATES = {"Avg", "Max", "BbAv", "BbMx"}

_1X2 = re.compile(r"^(B365|BW|IW|PS|WH|VC|LB|Avg|Max|BbAv|BbMx)(C?)([HDA])$")
_OU = re.compile(r"^(B365|P|Avg|Max|BbAv|BbMx)(C?)([<>])2\.5$")

_STATS = {
    "S": "shots_total",
    "ST": "shots_on_goal",
    "F": "fouls",
    "C": "corners",
    "Y": "yellow_cards",
    "R": "red_cards",
}


def stable_id(*parts: str) -> int:
    """Deterministic 60-bit positive integer id derived from text parts."""
    digest = hashlib.sha1("|".join(p.strip().casefold() for p in parts).encode()).hexdigest()
    return int(digest[:15], 16)


def season_code(season: int) -> str:
    """2023 -> '2324' (football-data.co.uk URL convention)."""
    return f"{season % 100:02d}{(season + 1) % 100:02d}"


@dataclass(slots=True)
class CsvSeason:
    fixtures: list[Fixture] = field(default_factory=list)
    statistics: list[TeamFixtureStatistics] = field(default_factory=list)
    odds: list[OddsQuote] = field(default_factory=list)
    issues: list[DataQualityIssue] = field(default_factory=list)
    rows: int = 0


def decode(content: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("undecodable CSV content")  # pragma: no cover - latin-1 never fails


def _int(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_kickoff(raw_date: str, raw_time: str | None) -> tuple[datetime, bool]:
    raw_date = raw_date.strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            day = datetime.strptime(raw_date, fmt).date()
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"invalid date {raw_date!r}")
    if raw_time and raw_time.strip():
        kick = datetime.strptime(raw_time.strip(), "%H:%M").time()
        local = datetime.combine(day, kick, tzinfo=SOURCE_TZ)
        return local.astimezone(UTC), True
    return datetime.combine(day, time(0, 0), tzinfo=SOURCE_TZ).astimezone(UTC), False


def parse_season_csv(
    text: str, *, division: str, season: int, competition_key: str, league_name: str
) -> CsvSeason:
    result = CsvSeason()
    league_id = stable_id(PROVIDER, "league", division)
    reader = csv.DictReader(io.StringIO(text))
    seen: set[int] = set()
    for line_no, row in enumerate(reader, start=2):
        home_name = (row.get("HomeTeam") or row.get("HT") or "").strip()
        away_name = (row.get("AwayTeam") or row.get("AT") or "").strip()
        if not home_name and not away_name and not (row.get("Date") or "").strip():
            continue  # trailing empty line
        result.rows += 1
        ref = f"{division}:{season}:L{line_no}"
        if not home_name or not away_name or not (row.get("Date") or "").strip():
            result.issues.append(
                DataQualityIssue("fixture", ref, "row", Severity.ERROR, "missing date or team")
            )
            continue
        try:
            kickoff, time_known = _parse_kickoff(row["Date"], row.get("Time"))
        except ValueError as exc:
            result.issues.append(DataQualityIssue("fixture", ref, "Date", Severity.ERROR, str(exc)))
            continue

        fixture_id = stable_id(PROVIDER, division, kickoff.date().isoformat(), home_name, away_name)
        if fixture_id in seen:
            result.issues.append(
                DataQualityIssue("fixture", ref, "row", Severity.WARNING, "duplicate row skipped")
            )
            continue
        seen.add(fixture_id)

        fthg, ftag = _int(row.get("FTHG") or row.get("HG")), _int(row.get("FTAG") or row.get("AG"))
        played = fthg is not None and ftag is not None
        home = TeamRef(provider_id=stable_id(PROVIDER, "team", home_name), name=home_name)
        away = TeamRef(provider_id=stable_id(PROVIDER, "team", away_name), name=away_name)
        result.fixtures.append(
            Fixture(
                provider=PROVIDER,
                provider_id=fixture_id,
                competition_key=competition_key,
                league_provider_id=league_id,
                league_name=league_name,
                season=season,
                kickoff_utc=kickoff,
                kickoff_time_known=time_known,
                provider_timezone="Europe/London",
                status_short="FT" if played else "NS",
                status_long="Match Finished" if played else "Not Started",
                status_group=FixtureStatusGroup.FINISHED
                if played
                else FixtureStatusGroup.SCHEDULED,
                referee=(row.get("Referee") or "").strip() or None,
                home_team=home,
                away_team=away,
                goals=Score(home=fthg, away=ftag),
                score_halftime=Score(home=_int(row.get("HTHG")), away=_int(row.get("HTAG"))),
                score_fulltime=Score(home=fthg, away=ftag),
            )
        )
        for side, team in (("H", home), ("A", away)):
            values = {key: _float(row.get(f"{side}{col}")) for col, key in _STATS.items()}
            if any(v is not None for v in values.values()):
                result.statistics.append(
                    TeamFixtureStatistics(
                        provider=PROVIDER, fixture_provider_id=fixture_id, team=team, values=values
                    )
                )
        result.odds.extend(_parse_odds(row, fixture_id))
    return result


def _quote(
    fixture_id: int, prefix: str, closing: bool, market: str, sel: str, odd: float
) -> OddsQuote:
    if prefix in ("Avg", "BbAv"):
        odds_type = OddsType.MARKET_AVERAGE
    elif prefix in ("Max", "BbMx"):
        odds_type = OddsType.MARKET_MAX
    else:
        odds_type = OddsType.PRE_MATCH
    if closing:
        odds_type = OddsType.CLOSING
    bookmaker = BOOKMAKERS[prefix]
    if prefix in _AGGREGATES and closing:
        bookmaker = f"{bookmaker}_closing"
    return OddsQuote(
        provider=PROVIDER,
        fixture_provider_id=fixture_id,
        bookmaker=bookmaker,
        market=market,
        selection=sel,
        odd=odd,
        odds_type=odds_type,
    )


def _parse_odds(row: dict[str, str | None], fixture_id: int) -> list[OddsQuote]:
    quotes: list[OddsQuote] = []
    for column, raw in row.items():
        if column is None:
            continue
        odd = _float(raw)
        if odd is None or odd <= 1.0:
            continue
        if m := _1X2.match(column):
            sel = {"H": "home", "D": "draw", "A": "away"}[m.group(3)]
            quotes.append(_quote(fixture_id, m.group(1), m.group(2) == "C", "1x2", sel, odd))
        elif m := _OU.match(column):
            sel = "over" if m.group(3) == ">" else "under"
            quotes.append(
                _quote(fixture_id, m.group(1), m.group(2) == "C", "over_under_2.5", sel, odd)
            )
    return quotes
