"""Shared CLI helpers."""

from __future__ import annotations

import argparse
from datetime import date

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONFIG = 2
EXIT_QUOTA = 3
EXIT_AUTH = 4


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}, expected YYYY-MM-DD") from exc


def parse_seasons(value: str) -> list[int]:
    """'2021' -> [2021]; '2019-2022' -> [2019..2022]; '2018,2020' -> [2018, 2020]."""
    seasons: set[int] = set()
    try:
        for part in value.split(","):
            part = part.strip()
            if "-" in part:
                start, end = (int(x) for x in part.split("-", 1))
                if start > end:
                    raise ValueError
                seasons.update(range(start, end + 1))
            elif part:
                seasons.add(int(part))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid seasons {value!r}") from exc
    if not seasons or min(seasons) < 1888 or max(seasons) > 2100:
        raise argparse.ArgumentTypeError(f"invalid seasons {value!r}")
    return sorted(seasons)
