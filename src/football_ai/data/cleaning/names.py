"""Team-name normalization (used as a secondary signal / for aliases)."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

# Generic club-type tokens that carry no identity information.
_NOISE = frozenset(
    {"fc", "cf", "afc", "ac", "as", "sc", "ssc", "us", "sv", "vfb", "vfl", "tsg", "fsv",
     "rc", "rcd", "ud", "cd", "sd", "ca", "club", "calcio", "de", "del", "la", "1"}
)  # fmt: skip


def normalize_team_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    text = text.replace("&", " and ").replace("'", "")
    tokens = [t for t in re.split(r"[^a-z0-9]+", text) if t and t not in _NOISE]
    return " ".join(tokens)


def name_similarity(a: str, b: str) -> float:
    na, nb = normalize_team_name(a), normalize_team_name(b)
    if not na or not nb:
        return 0.0
    if na == nb or na in nb.split() or nb in na.split():
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()
