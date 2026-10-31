"""Data-driven matching of team identities between two sources.

Two sources rarely share team identifiers or spellings ("Man United" vs
"Manchester United"), but they describe the same matches. A source team and a
target team are the same club when they keep playing the same fixtures:

1. **Vote** - every source fixture votes for the target fixtures played within
   ``day_tolerance`` days *with the same final score* (when both are known):
   ``(source home -> target home)`` and ``(source away -> target away)``.
   Each fixture distributes one vote among its candidates.
2. **Assign** - greedy one-to-one assignment by decreasing votes.
3. **Verify** - confidence = share of the source team's fixtures *in the period
   covered by both sources* for which a target fixture exists within tolerance
   with *both* mapped teams identical.

Name similarity is reported as evidence but never decides on its own.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True, slots=True)
class FixtureKey:
    day: date
    home: int
    away: int
    home_goals: int | None = None
    away_goals: int | None = None

    def same_score(self, other: FixtureKey) -> bool:
        if None in (self.home_goals, self.away_goals, other.home_goals, other.away_goals):
            return True  # unknown score: cannot discriminate
        return (self.home_goals, self.away_goals) == (other.home_goals, other.away_goals)


@dataclass(frozen=True, slots=True)
class TeamMatch:
    source_team: int
    target_team: int
    votes: float
    appearances: int
    verified: int
    confidence: float


def _index_by_day(fixtures: Iterable[FixtureKey]) -> dict[date, list[FixtureKey]]:
    index: dict[date, list[FixtureKey]] = defaultdict(list)
    for f in fixtures:
        index[f.day].append(f)
    return index


def _nearby(index: Mapping[date, list[FixtureKey]], day: date, tolerance: int) -> list[FixtureKey]:
    out: list[FixtureKey] = []
    for delta in range(-tolerance, tolerance + 1):
        out.extend(index.get(day + timedelta(days=delta), ()))
    return out


def resolve_teams(
    source: Sequence[FixtureKey],
    target: Sequence[FixtureKey],
    *,
    day_tolerance: int = 1,
) -> dict[int, TeamMatch]:
    """Return the best target team for each source team (all candidates, unfiltered)."""
    target_index = _index_by_day(target)
    votes: dict[tuple[int, int], float] = defaultdict(float)
    appearances: dict[int, int] = defaultdict(int)

    overlapping: list[FixtureKey] = []
    for fx in source:
        nearby = _nearby(target_index, fx.day, day_tolerance)
        if not nearby:
            continue  # period not covered by the target source: no evidence either way
        overlapping.append(fx)
        appearances[fx.home] += 1
        appearances[fx.away] += 1
        candidates = [t for t in nearby if fx.same_score(t)]
        if not candidates:
            continue
        weight = 1.0 / len(candidates)
        for t in candidates:
            votes[(fx.home, t.home)] += weight
            votes[(fx.away, t.away)] += weight

    mapping: dict[int, int] = {}
    used_targets: set[int] = set()
    for (src, tgt), _ in sorted(votes.items(), key=lambda kv: (-kv[1], kv[0])):
        if src in mapping or tgt in used_targets:
            continue
        mapping[src] = tgt
        used_targets.add(tgt)

    verified: dict[int, int] = defaultdict(int)
    target_pairs: dict[date, set[tuple[int, int]]] = defaultdict(set)
    for t in target:
        target_pairs[t.day].add((t.home, t.away))
    for fx in overlapping:
        mapped = (mapping.get(fx.home), mapping.get(fx.away))
        if None in mapped:
            continue
        for delta in range(-day_tolerance, day_tolerance + 1):
            if mapped in target_pairs.get(fx.day + timedelta(days=delta), set()):
                verified[fx.home] += 1
                verified[fx.away] += 1
                break

    return {
        src: TeamMatch(
            source_team=src,
            target_team=tgt,
            votes=round(votes[(src, tgt)], 3),
            appearances=appearances[src],
            verified=verified[src],
            confidence=round(verified[src] / appearances[src], 4) if appearances[src] else 0.0,
        )
        for src, tgt in mapping.items()
    }
