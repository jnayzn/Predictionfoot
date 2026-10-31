from __future__ import annotations

import random
from datetime import date, timedelta

from football_ai.data.cleaning.names import name_similarity, normalize_team_name
from football_ai.data.cleaning.team_resolution import FixtureKey, resolve_teams


def round_robin(teams: list[int], start: date, rng: random.Random) -> list[FixtureKey]:
    """Double round robin, ~half the league plays each weekend day (synthetic)."""
    n = len(teams)
    rounds = []
    order = teams[:]
    for _ in range(n - 1):
        rounds.append([(order[i], order[n - 1 - i]) for i in range(n // 2)])
        order = [order[0], order[-1], *order[1:-1]]
    rounds += [[(a, h) for h, a in r] for r in rounds]
    out = []
    for k, rnd in enumerate(rounds):
        day = start + timedelta(days=7 * k)
        for i, (h, a) in enumerate(rnd):
            out.append(
                FixtureKey(day + timedelta(days=i % 2), h, a, rng.randint(0, 3), rng.randint(0, 3))
            )
    return out


def test_resolution_maps_every_team_with_high_confidence() -> None:
    rng = random.Random(7)
    target = round_robin(list(range(1, 21)), date(2023, 8, 12), rng)
    mapping = {t: 1000 + t * 7 for t in range(1, 21)}  # source ids unrelated to target ids
    source = [
        FixtureKey(f.day, mapping[f.home], mapping[f.away], f.home_goals, f.away_goals)
        for f in target
    ]
    result = resolve_teams(source, target)
    assert {src: m.target_team for src, m in result.items()} == {v: k for k, v in mapping.items()}
    assert all(m.confidence == 1.0 and m.appearances == 38 for m in result.values())


def test_resolution_tolerates_one_day_shift_and_extra_history() -> None:
    rng = random.Random(3)
    target = round_robin(list(range(1, 11)), date(2023, 8, 12), rng)
    older = round_robin(list(range(1, 11)), date(2015, 8, 12), rng)  # not covered by target
    mapping = {t: 500 + t for t in range(1, 11)}
    source = [
        FixtureKey(
            f.day + timedelta(days=1), mapping[f.home], mapping[f.away], f.home_goals, f.away_goals
        )
        for f in [*target, *older]
    ]
    result = resolve_teams(source, target, day_tolerance=1)
    assert all(result[mapping[t]].target_team == t for t in range(1, 11))
    # confidence only measured on the overlapping period
    assert all(m.confidence == 1.0 and m.appearances == 18 for m in result.values())


def test_resolution_reports_partial_evidence() -> None:
    target = [FixtureKey(date(2024, 1, 1), 1, 2, 1, 0)]
    source = [FixtureKey(date(2024, 1, 1), 10, 20, 1, 0)]
    result = resolve_teams(source, target)
    assert result[10].target_team == 1 and result[10].appearances == 1


def test_name_normalization() -> None:
    assert normalize_team_name("1. FC Köln") == "koln"
    assert normalize_team_name("Brighton & Hove Albion") == "brighton and hove albion"
    assert name_similarity("Paris Saint Germain", "Paris Saint-Germain") == 1.0
    assert name_similarity("Inter", "FC Internazionale") < 1.0
