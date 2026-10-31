"""Feature engineering: correctness on hand-made data, leakage guard, Elo sanity."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from football_ai.features import history
from football_ai.features.builder import FeatureBuilder, feature_columns, feature_group
from football_ai.features.dataset import BaseFrames, implied_probabilities, injury_counts
from football_ai.features.elo_features import EloConfig, elo_states, goal_multiplier
from football_ai.features.leakage import check_leakage
from football_ai.features.store import latest_feature_set, load_feature_set, save_feature_set
from tests.synthetic import generate_league

T0 = datetime(2024, 8, 10, 15, tzinfo=UTC)


def make_frames(games: list[tuple[int, int, int | None, int | None, int, int]]) -> BaseFrames:
    """games: (home, away, home_goals, away_goals, day offset, season)."""
    rows = [
        {
            "match_id": i + 1, "competition_key": "premier_league", "season_year": season,
            "kickoff_utc": T0 + timedelta(days=day), "home_team_id": h, "away_team_id": a,
            "status_group": "finished" if hg is not None else "scheduled",
            "home_goals": hg, "away_goals": ag, "valid_for_training": hg is not None,
        }
        for i, (h, a, hg, ag, day, season) in enumerate(games)
    ]  # fmt: skip
    empty = BaseFrames.empty()
    return BaseFrames(pd.DataFrame(rows), empty.team_stats, empty.odds, empty.injuries)


GAMES = [
    (1, 2, 2, 0, 0, 2024),  # team 1 W
    (3, 1, 1, 0, 7, 2024),  # team 1 L (away)
    (1, 4, 1, 1, 10, 2024),  # team 1 D
    (2, 1, 0, 3, 17, 2024),  # team 1 W (away) <- target checked below
    (1, 3, None, None, 24, 2024),  # upcoming
]


def row(df: pd.DataFrame, match_id: int) -> pd.Series:
    return df[df["match_id"] == match_id].iloc[0]


def test_form_rest_and_season_values_use_only_previous_matches() -> None:
    df = FeatureBuilder().build(make_frames(GAMES))
    r = row(df, 4)  # team 1 is away: previous results W, L, D
    assert r["away_ppg_5"] == pytest.approx(4 / 3)
    assert r["away_win_rate_5"] == pytest.approx(1 / 3)
    assert r["away_matches_played"] == 3
    assert r["away_rest_days"] == pytest.approx(7.0)
    assert r["away_games_last_14d"] == 2  # days 7 and 10 (day 0 is 17 days before)
    assert r["away_season_ppg"] == pytest.approx(4 / 3)
    assert r["away_gf_avg_5"] == pytest.approx(1.0)  # 2, 0, 1
    assert r["away_ga_avg_5"] == pytest.approx(2 / 3)  # 0, 1, 1
    assert r["away_unbeaten_streak"] == 1 and r["away_win_streak"] == 0
    # venue form: team 1 away games before match 4 -> only the loss at team 3
    assert r["away_venue_ppg_5"] == 0.0
    # h2h team 2 vs team 1: one meeting, team 2 lost 0-2
    assert r["home_h2h_ppg"] == 0.0 and r["home_h2h_count"] == 1
    assert r["home_h2h_gd_avg"] == -2.0
    assert r["outcome"] == 2.0 and r["total_goals"] == 3

    first = row(df, 1)
    assert np.isnan(first["home_ppg_5"]) and first["home_matches_played"] == 0
    assert np.isnan(first["home_rest_days"])

    upcoming = row(df, 5)
    assert np.isnan(upcoming["outcome"]) and np.isnan(upcoming["home_goals"])
    assert upcoming["home_ppg_5"] == pytest.approx(7 / 4)  # W L D W
    assert upcoming["home_matches_played"] == 4


def test_season_features_reset_and_elo_regression() -> None:
    games = [(1, 2, 3, 0, 0, 2023), (1, 2, None, None, 365, 2024)]
    cfg = EloConfig(season_regression=0.5)
    df = FeatureBuilder().build(make_frames(games))
    r = row(df, 2)
    assert r["home_season_played"] == 0 and np.isnan(r["home_season_ppg"])
    assert r["home_ppg_5"] == 3.0  # overall form crosses seasons
    states = elo_states(make_frames(games).normalized().matches.iloc[:1], cfg)
    post = states.set_index("team_id")["elo_post"]
    from football_ai.features.builder import FeatureConfig

    df2 = FeatureBuilder(FeatureConfig(elo=cfg)).build(make_frames(games))
    expected = cfg.initial_rating + (post[1] - cfg.initial_rating) * 0.5
    assert row(df2, 2)["home_elo"] == pytest.approx(expected)


def test_elo_basics() -> None:
    assert goal_multiplier(1) == 1.0 and goal_multiplier(2) == 1.5 and goal_multiplier(4) == 15 / 8
    states = elo_states(make_frames([(1, 2, 1, 0, 0, 2024)]).normalized().matches, EloConfig())
    ratings = states.set_index("team_id")["elo_post"]
    assert ratings[1] > 1450 > ratings[2]
    assert ratings[1] + ratings[2] == pytest.approx(2 * 1450)  # zero-sum


@pytest.mark.parametrize("seed", [0, 1])
def test_no_leakage_on_synthetic_league(seed: int) -> None:
    league = generate_league(n_teams=8, seasons=(2022, 2023), seed=seed)
    result = check_leakage(league.frames, FeatureBuilder(), sample=20, seed=seed)
    assert result.checked == 20
    assert result.ok, result.leaking_features


def test_leakage_checker_detects_a_leak(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sanity check of the guard itself: allowing exact timestamps must be caught."""
    original = pd.merge_asof

    def leaky(*args: object, **kwargs: object) -> pd.DataFrame:
        kwargs["allow_exact_matches"] = True
        return original(*args, **kwargs)  # type: ignore[call-overload,no-any-return]

    monkeypatch.setattr(history.pd, "merge_asof", leaky)
    league = generate_league(n_teams=6, seasons=(2023,), seed=3)
    result = check_leakage(league.frames, FeatureBuilder(), sample=10)
    assert not result.ok
    assert any(c.endswith("ppg_5") for c in result.leaking_features)


def test_elo_and_form_recover_latent_strength() -> None:
    league = generate_league(n_teams=12, seasons=(2020, 2021, 2022, 2023), seed=1)
    df = FeatureBuilder().build(league.frames)
    last = df[df["season_year"] == 2023].groupby("home_team_id")["home_elo"].last()
    strength = [league.attack[t] + league.defence[t] for t in last.index]
    assert np.corrcoef(strength, last.to_numpy())[0, 1] > 0.7
    # xG features exist and correlate with goals scored
    played = df[df["outcome"].notna()]
    assert played["home_xg_for_avg_10"].notna().mean() > 0.9
    assert played[["home_xg_for_avg_10", "home_goals"]].corr().iloc[0, 1] > 0.1


def test_feature_columns_and_groups() -> None:
    df = FeatureBuilder().build(generate_league(n_teams=6, seasons=(2023,), seed=4).frames)
    cols = feature_columns(df)
    assert "match_id" not in cols and "outcome" not in cols and "home_team_id" not in cols
    assert "market_p_home" in cols and "diff_elo" in cols
    assert "other" not in {feature_group(c) for c in cols}
    assert df["market_available"].eq(1.0).all()
    assert df["home_missing_players"].isna().all()  # no injury coverage -> NaN, not 0


def test_implied_probabilities_and_injury_counts() -> None:
    raw = pd.DataFrame(
        [
            {"match_id": 1, "provider": "p", "bookmaker": "A", "selection": s, "odd": o}
            for s, o in (("home", 2.0), ("draw", 3.5), ("away", 4.0))
        ]
        + [{"match_id": 2, "provider": "p", "bookmaker": "A", "selection": "home", "odd": 1.5}]
    )
    probs = implied_probabilities(raw).set_index("match_id")
    assert probs.loc[1, ["implied_home", "implied_draw", "implied_away"]].sum() == pytest.approx(1)
    assert 2 not in probs.index  # incomplete market dropped
    matches = pd.DataFrame([{"match_id": 1, "home_team_id": 10, "away_team_id": 20}])
    inj = injury_counts(matches, pd.DataFrame([{"match_id": 1, "team_id": 10, "id": 5}]), {1})
    assert dict(zip(inj["team_id"], inj["missing_players"], strict=True)) == {10: 1.0, 20: 0.0}


def test_feature_store_roundtrip(tmp_path: Path) -> None:
    builder = FeatureBuilder()
    df = builder.build(generate_league(n_teams=6, seasons=(2023,), seed=5).frames)
    path = save_feature_set(df, builder.config, tmp_path)
    assert latest_feature_set(tmp_path) == path
    loaded, meta = load_feature_set(path)
    assert len(loaded) == len(df) and meta["rows"] == len(df)
    assert meta["config_fingerprint"] == builder.config.fingerprint()
    assert set(meta["features"]) == set(feature_columns(df))
