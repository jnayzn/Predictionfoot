"""Significance tests and the scientific backtest report (synthetic data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football_ai.features.builder import FeatureBuilder
from football_ai.training.backtest_report import run_backtest
from football_ai.training.datasets import split_by_seasons
from football_ai.training.significance import compare_losses, per_match_log_loss, per_match_rps
from football_ai.training.train import ALL_FACTORIES
from tests.synthetic import generate_league


def test_per_match_losses() -> None:
    y = np.array([0, 2])
    p = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    assert np.allclose(per_match_log_loss(y, p), [-np.log(0.5), -np.log(0.5)])
    assert per_match_rps(np.array([0]), np.array([[1.0, 0, 0]]))[0] == 0.0


def test_compare_losses_detects_real_and_absent_differences() -> None:
    rng = np.random.default_rng(0)
    blocks = np.repeat(np.arange(100), 5)
    base = rng.gamma(2.0, 0.5, size=500)
    better = compare_losses(
        base - 0.05 + rng.normal(0, 0.01, 500), base, blocks, model="m", reference="r"
    )
    assert better.significantly_better and better.p_value < 0.01
    assert better.mean_difference == pytest.approx(-0.05, abs=0.01)
    same = compare_losses(base + rng.normal(0, 0.3, 500), base, blocks, model="m", reference="r")
    assert not same.significantly_better and not same.significantly_worse


@pytest.fixture(scope="module")
def setup() -> tuple[pd.DataFrame, object, pd.DataFrame]:
    league = generate_league(n_teams=12, seasons=(2019, 2020, 2021, 2022, 2023), seed=13)
    frame = FeatureBuilder().build(league.frames)
    # synthetic "closing market": the generator's true probabilities
    odds = league.frames.odds.rename(
        columns={
            "implied_home": "market__home",
            "implied_draw": "market__draw",
            "implied_away": "market__away",
        }
    )[["match_id", "market__home", "market__draw", "market__away"]]
    return frame, league, odds.iloc[: len(odds) * 9 // 10]  # some matches without odds


def test_backtest_report(setup: tuple[pd.DataFrame, object, pd.DataFrame]) -> None:
    frame, _, market = setup
    split = split_by_seasons(frame, [2021, 2022], [2023])
    factories = {k: ALL_FACTORIES[k] for k in ("frequency", "elo", "dixon_coles")}
    report = run_backtest(frame, split, factories, market=market, refit_every_days=120, n_boot=300)
    assert set(report.overall) == {"frequency", "elo", "dixon_coles", "ensemble"}
    assert report.period["test_matches"] == len(split.test)
    assert set(report.by_competition) == {"premier_league"}
    assert set(report.by_season) == {"2023"}
    assert report.market_subset is not None and "market" in report.market_subset
    pairs = {(c["model"], c["reference"], c["metric"]) for c in report.comparisons}
    assert ("dixon_coles", "frequency", "log_loss") in pairs
    assert ("ensemble", "market", "rps") in pairs
    # every model is compared to the market on the SAME subset of matches
    n_market = {c["n"] for c in report.comparisons if c["reference"] == "market"}
    assert len(n_market) == 1 and n_market.pop() < report.period["test_matches"]
    # the true-probability "market" cannot be beaten significantly
    assert not any(c["reference"] == "market" and c["ci_high"] < 0 for c in report.comparisons)
    dc_vs_freq = next(
        c
        for c in report.comparisons
        if c["model"] == "dixon_coles"
        and c["reference"] == "frequency"
        and c["metric"] == "log_loss"
    )
    assert dc_vs_freq["mean_difference"] < 0
    md = report.to_markdown()
    assert "# Backtest report" in md and "| dixon_coles |" in md and "closing odds" in md
    assert report.ensemble is not None and "weights" in report.ensemble


def test_backtest_requires_test_period(setup: tuple[pd.DataFrame, object, pd.DataFrame]) -> None:
    frame, _, _ = setup
    with pytest.raises(ValueError):
        run_backtest(frame, split_by_seasons(frame, [2023]), {"elo": ALL_FACTORIES["elo"]})


def test_backtest_without_ensemble_or_market(
    setup: tuple[pd.DataFrame, object, pd.DataFrame],
) -> None:
    frame, _, _ = setup
    split = split_by_seasons(frame, [2022], [2023])
    report = run_backtest(
        frame,
        split,
        {"elo": ALL_FACTORIES["elo"], "frequency": ALL_FACTORIES["frequency"]},
        with_ensemble=False,
        refit_every_days=180,
        n_boot=200,
    )
    assert set(report.overall) == {"elo", "frequency"}
    assert report.market_subset is None
    assert any("market benchmark unavailable" in n for n in report.notes)


def test_insufficient_history_is_a_clear_error(
    setup: tuple[pd.DataFrame, object, pd.DataFrame],
) -> None:
    from football_ai.core.exceptions import InsufficientDataError
    from football_ai.training.backtesting import walk_forward

    frame, _, _ = setup
    with pytest.raises(InsufficientDataError, match="no training season before 2019"):
        split_by_seasons(frame, [2019], [2020])
    early = frame[frame["season_year"] == 2019].head(20)
    with pytest.raises(InsufficientDataError, match="played matches before"):
        walk_forward(ALL_FACTORIES["elo"], frame, early)
