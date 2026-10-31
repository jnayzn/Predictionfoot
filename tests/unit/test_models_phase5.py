"""Poisson / Dixon-Coles / baselines, markets, metrics, splits, walk-forward."""

from __future__ import annotations

from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad

from football_ai.features.builder import FeatureBuilder
from football_ai.models.base import OutcomeModel, check_probabilities
from football_ai.models.baselines import EloModel, FrequencyModel
from football_ai.models.dixon_coles import DixonColesModel, tau
from football_ai.models.poisson import PoissonModel, poisson_nll
from football_ai.prediction.market_prediction import markets_from_matrix, over_probability
from football_ai.prediction.score_prediction import (
    outcome_probabilities,
    poisson_score_matrix,
    top_scores,
)
from football_ai.training.backtesting import walk_forward
from football_ai.training.datasets import split_by_dates, split_by_seasons
from football_ai.training.evaluate import (
    brier_score,
    calibration_table,
    evaluate_probabilities,
    log_loss,
    ranked_probability_score,
)
from football_ai.training.registry import load_model, save_model
from football_ai.training.train import comparison_table, evaluate_models
from tests.synthetic import generate_league


@pytest.fixture(scope="module")
def league_features() -> tuple[pd.DataFrame, object]:
    league = generate_league(
        n_teams=14, seasons=(2019, 2020, 2021, 2022, 2023), seed=21, upcoming_rounds=0
    )
    return FeatureBuilder().build(league.frames), league


# ------------------------------------------------------------------- poisson
def test_poisson_gradient_matches_numerical() -> None:
    rng = np.random.default_rng(0)
    n, teams = 60, 6
    h, a = rng.integers(0, teams, n), rng.integers(0, teams, n)
    yh, ya = rng.poisson(1.4, n).astype(float), rng.poisson(1.1, n).astype(float)
    w = rng.uniform(0.5, 1.0, n)
    theta = rng.normal(0, 0.1, 2 + 2 * teams)
    err = check_grad(
        lambda t: poisson_nll(t, h, a, yh, ya, w, teams, 0.7)[0],
        lambda t: poisson_nll(t, h, a, yh, ya, w, teams, 0.7)[1],
        theta,
    )
    assert err < 1e-4


def test_poisson_recovers_latent_strengths(league_features: tuple[pd.DataFrame, object]) -> None:
    frame, league = league_features
    model = PoissonModel().fit(frame)
    assert model.params is not None
    teams = sorted(league.attack)  # type: ignore[attr-defined]
    att = [model.params.attack[t] for t in teams]
    true = [league.attack[t] for t in teams]  # type: ignore[attr-defined]
    assert np.corrcoef(att, true)[0, 1] > 0.9
    assert 0.1 < model.params.home_advantage < 0.4  # true value 0.25
    p = model.predict_proba(frame.head(5))
    assert p.shape == (5, 3) and np.allclose(p.sum(axis=1), 1)


def test_unknown_team_gets_average_strength(league_features: tuple[pd.DataFrame, object]) -> None:
    frame, _ = league_features
    model = PoissonModel().fit(frame)
    rows = frame.head(1).assign(home_team_id=99999)
    assert not model.known_teams(rows)[0]
    assert np.isfinite(model.expected_goals(rows)).all()


def test_dixon_coles_tau_and_matrix() -> None:
    lam_h, lam_a = np.array([1.3]), np.array([0.9])
    t = tau(np.array([0, 0, 1, 1, 2]), np.array([0, 1, 0, 1, 2]),
            np.repeat(lam_h, 5), np.repeat(lam_a, 5), 0.1)  # fmt: skip
    assert np.allclose(t, [1 - 1.3 * 0.9 * 0.1, 1 + 1.3 * 0.1, 1 + 0.9 * 0.1, 0.9, 1.0])
    model = DixonColesModel()
    model.params = (
        PoissonModel()
        .fit(
            generate_league(
                n_teams=6, seasons=(2023,), seed=1, upcoming_rounds=0
            ).frames.matches.assign(outcome=0.0)
        )
        .params
    )
    model.rho = -0.1
    rows = pd.DataFrame({"home_team_id": [1], "away_team_id": [2]})
    m = model.score_matrix(rows)
    assert m.sum() == pytest.approx(1.0)
    base = PoissonModel()
    base.params = model.params
    assert m[0, 0, 0] > base.score_matrix(rows)[0, 0, 0]  # rho < 0 inflates 0-0


# ------------------------------------------------------------------- markets
def test_score_matrix_and_markets() -> None:
    m = poisson_score_matrix(np.array([1.5, 0.8]), np.array([1.0, 2.0]))
    assert np.allclose(m.sum(axis=(1, 2)), 1)
    p = outcome_probabilities(m)
    assert np.allclose(p.sum(axis=1), 1) and p[0, 0] > p[0, 2] and p[1, 2] > p[1, 0]
    markets = markets_from_matrix(m)
    first = markets[0]
    assert first["1x2"]["home"] == pytest.approx(p[0, 0])
    assert first["double_chance"]["1x"] == pytest.approx(p[0, 0] + p[0, 1])
    assert first["over_under"]["2.5"]["over"] + first["over_under"]["2.5"][
        "under"
    ] == pytest.approx(1)
    assert first["expected_goals"]["home"] == pytest.approx(1.5, abs=1e-3)
    assert first["over_under"]["1.5"]["over"] > first["over_under"]["3.5"]["over"]
    probs = [s["probability"] for s in first["top_scores"]]
    assert probs == sorted(probs, reverse=True) and len(probs) == 5
    assert first["most_likely_score"] == first["top_scores"][0]["score"]
    assert first["team_goals"]["home"]["1+"] > first["team_goals"]["home"]["2+"]
    # independent Poisson: P(total > 2.5) equals the Poisson(2.5) tail
    from scipy.stats import poisson

    assert over_probability(m, 2.5)[0] == pytest.approx(1 - poisson.cdf(2, 2.5), abs=1e-6)
    assert top_scores(m, 1)[0][0][:2] == (1, 1)


# -------------------------------------------------------------------- metrics
def test_metrics_known_values() -> None:
    y = np.array([0, 1, 2])
    perfect = np.eye(3)
    uniform = np.full((3, 3), 1 / 3)
    assert log_loss(y, perfect) == pytest.approx(0, abs=1e-9)
    assert log_loss(y, uniform) == pytest.approx(np.log(3))
    assert brier_score(y, perfect) == 0 and brier_score(y, uniform) == pytest.approx(2 / 3)
    # RPS: predicting away (0,0,1) when home happened = worst case 1
    assert ranked_probability_score(np.array([0]), np.array([[0, 0, 1.0]])) == pytest.approx(1)
    # ...and it prefers "draw" to "away" when home happened (ordinal metric)
    assert ranked_probability_score(np.array([0]), np.array([[0, 1.0, 0]])) == pytest.approx(0.5)
    table = calibration_table(y, uniform)
    assert table[0]["mean_predicted"] == pytest.approx(1 / 3)
    metrics = evaluate_probabilities(y, perfect)
    assert metrics["accuracy"] == 1.0 and metrics["f1_macro"] == 1.0


def test_check_probabilities_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        check_probabilities(np.ones((2, 2)))
    with pytest.raises(ValueError):
        check_probabilities(np.array([[np.nan, 0.5, 0.5]]))


# --------------------------------------------------------------------- splits
def test_split_by_seasons_is_chronological(league_features: tuple[pd.DataFrame, object]) -> None:
    frame, _ = league_features
    split = split_by_seasons(frame, [2022], [2023])
    assert split.train["season_year"].max() == 2021
    assert split.train["kickoff_utc"].max() < split.validation["kickoff_utc"].min()
    assert split.validation["kickoff_utc"].max() < split.test["kickoff_utc"].min()
    with pytest.raises(ValueError):
        split_by_seasons(frame, [2023], [2022])  # test before validation
    with pytest.raises(ValueError):
        split_by_seasons(frame, [2022], [2022])
    by_date = split_by_dates(frame, pd.Timestamp("2022-07-01", tz="UTC"))
    assert by_date.train["kickoff_utc"].max() < pd.Timestamp("2022-07-01", tz="UTC")


def test_walk_forward_never_trains_on_the_future(
    league_features: tuple[pd.DataFrame, object],
) -> None:
    frame, _ = league_features
    seen: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    class Spy(FrequencyModel):
        def fit(self, train: pd.DataFrame) -> Self:
            self.last_train = train["kickoff_utc"].max()
            return super().fit(train)

        def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
            seen.append((self.last_train, rows["kickoff_utc"].min()))
            return super().predict_proba(rows)

    split = split_by_seasons(frame, [2022], [2023])
    result = walk_forward(Spy, frame, split.validation, refit_every_days=30)
    assert result.refits >= 6 and len(result.rows) == len(split.validation)
    assert all(train_max < first_target for train_max, first_target in seen)


def test_models_beat_frequency_baseline(league_features: tuple[pd.DataFrame, object]) -> None:
    frame, _ = league_features
    split = split_by_seasons(frame, [2022, 2023])
    results = {r.name: r.metrics for r in evaluate_models(frame, split)}
    assert results["dixon_coles"]["log_loss"] < results["frequency"]["log_loss"]
    assert results["poisson"]["log_loss"] < results["frequency"]["log_loss"]
    assert results["elo"]["log_loss"] < results["frequency"]["log_loss"]
    assert results["poisson"]["mae_total_goals"] < 2.0
    table = comparison_table(
        [r for r in evaluate_models(frame, split, {"frequency": FrequencyModel})]
    )
    assert list(table.index) == ["frequency"]
    assert isinstance(EloModel(), OutcomeModel)


def test_registry_roundtrip(league_features: tuple[pd.DataFrame, object], tmp_path: Path) -> None:
    frame, _ = league_features
    model = DixonColesModel().fit(frame)
    reg = save_model(
        model, name="dixon_coles", algorithm="DixonColesModel", params={"l2": 1.0},
        metrics={"log_loss": 0.9}, data_fingerprint="abc", training_start=None,
        training_end=None, root=tmp_path,
    )  # fmt: skip
    loaded = load_model(reg.path)
    assert np.allclose(loaded.predict_proba(frame.head(3)), model.predict_proba(frame.head(3)))
    assert (reg.path.parent / "metadata.json").is_file()
