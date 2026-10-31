"""Calibration methods, ensemble weights, matrix reconciliation, ensemble pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from football_ai.features.builder import FeatureBuilder
from football_ai.models.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureCalibrator,
    select_calibrator,
)
from football_ai.models.ensemble import (
    combine,
    combine_matrices,
    fit_weights,
    reconcile_matrix,
    score_log_likelihood,
)
from football_ai.prediction.score_prediction import outcome_probabilities, poisson_score_matrix
from football_ai.training.datasets import split_by_seasons
from football_ai.training.ensemble_pipeline import build_ensemble
from football_ai.training.evaluate import expected_calibration_error, log_loss
from football_ai.training.registry import load_model, save_model
from football_ai.training.train import ALL_FACTORIES
from tests.synthetic import generate_league


def simulated(n: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """True probabilities + outcomes drawn from them."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 0.8, (n, 3)) + np.array([0.4, 0.0, -0.2])
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(3, p=row) for row in p])
    return p, y


def overconfident(p: np.ndarray, power: float = 2.5) -> np.ndarray:
    q = p**power
    return q / q.sum(axis=1, keepdims=True)


@pytest.mark.parametrize("cls", [TemperatureCalibrator, PlattCalibrator, IsotonicCalibrator])
def test_calibrators_repair_overconfidence(cls: type) -> None:
    p, y = simulated()
    bad = overconfident(p)
    fit_p, fit_y, eval_p, eval_y = bad[:2000], y[:2000], bad[2000:], y[2000:]
    cal = cls().fit(fit_p, fit_y)
    fixed = cal.transform(eval_p)
    assert np.allclose(fixed.sum(axis=1), 1)
    assert log_loss(eval_y, fixed) < log_loss(eval_y, eval_p)
    assert expected_calibration_error(eval_y, fixed) < expected_calibration_error(eval_y, eval_p)


def test_temperature_recovers_power() -> None:
    p, y = simulated(8000, 1)
    cal = TemperatureCalibrator().fit(overconfident(p, 2.0), y)
    assert cal.temperature == pytest.approx(2.0, rel=0.15)


def test_selection_prefers_none_on_calibrated_input_and_low_variance_on_small_data() -> None:
    p, y = simulated(3000, 2)
    chosen, scores = select_calibrator(p, y)
    assert set(scores) == {"none", "temperature", "platt", "isotonic"}
    assert scores["isotonic"] > scores["none"]  # flexible method overfits calibrated input
    assert chosen.name in ("none", "temperature", "platt")
    _, small = select_calibrator(p[:100], y[:100])
    assert set(small) == {"none", "temperature"}


def test_ensemble_weights_favor_the_better_model_and_prune_noise() -> None:
    p, y = simulated(3000, 3)
    rng = np.random.default_rng(0)
    noise = rng.dirichlet(np.ones(3), size=len(y))
    good = 0.85 * p + 0.15 / 3
    probas = {"good": good, "noise": noise, "flat": np.full_like(p, 1 / 3)}
    for pool in ("linear", "log_linear"):
        result = fit_weights(probas, y, pool=pool)
        assert result.weights["good"] > 0.8
        assert "noise" in result.pruned
        assert result.validation_log_loss <= log_loss(y, good) + 1e-6
        assert np.allclose(combine(probas, result.weights, pool).sum(axis=1), 1)
    with pytest.raises(ValueError):
        fit_weights(probas, y, pool="median")


def test_matrix_reconciliation_and_mixing() -> None:
    m = poisson_score_matrix(np.array([1.4, 0.9]), np.array([1.1, 1.6]))
    target = np.array([[0.5, 0.3, 0.2], [0.2, 0.25, 0.55]])
    r = reconcile_matrix(m, target)
    assert np.allclose(outcome_probabilities(r), target)
    # shape inside a region is preserved: ratio of two home-win scores unchanged
    assert r[0, 2, 0] / r[0, 1, 0] == pytest.approx(m[0, 2, 0] / m[0, 1, 0])
    mix = combine_matrices({"a": m, "b": m[::-1]}, {"a": 0.5, "b": 0.5})
    assert np.allclose(mix.sum(axis=(1, 2)), 1)
    ll = score_log_likelihood(m, np.array([1, 0]), np.array([1, 2]))
    assert ll == pytest.approx(np.mean(np.log([m[0, 1, 1], m[1, 0, 2]])))


def test_build_ensemble_end_to_end(tmp_path: Path) -> None:
    league = generate_league(n_teams=12, seasons=(2019, 2020, 2021, 2022, 2023), seed=5)
    frame = FeatureBuilder().build(league.frames)
    split = split_by_seasons(frame, [2021, 2022], [2023])
    factories = {k: ALL_FACTORIES[k] for k in ("frequency", "elo", "dixon_coles", "lightgbm_goals")}
    ensemble, report = build_ensemble(
        frame, split, factories, refit_every_days=120, evaluate_test=True
    )
    assert report.pool in ("linear", "log_linear")
    assert sum(report.weights.values()) == pytest.approx(1.0)
    assert set(report.goal_weights) == {"dixon_coles", "lightgbm_goals"}
    assert report.test is not None and "ensemble" in report.test
    best_member = min(v["log_loss"] for k, v in report.test.items() if k != "ensemble")
    assert report.test["ensemble"]["log_loss"] < best_member + 0.01
    upcoming = frame[frame["outcome"].isna()]
    p = ensemble.predict_proba(upcoming)
    assert np.allclose(outcome_probabilities(ensemble.score_matrix(upcoming)), p, atol=1e-6)
    # production members only for non-zero weights, persisted with the registry
    assert set(ensemble.members) == {
        n for n, w in {**report.weights, **report.goal_weights}.items() if w > 0
    }
    reg = save_model(
        ensemble, name="ensemble", algorithm="EnsembleModel", params={}, metrics={},
        data_fingerprint="x", training_start=None, training_end=None, root=tmp_path,
    )  # fmt: skip
    assert np.allclose(load_model(reg.path).predict_proba(upcoming), p)
    assert isinstance(pd.DataFrame(report.validation_members).T, pd.DataFrame)
