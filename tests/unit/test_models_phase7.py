"""PyTorch siamese multi-task network (skipped when torch is not installed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from football_ai.features.builder import FeatureBuilder, feature_group
from football_ai.models.baselines import FrequencyModel
from football_ai.models.deep_learning.model import NeuralModel, Preprocessor
from football_ai.models.tabular.common import select_features
from football_ai.training.datasets import split_by_seasons
from football_ai.training.evaluate import log_loss
from football_ai.training.registry import load_model, save_model
from football_ai.training.train import ALL_FACTORIES
from tests.synthetic import generate_league


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    league = generate_league(
        n_teams=12, seasons=(2019, 2020, 2021, 2022, 2023), seed=8, upcoming_rounds=1
    )
    return FeatureBuilder().build(league.frames)


@pytest.fixture(scope="module")
def fitted(frame: pd.DataFrame) -> NeuralModel:
    split = split_by_seasons(frame, [2023])
    return NeuralModel(max_epochs=60, patience=8).fit(split.train)


def test_preprocessor_layout_and_train_only_statistics(frame: pd.DataFrame) -> None:
    played = frame[frame["outcome"].notna()]
    cols = select_features(played, use_market=False)
    prep = Preprocessor.fit(played, cols)
    assert "ppg_5" in prep.team_suffixes and "elo" in prep.team_suffixes
    assert all(not c.startswith(("home_", "away_")) or c.endswith("h2h_count") is False
               for c in prep.context_columns if c.startswith("diff_"))  # fmt: skip
    assert not any(feature_group(c) == "market" for c in prep.context_columns)
    arrays = prep.transform(frame.head(5))
    assert arrays["home"].shape == arrays["away"].shape == (5, len(prep.team_suffixes))
    assert np.isfinite(arrays["context"]).all()  # NaN imputed + indicators
    assert (arrays["home_team"] > 0).all()


def test_network_beats_frequency_and_outputs_markets(
    frame: pd.DataFrame, fitted: NeuralModel
) -> None:
    split = split_by_seasons(frame, [2023])
    y = split.validation["outcome"].to_numpy(dtype=int)
    nn_loss = log_loss(y, fitted.predict_proba(split.validation))
    base = log_loss(y, FrequencyModel().fit(split.train).predict_proba(split.validation))
    assert nn_loss < base
    p = fitted.predict_proba(split.validation)
    assert np.allclose(p.sum(axis=1), 1)
    rates = fitted.expected_goals(split.validation)
    assert (rates > 0).all() and 0.5 < rates.mean() < 3.0
    assert np.allclose(fitted.score_matrix(split.validation.head(3)).sum(axis=(1, 2)), 1)
    # the goal head is consistent with the outcome head (same direction on average)
    goal_p = fitted.goal_head_outcome_proba(split.validation)
    assert np.corrcoef(goal_p[:, 0], p[:, 0])[0, 1] > 0.5
    assert 0 < fitted.epochs_trained <= 60 and fitted.history


def test_unknown_teams_and_upcoming_matches(frame: pd.DataFrame, fitted: NeuralModel) -> None:
    upcoming = frame[frame["outcome"].isna()]
    assert len(upcoming) > 0 and np.isfinite(fitted.predict_proba(upcoming)).all()
    unknown = frame.head(2).assign(home_team_id=987654)
    assert not fitted.known_teams(unknown).any()
    assert np.isfinite(fitted.predict_proba(unknown)).all()


def test_seeded_training_is_reproducible(frame: pd.DataFrame) -> None:
    data = frame[frame["season_year"] <= 2021]
    a = NeuralModel(max_epochs=5, seed=3).fit(data).predict_proba(frame.tail(5))
    b = NeuralModel(max_epochs=5, seed=3).fit(data).predict_proba(frame.tail(5))
    assert np.allclose(a, b)


def test_persistence_roundtrip(frame: pd.DataFrame, fitted: NeuralModel, tmp_path: Path) -> None:
    reg = save_model(
        fitted, name="neural", algorithm="NeuralModel", params={}, metrics={},
        data_fingerprint="x", training_start=None, training_end=None, root=tmp_path,
    )  # fmt: skip
    loaded = load_model(reg.path)
    assert np.allclose(loaded.predict_proba(frame.tail(4)), fitted.predict_proba(frame.tail(4)))
    assert "neural" in ALL_FACTORIES and "neural_market" in ALL_FACTORIES
