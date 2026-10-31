"""XGBoost / LightGBM models, tuning, experiment tracking, train-models CLI."""

from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from football_ai.cli import EXIT_OK, main
from football_ai.database.connection import create_db_engine
from football_ai.database.models import Base
from football_ai.features.builder import FeatureBuilder, feature_group
from football_ai.features.store import save_feature_set
from football_ai.models.lightgbm import LightGBMGoalsModel, LightGBMModel
from football_ai.models.tabular.common import chronological_holdout
from football_ai.models.xgboost import XGBoostModel
from football_ai.training.datasets import TemporalSplit, split_by_seasons
from football_ai.training.evaluate import log_loss
from football_ai.training.tracking import ExperimentTracker
from football_ai.training.tuning import SEARCH_SPACES, random_search
from tests.synthetic import generate_league


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    league = generate_league(
        n_teams=12, seasons=(2019, 2020, 2021, 2022, 2023), seed=8, upcoming_rounds=0
    )
    return FeatureBuilder().build(league.frames)


@pytest.mark.parametrize("cls", [XGBoostModel, LightGBMModel])
def test_classifiers_fit_predict_and_exclude_market(cls: Any, frame: pd.DataFrame) -> None:
    model = cls(n_estimators=150).fit(frame)
    p = model.predict_proba(frame.tail(10))
    assert p.shape == (10, 3) and np.allclose(p.sum(axis=1), 1)
    assert not any(feature_group(c) == "market" for c in model.columns)
    with_market = cls(use_market=True, n_estimators=50).fit(frame)
    assert "market_p_home" in with_market.columns and with_market.name.endswith("_market")
    imp = model.feature_importance()
    assert imp["importance"].sum() == pytest.approx(1.0) and imp.iloc[0]["importance"] > 0
    # robust to missing columns at prediction time (NaN, handled natively)
    p2 = model.predict_proba(frame.tail(3).drop(columns=["home_ppg_5"]))
    assert np.isfinite(p2).all()


def test_early_stopping_uses_chronological_training_holdout(frame: pd.DataFrame) -> None:
    fit_part, holdout = chronological_holdout(frame[frame["outcome"].notna()])
    assert holdout is not None
    assert fit_part["kickoff_utc"].max() <= holdout["kickoff_utc"].min()
    tiny_fit, tiny_hold = chronological_holdout(frame.head(40))
    assert tiny_hold is None and len(tiny_fit) == 40
    model = XGBoostModel(n_estimators=2000, learning_rate=0.1).fit(frame)
    assert model.best_iteration is not None and model.best_iteration < 1999


def test_goals_model_tracks_true_intensity(frame: pd.DataFrame) -> None:
    model = LightGBMGoalsModel(n_estimators=200).fit(frame)
    lam = model.expected_goals(frame)
    assert (lam > 0).all()
    corr = np.corrcoef(lam[:, 0], frame["home_xg_for_avg_10"].fillna(frame["home_goals"].mean()))
    assert corr[0, 1] > 0.2
    m = model.score_matrix(frame.head(2))
    assert np.allclose(m.sum(axis=(1, 2)), 1)


def test_gbm_beats_frequency(frame: pd.DataFrame) -> None:
    split = split_by_seasons(frame, [2023])
    y = split.validation["outcome"].to_numpy(dtype=int)
    from football_ai.models.baselines import FrequencyModel

    base = log_loss(y, FrequencyModel().fit(split.train).predict_proba(split.validation))
    gbm = log_loss(y, LightGBMModel().fit(split.train).predict_proba(split.validation))
    assert gbm < base


def test_random_search_uses_validation_only(frame: pd.DataFrame) -> None:
    split = split_by_seasons(frame, [2023])
    guarded = TemporalSplit(split.train, split.validation, object())  # type: ignore[arg-type]
    result = random_search(
        LightGBMModel, SEARCH_SPACES["lightgbm"], guarded, n_trials=3, fixed={"n_estimators": 80}
    )
    assert len(result.trials) == 3
    assert result.best_log_loss == result.trials["log_loss"].min()
    assert set(result.best_params) == set(SEARCH_SPACES["lightgbm"](np.random.default_rng(0)))
    assert isinstance(result.best_params["num_leaves"], int)  # types preserved


def test_tracker_mlflow_and_json_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = ExperimentTracker(tmp_path / "exp", "unit")
    assert tracker.backend == "mlflow"
    with tracker.run("r1", {"a": 1, "b": [1, 2]}) as run:
        run.log_metrics({"log_loss": 0.9, "n": 10, "label": "x"}, prefix="val_")
    assert run.metrics == {"val_log_loss": 0.9, "val_n": 10.0}
    assert list((tmp_path / "exp" / "runs" / "unit").glob("r1_*.json"))

    real_import = builtins.__import__

    def no_mlflow(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "mlflow":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_mlflow)
    fallback = ExperimentTracker(tmp_path / "exp2", "unit")
    assert fallback.backend == "json"
    with fallback.run("r2", {}) as run2:
        run2.log_metrics({"x": 1.0})
    assert list((tmp_path / "exp2" / "runs" / "unit").glob("r2_*.json"))


def test_train_models_cli_end_to_end(
    frame: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "var"
    save_feature_set(frame, FeatureBuilder().config, data_dir / "features")
    db_url = f"sqlite:///{tmp_path / 'cli.db'}"
    Base.metadata.create_all(create_db_engine(db_url))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    code = main([
        "train-models", "--models", "frequency,dixon_coles,lightgbm",
        "--validation-seasons", "2022", "--test-seasons", "2023", "--refit-days", "120",
        "--tune", "2", "--register",
    ])  # fmt: skip
    assert code == EXIT_OK
    assert list((data_dir / "reports").glob("models_*.json"))
    assert list((data_dir / "models" / "lightgbm").glob("*/feature_importance.csv"))
    assert list((data_dir / "models" / "dixon_coles").glob("*/model.joblib"))
