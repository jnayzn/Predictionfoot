"""Model commands: train / tune / evaluate / register (Phases 5-6)."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from football_ai.bootstrap import build_session_factory
from football_ai.cli.common import EXIT_ERROR, EXIT_OK, parse_seasons
from football_ai.config.settings import Settings
from football_ai.core.io import atomic_write_json
from football_ai.database.connection import session_scope
from football_ai.features.store import data_fingerprint, latest_feature_set, load_feature_set
from football_ai.models.base import OutcomeModel
from football_ai.training.datasets import TemporalSplit, split_by_seasons
from football_ai.training.registry import save_model
from football_ai.training.tracking import ExperimentTracker
from football_ai.training.train import (
    ALL_FACTORIES,
    BASELINE_FACTORIES,
    MODEL_CLASSES,
    ModelFactory,
    comparison_table,
    evaluate_models,
)
from football_ai.training.tuning import SEARCH_SPACES, random_search


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--features", type=Path, default=None, help="feature set (default: latest)")
    p.add_argument("--competition", action="append", dest="competitions")
    p.add_argument("--validation-seasons", type=parse_seasons, required=True)
    p.add_argument("--test-seasons", type=parse_seasons, default=[])
    p.add_argument(
        "--evaluate-test",
        action="store_true",
        help="ALSO report test metrics (final evaluation only - never use them to tune)",
    )
    p.add_argument("--refit-days", type=int, default=30, help="walk-forward refit period")
    p.add_argument("--register", action="store_true", help="fit on train+validation and register")


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("train-baselines", help="Evaluate frequency/Elo/Poisson/Dixon-Coles")
    _common(p)
    p.set_defaults(handler=lambda a, s: cmd_train(a, s, list(BASELINE_FACTORIES)))

    p = sub.add_parser("train-models", help="Evaluate, tune and register any set of models")
    _common(p)
    p.add_argument(
        "--models",
        default="frequency,elo,dixon_coles,xgboost,lightgbm,lightgbm_goals",
        help=f"comma-separated, among: {','.join(ALL_FACTORIES)}",
    )
    p.add_argument("--tune", type=int, default=0, metavar="N", help="random-search trials")
    p.set_defaults(handler=lambda a, s: cmd_train(a, s, a.models.split(",")))

    p = sub.add_parser("train-ensemble", help="Learn ensemble weights + calibration on validation")
    _common(p)
    p.add_argument("--models", default="elo,dixon_coles,xgboost,lightgbm,lightgbm_goals")
    p.add_argument("--pool", choices=["auto", "linear", "log_linear"], default="auto")
    p.set_defaults(handler=cmd_train_ensemble)


def load_features(settings: Settings, path: Path | None) -> pd.DataFrame:
    path = path or latest_feature_set(settings.data_dir / "features")
    if path is None:
        raise FileNotFoundError("no feature set found - run `football-ai build-features` first")
    frame, _ = load_feature_set(path)
    frame["kickoff_utc"] = pd.to_datetime(frame["kickoff_utc"], utc=True)
    return frame


def _tuned_factories(
    names: list[str], split: TemporalSplit, trials: int, tracker: ExperimentTracker
) -> tuple[dict[str, ModelFactory], dict[str, dict[str, Any]]]:
    factories: dict[str, ModelFactory] = {}
    best: dict[str, dict[str, Any]] = {}
    for name in names:
        if trials and name in SEARCH_SPACES and name in MODEL_CLASSES:
            result = random_search(MODEL_CLASSES[name], SEARCH_SPACES[name], split, n_trials=trials)
            best[name] = result.best_params
            with tracker.run(f"tune_{name}", {"trials": trials}) as run:
                run.log_metrics({"best_validation_log_loss": result.best_log_loss})
            print(f"tuned {name}: log_loss={result.best_log_loss:.4f} params={result.best_params}")
            cls, params = MODEL_CLASSES[name], result.best_params
            factories[name] = lambda cls=cls, params=params: cls(**params)  # type: ignore[misc]
        else:
            factories[name] = ALL_FACTORIES[name]
    return factories, best


def cmd_train(args: argparse.Namespace, settings: Settings, names: list[str]) -> int:
    unknown = set(names) - set(ALL_FACTORIES)
    if unknown:
        print(f"Unknown models: {sorted(unknown)}; available: {sorted(ALL_FACTORIES)}")
        return EXIT_ERROR
    try:
        frame = load_features(settings, args.features)
    except FileNotFoundError as exc:
        print(exc)
        return EXIT_ERROR
    if args.competitions:
        frame = frame[frame["competition_key"].isin(args.competitions)]
    split = split_by_seasons(frame, args.validation_seasons, args.test_seasons)
    print(json.dumps(split.describe(), indent=2))
    tracker = ExperimentTracker(settings.data_dir / "experiments", "football-ai")
    factories, tuned = _tuned_factories(names, split, getattr(args, "tune", 0), tracker)

    report: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "split": split.describe(),
        "tuned_params": tuned,
        "tracking_backend": tracker.backend,
    }
    validation = evaluate_models(frame, split, factories, refit_every_days=args.refit_days)
    print("\nVALIDATION (walk-forward)\n" + comparison_table(validation).round(4).to_string())
    report["validation"] = {r.name: r.metrics for r in validation}
    for result in validation:
        with tracker.run(
            f"eval_{result.name}", {"model": result.name, **tuned.get(result.name, {})}
        ) as run:
            run.log_metrics(result.metrics, prefix="validation_")
    if args.evaluate_test:
        test = evaluate_models(frame, split, factories, on="test", refit_every_days=args.refit_days)
        print("\nTEST (final, do not tune on it)\n" + comparison_table(test).round(4).to_string())
        report["test"] = {r.name: r.metrics for r in test}

    path = settings.data_dir / "reports" / f"models_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    atomic_write_json(path, report)
    print(f"\nreport saved to {path} (tracking: {tracker.backend})")
    if args.register:
        _register(settings, split, factories, validation, tuned)
    return EXIT_OK


def _register(
    settings: Settings,
    split: TemporalSplit,
    factories: dict[str, ModelFactory],
    validation: list[Any],
    tuned: dict[str, dict[str, Any]],
) -> None:
    fit_data = pd.concat([split.train, split.validation])
    fingerprint = data_fingerprint(fit_data)
    with session_scope(build_session_factory(settings)) as session:
        for result in validation:
            model: OutcomeModel = factories[result.name]().fit(fit_data)
            registered = save_model(
                model,
                name=result.name,
                algorithm=type(model).__name__,
                params=tuned.get(result.name, {}),
                metrics={"validation": result.metrics},
                data_fingerprint=fingerprint,
                training_start=fit_data["kickoff_utc"].min().to_pydatetime(),
                training_end=fit_data["kickoff_utc"].max().to_pydatetime(),
                root=settings.data_dir / "models",
                session=session,
            )
            importance = getattr(model, "feature_importance", None)
            if callable(importance):
                importance().to_csv(registered.path.parent / "feature_importance.csv", index=False)
            print(f"registered {registered.name} {registered.version}")


def cmd_train_ensemble(args: argparse.Namespace, settings: Settings) -> int:
    from football_ai.training.ensemble_pipeline import build_ensemble

    names = [n.strip() for n in args.models.split(",") if n.strip()]
    unknown = set(names) - set(ALL_FACTORIES)
    if unknown:
        print(f"Unknown models: {sorted(unknown)}; available: {sorted(ALL_FACTORIES)}")
        return EXIT_ERROR
    try:
        frame = load_features(settings, args.features)
    except FileNotFoundError as exc:
        print(exc)
        return EXIT_ERROR
    if args.competitions:
        frame = frame[frame["competition_key"].isin(args.competitions)]
    split = split_by_seasons(frame, args.validation_seasons, args.test_seasons)
    factories = {n: ALL_FACTORIES[n] for n in names}
    ensemble, report = build_ensemble(
        frame,
        split,
        factories,
        pool=args.pool,
        refit_every_days=args.refit_days,
        evaluate_test=args.evaluate_test,
        fit_production_members=args.register,
    )
    print(f"pool: {report.pool}")
    print("weights: " + ", ".join(f"{k}={v:.3f}" for k, v in report.weights.items()))
    print(f"pruned: {report.pruned}")
    print(f"calibration: {report.calibration_method} {report.calibration_scores}")
    print("goal weights: " + ", ".join(f"{k}={v:.3f}" for k, v in report.goal_weights.items()))
    rows = {**report.validation_members, **report.validation_cv}
    print("\nVALIDATION (members walk-forward, ensemble cross-fitted)")
    print(
        pd.DataFrame(rows).T[["log_loss", "brier", "rps", "accuracy", "ece"]].round(4).to_string()
    )
    if report.test:
        print("\nTEST (final, frozen ensemble)")
        print(
            pd.DataFrame(report.test)
            .T[["log_loss", "brier", "rps", "accuracy", "ece"]]
            .round(4)
            .to_string()
        )
    path = settings.data_dir / "reports" / f"ensemble_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    atomic_write_json(path, {"split": split.describe(), **report.to_dict()})
    print(f"\nreport saved to {path}")
    if args.register:
        fit_data = pd.concat([split.train, split.validation])
        with session_scope(build_session_factory(settings)) as session:
            registered = save_model(
                ensemble,
                name="ensemble",
                algorithm=f"EnsembleModel[{report.pool}]",
                params={
                    "members": names,
                    "weights": report.weights,
                    "goal_weights": report.goal_weights,
                    "calibration": report.calibration_method,
                },
                metrics={"validation_cv": report.validation_cv, "test": report.test},
                data_fingerprint=data_fingerprint(fit_data),
                training_start=fit_data["kickoff_utc"].min().to_pydatetime(),
                training_end=fit_data["kickoff_utc"].max().to_pydatetime(),
                root=settings.data_dir / "models",
                session=session,
            )
        print(f"registered ensemble {registered.version}")
    return EXIT_OK
