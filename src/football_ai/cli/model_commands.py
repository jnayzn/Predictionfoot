"""Phase 5+ commands: train, evaluate and register models."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from football_ai.bootstrap import build_session_factory
from football_ai.cli.common import EXIT_ERROR, EXIT_OK, parse_seasons
from football_ai.config.settings import Settings
from football_ai.core.io import atomic_write_json
from football_ai.database.connection import session_scope
from football_ai.features.store import data_fingerprint, latest_feature_set, load_feature_set
from football_ai.training.datasets import split_by_seasons
from football_ai.training.registry import save_model
from football_ai.training.train import BASELINE_FACTORIES, comparison_table, evaluate_models


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("train-baselines", help="Evaluate frequency/Elo/Poisson/Dixon-Coles models")
    p.add_argument("--features", type=Path, default=None, help="feature set (default: latest)")
    p.add_argument("--competition", action="append", dest="competitions")
    p.add_argument("--validation-seasons", type=parse_seasons, required=True)
    p.add_argument("--test-seasons", type=parse_seasons, default=[])
    p.add_argument(
        "--evaluate-test", action="store_true",
        help="ALSO report test metrics (final evaluation only - never use them to tune)",
    )  # fmt: skip
    p.add_argument("--refit-days", type=int, default=30, help="walk-forward refit period")
    p.add_argument("--register", action="store_true", help="fit on train+validation and register")
    p.set_defaults(handler=cmd_train_baselines)


def load_features(settings: Settings, path: Path | None) -> pd.DataFrame:
    path = path or latest_feature_set(settings.data_dir / "features")
    if path is None:
        raise FileNotFoundError("no feature set found - run `football-ai build-features` first")
    frame, _ = load_feature_set(path)
    frame["kickoff_utc"] = pd.to_datetime(frame["kickoff_utc"], utc=True)
    return frame


def cmd_train_baselines(args: argparse.Namespace, settings: Settings) -> int:
    try:
        frame = load_features(settings, args.features)
    except FileNotFoundError as exc:
        print(exc)
        return EXIT_ERROR
    if args.competitions:
        frame = frame[frame["competition_key"].isin(args.competitions)]
    split = split_by_seasons(frame, args.validation_seasons, args.test_seasons)
    print(json.dumps(split.describe(), indent=2))
    report: dict[str, object] = {
        "split": split.describe(),
        "created_at": datetime.now(UTC).isoformat(),
    }

    validation = evaluate_models(frame, split, on="validation", refit_every_days=args.refit_days)
    table = comparison_table(validation)
    print("\nVALIDATION (walk-forward)\n" + table.round(4).to_string())
    report["validation"] = {r.name: r.metrics for r in validation}
    if args.evaluate_test:
        test = evaluate_models(frame, split, on="test", refit_every_days=args.refit_days)
        print("\nTEST (final, do not tune on it)\n" + comparison_table(test).round(4).to_string())
        report["test"] = {r.name: r.metrics for r in test}

    reports_dir = settings.data_dir / "reports"
    path = reports_dir / f"baselines_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    atomic_write_json(path, report)
    print(f"\nreport saved to {path}")

    if args.register:
        fit_data = pd.concat([split.train, split.validation])
        fingerprint = data_fingerprint(fit_data)
        with session_scope(build_session_factory(settings)) as session:
            for result in validation:
                model = BASELINE_FACTORIES[result.name]().fit(fit_data)
                registered = save_model(
                    model,
                    name=result.name,
                    algorithm=type(model).__name__,
                    params={},
                    metrics={"validation": result.metrics},
                    data_fingerprint=fingerprint,
                    training_start=fit_data["kickoff_utc"].min().to_pydatetime(),
                    training_end=fit_data["kickoff_utc"].max().to_pydatetime(),
                    root=settings.data_dir / "models",
                    session=session,
                )
                print(f"registered {registered.name} {registered.version}")
    return EXIT_OK
