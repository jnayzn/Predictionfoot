"""Phase 9 commands: scientific backtest, live prediction evaluation."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from football_ai.bootstrap import build_session_factory
from football_ai.cli.common import EXIT_ERROR, EXIT_OK, parse_seasons
from football_ai.cli.model_commands import load_features
from football_ai.config.settings import Settings
from football_ai.core.io import atomic_write_json
from football_ai.database.connection import session_scope
from football_ai.database.repositories import PredictionRepository
from football_ai.training.backtest_report import run_backtest
from football_ai.training.benchmark import load_market_benchmark
from football_ai.training.datasets import split_by_seasons
from football_ai.training.train import ALL_FACTORIES


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("backtest", help="Full scientific backtest on the locked test seasons")
    p.add_argument("--features", type=Path, default=None)
    p.add_argument("--competition", action="append", dest="competitions")
    p.add_argument("--validation-seasons", type=parse_seasons, required=True)
    p.add_argument("--test-seasons", type=parse_seasons, required=True)
    p.add_argument(
        "--models", default="frequency,elo,poisson,dixon_coles,xgboost,lightgbm,lightgbm_goals"
    )
    p.add_argument("--no-ensemble", action="store_true")
    p.add_argument("--no-market", action="store_true", help="skip the closing-odds benchmark")
    p.add_argument("--refit-days", type=int, default=30)
    p.set_defaults(handler=cmd_backtest)

    p = sub.add_parser("evaluate-predictions", help="Score stored predictions of finished matches")
    p.set_defaults(handler=cmd_evaluate_predictions)


def cmd_backtest(args: argparse.Namespace, settings: Settings) -> int:
    names = [n.strip() for n in args.models.split(",") if n.strip()]
    unknown = set(names) - set(ALL_FACTORIES)
    if unknown:
        print(f"Unknown models: {sorted(unknown)}")
        return EXIT_ERROR
    try:
        frame = load_features(settings, args.features)
    except FileNotFoundError as exc:
        print(exc)
        return EXIT_ERROR
    if args.competitions:
        frame = frame[frame["competition_key"].isin(args.competitions)]
    market = None
    if not args.no_market:
        with session_scope(build_session_factory(settings)) as session:
            market = load_market_benchmark(session)
    split = split_by_seasons(frame, args.validation_seasons, args.test_seasons)
    report = run_backtest(
        frame, split, {n: ALL_FACTORIES[n] for n in names}, market=market,
        with_ensemble=not args.no_ensemble, refit_every_days=args.refit_days,
    )  # fmt: skip
    stamp = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    directory = settings.data_dir / "reports"
    atomic_write_json(directory / f"backtest_{stamp}.json", report.to_dict())
    markdown = report.to_markdown()
    (directory / f"backtest_{stamp}.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"saved to {directory / f'backtest_{stamp}.md'}")
    return EXIT_OK


def cmd_evaluate_predictions(args: argparse.Namespace, settings: Settings) -> int:
    with session_scope(build_session_factory(settings)) as session:
        repo = PredictionRepository(session)
        created = repo.evaluate_pending()
        summary = repo.performance()
    print(f"new evaluations: {created}")
    print(
        f"track record (stored pre-match predictions only): n={summary.n} "
        f"log_loss={summary.log_loss} brier={summary.brier} accuracy={summary.accuracy} "
        f"mae_total_goals={summary.mae_total_goals}"
    )
    return EXIT_OK
