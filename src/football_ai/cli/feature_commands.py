"""Phase 4 commands: build / inspect the feature set."""

from __future__ import annotations

import argparse

from football_ai.bootstrap import build_session_factory
from football_ai.cli.common import EXIT_ERROR, EXIT_OK
from football_ai.config.settings import Settings
from football_ai.database.connection import session_scope
from football_ai.features.builder import FeatureBuilder
from football_ai.features.dataset import load_base_frames
from football_ai.features.leakage import check_leakage
from football_ai.features.store import describe, save_feature_set, to_json


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("build-features", help="Compute the leakage-free feature set")
    p.add_argument("--competition", action="append", dest="competitions", help="(repeatable)")
    p.add_argument(
        "--check-leakage",
        type=int,
        default=0,
        metavar="N",
        help="recompute N random matches from truncated data and compare",
    )
    p.set_defaults(handler=cmd_build_features)


def cmd_build_features(args: argparse.Namespace, settings: Settings) -> int:
    with session_scope(build_session_factory(settings)) as session:
        frames = load_base_frames(session)
    builder = FeatureBuilder()
    features = builder.build(frames, competition_keys=args.competitions)
    if features.empty:
        print("No match in the canonical layer - run build-matches first.")
        return EXIT_ERROR
    path = save_feature_set(features, builder.config, settings.data_dir / "features")
    print(to_json(describe(features)))
    print(f"saved to {path}")
    if args.check_leakage:
        result = check_leakage(frames, builder, sample=args.check_leakage, full=features)
        if result.ok:
            print(f"Leakage check OK on {result.checked} matches.")
        else:
            print(f"LEAKAGE DETECTED on {result.checked} matches: {result.leaking_features}")
            return EXIT_ERROR
    return EXIT_OK
