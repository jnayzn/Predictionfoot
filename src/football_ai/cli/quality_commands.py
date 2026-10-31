"""Phase 3 commands: canonical layer build and data-quality report."""

from __future__ import annotations

import argparse
import json

from football_ai.bootstrap import build_session_factory
from football_ai.cli.common import EXIT_ERROR, EXIT_OK
from football_ai.config.aliases import load_aliases
from football_ai.config.competitions import load_competitions
from football_ai.config.settings import Settings
from football_ai.core.exceptions import ConfigurationError
from football_ai.data.cleaning.canonical import CanonicalBuilder
from football_ai.data.quality.checks import load_frames, run_checks
from football_ai.data.quality.report import format_report, save_report
from football_ai.database.connection import session_scope


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("build-matches", help="Link sources and build canonical teams/matches")
    p.add_argument("--competition", action="append", dest="competitions", help="(repeatable)")
    p.add_argument("--min-confidence", type=float, default=0.6)
    p.set_defaults(handler=cmd_build_matches)

    p = sub.add_parser("quality-report", help="Dataset-level data-quality checks")
    p.add_argument("--competition", default=None)
    p.add_argument("--json", action="store_true")
    p.add_argument("--fail-on-error", action="store_true", help="exit 1 if errors are found")
    p.set_defaults(handler=cmd_quality_report)


def cmd_build_matches(args: argparse.Namespace, settings: Settings) -> int:
    keys_all = [c.key for c in load_competitions(settings.competitions_file)]
    keys = args.competitions or keys_all
    unknown = set(keys) - set(keys_all)
    if unknown:
        raise ConfigurationError(f"Unknown competition keys: {sorted(unknown)}")
    aliases = load_aliases(settings.team_aliases_file)
    factory = build_session_factory(settings)
    for key in keys:
        with session_scope(factory) as session:
            report = CanonicalBuilder(
                session, aliases=aliases, min_confidence=args.min_confidence
            ).build(key)
        print(report.summary())
        for item in report.low_confidence:
            print(f"  LOW CONFIDENCE (not linked): {item}")
    return EXIT_OK


def cmd_quality_report(args: argparse.Namespace, settings: Settings) -> int:
    factory = build_session_factory(settings)
    with session_scope(factory) as session:
        report = run_checks(load_frames(session, args.competition), scope=args.competition or "all")
        path = save_report(report, session, settings.data_dir / "reports")
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False, default=str))
    else:
        print(format_report(report))
        print(f"(saved to {path})")
    return EXIT_ERROR if args.fail_on_error and report.errors else EXIT_OK
