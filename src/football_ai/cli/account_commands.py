"""Phase 14 commands: commercial API clients (keys, plans, usage)."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta

from football_ai.bootstrap import build_session_factory
from football_ai.cli.common import EXIT_ERROR, EXIT_OK
from football_ai.config.settings import Settings
from football_ai.database.connection import session_scope
from football_ai.database.repositories import ApiClientRepository
from football_ai.database.repositories.api_clients import load_plans


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("api-keys", help="Manage API clients (create / list / revoke)")
    actions = p.add_subparsers(dest="action", required=True)
    create = actions.add_parser("create", help="Create a client - the key is shown ONCE")
    create.add_argument("--name", required=True)
    create.add_argument("--plan", default="free")
    actions.add_parser("list", help="List clients (never shows keys)")
    revoke = actions.add_parser("revoke", help="Deactivate a client by key prefix")
    revoke.add_argument("--prefix", required=True)
    p.set_defaults(handler=cmd_api_keys)

    p = sub.add_parser("api-usage", help="Requests per client and day (billing)")
    p.add_argument("--days", type=int, default=30)
    p.set_defaults(handler=cmd_api_usage)


def cmd_api_keys(args: argparse.Namespace, settings: Settings) -> int:
    factory = build_session_factory(settings)
    with session_scope(factory) as session:
        repo = ApiClientRepository(session)
        if args.action == "create":
            plans = load_plans(settings.plans_file)
            if args.plan not in plans:
                print(f"unknown plan {args.plan!r}; available: {sorted(plans)}")
                return EXIT_ERROR
            row, key = repo.create(args.name, plans[args.plan])
            print(
                f"client #{row.id} {row.name} plan={row.plan} "
                f"({row.daily_quota}/day, {row.rate_limit_per_minute}/min)"
            )
            print(f"API key (shown once, store it securely): {key}")
        elif args.action == "list":
            for c in repo.list():
                last = c.last_used_at.isoformat(timespec="minutes") if c.last_used_at else "never"
                print(
                    f"#{c.id:<4} {c.key_prefix:<13} {c.name:<24} {c.plan:<9} "
                    f"{'active' if c.active else 'revoked':<8} last_used={last}"
                )
        else:
            n = repo.revoke(args.prefix)
            print(f"revoked {n} client(s)")
            if n == 0:
                return EXIT_ERROR
    return EXIT_OK


def cmd_api_usage(args: argparse.Namespace, settings: Settings) -> int:
    since = datetime.now(UTC).date() - timedelta(days=args.days)
    with session_scope(build_session_factory(settings)) as session:
        rows = ApiClientRepository(session).usage(since=since)
    totals: dict[str, int] = {}
    for name, day, n in rows:
        print(f"{day.isoformat()} {name:<24} {n}")
        totals[name] = totals.get(name, 0) + n
    for name, total in sorted(totals.items()):
        print(f"TOTAL {name:<24} {total}")
    return EXIT_OK
