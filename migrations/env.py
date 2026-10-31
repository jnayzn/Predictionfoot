"""Alembic environment: URL from settings (DATABASE_URL), metadata from the ORM models."""

from __future__ import annotations

from alembic import context
from sqlalchemy import engine_from_config, pool

from football_ai.config.settings import Settings
from football_ai.database.models import Base

config = context.config
target_metadata = Base.metadata


def _url() -> str:
    # Explicit -x url=... (used by tests) > DATABASE_URL / .env
    return context.get_x_argument(as_dictionary=True).get("url") or Settings().database_url


def run_migrations_offline() -> None:
    context.configure(
        url=_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _url()
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            render_as_batch=connection.dialect.name == "sqlite",
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
