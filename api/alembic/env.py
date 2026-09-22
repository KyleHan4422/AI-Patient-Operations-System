"""Alembic environment: how migrations find the database and the models.

Migrations run synchronously. psycopg3 serves both modes, so the URL is the
same one the async app uses -- only the engine type differs.

URL resolution, in order:
  1. `sqlalchemy.url` set on the Alembic Config by the caller -- the test suite
     does this to point migrations at the throwaway test database;
  2. otherwise the app's Settings (DATABASE_URL from the repo-root .env).

The database also holds LangGraph's checkpoint tables, which LangGraph creates
and versions itself. Alembic neither owns nor compares them -- see
include_name below.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from patient_ops.config import get_settings
from patient_ops.db.models import LANGGRAPH_TABLES, Base

config = context.config

# Only configure logging when invoked from the CLI. When the test suite calls
# alembic programmatically it passes no ini file, and must keep its own
# logging setup.
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def database_url() -> str:
    return config.get_main_option("sqlalchemy.url") or get_settings().sqlalchemy_url


def include_name(name: str | None, type_: str, parent_names: object) -> bool:
    # Skip exactly LangGraph's tables. Not "every table the models don't
    # know": that would also hide a table a migration forgot to drop.
    return not (type_ == "table" and name in LANGGRAPH_TABLES)


def run_migrations_offline() -> None:
    """`alembic upgrade --sql`: emit the SQL instead of running it."""
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        include_name=include_name,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    # NullPool: a migration run is one short-lived connection, not a service.
    engine = create_engine(database_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata, include_name=include_name
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
