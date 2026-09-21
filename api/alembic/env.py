"""Alembic environment: how migrations find the database and the models.

Migrations run synchronously. psycopg3 serves both modes, so the URL is the
same one the async app uses -- only the engine type differs.

URL resolution, in order:
  1. `sqlalchemy.url` set on the Alembic Config by the caller -- the test suite
     does this to point migrations at the throwaway test database;
  2. otherwise the app's Settings (DATABASE_URL from the repo-root .env).
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from patient_ops.config import get_settings
from patient_ops.db.models import Base

config = context.config

# Only configure logging when invoked from the CLI. When the test suite calls
# alembic programmatically it passes no ini file, and must keep its own
# logging setup.
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def database_url() -> str:
    return config.get_main_option("sqlalchemy.url") or get_settings().sqlalchemy_url


def run_migrations_offline() -> None:
    """`alembic upgrade --sql`: emit the SQL instead of running it."""
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    # NullPool: a migration run is one short-lived connection, not a service.
    engine = create_engine(database_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
