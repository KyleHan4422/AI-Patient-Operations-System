"""Shared fixtures.

Two kinds of test live in this suite:

  pure tests  No database, no network. `make test-unit` runs only these, and
              they pass with every container stopped.
  db tests    Run against a real Postgres, because the guarantees they check --
              the EXCLUDE constraint, the UNIQUE idempotency key -- exist only
              in Postgres. A mock would only test the mock. Any test that uses
              a database fixture is marked `db` automatically.

The test database is dropped and rebuilt at the start of every session by
running the real migrations, so what gets tested is the schema the migrations
produce -- not the one the ORM models describe.

Why not the usual "wrap each test in a transaction and roll it back" fixture:
the concurrency tests need several connections that see each other's commits.
Inside one uncommitted transaction there is no race to test. So tests commit
for real, and every table is truncated afterwards.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from hypothesis import settings as hypothesis_settings
from sqlalchemy import create_engine, make_url, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool

from patient_ops.adapters.calendar.fake import FakeCalendar
from patient_ops.config import Settings, get_settings
from patient_ops.db.models import Base
from patient_ops.db.session import build_engine, build_session_factory
from patient_ops.domain.availability import SchedulingPolicy
from tests.factories import FIXED_NOW, TZ, Clinic, build_minimal_clinic

API_DIR = Path(__file__).resolve().parents[1]
TEST_DB_NAME = "patient_ops_test"
DB_FIXTURES = {"test_database_url", "engine", "session_factory", "clinic", "calendar"}

# Property tests: reproducible in CI (same examples every run), exploratory
# locally (new examples each run; failures are replayed from .hypothesis/).
hypothesis_settings.register_profile("ci", derandomize=True, max_examples=200, deadline=None)
hypothesis_settings.register_profile("dev", max_examples=100, deadline=None)
hypothesis_settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if DB_FIXTURES & set(getattr(item, "fixturenames", ())):
            item.add_marker(pytest.mark.db)


def alembic_config(url: str) -> Config:
    """An Alembic config aimed at `url`, without reading alembic.ini.

    No ini file means env.py leaves pytest's logging alone.
    """
    cfg = Config()
    cfg.set_main_option("script_location", str(API_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))  # ini-style escaping
    return cfg


# ---------------------------------------------------------------------------
# Session scope: one fresh, migrated database per test run.
# Synchronous on purpose -- a session-scoped async fixture would need its own
# event loop under pytest-asyncio 1.x, and this runs once anyway.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def test_database_url() -> str:
    base = make_url(get_settings().sqlalchemy_url)
    admin = create_engine(
        base.set(database="postgres"), isolation_level="AUTOCOMMIT", poolclass=NullPool
    )
    try:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{TEST_DB_NAME}" WITH (FORCE)'))
            conn.execute(text(f'CREATE DATABASE "{TEST_DB_NAME}"'))
    except OperationalError as exc:
        # Fail, never skip: a constraint test that silently skips is a
        # constraint nobody is checking.
        pytest.fail(
            f"Postgres is not reachable at {base.host}:{base.port} -- run `make up`.\n{exc.orig}",
            pytrace=False,
        )
    finally:
        admin.dispose()

    url = base.set(database=TEST_DB_NAME).render_as_string(hide_password=False)
    command.upgrade(alembic_config(url), "head")
    return url


# ---------------------------------------------------------------------------
# Function scope: a private engine per test, every table emptied afterwards.
# ---------------------------------------------------------------------------
@pytest.fixture
def test_settings(test_database_url: str) -> Settings:
    # A pool wide enough for the concurrency tests to hold 10+ connections at once.
    return Settings(
        app_env="test", database_url=test_database_url, db_pool_min_size=1, db_pool_max_size=15
    )


async def truncate_all(engine: AsyncEngine) -> None:
    # The guard that makes this fixture safe to have: it cannot empty a
    # development database, whatever DATABASE_URL says.
    database = engine.url.database or ""
    if not database.endswith("_test"):
        raise RuntimeError(f"refusing to TRUNCATE {database!r}: not a *_test database")
    tables = ", ".join(t.name for t in Base.metadata.sorted_tables)
    async with engine.begin() as conn:
        await conn.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))


@pytest.fixture
async def engine(test_settings: Settings) -> AsyncIterator[AsyncEngine]:
    engine = build_engine(test_settings)
    try:
        yield engine
    finally:
        await truncate_all(engine)
        await engine.dispose()


@pytest.fixture
def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return build_session_factory(engine)


@pytest.fixture
async def clinic(session_factory: async_sessionmaker[AsyncSession]) -> Clinic:
    return await build_minimal_clinic(session_factory)


@pytest.fixture
def calendar(session_factory: async_sessionmaker[AsyncSession]) -> FakeCalendar:
    return FakeCalendar(session_factory, tz=TZ, policy=SchedulingPolicy(), clock=lambda: FIXED_NOW)
