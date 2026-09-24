"""Shared fixtures.

Two kinds of test live in this suite:

  pure tests  No database, no network. `make test-unit` runs only these, and
              they pass with every container stopped.
  db tests    Run against a real Postgres, because the guarantees they check --
              the EXCLUDE constraint, the UNIQUE idempotency key -- exist only
              in Postgres. A mock would only test the mock. Any test that uses
              a database fixture is marked `db` automatically.
  redis tests Run against a real Redis, for the same reason: what they check
              is that a Lua script is atomic and a key expires, which only the
              server can say. Database 15, emptied before every test, so the
              development data in database 0 is never touched. Marked `redis`
              automatically.

The test database is dropped and rebuilt at the start of every session by
running the real migrations (and LangGraph's checkpointer setup, exactly as
`make migrate` does), so what gets tested is the schema the migrations
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
from urllib.parse import urlsplit, urlunsplit

import pytest
import redis as sync_redis
from alembic import command
from alembic.config import Config
from hypothesis import settings as hypothesis_settings
from sqlalchemy import create_engine, make_url, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool

from patient_ops.adapters.calendar.fake import FakeCalendar
from patient_ops.config import Settings, get_settings
from patient_ops.db.models import LANGGRAPH_TABLES, Base
from patient_ops.db.session import build_engine, build_session_factory
from patient_ops.domain.availability import SchedulingPolicy
from patient_ops.graph.checkpointer import setup_checkpointer
from patient_ops.redis_layer.client import Coordinator, build_redis_client
from tests.factories import FIXED_NOW, TZ, Clinic, build_minimal_clinic

API_DIR = Path(__file__).resolve().parents[1]
TEST_DB_NAME = "patient_ops_test"
# Emptied between tests. Not checkpoint_migrations: it records which version
# of LangGraph's schema is installed, not test data.
LANGGRAPH_DATA_TABLES = sorted(LANGGRAPH_TABLES - {"checkpoint_migrations"})
DB_FIXTURES = {"test_database_url", "engine", "session_factory", "clinic", "calendar"}
REDIS_FIXTURES = {"test_redis_url", "redis_client", "coordinator"}
TEST_REDIS_DB = 15
# FLUSHDB is only ever sent to Redis on this machine or CI's service container.
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def redis_test_url(url: str) -> str:
    """REDIS_URL with its database swapped for the test one, everything else kept.

    Refuses a Redis that is not local -- the same promise truncate_all makes
    for Postgres: whatever REDIS_URL says, the suite cannot empty a shared
    server.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("redis", "rediss"):
        raise ValueError(f"REDIS_URL must be redis:// or rediss://, got {url!r}")
    if parts.hostname not in LOCAL_HOSTS:
        raise ValueError(f"refusing to run tests against non-local Redis {parts.hostname!r}")
    return urlunsplit(parts._replace(path=f"/{TEST_REDIS_DB}"))


# Property tests: reproducible in CI (same examples every run), exploratory
# locally (new examples each run; failures are replayed from .hypothesis/).
hypothesis_settings.register_profile("ci", derandomize=True, max_examples=200, deadline=None)
hypothesis_settings.register_profile("dev", max_examples=100, deadline=None)
hypothesis_settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        fixtures = set(getattr(item, "fixturenames", ()))
        if DB_FIXTURES & fixtures:
            item.add_marker(pytest.mark.db)
        if REDIS_FIXTURES & fixtures:
            item.add_marker(pytest.mark.redis)


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

    test_url = base.set(database=TEST_DB_NAME)
    url = test_url.render_as_string(hide_password=False)
    command.upgrade(alembic_config(url), "head")
    setup_checkpointer(test_url.set(drivername="postgresql").render_as_string(hide_password=False))
    return url


# ---------------------------------------------------------------------------
# Function scope: a private engine per test, every table emptied afterwards.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def test_redis_url() -> str:
    try:
        url = redis_test_url(get_settings().redis_url)
    except ValueError as exc:
        pytest.fail(str(exc), pytrace=False)
    client = sync_redis.Redis.from_url(url, socket_connect_timeout=2)
    try:
        client.ping()
    except sync_redis.ConnectionError as exc:
        # Fail, never skip -- the same rule as Postgres above.
        pytest.fail(f"Redis is not reachable at {url} -- run `make up`.\n{exc}", pytrace=False)
    finally:
        client.close()
    return url


@pytest.fixture
def test_settings(test_database_url: str) -> Settings:
    # A pool wide enough for the concurrency tests to hold 10+ connections at once.
    # Rate limiting is off: it is per client, every test client is the same
    # address, and a chat test must not fail for having run after others.
    # Tests of the limiter turn it back on, against the test Redis.
    return Settings(
        app_env="test",
        database_url=test_database_url,
        db_pool_min_size=1,
        db_pool_max_size=15,
        # Computed, not the `test_redis_url` fixture: that one pings Redis,
        # and a database test must not start needing Redis to run.
        redis_url=redis_test_url(get_settings().redis_url),
        rate_limit_enabled=False,
    )


@pytest.fixture
async def redis_client(test_redis_url: str) -> AsyncIterator:
    client = build_redis_client(test_redis_url, connect_timeout_s=2, socket_timeout_s=2)
    await client.flushdb()
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
def coordinator(redis_client) -> Coordinator:
    return Coordinator(redis_client)


async def truncate_all(engine: AsyncEngine) -> None:
    # The guard that makes this fixture safe to have: it cannot empty a
    # development database, whatever DATABASE_URL says.
    database = engine.url.database or ""
    if not database.endswith("_test"):
        raise RuntimeError(f"refusing to TRUNCATE {database!r}: not a *_test database")
    tables = ", ".join([*(t.name for t in Base.metadata.sorted_tables), *LANGGRAPH_DATA_TABLES])
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
