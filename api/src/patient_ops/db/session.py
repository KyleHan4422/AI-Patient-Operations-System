"""Engine and session factory: how the process talks to Postgres.

One engine per process -- built in the FastAPI lifespan, or by a script --
holding the connection pool. A session is a short-lived unit of work borrowed
from it: open, do one thing, commit or roll back, give the connection back.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from patient_ops.config import Settings


def build_engine(settings: Settings) -> AsyncEngine:
    """Create the engine. Does not connect: the first query does.

    That laziness keeps Phase 0's rule intact -- the process boots even when
    Postgres is down, and /health reports it instead of the process crashing.
    """
    return create_async_engine(
        settings.sqlalchemy_url,
        pool_size=settings.db_pool_min_size,
        max_overflow=settings.db_pool_max_size - settings.db_pool_min_size,
        # How long a request waits for a free pooled connection before failing.
        pool_timeout=settings.db_connect_timeout_s,
        # Test each connection with a cheap round trip before handing it out,
        # so a Postgres restart costs one reconnect instead of one failed request.
        pool_pre_ping=True,
        # libpq's own TCP connect timeout, in whole seconds.
        connect_args={"connect_timeout": max(1, int(settings.db_connect_timeout_s))},
    )


def build_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    # expire_on_commit=False: by default SQLAlchemy forgets every loaded value
    # at commit and reloads it on next access. Under asyncio that reload is
    # hidden IO outside an await and fails (MissingGreenlet), so objects keep
    # the values they had when the transaction committed.
    return async_sessionmaker(engine, expire_on_commit=False)
