"""The graph's memory: LangGraph's Postgres checkpointer, on a pool of its own.

After every step of the graph, the checkpointer saves the whole conversation
state under its thread_id. The next turn -- even in a freshly restarted
process -- loads the latest save and carries on. That is what lets the
conversation survive a restart.

Why a second pool beside the SQLAlchemy engine's: the checkpointer requires
connections with

    autocommit=True        it runs its own statements and manages its own
                           transactions; an implicit open transaction would
                           hold its writes back
    row_factory=dict_row   it reads rows as dicts
    prepare_threshold=0    no server-side prepared statements, which break
                           behind connection poolers like PgBouncer

Same Postgres, same psycopg driver -- different connection semantics.

Its tables are LangGraph's, created and migrated by LangGraph's own setup(),
never at boot: `make migrate` runs it after Alembic. Alembic ignores them by
name (db.models.LANGGRAPH_TABLES).
"""

from __future__ import annotations

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from patient_ops.config import Settings


def build_checkpointer_pool(settings: Settings) -> AsyncConnectionPool:
    """Build the pool, unopened. The caller opens it with `open(wait=False)`.

    Not waiting keeps Phase 0's rule: the process boots with Postgres down,
    and /health reports it.
    """
    return AsyncConnectionPool(
        settings.libpq_url,
        min_size=settings.db_pool_min_size,
        max_size=settings.db_pool_max_size,
        open=False,
        name="checkpointer",
        # Fail a request after this long without a free connection. The 30s
        # default would leave a chat turn hanging while Postgres is down.
        timeout=settings.db_connect_timeout_s,
        # Test a connection before lending it out -- the psycopg counterpart
        # of SQLAlchemy's pool_pre_ping, so a Postgres restart costs a
        # reconnect rather than a failed turn.
        check=AsyncConnectionPool.check_connection,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
            "connect_timeout": max(1, int(settings.db_connect_timeout_s)),
        },
    )


def build_checkpointer(pool: AsyncConnectionPool) -> AsyncPostgresSaver:
    return AsyncPostgresSaver(pool)  # type: ignore[arg-type]


def setup_checkpointer(libpq_url: str) -> None:
    """Create or upgrade the checkpoint tables. Idempotent.

    LangGraph versions its own schema (the checkpoint_migrations table), so
    running this on every `make migrate` brings the tables up to whatever
    version of LangGraph is installed.
    """
    with PostgresSaver.from_conn_string(libpq_url) as saver:
        saver.setup()
