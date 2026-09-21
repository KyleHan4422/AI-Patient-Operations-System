"""FastAPI application: the HTTP boundary.

This layer translates, validates and dispatches. It holds no business logic --
Phase 13 adds a voice channel that is not HTTP at all, and it must reuse the
same graph rather than re-implementing it.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import redis.asyncio as aioredis
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from psycopg_pool import AsyncConnectionPool

from patient_ops import __version__
from patient_ops.config import Settings, get_settings
from patient_ops.health import HealthReport, Probe, check_health
from patient_ops.obs.logging import RequestIdMiddleware, configure_logging, get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dependency clients
# ---------------------------------------------------------------------------
def build_pg_pool(settings: Settings) -> AsyncConnectionPool:
    return AsyncConnectionPool(
        conninfo=settings.database_url,
        min_size=settings.db_pool_min_size,
        max_size=settings.db_pool_max_size,
        timeout=settings.db_connect_timeout_s,
        # Do not connect in the constructor: opening is an awaitable step we
        # want to control, and we must not block import.
        open=False,
    )


def build_redis(settings: Settings) -> aioredis.Redis:
    return aioredis.from_url(
        settings.redis_url,
        socket_connect_timeout=settings.redis_connect_timeout_s,
        socket_timeout=settings.redis_socket_timeout_s,
        decode_responses=True,
    )


# ---------------------------------------------------------------------------
# Probes -- one per dependency, raising on failure.
# ---------------------------------------------------------------------------
def make_postgres_probe(pool: AsyncConnectionPool) -> Probe:
    async def probe() -> None:
        async with pool.connection() as conn:
            await conn.execute("SELECT 1")

    return probe


def make_redis_probe(client: aioredis.Redis) -> Probe:
    async def probe() -> None:
        await client.ping()

    return probe


# ---------------------------------------------------------------------------
# Lifespan: open shared clients on boot, close them on shutdown.
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    configure_logging(settings)

    pool = build_pg_pool(settings)
    redis_client = build_redis(settings)

    # wait=False: boot must succeed even when a dependency is down, so that
    # /health can report the truth. A process that crashes on boot reports
    # nothing at all.
    await pool.open(wait=False)

    app.state.pg_pool = pool
    app.state.redis = redis_client
    app.state.probes = {
        "postgres": make_postgres_probe(pool),
        "redis": make_redis_probe(redis_client),
    }

    log.info("startup_complete", version=__version__, app_env=settings.app_env)
    try:
        yield
    finally:
        await pool.close()
        await redis_client.aclose()
        log.info("shutdown_complete")


# ---------------------------------------------------------------------------
def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title="AI Patient Operations System",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = settings

    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthReport)
    async def health(request: Request) -> JSONResponse:
        report = await check_health(
            probes=request.app.state.probes,
            timeout_s=settings.health_probe_timeout_s,
            version=__version__,
        )
        if report.degraded:
            log.warning("health_degraded", status=report.status, degraded=report.degraded)
        return JSONResponse(content=report.model_dump(), status_code=report.http_status)

    return app


app = create_app()
