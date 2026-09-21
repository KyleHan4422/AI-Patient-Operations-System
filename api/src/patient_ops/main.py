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
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from patient_ops import __version__
from patient_ops.config import Settings, get_settings
from patient_ops.db.session import build_engine, build_session_factory
from patient_ops.health import HealthReport, Probe, check_health
from patient_ops.obs.logging import RequestIdMiddleware, configure_logging, get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dependency clients
# ---------------------------------------------------------------------------
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
def make_postgres_probe(engine: AsyncEngine) -> Probe:
    # Probe through the same pool that serves real traffic. A separate probe
    # connection can be green while the application pool is exhausted.
    async def probe() -> None:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

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

    # Neither client connects here. Boot must succeed even when a dependency
    # is down, so that /health can report the truth: a process that crashes
    # on boot reports nothing at all.
    engine = build_engine(settings)
    redis_client = build_redis(settings)

    app.state.engine = engine
    app.state.session_factory = build_session_factory(engine)
    app.state.redis = redis_client
    app.state.probes = {
        "postgres": make_postgres_probe(engine),
        "redis": make_redis_probe(redis_client),
    }

    log.info("startup_complete", version=__version__, app_env=settings.app_env)
    try:
        yield
    finally:
        await engine.dispose()
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
