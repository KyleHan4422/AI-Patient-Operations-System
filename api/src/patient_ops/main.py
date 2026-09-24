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
from langchain_core.language_models import BaseChatModel
from psycopg_pool import AsyncConnectionPool
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from patient_ops import __version__
from patient_ops.adapters.llm.client import build_chat_model
from patient_ops.adapters.llm.embeddings import build_embeddings
from patient_ops.api.routes_chat import router as chat_router
from patient_ops.config import Settings, get_settings
from patient_ops.db.session import build_engine, build_session_factory
from patient_ops.db.transcript import SqlTranscript
from patient_ops.faults import FaultInjector
from patient_ops.graph.build import build_graph
from patient_ops.graph.checkpointer import build_checkpointer, build_checkpointer_pool
from patient_ops.health import HealthReport, Probe, check_health
from patient_ops.obs.logging import RequestIdMiddleware, configure_logging, get_logger
from patient_ops.redis_layer.client import Coordinator, build_redis_client
from patient_ops.redis_layer.ratelimit import RateLimiter

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dependency clients
# ---------------------------------------------------------------------------
def build_redis(settings: Settings) -> aioredis.Redis:
    return build_redis_client(
        settings.redis_url,
        connect_timeout_s=settings.redis_connect_timeout_s,
        socket_timeout_s=settings.redis_socket_timeout_s,
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


def make_checkpointer_probe(pool: AsyncConnectionPool) -> Probe:
    # The graph's memory has its own pool (graph/checkpointer.py), so it gets
    # its own probe: the engine's pool being fine says nothing about this one.
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

    # Neither client connects here. Boot must succeed even when a dependency
    # is down, so that /health can report the truth: a process that crashes
    # on boot reports nothing at all.
    engine = build_engine(settings)
    redis_client = build_redis(settings)
    checkpointer_pool = build_checkpointer_pool(settings)
    await checkpointer_pool.open(wait=False)  # connects in the background

    app.state.engine = engine
    app.state.session_factory = build_session_factory(engine)
    app.state.redis = redis_client
    # Every coordination feature reaches Redis through this, so "Redis is not
    # there" -- real or injected with FAULT_INJECT=redis:unavailable -- means
    # the same thing everywhere: take the fallback, and say so.
    app.state.coordinator = Coordinator(
        redis_client,
        FaultInjector(settings.fault_specs),
        down_backoff_s=settings.redis_down_backoff_s,
    )
    app.state.rate_limiter = RateLimiter(
        app.state.coordinator,
        capacity=settings.rate_limit_capacity,
        refill_per_s=settings.rate_limit_refill_per_s,
    )
    # Compiled once; each turn supplies its own context (graph/context.py).
    app.state.graph = build_graph(build_checkpointer(checkpointer_pool))
    app.state.transcript = SqlTranscript(app.state.session_factory)
    app.state.chat_model = app.state.injected_chat_model or build_chat_model(settings)
    # Built once and shared: an OpenAIEmbeddings holds an HTTP client. The
    # per-turn object is the toolset around it (api/routes_chat.py), because
    # that one accumulates a single turn's evidence.
    app.state.embeddings = build_embeddings(settings)
    app.state.probes = {
        "postgres": make_postgres_probe(engine),
        "checkpointer": make_checkpointer_probe(checkpointer_pool),
        "redis": make_redis_probe(redis_client),
    }

    if app.state.chat_model is None:
        log.warning("llm_not_configured", hint="set OPENAI_API_KEY, or LLM_PROVIDER=fake")
    log.info(
        "startup_complete",
        version=__version__,
        app_env=settings.app_env,
        llm_provider=settings.llm_provider,
    )
    try:
        yield
    finally:
        await checkpointer_pool.close()
        await engine.dispose()
        await redis_client.aclose()
        log.info("shutdown_complete")


# ---------------------------------------------------------------------------
def create_app(
    settings: Settings | None = None, *, chat_model: BaseChatModel | None = None
) -> FastAPI:
    """Build the app. `chat_model` replaces the configured model -- a seam for
    tests, like the injectable health probes."""
    settings = settings or get_settings()

    app = FastAPI(
        title="AI Patient Operations System",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.injected_chat_model = chat_model

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

    app.include_router(chat_router)
    return app


app = create_app()
