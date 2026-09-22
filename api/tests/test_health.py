"""The three-state health semantics, as executable acceptance criteria.

The point of this file is that "the dependency is down" is tested WITHOUT
stopping a container. Probes are injected at a seam, so a failure is a fake
coroutine that raises -- deterministic, millisecond-fast, and safe to run in
parallel with a real stack on the same machine.

That seam is the ancestor of the fault-injection framework in Phase 8: the same
idea applied to calendar timeouts, Redis outages and expired holds.
"""

from __future__ import annotations

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from patient_ops.config import Settings
from patient_ops.health import (
    CRITICAL_DEPENDENCIES,
    ProbeResult,
    check_health,
    classify,
    run_probe,
)
from patient_ops.main import create_app

# ---------------------------------------------------------------------------
# Fake probes
# ---------------------------------------------------------------------------


async def probe_ok() -> None:
    return None


async def probe_fails() -> None:
    raise ConnectionError("Connection refused")


async def probe_hangs() -> None:
    await asyncio.sleep(60)


# ---------------------------------------------------------------------------
# classify() -- a pure function, testable with no IO at all
# ---------------------------------------------------------------------------


def test_classify_all_ok():
    status, degraded = classify({"postgres": ProbeResult(ok=True), "redis": ProbeResult(ok=True)})
    assert status == "ok"
    assert degraded == []


def test_classify_non_critical_failure_is_degraded():
    """Redis is the coordination layer: losing it degrades, never breaks."""
    status, degraded = classify({"postgres": ProbeResult(ok=True), "redis": ProbeResult(ok=False)})
    assert status == "degraded"
    assert degraded == ["redis"]


def test_classify_critical_failure_is_unhealthy():
    status, degraded = classify({"postgres": ProbeResult(ok=False), "redis": ProbeResult(ok=True)})
    assert status == "unhealthy"
    # A failed critical dependency is an outage, not a degradation. Keeping it
    # out of `degraded` keeps the field's name honest for its three consumers:
    # graph state, structured logs, and the /ops banner.
    assert degraded == []


def test_classify_critical_failure_dominates():
    status, degraded = classify({"postgres": ProbeResult(ok=False), "redis": ProbeResult(ok=False)})
    assert status == "unhealthy"
    assert degraded == ["redis"]


def test_postgres_is_critical_and_redis_is_not():
    """Guards the project's central infrastructure claim against silent drift."""
    assert "postgres" in CRITICAL_DEPENDENCIES
    assert "redis" not in CRITICAL_DEPENDENCIES


def test_classify_checkpointer_failure_is_unhealthy():
    """Without the checkpointer's pool every conversation forgets itself."""
    status, degraded = classify(
        {
            "postgres": ProbeResult(ok=True),
            "checkpointer": ProbeResult(ok=False),
            "redis": ProbeResult(ok=True),
        }
    )
    assert status == "unhealthy"
    assert degraded == []


# ---------------------------------------------------------------------------
# run_probe() -- timeout budget
# ---------------------------------------------------------------------------


async def test_run_probe_records_latency():
    result = await run_probe(probe_ok, timeout_s=1.0)
    assert result.ok
    assert result.latency_ms is not None
    assert result.error is None


async def test_run_probe_captures_error_type():
    result = await run_probe(probe_fails, timeout_s=1.0)
    assert not result.ok
    assert "ConnectionError" in (result.error or "")


async def test_run_probe_enforces_its_own_timeout():
    """A health endpoint that can hang is worse than one that fails."""
    started = asyncio.get_running_loop().time()
    result = await run_probe(probe_hangs, timeout_s=0.05)
    elapsed = asyncio.get_running_loop().time() - started

    assert not result.ok
    assert "timeout" in (result.error or "")
    assert elapsed < 1.0, "probe must be bounded by its budget, not by the dependency"


async def test_check_health_runs_probes_concurrently():
    """Worst-case latency is max(probes), not sum(probes)."""
    started = asyncio.get_running_loop().time()
    report = await check_health(
        probes={"postgres": probe_hangs, "redis": probe_hangs},
        timeout_s=0.1,
        version="test",
    )
    elapsed = asyncio.get_running_loop().time() - started

    assert report.status == "unhealthy"
    assert elapsed < 0.2, "two 0.1s probes run concurrently must not take 0.2s"


# ---------------------------------------------------------------------------
# /health over HTTP -- status code contract
# ---------------------------------------------------------------------------


@pytest.fixture
def settings() -> Settings:
    return Settings(app_env="test", health_probe_timeout_s=0.2)


async def call_health(settings: Settings, probes: dict) -> tuple[int, dict]:
    """Drive the real app over ASGI with probes swapped out.

    No uvicorn, no port binding, no container teardown.
    """
    app = create_app(settings)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        app.state.probes = probes  # replaces what lifespan installed
        response = await client.get("/health")
    return response.status_code, response.json()


async def test_health_endpoint_ok(settings: Settings):
    code, body = await call_health(settings, {"postgres": probe_ok, "redis": probe_ok})
    assert code == 200
    assert body["status"] == "ok"
    assert body["degraded"] == []


async def test_health_endpoint_redis_down_is_200_degraded(settings: Settings):
    """The acceptance criterion for Phase 0.

    Returning 503 here would let a load balancer evict an instance that is
    fully correct and merely running on its fallback coordination paths.
    """
    code, body = await call_health(settings, {"postgres": probe_ok, "redis": probe_fails})
    assert code == 200, "Redis is degradable -- it must not take the service out of rotation"
    assert body["status"] == "degraded"
    assert body["degraded"] == ["redis"]
    assert body["checks"]["redis"]["ok"] is False
    assert body["checks"]["postgres"]["ok"] is True


async def test_health_endpoint_postgres_down_is_503_unhealthy(settings: Settings):
    code, body = await call_health(settings, {"postgres": probe_fails, "redis": probe_ok})
    assert code == 503
    assert body["status"] == "unhealthy"


async def test_health_endpoint_reports_error_detail(settings: Settings):
    """Diagnostics live in the body, which is why a 503 must still be parsed."""
    _, body = await call_health(settings, {"postgres": probe_ok, "redis": probe_fails})
    assert "Connection refused" in body["checks"]["redis"]["error"]


async def test_health_endpoint_echoes_request_id(settings: Settings):
    app = create_app(settings)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        app.state.probes = {"postgres": probe_ok, "redis": probe_ok}
        response = await client.get("/health", headers={"X-Request-ID": "trace-me-123"})
    assert response.headers["x-request-id"] == "trace-me-123"
