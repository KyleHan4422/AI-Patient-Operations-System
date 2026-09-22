"""Dependency health probing.

The distinction this module encodes is the first appearance of the project's
central claim about where correctness lives:

    Postgres is the system of record -- losing it means we cannot be correct,
    so it is CRITICAL and its loss makes the service `unhealthy` (503). The
    same holds for the checkpointer's own pool into it: without it every
    conversation forgets itself.

    Redis is the coordination layer -- slot holds, circuit breaker state,
    idempotency reservations, the async queue. Losing it degrades throughput
    and raises the booking conflict rate, but every correctness guarantee is
    still enforced by Postgres constraints. Its loss makes the service
    `degraded` (200), never `unhealthy`.

Probes are injected rather than imported so that "the dependency is down" is
testable without actually stopping a container. That injection seam is the
ancestor of the fault-injection framework in Phase 8.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Literal

from pydantic import BaseModel

# Dependencies whose loss breaks correctness. Everything else is degradable.
CRITICAL_DEPENDENCIES: frozenset[str] = frozenset({"postgres", "checkpointer"})

HealthStatus = Literal["ok", "degraded", "unhealthy"]

# A probe answers one question -- "can I reach you?" -- and raises on failure.
Probe = Callable[[], Awaitable[None]]


class ProbeResult(BaseModel):
    ok: bool
    latency_ms: float | None = None
    error: str | None = None


class HealthReport(BaseModel):
    status: HealthStatus
    version: str
    checks: dict[str, ProbeResult]
    # Non-critical dependencies that are down. This list is the ancestor of
    # the graph state's `degraded_modes` field and of the /ops banner: a
    # degradation nobody can see is how silent production incidents start.
    degraded: list[str]

    @property
    def http_status(self) -> int:
        # `degraded` is deliberately 200: a load balancer must not evict an
        # instance that is fully correct and merely slower.
        return 503 if self.status == "unhealthy" else 200


async def run_probe(probe: Probe, timeout_s: float) -> ProbeResult:
    """Run one probe under its own timeout budget.

    A health endpoint that can hang is worse than one that fails -- probes
    queue up behind it and the orchestrator learns nothing.
    """
    started = time.perf_counter()
    try:
        await asyncio.wait_for(probe(), timeout=timeout_s)
    except TimeoutError:
        return ProbeResult(ok=False, error=f"timeout after {timeout_s}s")
    except Exception as exc:
        return ProbeResult(ok=False, error=f"{type(exc).__name__}: {exc}")
    elapsed_ms = (time.perf_counter() - started) * 1000
    return ProbeResult(ok=True, latency_ms=round(elapsed_ms, 2))


def classify(checks: dict[str, ProbeResult]) -> tuple[HealthStatus, list[str]]:
    """Pure function: probe results -> (status, degraded dependency names).

    Kept free of IO so the three-state semantics can be unit tested on their
    own, with no database, no Redis and no HTTP.

    `degraded` lists only the non-critical failures, so the name stays honest:
    a failed critical dependency is not a degradation, it is an outage.
    """
    failed = sorted(name for name, result in checks.items() if not result.ok)
    critical_failures = [n for n in failed if n in CRITICAL_DEPENDENCIES]
    degraded = [n for n in failed if n not in CRITICAL_DEPENDENCIES]

    if critical_failures:
        return "unhealthy", degraded
    if degraded:
        return "degraded", degraded
    return "ok", []


async def check_health(probes: dict[str, Probe], timeout_s: float, version: str) -> HealthReport:
    """Probe every dependency concurrently and classify the result."""
    names = list(probes)
    results = await asyncio.gather(*(run_probe(probes[n], timeout_s) for n in names))
    checks = dict(zip(names, results, strict=True))
    status, degraded = classify(checks)
    return HealthReport(status=status, version=version, checks=checks, degraded=degraded)
