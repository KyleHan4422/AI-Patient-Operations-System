"""Redis is gone. Every coordination feature takes its fallback, and says so.

This file is the degradation matrix in the README, as tests:

    capability        with Redis down                        correctness
    R1 holds          offer anyway; EXCLUDE decides           unaffected -- more conflicts
    R4 dedup          both requests write; UNIQUE decides     unaffected -- one extra call
    R3 breaker        per-process breaker                     unaffected -- weaker protection
    R5 rate limit     let the request through                 unaffected

and in every row, the turn's degraded modes name the fallback.

"Redis is gone" two ways, and every test runs under both. `injected` is
FAULT_INJECT=redis:unavailable against a Redis that is up: deterministic and
instant. `unreachable` is a port nothing listens on: a real refused
connection, so the injection is not a stand-in for a failure that behaves
differently.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from patient_ops.adapters.calendar.breaker import CIRCUIT_OPEN, BreakerCalendar
from patient_ops.degradation import DegradedModes
from patient_ops.errors import ErrorCode, ToolError
from patient_ops.faults import FaultInjector
from patient_ops.redis_layer.breaker import build_breaker
from patient_ops.redis_layer.client import Coordinator, build_redis_client
from patient_ops.redis_layer.holds import HoldResult, HoldStatus, SlotHolds
from patient_ops.redis_layer.idempotency import InFlightDedup
from patient_ops.redis_layer.ratelimit import RateLimiter
from tests.factories import booked_count
from tests.fakes import PinnedIntent
from tests.test_breaker import ScriptedCalendar, _book_n
from tests.test_chat_api import parse_sse, running_app
from tests.test_redis_layer import CountingCalendar, book_once, exam_request, slot

UNREACHABLE_URL = "redis://127.0.0.1:1/0"  # port 1: nothing listens, the connect is refused


@pytest.fixture(params=["injected", "unreachable"])
async def down(request, redis_client):
    """A Coordinator for a Redis that is not there."""
    if request.param == "injected":
        # `redis_client` is up and answering: only the injection makes it "down".
        yield Coordinator(redis_client, FaultInjector.from_string("redis:unavailable"))
        return
    client = build_redis_client(UNREACHABLE_URL, connect_timeout_s=0.2, socket_timeout_s=0.2)
    try:
        yield Coordinator(client)
    finally:
        await client.aclose()


async def test_coordinator_turns_every_outage_into_degraded(down):
    with pytest.raises(ToolError) as exc:
        await down.run("ping", lambda r: r.ping())
    assert exc.value.code is ErrorCode.DEGRADED


# ---------------------------------------------------------------------------
# R1 holds
# ---------------------------------------------------------------------------
async def test_holds_degrade_to_offering_without_a_hold(down):
    degraded = DegradedModes()
    holds = SlotHolds(
        down, ttl=timedelta(seconds=120), step=timedelta(minutes=30), degraded=degraded
    )

    assert await holds.place(slot(10), "thread-a") is HoldResult.DEGRADED
    # DEGRADED, not TAKEN: the slot is offered, and the constraint decides.
    assert await holds.check(slot(10), "thread-a") is HoldStatus.DEGRADED
    assert await holds.release_all("thread-a") == 0
    assert degraded.modes == ["holds"]


# ---------------------------------------------------------------------------
# R4 in-flight dedup
# ---------------------------------------------------------------------------
async def test_dedup_degrades_to_the_unique_key(down, calendar, clinic, session_factory):
    degraded = DegradedModes()
    dedup = InFlightDedup(down, inflight_ttl_s=30, wait_s=1, degraded=degraded)
    counting = CountingCalendar(calendar, delay_s=0)
    request = exam_request(clinic)

    first = await book_once(dedup, counting, request)
    second = await book_once(dedup, counting, request)

    assert counting.book_calls == 2, "without Redis the duplicate reaches the calendar..."
    assert second.appointment_id == first.appointment_id, "...and Postgres returns the same row"
    assert first.created and not second.created
    assert await booked_count(session_factory) == 1
    assert degraded.modes == ["idempotency"]


# ---------------------------------------------------------------------------
# R3 breaker
# ---------------------------------------------------------------------------
async def test_breaker_degrades_to_a_per_process_breaker_that_still_opens(down):
    degraded = DegradedModes()
    inner = ScriptedCalendar(ToolError(ErrorCode.TRANSIENT, "timed out"))
    breaker = build_breaker(down, "calendar", failure_threshold=3, cooldown_s=30)
    calendar = BreakerCalendar(inner, breaker, degraded=degraded)

    errors = await _book_n(calendar, 4)

    assert inner.calls == 3, "weaker protection is still protection"
    assert errors[-1].cause == CIRCUIT_OPEN
    assert degraded.modes == ["breaker"]


# ---------------------------------------------------------------------------
# R5 rate limit
# ---------------------------------------------------------------------------
async def test_rate_limit_fails_open(down):
    degraded = DegradedModes()
    limiter = RateLimiter(down, capacity=1, refill_per_s=0.001)

    decisions = [await limiter.take("chat", "1.2.3.4", degraded) for _ in range(5)]

    assert all(d.allowed for d in decisions), "a cache outage must not become a front-desk outage"
    assert degraded.modes == ["rate_limit"]


# ---------------------------------------------------------------------------
# A whole chat turn
# ---------------------------------------------------------------------------
@pytest.fixture(params=["injected", "unreachable"])
def degraded_settings(request, test_settings, engine, test_redis_url):
    base = {"llm_provider": "fake", "rate_limit_enabled": True}
    if request.param == "injected":
        return test_settings.model_copy(update=base | {"fault_inject": "redis:unavailable"})
    return test_settings.model_copy(
        update=base
        | {
            "redis_url": UNREACHABLE_URL,
            "redis_connect_timeout_s": 0.2,
            "redis_socket_timeout_s": 0.2,
        }
    )


async def test_a_chat_turn_with_redis_down_is_answered_and_says_it_degraded(degraded_settings):
    async with running_app(degraded_settings, PinnedIntent()) as client:
        response = await client.post("/api/chat/turn", json={"message": "hello there"})

    assert response.status_code == 200
    events = parse_sse(response.text)
    assert events[-1].name == "done"
    assert events[-1].data["text"]
    assert events[-1].data["degraded"] == ["rate_limit"]


# ---------------------------------------------------------------------------
# A Redis that answers but refuses writes -- the real server, really out of memory
# ---------------------------------------------------------------------------
@pytest.fixture
async def out_of_memory(redis_client):
    """maxmemory set below what the server already uses, with no eviction:
    every write is refused with OOM. Restored afterwards whatever happens --
    this is server-wide configuration."""
    before = await redis_client.config_get("maxmemory", "maxmemory-policy")
    await redis_client.config_set("maxmemory-policy", "noeviction")
    await redis_client.config_set("maxmemory", 1)
    try:
        yield
    finally:
        await redis_client.config_set("maxmemory", before["maxmemory"])
        await redis_client.config_set("maxmemory-policy", before["maxmemory-policy"])


async def test_a_redis_out_of_memory_degrades_and_does_not_break_chat(
    out_of_memory, limited_settings
):
    """The audit's case: a reachable Redis whose token-bucket script is refused.
    Before the fix this was a 500 on every chat turn."""
    async with running_app(limited_settings, PinnedIntent()) as client:
        response = await client.post("/api/chat/turn", json={"message": "hello there"})

    assert response.status_code == 200
    assert parse_sse(response.text)[-1].data["degraded"] == ["rate_limit"]


# ---------------------------------------------------------------------------
# ...and the same route with Redis up, for contrast
# ---------------------------------------------------------------------------
@pytest.fixture
def limited_settings(test_settings, engine, redis_client):
    # `redis_client` is requested to empty the test database first.
    return test_settings.model_copy(
        update={"llm_provider": "fake", "rate_limit_enabled": True, "rate_limit_capacity": 2}
    )


async def test_with_redis_up_nothing_degrades_and_the_limit_is_a_real_429(limited_settings):
    async with running_app(limited_settings, PinnedIntent()) as client:
        ok = [await client.post("/api/chat/turn", json={"message": "hi"}) for _ in range(2)]
        refused = await client.post("/api/chat/turn", json={"message": "hi"})

    assert [r.status_code for r in ok] == [200, 200]
    assert parse_sse(ok[0].text)[-1].data["degraded"] == []
    assert refused.status_code == 429, "refused before the stream opens, so a real status code"
    assert int(refused.headers["Retry-After"]) >= 1
    assert refused.json()["detail"]["code"] == "transient"
