"""R3 -- the circuit breaker: one state machine, two implementations, one table.

breaker_step() is the specification and lua/breaker.lua repeats it in another
language. The only way to keep two copies of a rule honest is to hold them to
the same examples, so TIMELINE below is run against the in-process breaker and
the Redis one, step by step, with the same clock.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from patient_ops.adapters.calendar.base import Booking, BookingRequest, CalendarProvider
from patient_ops.adapters.calendar.breaker import CIRCUIT_OPEN, BreakerCalendar
from patient_ops.errors import ErrorCode, ToolError
from patient_ops.faults import FaultInjector
from patient_ops.redis_layer.breaker import (
    BreakerEvent,
    BreakerPolicy,
    BreakerState,
    LocalBreaker,
    RedisBreaker,
    build_breaker,
)
from patient_ops.redis_layer.client import Coordinator

POLICY = BreakerPolicy(failure_threshold=3, cooldown_ms=1_000)
A, S, F, X = BreakerEvent.ALLOW, BreakerEvent.SUCCESS, BreakerEvent.FAILURE, BreakerEvent.ABANDON
CLOSED, OPEN, HALF = BreakerState.CLOSED, BreakerState.OPEN, BreakerState.HALF_OPEN

# (advance the clock by ms, event, state afterwards, allowed -- checked for ALLOW only)
TIMELINE: list[tuple[int, BreakerEvent, BreakerState, bool | None]] = [
    (0, A, CLOSED, True),
    (0, X, CLOSED, None),  # a cancelled call says nothing about the backend
    (0, F, CLOSED, None),
    (0, F, CLOSED, None),
    (0, S, CLOSED, None),  # a success resets the count: failures must be consecutive
    (0, F, CLOSED, None),
    (0, F, CLOSED, None),
    (0, F, OPEN, None),  # third in a row: open
    (0, A, OPEN, False),  # fail fast
    (0, S, OPEN, None),  # a late success from before it opened changes nothing
    (999, A, OPEN, False),  # cooldown not over
    (1, A, HALF, True),  # cooldown over: this caller is the probe
    (0, A, HALF, False),  # ...and the only one
    (0, F, OPEN, None),  # probe failed: open again, cooldown restarts
    (500, A, OPEN, False),
    (500, A, HALF, True),
    (0, S, CLOSED, None),  # probe succeeded: closed
    (0, A, CLOSED, True),
    # A probe that never reports back (its worker died) must not wedge the
    # breaker half-open: after another cooldown, a new probe goes through.
    (0, F, CLOSED, None),
    (0, F, CLOSED, None),
    (0, F, OPEN, None),
    (1_000, A, HALF, True),
    (999, A, HALF, False),
    (1, A, HALF, True),
    # A probe cancelled mid-call (the client went away) hands its turn back
    # at once: the next caller probes without waiting another cooldown.
    (0, A, HALF, False),
    (0, X, HALF, None),
    (0, A, HALF, True),
    (0, A, HALF, False),  # ...and that new probe is, again, the only one
    (0, S, CLOSED, None),
    (0, X, CLOSED, None),
    (0, F, CLOSED, None),
    (0, F, CLOSED, None),
    (0, F, OPEN, None),
    (0, X, OPEN, None),  # abandoning while open does not shorten the cooldown
    (0, A, OPEN, False),
]


class Clock:
    def __init__(self) -> None:
        self.ms = 1_700_000_000_000

    def __call__(self) -> int:
        return self.ms


# The Redis case is marked by hand: it reaches its fixture through
# getfixturevalue, which the automatic marker in conftest.py cannot see.
@pytest.fixture(params=["local", pytest.param("redis", marks=pytest.mark.redis)])
def breaker_and_clock(request):
    clock = Clock()
    if request.param == "local":
        return LocalBreaker(POLICY, clock=clock), clock
    coordinator = request.getfixturevalue("coordinator")
    return RedisBreaker(coordinator, "calendar", POLICY, clock=clock), clock


async def test_both_implementations_follow_the_same_timeline(breaker_and_clock):
    breaker, clock = breaker_and_clock
    for i, (advance, event, state, allowed) in enumerate(TIMELINE):
        clock.ms += advance
        got = await breaker._step(event)
        snap = await breaker.snapshot()
        assert snap.state is state, f"step {i}: {event} -> {snap.state}, expected {state}"
        if allowed is not None:
            assert got is allowed, f"step {i}: allow() -> {got}, expected {allowed}"


async def test_when_the_cooldown_ends_exactly_one_worker_probes(coordinator):
    """Five workers ask at the same moment; the Lua script lets one through."""
    clock = Clock()
    workers = [RedisBreaker(coordinator, "calendar", POLICY, clock=clock) for _ in range(5)]
    for _ in range(3):
        await workers[0].record_failure()
    clock.ms += 1_000

    allowed = await asyncio.gather(*(w.allow() for w in workers))

    assert allowed.count(True) == 1


async def test_workers_share_the_state(coordinator):
    """Failures seen by one worker open the breaker for all of them."""
    clock = Clock()
    one, two = (RedisBreaker(coordinator, "calendar", POLICY, clock=clock) for _ in range(2))
    await one.record_failure()
    await two.record_failure()
    await one.record_failure()
    assert not await two.allow()


# ---------------------------------------------------------------------------
# BreakerCalendar: what counts as a failure
# ---------------------------------------------------------------------------
class ScriptedCalendar(CalendarProvider):
    """Fails every book() with the given error, and counts the calls that reached it."""

    def __init__(self, error: Exception | None, *, hang: asyncio.Event | None = None) -> None:
        self.error = error
        self.calls = 0
        self.hang = hang  # book() waits on this, so a test can cancel it mid-call

    async def find_slots(self, *args, **kwargs):
        self.calls += 1
        return []

    async def book(self, request: BookingRequest) -> Booking:
        self.calls += 1
        if self.hang is not None:
            await self.hang.wait()
        if self.error:
            raise self.error
        raise AssertionError("not used")

    async def get_booking(self, **kwargs):
        self.calls += 1
        return None


def request_() -> BookingRequest:
    t = datetime(2026, 10, 5, 13, tzinfo=UTC)
    return BookingRequest(1, 1, "EXAM", t, t, idempotency_key="k")


async def _book_n(calendar: CalendarProvider, n: int) -> list[ToolError]:
    errors = []
    for _ in range(n):
        with pytest.raises(ToolError) as exc:
            await calendar.book(request_())
        errors.append(exc.value)
    return errors


async def test_timeouts_open_the_breaker_and_then_the_calendar_is_not_called(coordinator):
    inner = ScriptedCalendar(ToolError(ErrorCode.TRANSIENT, "timed out"))
    breaker = build_breaker(coordinator, "calendar", failure_threshold=3, cooldown_s=30)
    calendar = BreakerCalendar(inner, breaker)

    errors = await _book_n(calendar, 4)

    assert inner.calls == 3, "the fourth call failed fast, without reaching the calendar"
    assert errors[-1].code is ErrorCode.TRANSIENT
    assert errors[-1].cause == CIRCUIT_OPEN


async def test_conflicts_never_open_the_breaker(coordinator):
    """A taken slot is the calendar working. Counting it would let one popular
    slot switch booking off for everyone."""
    inner = ScriptedCalendar(ToolError(ErrorCode.CONFLICT, "slot taken"))
    breaker = build_breaker(coordinator, "calendar", failure_threshold=3, cooldown_s=30)
    calendar = BreakerCalendar(inner, breaker)

    errors = await _book_n(calendar, 10)

    assert inner.calls == 10
    assert all(e.code is ErrorCode.CONFLICT for e in errors)


async def test_an_open_breaker_also_guards_reads(coordinator):
    inner = ScriptedCalendar(ToolError(ErrorCode.TRANSIENT, "timed out"))
    breaker = build_breaker(coordinator, "calendar", failure_threshold=3, cooldown_s=30)
    calendar = BreakerCalendar(inner, breaker)
    await _book_n(calendar, 3)

    with pytest.raises(ToolError) as exc:
        await calendar.get_booking(idempotency_key="k")
    assert exc.value.cause == CIRCUIT_OPEN
    assert inner.calls == 3


async def test_an_unclassified_error_counts_as_a_failure(coordinator):
    """Whatever the adapter failed to turn into a ToolError is UNKNOWN, and
    UNKNOWN is unwell -- an unclassified failure must not be free."""
    inner = ScriptedCalendar(RuntimeError("driver exploded"))
    breaker = build_breaker(coordinator, "calendar", failure_threshold=3, cooldown_s=30)
    calendar = BreakerCalendar(inner, breaker)

    for _ in range(3):
        with pytest.raises(RuntimeError):
            await calendar.book(request_())
    with pytest.raises(ToolError) as exc:
        await calendar.book(request_())

    assert exc.value.cause == CIRCUIT_OPEN
    assert inner.calls == 3


async def test_a_cancelled_probe_lets_the_next_caller_probe_at_once(coordinator):
    """The patient closes the tab while the only half-open probe is waiting on
    the calendar. Without handing the probe back, every worker would fail fast
    for another full cooldown although nobody is testing the calendar."""
    clock = Clock()
    breaker = build_breaker(coordinator, "calendar", failure_threshold=1, cooldown_s=1, clock=clock)
    await breaker.shared.record_failure()  # open
    clock.ms += 1_000

    hang = asyncio.Event()
    calendar = BreakerCalendar(ScriptedCalendar(None, hang=hang), breaker)
    probe = asyncio.create_task(calendar.book(request_()))
    await asyncio.sleep(0.05)  # the probe is admitted and waiting on the calendar
    assert (await breaker.shared.snapshot()).state is BreakerState.HALF_OPEN
    probe.cancel()
    with pytest.raises(asyncio.CancelledError):
        await probe

    assert await breaker.allow() is not None, "no second cooldown"


async def test_an_outcome_goes_back_to_the_breaker_that_admitted_the_call(redis_client):
    """Redis answers "may I?" and is gone before "it worked". The outcome must
    not be filed with the local breaker, which never admitted anything."""
    # Redis works for the first operation (allow) and fails from the second on.
    flaky = Coordinator(redis_client, FaultInjector.from_string("redis:unavailable@2"))
    breaker = build_breaker(flaky, "calendar", failure_threshold=1, cooldown_s=30)

    permit = await breaker.allow()
    assert permit is not None and permit.breaker is breaker.shared
    await breaker.record(permit, BreakerEvent.FAILURE)  # lost, and logged as lost

    local = await breaker.local.snapshot()
    assert local.state is BreakerState.CLOSED and local.failures == 0, "not misfiled"
