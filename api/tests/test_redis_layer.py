"""The coordination layer against a real Redis: holds, in-flight dedup, rate limiting.

Real, not fakeredis, for the reason the database tests use a real Postgres:
what is under test is that a Lua script runs as one step and that a key
expires -- properties of the server, which a stand-in would only imitate.

What happens when Redis is *not* there is test_degradation.py.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from patient_ops.adapters.calendar.base import Booking, BookingRequest, CalendarProvider, Slot
from patient_ops.degradation import DegradedModes
from patient_ops.errors import ErrorCode, ToolError
from patient_ops.redis_layer.holds import HoldResult, HoldStatus, SlotHolds, cells, hold_key
from patient_ops.redis_layer.idempotency import ClaimState, InFlightDedup
from patient_ops.redis_layer.ratelimit import RateLimiter
from tests.factories import OPEN_MONDAY, local

STEP = timedelta(minutes=30)
TTL = timedelta(seconds=120)


def slot(hour: int, minute: int = 0, minutes: int = 30, provider_id: int = 3) -> Slot:
    start = local(OPEN_MONDAY, hour, minute)
    return Slot(provider_id, "EXAM", start, start + timedelta(minutes=minutes))


# ---------------------------------------------------------------------------
# Cells -- pure
# ---------------------------------------------------------------------------
def test_a_slot_covers_every_grid_cell_it_touches():
    assert len(cells(slot(10, minutes=90), STEP)) == 3
    assert len(cells(slot(10, minutes=30), STEP)) == 1


def test_cells_are_half_open():
    """A slot ending at 10:30 does not touch the 10:30 cell."""
    assert cells(slot(10), STEP) == [slot(10).start_at]
    assert not set(cells(slot(10), STEP)) & set(cells(slot(10, 30), STEP))


def test_a_misaligned_slot_still_shares_a_cell_with_what_it_overlaps():
    """The grid is anchored to the epoch, not to the slot, so overlap always means a shared cell."""
    assert set(cells(slot(10, 15), STEP)) & set(cells(slot(10, 30), STEP))


def test_overlapping_slots_of_different_lengths_share_a_cell():
    assert set(cells(slot(10, minutes=60), STEP)) & set(cells(slot(10, 30), STEP))


# ---------------------------------------------------------------------------
# R1 holds
# ---------------------------------------------------------------------------
@pytest.fixture
def holds(coordinator) -> SlotHolds:
    return SlotHolds(coordinator, ttl=TTL, step=STEP)


async def test_a_hold_is_a_key_with_the_owner_and_a_ttl(holds, redis_client):
    assert await holds.place(slot(10), "thread-a") is HoldResult.HELD

    key = hold_key(3, cells(slot(10), STEP)[0])
    assert await redis_client.get(key) == "thread-a"
    assert 0 < await redis_client.pttl(key) <= 120_000


async def test_a_hold_expires_by_itself(coordinator):
    short = SlotHolds(coordinator, ttl=timedelta(milliseconds=50), step=STEP)
    assert await short.place(slot(10), "thread-a") is HoldResult.HELD
    await asyncio.sleep(0.15)
    assert await short.check(slot(10), "thread-a") is HoldStatus.GONE
    assert await short.place(slot(10), "thread-b") is HoldResult.HELD, "no sweeper needed"


async def test_ten_conversations_race_for_one_slot_and_exactly_one_holds_it(holds):
    results = await asyncio.gather(*(holds.place(slot(10), f"thread-{i}") for i in range(10)))
    assert results.count(HoldResult.HELD) == 1
    assert results.count(HoldResult.TAKEN) == 9


async def test_holding_again_is_a_refresh_not_a_conflict(holds):
    assert await holds.place(slot(10), "thread-a") is HoldResult.HELD
    assert await holds.place(slot(10), "thread-a") is HoldResult.HELD


async def test_an_overlapping_slot_with_a_different_start_is_taken(holds):
    """10:00 x 60min and 10:30 x 30min: different start times, same half hour."""
    assert await holds.place(slot(10, minutes=60), "thread-a") is HoldResult.HELD
    assert await holds.place(slot(10, 30), "thread-b") is HoldResult.TAKEN


async def test_another_providers_slot_at_the_same_time_is_free(holds):
    assert await holds.place(slot(10, provider_id=3), "thread-a") is HoldResult.HELD
    assert await holds.place(slot(10, provider_id=4), "thread-b") is HoldResult.HELD


async def test_a_refused_hold_leaves_no_cells_behind(holds, redis_client):
    """All or nothing: 10:00-11:30 is refused because 11:00 is taken, and
    10:00 and 10:30 must not stay held by the conversation that was refused."""
    assert await holds.place(slot(11), "thread-a") is HoldResult.HELD
    assert await holds.place(slot(10, minutes=90), "thread-b") is HoldResult.TAKEN

    assert await redis_client.get(hold_key(3, cells(slot(10), STEP)[0])) is None
    assert await redis_client.get(hold_key(3, cells(slot(10, 30), STEP)[0])) is None
    assert await holds.place(slot(10, minutes=60), "thread-c") is HoldResult.HELD


async def test_check_tells_mine_from_gone_from_someone_elses(holds):
    assert await holds.check(slot(10), "thread-a") is HoldStatus.GONE
    await holds.place(slot(10), "thread-a")
    assert await holds.check(slot(10), "thread-a") is HoldStatus.MINE
    assert await holds.check(slot(10), "thread-b") is HoldStatus.OTHERS


async def test_releasing_an_expired_hold_does_not_delete_the_new_owners(coordinator):
    """The timeline a plain DEL gets wrong:
    A holds -> A's hold expires -> B holds -> A releases "its" hold."""
    short = SlotHolds(coordinator, ttl=timedelta(milliseconds=50), step=STEP)
    await short.place(slot(10), "thread-a")
    await asyncio.sleep(0.15)
    long = SlotHolds(coordinator, ttl=TTL, step=STEP)
    assert await long.place(slot(10), "thread-b") is HoldResult.HELD

    assert await short.release(slot(10), "thread-a") == 0
    assert await long.check(slot(10), "thread-b") is HoldStatus.MINE


async def test_release_all_keeps_the_booked_slot_and_frees_the_other_offers(holds):
    offered = [slot(9), slot(10), slot(11)]
    for s in offered:
        await holds.place(s, "thread-a")

    released = await holds.release_all("thread-a", keep=slot(10))

    assert released == 2
    assert await holds.check(slot(10), "thread-a") is HoldStatus.MINE
    assert await holds.place(slot(9), "thread-b") is HoldResult.HELD
    assert await holds.place(slot(11), "thread-b") is HoldResult.HELD


# ---------------------------------------------------------------------------
# R4 in-flight dedup, over the real calendar
# ---------------------------------------------------------------------------
class CountingCalendar(CalendarProvider):
    """Counts book() calls, and makes each one slow enough that two overlap."""

    def __init__(self, inner: CalendarProvider, delay_s: float = 0.2) -> None:
        self.inner = inner
        self.delay_s = delay_s
        self.book_calls = 0

    async def find_slots(self, *args, **kwargs):  # pragma: no cover - unused
        return await self.inner.find_slots(*args, **kwargs)

    async def book(self, request: BookingRequest) -> Booking:
        self.book_calls += 1
        await asyncio.sleep(self.delay_s)
        return await self.inner.book(request)

    async def get_booking(self, **kwargs):
        return await self.inner.get_booking(**kwargs)


def exam_request(clinic, key: str = "req-1") -> BookingRequest:
    start = local(OPEN_MONDAY, 9)
    return BookingRequest(
        patient_id=clinic.patient_id,
        provider_id=clinic.dentist_id,
        procedure_code="EXAM",
        start_at=start,
        end_at=start + timedelta(minutes=30),
        idempotency_key=key,
    )


def book_once(dedup: InFlightDedup, calendar: CalendarProvider, request: BookingRequest):
    return dedup.run_once(
        request.idempotency_key,
        lambda: calendar.book(request),
        read_back=lambda: calendar.get_booking(idempotency_key=request.idempotency_key),
        ref=lambda b: b.appointment_id,
    )


@pytest.fixture
def dedup(coordinator) -> InFlightDedup:
    return InFlightDedup(coordinator, inflight_ttl_s=30, wait_s=5)


async def test_a_double_click_calls_the_calendar_once(dedup, calendar, clinic):
    counting = CountingCalendar(calendar)
    request = exam_request(clinic)

    first, second = await asyncio.gather(
        book_once(dedup, counting, request), book_once(dedup, counting, request)
    )

    assert counting.book_calls == 1, "the duplicate waited instead of racing"
    assert first.appointment_id == second.appointment_id


async def test_a_retry_after_success_is_read_back_from_postgres(dedup, calendar, clinic):
    counting = CountingCalendar(calendar, delay_s=0)
    request = exam_request(clinic)
    first = await book_once(dedup, counting, request)

    again = await book_once(dedup, counting, request)

    assert counting.book_calls == 1
    assert again.appointment_id == first.appointment_id
    assert again.created is False, "the result came from the database row, not from Redis"


async def test_a_failed_attempt_gives_the_key_back(dedup, clinic):
    async def fails():
        raise ToolError(ErrorCode.TRANSIENT, "timed out")

    with pytest.raises(ToolError):
        await dedup.run_once("req-1", fails, read_back=_nothing, ref=str)

    claim = await dedup.claim("req-1")
    assert claim.state is ClaimState.CLAIMED, "a retry must not be told it is in flight"


async def test_waiting_too_long_for_a_duplicate_is_unknown_not_a_second_write(coordinator):
    dedup = InFlightDedup(coordinator, inflight_ttl_s=30, wait_s=0.2)
    assert (await dedup.claim("req-1")).state is ClaimState.CLAIMED  # someone else, still running

    calls = 0

    async def action():
        nonlocal calls
        calls += 1

    with pytest.raises(ToolError) as exc:
        await dedup.run_once("req-1", action, read_back=_nothing, ref=str)
    assert exc.value.code is ErrorCode.UNKNOWN
    assert calls == 0


async def test_finishing_a_claim_that_expired_does_not_overwrite_the_next_one(coordinator):
    dedup = InFlightDedup(coordinator, inflight_ttl_s=0.05, wait_s=1)
    mine = await dedup.claim("req-1")
    await asyncio.sleep(0.15)
    theirs = await dedup.claim("req-1")
    assert theirs.state is ClaimState.CLAIMED

    assert await dedup.finish("req-1", mine.token, "42") is False
    assert (await dedup.peek("req-1")).state is ClaimState.IN_FLIGHT


async def _nothing():
    return None


# ---------------------------------------------------------------------------
# R5 rate limiting
# ---------------------------------------------------------------------------
class Clock:
    def __init__(self) -> None:
        self.ms = 1_000_000

    def __call__(self) -> int:
        return self.ms


async def test_the_bucket_empties_then_refills_with_time(coordinator):
    clock = Clock()
    limiter = RateLimiter(coordinator, capacity=3, refill_per_s=0.5, clock=clock)
    degraded = DegradedModes()

    for _ in range(3):
        assert (await limiter.take("chat", "1.2.3.4", degraded)).allowed
    refused = await limiter.take("chat", "1.2.3.4", degraded)
    assert not refused.allowed
    assert refused.retry_after_s == 2, "one token every two seconds"

    clock.ms += 2_000
    assert (await limiter.take("chat", "1.2.3.4", degraded)).allowed
    assert not (await limiter.take("chat", "1.2.3.4", degraded)).allowed
    assert not degraded


async def test_clients_have_separate_buckets(coordinator):
    limiter = RateLimiter(coordinator, capacity=1, refill_per_s=0.01, clock=Clock())
    assert (await limiter.take("chat", "a", DegradedModes())).allowed
    assert (await limiter.take("chat", "b", DegradedModes())).allowed
    assert not (await limiter.take("chat", "a", DegradedModes())).allowed


async def test_concurrent_requests_cannot_overdraw_the_bucket(coordinator):
    limiter = RateLimiter(coordinator, capacity=5, refill_per_s=0.001, clock=Clock())
    results = await asyncio.gather(*(limiter.take("chat", "a", DegradedModes()) for _ in range(20)))
    assert sum(r.allowed for r in results) == 5


async def test_a_redis_error_while_settling_never_replaces_the_bookings_own_outcome(dedup):
    """The booking conflicted; then Redis failed while giving the claim back.
    The caller must still see CONFLICT -- that is what it has a path for."""

    async def broken_settle(*args, **kwargs):
        raise RuntimeError("redis misbehaving after the write")

    dedup.abandon = broken_settle
    dedup.finish = broken_settle

    async def conflicts():
        raise ToolError(ErrorCode.CONFLICT, "slot taken")

    with pytest.raises(ToolError) as exc:
        await dedup.run_once("req-1", conflicts, read_back=_nothing, ref=str)
    assert exc.value.code is ErrorCode.CONFLICT

    async def succeeds():
        return 42

    result = await dedup.run_once("req-2", succeeds, read_back=_nothing, ref=str)
    assert result == 42, "a booking that happened is returned, whatever Redis does next"
