"""Property-based tests for compute_slots.

Example tests prove the cases I thought of are right. These generate random
clinics -- schedules, closures, bookings, timezones (some with DST), lead times
-- and check rules that must hold for *every* input:

  1. every slot is exactly the requested duration
  2. no slot overlaps a booking of the same provider
  3. every slot fits the provider's schedule (cross-checks fits_schedule)
  4. no slot falls on a closure, judged by the clinic-local date
  5. every slot starts inside [now + min lead, now + horizon]
  6. booking one more interval never creates a slot -- slots only disappear
  7. the result is sorted and has no duplicates

The checks (oracles) are written independently of the code under test. An
oracle that calls Interval.overlaps to verify compute_slots inherits any bug in
Interval.overlaps and agrees with it forever -- planting a bug there and
watching the suite stay green is how this file found that out.

When a property fails, Hypothesis shrinks the input to the smallest
counterexample it can find and saves it in .hypothesis/ so it is replayed on the
next run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from hypothesis import given
from hypothesis import strategies as st

from patient_ops.domain.availability import (
    Interval,
    ScheduleWindow,
    SchedulingPolicy,
    SlotCandidate,
    compute_slots,
    fits_schedule,
)

TIMEZONES = ["America/New_York", "Europe/Berlin", "Asia/Kolkata", "UTC"]  # DST, DST, +05:30, none
PROVIDERS = [1, 2, 3]
# Dates just before DST changes in New York and Berlin, so the generated
# windows regularly straddle a clock change instead of rarely.
DST_EDGES = [date(2026, 3, 6), date(2026, 3, 27), date(2026, 10, 23), date(2026, 10, 30)]


@dataclass(frozen=True)
class Scenario:
    schedules: list[ScheduleWindow]
    closures: set[date]
    booked: list[tuple[int, Interval]]
    extra_booking: tuple[int, Interval]
    duration: timedelta
    date_from: date
    date_to: date
    tz: ZoneInfo
    now: datetime
    policy: SchedulingPolicy

    def slots(self, booked: list[tuple[int, Interval]] | None = None) -> list[SlotCandidate]:
        return compute_slots(
            schedules=self.schedules,
            closures=self.closures,
            booked=self.booked if booked is None else booked,
            duration=self.duration,
            date_from=self.date_from,
            date_to=self.date_to,
            tz=self.tz,
            now=self.now,
            policy=self.policy,
        )


def _minute_of_day(minutes: int) -> time:
    return time(minutes // 60, minutes % 60)


# --- Independent oracles: none of these call into domain/availability.py ----
def _overlap(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> bool:
    return a_start < b_end and b_start < a_end  # half-open ranges


def _inside_raw_schedule(s: Scenario, slot: SlotCandidate) -> bool:
    """Is the slot covered by the provider's schedule rows, judged in local
    wall-clock time straight from the rows (no merging, no UTC)?"""
    start, end = slot.start.astimezone(s.tz), slot.end.astimezone(s.tz)
    if start.date() != end.date():
        return False
    rows = sorted(
        (w.start, w.end)
        for w in s.schedules
        if w.provider_id == slot.provider_id and w.weekday == start.weekday()
    )
    covered_until = start.time()
    for row_start, row_end in rows:
        if row_start <= covered_until < row_end:
            covered_until = row_end
    return covered_until >= end.time()


@st.composite
def weekly_blocks(draw: st.DrawFn) -> list[ScheduleWindow]:
    """One provider working the same hours on several weekdays -- the shape of
    a real schedule, and dense enough that most scenarios produce slots."""
    provider_id = draw(st.sampled_from(PROVIDERS))
    weekdays = draw(st.sets(st.integers(0, 6), min_size=1))
    start = draw(st.integers(6 * 4, 20 * 4)) * 15  # 06:00-20:00 on a 15-minute grid
    length = draw(st.integers(1, 8 * 4)) * 15  # 15 min .. 8 h
    end = min(start + length, 23 * 60 + 45)
    return [
        ScheduleWindow(provider_id, weekday, _minute_of_day(start), _minute_of_day(end))
        for weekday in weekdays
    ]


@st.composite
def bookings(draw: st.DrawFn, days: list[date], tz: ZoneInfo) -> tuple[int, Interval]:
    day = draw(st.sampled_from(days))
    start = datetime.combine(day, _minute_of_day(draw(st.integers(0, 95)) * 15), tzinfo=tz)
    length = timedelta(minutes=draw(st.integers(1, 16)) * 15)
    return draw(st.sampled_from(PROVIDERS)), Interval(start, start + length)


@st.composite
def scenarios(draw: st.DrawFn) -> Scenario:
    tz = ZoneInfo(draw(st.sampled_from(TIMEZONES)))
    date_from = draw(
        st.one_of(
            st.sampled_from(DST_EDGES),
            st.dates(min_value=date(2026, 1, 1), max_value=date(2027, 12, 31)),
        )
    )
    date_to = date_from + timedelta(days=draw(st.integers(0, 13)))
    days = [date_from + timedelta(days=i) for i in range((date_to - date_from).days + 1)]

    # Half the time "now" falls on the first requested day -- the case where
    # lead time decides which of today's slots are still offered.
    days_before = draw(st.one_of(st.just(0), st.integers(0, 3)))
    now_local = datetime.combine(
        date_from - timedelta(days=days_before),
        _minute_of_day(draw(st.integers(0, 95)) * 15),
        tzinfo=tz,
    )
    blocks = draw(st.lists(weekly_blocks(), min_size=1, max_size=5))
    return Scenario(
        schedules=[window for block in blocks for window in block],
        closures=set(draw(st.lists(st.sampled_from(days), max_size=3))),
        booked=draw(st.lists(bookings(days, tz), max_size=10)),
        extra_booking=draw(bookings(days, tz)),
        duration=timedelta(minutes=draw(st.sampled_from([15, 30, 45, 60, 90, 120]))),
        date_from=date_from,
        date_to=date_to,
        tz=tz,
        now=now_local.astimezone(UTC),
        policy=SchedulingPolicy(
            step=timedelta(minutes=draw(st.sampled_from([15, 30, 60]))),
            min_lead=timedelta(hours=draw(st.one_of(st.sampled_from([0, 2]), st.integers(0, 48)))),
            max_horizon=timedelta(days=draw(st.integers(1, 200))),
        ),
    )


@given(scenarios())
def test_every_slot_has_the_requested_duration(s: Scenario):
    assert all(slot.end - slot.start == s.duration for slot in s.slots())


@given(scenarios())
def test_no_slot_overlaps_a_booking_of_its_provider(s: Scenario):
    for slot in s.slots():
        for provider_id, booking in s.booked:
            if provider_id == slot.provider_id:
                assert not _overlap(slot.start, slot.end, booking.start, booking.end)


@given(scenarios())
def test_every_slot_fits_the_schedule(s: Scenario):
    for slot in s.slots():
        assert _inside_raw_schedule(s, slot)
        # ...and the calendar's pre-write check agrees with the slot generator.
        assert fits_schedule(slot.provider_id, slot.interval, s.schedules, s.closures, s.tz)


@given(scenarios())
def test_no_slot_on_a_closure(s: Scenario):
    assert all(slot.start.astimezone(s.tz).date() not in s.closures for slot in s.slots())


@given(scenarios())
def test_every_slot_respects_lead_time_and_horizon(s: Scenario):
    earliest, latest = s.now + s.policy.min_lead, s.now + s.policy.max_horizon
    assert all(earliest <= slot.start <= latest for slot in s.slots())


@given(scenarios())
def test_one_more_booking_never_creates_a_slot(s: Scenario):
    """Metamorphic property: compare two runs instead of predicting one.

    Catches off-by-one errors in the overlap test that no hand-picked example
    happens to hit.
    """
    before = set(s.slots())
    after = set(s.slots(booked=[*s.booked, s.extra_booking]))
    assert after <= before


@given(scenarios())
def test_result_is_sorted_and_unique(s: Scenario):
    slots = s.slots()
    assert slots == sorted(slots)
    assert len(slots) == len(set(slots))
