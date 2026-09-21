"""Example-based tests for the pure domain logic: slots and phone numbers.

Each test is one situation a person at the front desk would recognise -- the
lunch break, a holiday, back-to-back appointments, the week the clocks change.
No database: these run with every container stopped.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta

import pytest

from patient_ops.domain.availability import (
    Interval,
    ScheduleWindow,
    SchedulingPolicy,
    compute_slots,
    fits_schedule,
    working_windows,
)
from patient_ops.domain.phone import normalize_phone
from tests.factories import TZ, local

DENTIST = 1
MONDAY = date(2026, 10, 5)
HOLIDAY = date(2026, 10, 12)  # also a Monday

# Dentist works Monday-Friday, 09-12 and 13-17, with lunch in between.
SCHEDULE = [
    ScheduleWindow(DENTIST, weekday, time(start), time(end))
    for weekday in range(5)
    for start, end in ((9, 12), (13, 17))
]
POLICY = SchedulingPolicy(
    step=timedelta(minutes=30), min_lead=timedelta(hours=2), max_horizon=timedelta(days=180)
)
LONG_BEFORE = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)  # "now", well before every test date


def slots_on(day: date, minutes: int, *, booked=(), now=LONG_BEFORE, closures=(HOLIDAY,)):
    return compute_slots(
        schedules=SCHEDULE,
        closures=set(closures),
        booked=booked,
        duration=timedelta(minutes=minutes),
        date_from=day,
        date_to=day,
        tz=TZ,
        now=now,
        policy=POLICY,
    )


def local_starts(slots) -> list[str]:
    return [s.start.astimezone(TZ).strftime("%H:%M") for s in slots]


# ---------------------------------------------------------------------------
# Working windows
# ---------------------------------------------------------------------------
def test_working_windows_convert_local_hours_to_instants():
    windows = working_windows(SCHEDULE, set(), MONDAY, TZ)[DENTIST]
    assert windows == [
        Interval(local(MONDAY, 9), local(MONDAY, 12)),
        Interval(local(MONDAY, 13), local(MONDAY, 17)),
    ]


def test_no_windows_on_a_closure_or_a_day_off():
    assert working_windows(SCHEDULE, {HOLIDAY}, HOLIDAY, TZ) == {}
    assert working_windows(SCHEDULE, set(), date(2026, 10, 4), TZ) == {}  # Sunday


def test_overlapping_schedule_rows_are_merged():
    rows = [
        ScheduleWindow(DENTIST, 0, time(9), time(12)),
        ScheduleWindow(DENTIST, 0, time(11), time(14)),
    ]
    assert working_windows(rows, set(), MONDAY, TZ)[DENTIST] == [
        Interval(local(MONDAY, 9), local(MONDAY, 14))
    ]


# ---------------------------------------------------------------------------
# Daylight saving time -- same local opening hour, different UTC instant
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("day", "utc_hour"),
    [
        (date(2026, 10, 30), 13),  # Friday, still EDT (UTC-4)
        (date(2026, 11, 2), 14),  # Monday after clocks go back: EST (UTC-5)
        (date(2026, 3, 6), 14),  # Friday before clocks go forward: EST
        (date(2026, 3, 9), 13),  # Monday after: EDT
    ],
)
def test_nine_am_stays_nine_am_across_dst(day: date, utc_hour: int):
    first = working_windows(SCHEDULE, set(), day, TZ)[DENTIST][0]
    assert first.start.astimezone(TZ).time() == time(9)
    assert first.start.astimezone(UTC).hour == utc_hour


# ---------------------------------------------------------------------------
# Slots
# ---------------------------------------------------------------------------
def test_thirty_minute_slots_fill_both_windows():
    assert local_starts(slots_on(MONDAY, 30)) == [
        "09:00", "09:30", "10:00", "10:30", "11:00", "11:30",
        "13:00", "13:30", "14:00", "14:30", "15:00", "15:30", "16:00", "16:30",
    ]  # fmt: skip


def test_a_long_procedure_never_spans_lunch():
    starts = local_starts(slots_on(MONDAY, 90))
    assert starts == [
        "09:00", "09:30", "10:00", "10:30",
        "13:00", "13:30", "14:00", "14:30", "15:00", "15:30",
    ]  # fmt: skip
    assert "11:00" not in starts  # 11:00-12:30 would run into lunch


def test_booked_time_is_removed_but_neighbours_stay():
    booked = [(DENTIST, Interval(local(MONDAY, 10), local(MONDAY, 11)))]
    starts = local_starts(slots_on(MONDAY, 30, booked=booked))
    assert "10:00" not in starts and "10:30" not in starts
    # Half-open intervals: 09:30-10:00 and 11:00-11:30 only touch the booking.
    assert "09:30" in starts and "11:00" in starts


def test_bookings_of_other_providers_do_not_matter():
    booked = [(99, Interval(local(MONDAY, 9), local(MONDAY, 17)))]
    assert len(slots_on(MONDAY, 30, booked=booked)) == 14


def test_no_slots_on_a_closure():
    assert slots_on(HOLIDAY, 30) == []


def test_minimum_lead_time():
    now = local(MONDAY, 8)  # 08:00 -> nothing before 10:00
    assert local_starts(slots_on(MONDAY, 30, now=now))[0] == "10:00"


def test_booking_horizon():
    now = local(MONDAY, 8) - timedelta(days=180)  # MONDAY 08:00 is exactly the horizon
    assert slots_on(MONDAY, 30, now=now) == []


def test_now_must_be_timezone_aware():
    with pytest.raises(ValueError, match="aware"):
        slots_on(MONDAY, 30, now=datetime(2026, 9, 1, 12, 0))


# ---------------------------------------------------------------------------
# fits_schedule -- the check the calendar runs before it writes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("provider", "day", "start", "end", "expected"),
    [
        (DENTIST, MONDAY, (10, 0), (11, 0), True),
        (DENTIST, MONDAY, (9, 0), (12, 0), True),  # the whole morning window
        (DENTIST, MONDAY, (11, 30), (12, 30), False),  # runs into lunch
        (DENTIST, MONDAY, (8, 30), (9, 30), False),  # starts before opening
        (DENTIST, HOLIDAY, (10, 0), (11, 0), False),
        (DENTIST, date(2026, 10, 4), (10, 0), (11, 0), False),  # Sunday
        (2, MONDAY, (10, 0), (11, 0), False),  # a provider with no schedule
    ],
)
def test_fits_schedule(provider, day, start, end, expected):
    interval = Interval(local(day, *start), local(day, *end))
    assert fits_schedule(provider, interval, SCHEDULE, {HOLIDAY}, TZ) is expected


# ---------------------------------------------------------------------------
# Phone numbers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw", ["(212) 555-0143", "212.555.0143", "2125550143", "1-212-555-0143", "+1 212 555 0143"]
)
def test_phone_forms_normalise_to_one_value(raw: str):
    assert normalize_phone(raw) == "+12125550143"


def test_international_numbers_keep_their_country_code():
    assert normalize_phone("+86 138 0013 8000") == "+8613800138000"


@pytest.mark.parametrize("raw", ["", "555-0143", "call me", "212-555-0143 ext 5"])
def test_unrecognised_phone_numbers_are_rejected(raw: str):
    with pytest.raises(ValueError):
        normalize_phone(raw)
