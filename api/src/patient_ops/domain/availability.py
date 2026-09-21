"""Available appointment slots, computed as a pure function.

    available = working windows - clinic closures - booked intervals,
                keeping only starts between (now + lead time) and (now + horizon)

Pure: no database, no clock, no IO. Everything it needs is an argument --
including `now`. That is what lets the same rules be tested with thousands of
generated inputs, and reused unchanged by the calendar adapter and (Phase 7) by
the G5 guardrail.

Time model:
- Schedules are recurring *local wall-clock* windows ("Monday 09:00-12:00").
- Everything else is an *aware instant*, normalised to UTC.
- Conversion happens in one place, `_instant`, per calendar day, using the
  clinic's IANA timezone. So a DST change moves the UTC instant, never the
  local opening time.
- Arithmetic is done on UTC instants. Python adds a timedelta to an aware
  datetime in wall-clock terms when the zone stays the same, which is wrong
  across a DST change; UTC has no DST.
- Intervals are half-open [start, end), the same as Postgres tstzrange: an
  appointment ending at 11:00 does not collide with one starting at 11:00.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class ScheduleWindow:
    provider_id: int
    weekday: int  # Monday=0 .. Sunday=6, as date.weekday()
    start: time  # local wall-clock
    end: time


@dataclass(frozen=True, order=True)
class Interval:
    start: datetime  # aware
    end: datetime

    def overlaps(self, other: Interval) -> bool:
        return self.start < other.end and other.start < self.end

    def contains(self, other: Interval) -> bool:
        return self.start <= other.start and other.end <= self.end


@dataclass(frozen=True)
class SchedulingPolicy:
    step: timedelta = timedelta(minutes=30)  # slot starts fall on this grid
    min_lead: timedelta = timedelta(hours=2)  # no booking sooner than this
    max_horizon: timedelta = timedelta(days=180)  # no booking further out than this


@dataclass(frozen=True, order=True)
class SlotCandidate:
    start: datetime  # field order = sort order: by time, then provider
    provider_id: int
    end: datetime

    @property
    def interval(self) -> Interval:
        return Interval(self.start, self.end)


def _instant(day: date, wall_clock: time, tz: ZoneInfo) -> datetime:
    """Clinic-local wall-clock time on `day` -> aware UTC instant."""
    return datetime.combine(day, wall_clock, tzinfo=tz).astimezone(UTC)


def _merge(intervals: Iterable[Interval]) -> list[Interval]:
    """Union overlapping or touching intervals, so 09-12 + 12-13 acts as 09-13
    and a duplicated schedule row cannot produce duplicate slots."""
    merged: list[Interval] = []
    for iv in sorted(intervals):
        if merged and iv.start <= merged[-1].end:
            merged[-1] = Interval(merged[-1].start, max(merged[-1].end, iv.end))
        else:
            merged.append(iv)
    return merged


def working_windows(
    schedules: Iterable[ScheduleWindow], closures: Collection[date], day: date, tz: ZoneInfo
) -> dict[int, list[Interval]]:
    """Each provider's working intervals on one clinic-local day."""
    if day in closures:
        return {}
    by_provider: dict[int, list[Interval]] = defaultdict(list)
    for w in schedules:
        if w.weekday == day.weekday():
            by_provider[w.provider_id].append(
                Interval(_instant(day, w.start, tz), _instant(day, w.end, tz))
            )
    return {pid: _merge(ivs) for pid, ivs in by_provider.items()}


def fits_schedule(
    provider_id: int,
    interval: Interval,
    schedules: Iterable[ScheduleWindow],
    closures: Collection[date],
    tz: ZoneInfo,
) -> bool:
    """True if `interval` lies entirely inside one of the provider's working
    windows on its clinic-local day (and that day is not a closure)."""
    day = interval.start.astimezone(tz).date()
    windows = working_windows(schedules, closures, day, tz).get(provider_id, [])
    return any(w.contains(interval) for w in windows)


def compute_slots(
    *,
    schedules: Iterable[ScheduleWindow],
    closures: Collection[date],
    booked: Iterable[tuple[int, Interval]],
    duration: timedelta,
    date_from: date,
    date_to: date,
    tz: ZoneInfo,
    now: datetime,
    policy: SchedulingPolicy,
) -> list[SlotCandidate]:
    """Every bookable slot of `duration` between two clinic-local dates, inclusive.

    `booked` is (provider_id, interval) for each live booking. `now` is passed
    in rather than read from a clock, so the result depends only on the
    arguments.
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if duration <= timedelta(0):
        raise ValueError("duration must be positive")

    schedules = list(schedules)
    booked_by_provider: dict[int, list[Interval]] = defaultdict(list)
    for provider_id, iv in booked:
        booked_by_provider[provider_id].append(iv)

    earliest_start = now + policy.min_lead
    latest_start = now + policy.max_horizon

    slots: list[SlotCandidate] = []
    day = date_from
    while day <= date_to:
        for provider_id, windows in working_windows(schedules, closures, day, tz).items():
            taken = booked_by_provider[provider_id]
            for window in windows:
                start = window.start
                while start + duration <= window.end:
                    candidate = Interval(start, start + duration)
                    if earliest_start <= start <= latest_start and not any(
                        candidate.overlaps(b) for b in taken
                    ):
                        slots.append(SlotCandidate(start, provider_id, candidate.end))
                    start += policy.step
        day += timedelta(days=1)
    return sorted(slots)
