"""G5: the booking policy, checked once more just before an appointment is written.

The booking path only ever writes a slot it computed and offered, so on a good
day this finds nothing. It exists for the other days: a draft that was
checkpointed under an older policy, a clock that moved on between the offer and
the "yes", a bug in how an offered slot is carried from turn to turn. The
calendar checks what any calendar would -- the provider performs the
procedure, the time is inside their working hours -- but whether the visit is
the right length, far enough ahead and not too far, is this clinic's policy,
and the calendar has no business knowing it.

Pure: the caller loads the procedure, schedules and closures, and passes `now`.
The same rules as domain/availability.py, by reusing its functions -- so every
slot compute_slots offers passes here, and tests/test_booking_policy.py
generates thousands of them to check exactly that.

Staff are trusted with what staff legitimately do: a longer visit than the
procedure's standard length, a time off the grid, a same-day slot inside the
lead time. The schedule, closures and the horizon apply to everyone.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import StrEnum
from zoneinfo import ZoneInfo

from patient_ops.domain.availability import (
    Interval,
    ScheduleWindow,
    SchedulingPolicy,
    working_windows,
)


class Violation(StrEnum):
    NOT_POSITIVE = "not_positive"  # ends at or before it starts
    WRONG_DURATION = "wrong_duration"  # not the procedure's length (staff: shorter)
    CLOSED = "closed"  # the clinic is closed that day
    OUTSIDE_SCHEDULE = "outside_schedule"  # not inside one of the provider's windows
    OFF_GRID = "off_grid"  # does not start on a slot boundary of its window
    TOO_SOON = "too_soon"  # inside the minimum lead time
    TOO_FAR = "too_far"  # beyond the booking horizon


STAFF = "staff"


@dataclass(frozen=True)
class ProposedVisit:
    provider_id: int
    start_at: datetime  # aware
    end_at: datetime
    duration: timedelta  # the procedure's standard length
    source: str  # "web", "voice" or "staff"


def violations(
    visit: ProposedVisit,
    *,
    schedules: Iterable[ScheduleWindow],
    closures: Collection[date],
    tz: ZoneInfo,
    now: datetime,
    policy: SchedulingPolicy,
) -> list[Violation]:
    """Every rule the visit breaks, in a fixed order. Empty means book it."""
    if now.tzinfo is None or visit.start_at.tzinfo is None or visit.end_at.tzinfo is None:
        raise ValueError("times must be timezone-aware")
    found: list[Violation] = []
    staff = visit.source == STAFF
    length = visit.end_at - visit.start_at

    if length <= timedelta(0):
        return [Violation.NOT_POSITIVE]  # nothing else about it is meaningful
    if (length < visit.duration) if staff else (length != visit.duration):
        found.append(Violation.WRONG_DURATION)

    interval = Interval(visit.start_at, visit.end_at)
    day = visit.start_at.astimezone(tz).date()
    if day in closures:
        found.append(Violation.CLOSED)
    else:
        windows = working_windows(schedules, closures, day, tz).get(visit.provider_id, [])
        window = next((w for w in windows if w.contains(interval)), None)
        if window is None:
            found.append(Violation.OUTSIDE_SCHEDULE)
        elif not staff and (visit.start_at - window.start) % policy.step:
            # Relative to the window, as compute_slots steps: a window that
            # opens at 09:15 offers 09:15 and 09:45, not 09:30.
            found.append(Violation.OFF_GRID)

    # Staff may book inside the lead time -- a patient at the desk -- but not
    # in the past.
    if visit.start_at < (now if staff else now + policy.min_lead):
        found.append(Violation.TOO_SOON)
    if visit.start_at > now + policy.max_horizon:
        found.append(Violation.TOO_FAR)
    return found
