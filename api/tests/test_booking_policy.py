"""G5, the booking policy checked before every write.

Three layers. The pure function against generated clinics: everything
compute_slots offers passes, and a slot nudged off what it would offer does
not. The rules one at a time, by example. And the booking desk: a request that
breaks the policy is refused before the calendar is called -- unless it is the
replay of a booking that already exists.
"""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from patient_ops.adapters.calendar.base import BookingRequest
from patient_ops.db.session import build_engine, build_session_factory
from patient_ops.degradation import DegradedModes
from patient_ops.domain.availability import ScheduleWindow, SchedulingPolicy
from patient_ops.errors import ErrorCode, ToolError
from patient_ops.graph import replies
from patient_ops.graph.turn import turn_config
from patient_ops.guardrails.booking_policy import ProposedVisit, Violation, violations
from tests.factories import CLOSED_MONDAY, FIXED_NOW, TZ, booked_count, local
from tests.test_booking_flow import FRI, THU, BookingChat, appointments, offer_cleanings
from tests.test_domain_properties import Scenario, scenarios

POLICY = SchedulingPolicy()  # 30-minute steps, two hours' lead, 180 days out


# ---------------------------------------------------------------------------
# Generated: G5 agrees with compute_slots
# ---------------------------------------------------------------------------
def _check(s: Scenario, provider_id: int, start: datetime, end: datetime, source: str = "web"):
    return violations(
        ProposedVisit(provider_id, start, end, s.duration, source),
        schedules=s.schedules,
        closures=s.closures,
        tz=s.tz,
        now=s.now,
        policy=s.policy,
    )


@given(scenarios())
def test_every_slot_compute_slots_offers_passes(s: Scenario):
    for slot in s.slots(booked=[]):
        assert _check(s, slot.provider_id, slot.start, slot.end) == [], slot


@given(scenarios(), st.integers(1, 14))
def test_a_slot_nudged_off_its_grid_is_refused(s: Scenario, minutes: int):
    slots = s.slots(booked=[])
    assume(slots)
    slot = slots[0]
    nudge = timedelta(minutes=minutes)
    assume(nudge % s.policy.step)
    assert _check(s, slot.provider_id, slot.start + nudge, slot.end + nudge) != []


@given(scenarios())
def test_a_longer_visit_is_refused_for_patients_and_allowed_for_staff(s: Scenario):
    slots = s.slots(booked=[])
    assume(slots)
    slot = slots[0]
    longer = slot.end + s.policy.step
    assert Violation.WRONG_DURATION in _check(s, slot.provider_id, slot.start, longer)
    assert Violation.WRONG_DURATION not in _check(s, slot.provider_id, slot.start, longer, "staff")


# ---------------------------------------------------------------------------
# By example, one rule at a time. Mon-Fri 09:00-17:00; now is Thu 01 Oct 08:00.
# ---------------------------------------------------------------------------
WEEKDAYS = [ScheduleWindow(1, day, time(9), time(17)) for day in range(5)]
HOUR = timedelta(hours=1)


def check(start: datetime, end: datetime | None = None, *, source: str = "web", closures=()):
    return violations(
        ProposedVisit(1, start, end or start + HOUR, HOUR, source),
        schedules=WEEKDAYS,
        closures=set(closures),
        tz=TZ,
        now=FIXED_NOW,
        policy=POLICY,
    )


@pytest.mark.parametrize(
    ("start", "end", "source", "expected"),
    [
        (local(FRI, 9), None, "web", []),
        (local(THU, 10), None, "web", []),  # exactly the lead time
        (local(THU, 9, 30), None, "web", [Violation.TOO_SOON]),
        (local(THU, 9, 30), None, "staff", []),
        (local(THU, 7), None, "staff", [Violation.OUTSIDE_SCHEDULE, Violation.TOO_SOON]),
        (local(FRI, 9, 10), None, "web", [Violation.OFF_GRID]),
        (local(FRI, 9, 10), None, "staff", []),
        (local(FRI, 9), local(FRI, 9, 30), "web", [Violation.WRONG_DURATION]),
        (local(FRI, 9), local(FRI, 9, 30), "staff", [Violation.WRONG_DURATION]),
        (local(FRI, 9), local(FRI, 11), "staff", []),
        (local(FRI, 16, 30), None, "web", [Violation.OUTSIDE_SCHEDULE]),
        (local(FRI, 9), local(FRI, 9), "web", [Violation.NOT_POSITIVE]),
        (local(THU, 9) + timedelta(days=182), None, "web", [Violation.TOO_FAR]),
    ],
)
def test_each_rule(start, end, source, expected):
    assert check(start, end, source=source) == expected


def test_a_closure_is_named_as_such():
    assert check(local(CLOSED_MONDAY, 9), closures=[CLOSED_MONDAY]) == [Violation.CLOSED]


def test_naive_times_are_a_programming_error():
    with pytest.raises(ValueError):
        check(datetime(2026, 10, 2, 9))


def test_across_a_dst_change_the_local_opening_time_holds():
    """Sunday 01 Nov 2026 the clocks go back in New York; Monday 09:00 local is
    14:00 UTC, not 13:00."""
    monday = datetime(2026, 11, 2, 14, 0, tzinfo=UTC)
    assert check(monday) == []
    assert check(monday - HOUR) == [Violation.OUTSIDE_SCHEDULE]
    assert ZoneInfo("America/New_York").utcoffset(monday) == timedelta(hours=-5)


# ---------------------------------------------------------------------------
# The desk: refused before the calendar is called, except a replay
# ---------------------------------------------------------------------------
@pytest.fixture
def chat(session_factory, clinic, calendar, coordinator) -> BookingChat:
    return BookingChat(session_factory, calendar, coordinator)


def cleaning(clinic, start: datetime, end: datetime, key: str = "k-1") -> BookingRequest:
    return BookingRequest(
        patient_id=clinic.patient_id,
        provider_id=clinic.hygienist_id,
        procedure_code="CLEANING",
        start_at=start,
        end_at=end,
        idempotency_key=key,
    )


async def test_the_desk_refuses_a_policy_breach_without_writing(chat, clinic, session_factory):
    desk = chat.desk(DegradedModes())
    with pytest.raises(ToolError) as exc:
        await desk.book(cleaning(clinic, local(FRI, 9), local(FRI, 11)))  # two hours, not one

    assert exc.value.code is ErrorCode.INVALID
    assert exc.value.cause == "G5:wrong_duration"
    assert await booked_count(session_factory) == 0
    (call,) = desk.trace
    assert call.summary == "error: invalid (G5:wrong_duration)", "the refusal is on the record"


async def test_a_replay_is_not_refused_when_the_clock_has_moved_on(chat, clinic, session_factory):
    """Booked Thu 10:00 at 08:00 -- exactly the lead time. The "yes" is
    retried at 08:20 with the same key: the appointment exists, and saying
    otherwise would be the lie."""
    request = cleaning(clinic, local(THU, 10), local(THU, 11))
    first = await chat.desk(DegradedModes()).book(request)

    chat.now = FIXED_NOW + timedelta(minutes=20)
    again = await chat.desk(DegradedModes()).book(request)
    assert again.appointment_id == first.appointment_id and not again.created

    with pytest.raises(ToolError, match="too_soon"):  # a new key at 08:20 is refused
        await chat.desk(DegradedModes()).book(
            cleaning(clinic, local(THU, 10), local(THU, 11), key="k-2")
        )
    assert await booked_count(session_factory) == 1


async def test_a_tampered_draft_is_refused_and_fresh_times_offered(chat, session_factory):
    """Whatever put a longer visit into the checkpoint, the "yes" does not
    write it."""
    await offer_cleanings(chat)
    await chat.say("the second one")
    booking = (await chat.state())["booking"]
    offered = booking["offered"][booking["chosen"] - 1]
    offered["end_at"] = (datetime.fromisoformat(offered["end_at"]) + HOUR).isoformat()
    await chat.graph.aupdate_state(turn_config(chat.thread_id), {"booking": booking})

    reply = await chat.say("yes")
    assert reply.startswith(replies.SLOT_UNAVAILABLE)
    assert await appointments(session_factory) == []
    assert await chat.kind() == "booking_offer"


async def test_a_database_that_cannot_be_asked_is_transient_not_a_crash(
    chat, clinic, session_factory, test_settings
):
    """Found in the Phase 7 audit: the policy's own reads raised a raw
    OperationalError, which the booking path does not catch -- the patient got
    "something went wrong" instead of "the calendar did not answer, say yes to
    try again". Now classified like every other read on this path."""
    unreachable = test_settings.model_copy(
        update={"database_url": "postgresql://x:y@127.0.0.1:1/z", "db_connect_timeout_s": 0.5}
    )
    desk = chat.desk(DegradedModes())
    desk.session_factory = build_session_factory(build_engine(unreachable))
    with pytest.raises(ToolError) as exc:
        await desk.book(cleaning(clinic, local(FRI, 9), local(FRI, 10)))
    assert exc.value.code is ErrorCode.TRANSIENT
    assert desk.trace[-1].summary.startswith("error: transient")
    assert await booked_count(session_factory) == 0
