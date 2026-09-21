"""FakeCalendar availability, and the fault-injecting wrapper around it."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from patient_ops.adapters.calendar.base import BookingRequest
from patient_ops.adapters.calendar.fake import FakeCalendar
from patient_ops.adapters.calendar.faults import INJECTED, FaultyCalendar
from patient_ops.errors import ErrorCode, ToolError
from patient_ops.faults import FaultInjector
from tests.factories import CLOSED_MONDAY, OPEN_MONDAY, TZ, Clinic, booked_count, local


def starts(slots) -> list[str]:
    return [s.start_at.astimezone(TZ).strftime("%H:%M") for s in slots]


def exam_request(clinic: Clinic, key: str = "req-1") -> BookingRequest:
    return BookingRequest(
        patient_id=clinic.patient_id,
        provider_id=clinic.dentist_id,
        procedure_code="EXAM",
        start_at=local(OPEN_MONDAY, 10),
        end_at=local(OPEN_MONDAY, 10, 30),
        idempotency_key=key,
    )


def faulty(calendar: FakeCalendar, spec: str) -> FaultyCalendar:
    return FaultyCalendar(calendar, FaultInjector.from_string(spec))


# ---------------------------------------------------------------------------
# Availability through the real database
# ---------------------------------------------------------------------------
async def test_slots_come_only_from_providers_who_perform_the_procedure(calendar, clinic):
    exam_slots = await calendar.find_slots("EXAM", OPEN_MONDAY, OPEN_MONDAY)
    cleaning_slots = await calendar.find_slots("CLEANING", OPEN_MONDAY, OPEN_MONDAY)

    assert {s.provider_id for s in exam_slots} == {clinic.dentist_id}
    assert len(exam_slots) == 14  # 30-min slots in 09-12 and 13-17
    assert {s.provider_id for s in cleaning_slots} == {clinic.hygienist_id}
    assert starts(cleaning_slots)[0] == "09:00" and starts(cleaning_slots)[-1] == "16:00"


async def test_a_booking_removes_its_slot(calendar, clinic):
    await calendar.book(exam_request(clinic))
    remaining = starts(await calendar.find_slots("EXAM", OPEN_MONDAY, OPEN_MONDAY))
    assert "10:00" not in remaining
    assert "09:30" in remaining and "10:30" in remaining


async def test_no_slots_on_closures_or_weekends(calendar, clinic):
    assert await calendar.find_slots("EXAM", CLOSED_MONDAY, CLOSED_MONDAY) == []
    assert await calendar.find_slots("EXAM", date(2026, 10, 3), date(2026, 10, 4)) == []


async def test_lead_time_is_measured_from_the_injected_clock(calendar, clinic):
    # FIXED_NOW is Thursday 2026-10-01 08:00 in New York; minimum lead is 2 h.
    today = await calendar.find_slots("EXAM", date(2026, 10, 1), date(2026, 10, 1))
    assert starts(today)[0] == "10:00"


async def test_filter_by_provider(calendar, clinic):
    own = await calendar.find_slots("EXAM", OPEN_MONDAY, OPEN_MONDAY, provider_id=clinic.dentist_id)
    assert len(own) == 14
    with pytest.raises(ToolError) as exc:  # a hygienist does not perform exams here
        await calendar.find_slots("EXAM", OPEN_MONDAY, OPEN_MONDAY, provider_id=clinic.hygienist_id)
    assert exc.value.code is ErrorCode.INVALID


@pytest.mark.parametrize(
    ("code", "date_from", "date_to"),
    [
        ("WHITENING", OPEN_MONDAY, OPEN_MONDAY),  # unknown procedure
        ("EXAM", OPEN_MONDAY, OPEN_MONDAY - timedelta(days=1)),  # range backwards
        ("EXAM", OPEN_MONDAY, OPEN_MONDAY + timedelta(days=40)),  # range too long
    ],
)
async def test_impossible_searches_are_invalid(calendar, clinic, code, date_from, date_to):
    with pytest.raises(ToolError) as exc:
        await calendar.find_slots(code, date_from, date_to)
    assert exc.value.code is ErrorCode.INVALID


async def test_get_booking_by_id_or_key(calendar, clinic):
    booking = await calendar.book(exam_request(clinic, key="look-me-up"))
    by_id = await calendar.get_booking(appointment_id=booking.appointment_id)
    by_key = await calendar.get_booking(idempotency_key="look-me-up")
    assert by_id == by_key
    assert by_id.appointment_id == booking.appointment_id
    assert await calendar.get_booking(idempotency_key="never-written") is None


# ---------------------------------------------------------------------------
# Injected faults
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("spec", "code", "detail"),
    [
        ("book_appointment:timeout", ErrorCode.TRANSIENT, "timed out"),
        ("book_appointment:server_error", ErrorCode.TRANSIENT, "HTTP 500"),
        ("book_appointment:conflict", ErrorCode.CONFLICT, "slot taken"),
    ],
)
async def test_injected_booking_failures_write_nothing(
    calendar, clinic, session_factory, spec, code, detail
):
    with pytest.raises(ToolError) as exc:
        await faulty(calendar, spec).book(exam_request(clinic))
    assert exc.value.code is code
    assert detail in exc.value.detail
    assert exc.value.cause == INJECTED  # never mistaken for a real failure in the logs
    assert await booked_count(session_factory) == 0


async def test_timeout_after_write_the_booking_exists_anyway(calendar, clinic, session_factory):
    """★ The failure that matters most: the write committed, the answer was lost.

    The caller sees exactly what it would see for a timeout that wrote nothing.
    Only reading the booking back tells the two apart -- which is why, in
    Phase 8, execute_booking -> verify_booking is an unconditional edge.
    """
    cal = faulty(calendar, "book_appointment:timeout_after_write@1")
    request = exam_request(clinic, key="lost-ack")

    with pytest.raises(ToolError) as exc:
        await cal.book(request)
    assert exc.value.code is ErrorCode.TRANSIENT  # looks like an ordinary timeout...

    found = await cal.get_booking(idempotency_key="lost-ack")
    assert found is not None  # ...but the appointment is there

    retry = await cal.book(request)  # retrying with the SAME key is safe
    assert retry.appointment_id == found.appointment_id
    assert retry.created is False
    assert await booked_count(session_factory) == 1


async def test_fault_on_first_attempt_only(calendar, clinic):
    cal = faulty(calendar, "book_appointment:timeout@1")
    with pytest.raises(ToolError):
        await cal.book(exam_request(clinic))
    booking = await cal.book(exam_request(clinic))
    assert booking.created is True
    assert cal.injector.attempts("book_appointment") == 2


async def test_faults_on_the_read_paths(calendar, clinic):
    cal = faulty(calendar, "check_availability:server_error,get_appointment:timeout")
    with pytest.raises(ToolError) as slots_err:
        await cal.find_slots("EXAM", OPEN_MONDAY, OPEN_MONDAY)
    with pytest.raises(ToolError) as read_err:
        await cal.get_booking(idempotency_key="anything")
    assert slots_err.value.code is read_err.value.code is ErrorCode.TRANSIENT


async def test_without_faults_the_wrapper_is_transparent(calendar, clinic):
    booking = await faulty(calendar, "").book(exam_request(clinic))
    assert booking.created is True
