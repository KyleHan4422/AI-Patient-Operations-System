"""The correctness guarantees of the booking write path.

Everything here runs against real Postgres, because the guarantees *are*
Postgres: the no_overlap EXCLUDE constraint and the UNIQUE idempotency key.
Several tests go around the application entirely to show the database holds
the line on its own.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from patient_ops.adapters.calendar.base import Booking, BookingRequest
from patient_ops.adapters.calendar.fake import FakeCalendar
from patient_ops.db import repo
from patient_ops.db.errors import CHECK_VIOLATION, EXCLUSION_VIOLATION
from patient_ops.errors import ErrorCode, ToolError
from tests.factories import CLOSED_MONDAY, OPEN_MONDAY, Clinic, booked_count, local


def exam(clinic: Clinic, start: datetime, end: datetime, key: str, **overrides) -> BookingRequest:
    fields = dict(
        patient_id=clinic.patient_id,
        provider_id=clinic.dentist_id,
        procedure_code="EXAM",
        start_at=start,
        end_at=end,
        idempotency_key=key,
    )
    return BookingRequest(**(fields | overrides))


async def outcome(calendar: FakeCalendar, request: BookingRequest) -> Booking | ToolError:
    try:
        return await calendar.book(request)
    except ToolError as err:
        return err


TEN, TEN_THIRTY, ELEVEN = (
    local(OPEN_MONDAY, 10),
    local(OPEN_MONDAY, 10, 30),
    local(OPEN_MONDAY, 11),
)


# ---------------------------------------------------------------------------
# ★ Acceptance: concurrent requests for one slot -> exactly one wins
# ---------------------------------------------------------------------------
async def test_ten_concurrent_bookings_for_one_slot_yield_exactly_one(
    calendar, clinic, session_factory
):
    results = await asyncio.gather(
        *(outcome(calendar, exam(clinic, TEN, TEN_THIRTY, key=f"patient-{i}")) for i in range(10))
    )

    winners = [r for r in results if isinstance(r, Booking)]
    losers = [r for r in results if isinstance(r, ToolError)]
    assert len(winners) == 1
    assert len(losers) == 9
    assert {err.code for err in losers} == {ErrorCode.CONFLICT}
    assert {err.cause for err in losers} == {"ExclusionViolation"}  # the mapping is auditable
    assert await booked_count(session_factory) == 1


async def test_check_then_insert_has_a_race_that_the_constraint_closes(clinic, session_factory):
    """Why application-level checking is not enough, step by step.

    Two requests each ask "is 10:00 free?", both hear "yes", both insert.
    Nothing in the application can stop the second insert -- only the database.
    """
    async with session_factory() as a, session_factory() as b:
        for session in (a, b):  # 1. both check -- and both see a free slot
            free = not await repo.booked_intervals(session, [clinic.dentist_id], TEN, TEN_THIRTY)
            assert free

        insert = text(
            "INSERT INTO appointments (patient_id, provider_id, procedure_code, start_at, end_at,"
            " source, idempotency_key) VALUES (:p, :d, 'EXAM', :s, :e, 'web', :k)"
        )
        args = dict(p=clinic.patient_id, d=clinic.dentist_id, s=TEN, e=TEN_THIRTY)
        await a.execute(insert, args | {"k": "a"})  # 2. the first one writes
        await a.commit()

        with pytest.raises(IntegrityError) as exc:  # 3. the second is refused by Postgres
            await b.execute(insert, args | {"k": "b"})
        assert exc.value.orig.sqlstate == EXCLUSION_VIOLATION


# ---------------------------------------------------------------------------
# ★ Acceptance: one idempotency key -> one appointment
# ---------------------------------------------------------------------------
async def test_same_key_twice_returns_the_first_booking(calendar, clinic, session_factory):
    request = exam(clinic, TEN, TEN_THIRTY, key="double-click")
    first = await calendar.book(request)
    second = await calendar.book(request)

    assert first.created is True
    assert second.created is False  # a replay, not a new row
    assert second.appointment_id == first.appointment_id
    assert second.external_ref == first.external_ref
    assert await booked_count(session_factory) == 1


async def test_concurrent_requests_with_one_key_create_one_row(calendar, clinic, session_factory):
    request = exam(clinic, TEN, TEN_THIRTY, key="retried-five-times")
    results = await asyncio.gather(*(calendar.book(request) for _ in range(5)))

    assert len({r.appointment_id for r in results}) == 1
    assert sum(r.created for r in results) == 1
    assert await booked_count(session_factory) == 1


async def test_same_key_with_different_parameters_is_rejected(calendar, clinic):
    await calendar.book(exam(clinic, TEN, TEN_THIRTY, key="reused"))
    with pytest.raises(ToolError) as exc:
        await calendar.book(exam(clinic, TEN_THIRTY, ELEVEN, key="reused"))
    assert exc.value.code is ErrorCode.INVALID


# ---------------------------------------------------------------------------
# What counts as a conflict
# ---------------------------------------------------------------------------
async def test_partial_overlap_is_a_conflict(calendar, clinic):
    await calendar.book(exam(clinic, TEN, ELEVEN, key="first"))
    with pytest.raises(ToolError) as exc:
        await calendar.book(exam(clinic, TEN_THIRTY, local(OPEN_MONDAY, 11, 30), key="second"))
    assert exc.value.code is ErrorCode.CONFLICT


async def test_back_to_back_appointments_do_not_conflict(calendar, clinic, session_factory):
    await calendar.book(exam(clinic, TEN, ELEVEN, key="first"))
    await calendar.book(exam(clinic, ELEVEN, local(OPEN_MONDAY, 11, 30), key="second"))  # [) ranges
    assert await booked_count(session_factory) == 2


async def test_different_providers_may_book_the_same_time(calendar, clinic, session_factory):
    await calendar.book(exam(clinic, TEN, ELEVEN, key="dentist"))
    await calendar.book(
        exam(
            clinic,
            TEN,
            ELEVEN,
            key="hygienist",
            provider_id=clinic.hygienist_id,
            procedure_code="CLEANING",
        )
    )
    assert await booked_count(session_factory) == 2


async def test_a_cancelled_appointment_frees_its_slot(calendar, clinic, session_factory):
    first = await calendar.book(exam(clinic, TEN, ELEVEN, key="first"))
    async with session_factory() as session, session.begin():  # staff cancels it
        await session.execute(
            text("UPDATE appointments SET status = 'cancelled' WHERE id = :id"),
            {"id": first.appointment_id},
        )
    second = await calendar.book(exam(clinic, TEN, ELEVEN, key="second"))
    assert second.created is True


# ---------------------------------------------------------------------------
# The database refuses bad rows even when the application is bypassed
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("start_hour", "end_hour", "status", "sqlstate"),
    [
        (10, 11, "booked", EXCLUSION_VIOLATION),  # overlaps the existing 10:00-11:00
        (14, 13, "booked", CHECK_VIOLATION),  # ends before it starts
        (14, 15, "pending", CHECK_VIOLATION),  # not a known status
    ],
)
async def test_database_rejects_bad_rows_written_directly(
    calendar, clinic, session_factory, start_hour, end_hour, status, sqlstate
):
    await calendar.book(exam(clinic, TEN, ELEVEN, key="existing"))
    async with session_factory() as session:
        with pytest.raises(IntegrityError) as exc:
            await session.execute(
                text(
                    "INSERT INTO appointments (patient_id, provider_id, procedure_code, start_at,"
                    " end_at, status, source, idempotency_key)"
                    " VALUES (:p, :d, 'EXAM', :s, :e, :status, 'staff', 'direct')"
                ),
                dict(
                    p=clinic.patient_id,
                    d=clinic.dentist_id,
                    s=local(OPEN_MONDAY, start_hour),
                    e=local(OPEN_MONDAY, end_hour),
                    status=status,
                ),
            )
        assert exc.value.orig.sqlstate == sqlstate


# ---------------------------------------------------------------------------
# Requests the calendar refuses before writing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("start", "end"),
    [
        (local(OPEN_MONDAY, 11, 30), local(OPEN_MONDAY, 12, 30)),  # runs into lunch
        (local(OPEN_MONDAY, 7), local(OPEN_MONDAY, 7, 30)),  # before opening
        (local(CLOSED_MONDAY, 10), local(CLOSED_MONDAY, 10, 30)),  # clinic closed
    ],
)
async def test_bookings_outside_working_hours_are_invalid(
    calendar, clinic, session_factory, start, end
):
    with pytest.raises(ToolError) as exc:
        await calendar.book(exam(clinic, start, end, key="off-hours"))
    assert exc.value.code is ErrorCode.INVALID
    assert await booked_count(session_factory) == 0


async def test_a_provider_cannot_be_booked_for_work_they_do_not_do(
    calendar, clinic, session_factory
):
    """A crown with the hygienist: find_slots never offers it, and book() refuses it too."""
    crown_with_hygienist = exam(
        clinic,
        TEN,
        local(OPEN_MONDAY, 11, 30),
        key="wrong-chair",
        provider_id=clinic.hygienist_id,
        procedure_code="CROWN",
    )
    with pytest.raises(ToolError) as exc:
        await calendar.book(crown_with_hygienist)
    assert exc.value.code is ErrorCode.INVALID
    assert "does not perform CROWN" in exc.value.detail
    assert await booked_count(session_factory) == 0


@pytest.mark.parametrize("overrides", [{"provider_id": 999_999}, {"procedure_code": "NOPE"}])
async def test_unknown_provider_or_procedure_is_invalid(calendar, clinic, overrides):
    with pytest.raises(ToolError) as exc:
        await calendar.book(exam(clinic, TEN, TEN_THIRTY, key="unknown", **overrides))
    assert exc.value.code is ErrorCode.INVALID


async def test_unknown_patient_is_invalid(calendar, clinic):
    with pytest.raises(ToolError) as exc:
        await calendar.book(exam(clinic, TEN, TEN_THIRTY, key="ghost", patient_id=999_999))
    assert exc.value.code is ErrorCode.INVALID
    assert exc.value.cause == "ForeignKeyViolation"


def test_naive_datetimes_never_reach_the_database():
    with pytest.raises(ValueError, match="timezone-aware"):
        BookingRequest(
            patient_id=1,
            provider_id=1,
            procedure_code="EXAM",
            start_at=datetime(2026, 10, 5, 10, 0),
            end_at=datetime(2026, 10, 5, 10, 30),
            idempotency_key="naive",
        )
