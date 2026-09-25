"""Read-only queries over the system of record.

Every function here reads; none writes. The write paths -- FakeCalendar.book(),
db/transcript.py and scripts/seed.py -- all live elsewhere.
Reads and writes are separated at the module level so that the architectural
check can reason about imports rather than about method names.

Rows that feed pure domain logic are returned as domain objects
(ScheduleWindow, Interval), so domain/ never sees the ORM.
"""

from __future__ import annotations

import uuid
from collections.abc import Collection
from datetime import date, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from patient_ops.db.models import (
    Appointment,
    ClinicClosure,
    Conversation,
    InsurancePlan,
    Message,
    Patient,
    Procedure,
    Provider,
    ProviderSchedule,
)
from patient_ops.domain.availability import Interval, ScheduleWindow
from patient_ops.domain.phone import normalize_phone


def _normalize_name(name: str) -> str:
    return " ".join(name.split()).lower()


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------
async def get_procedure(session: AsyncSession, code: str) -> Procedure | None:
    return await session.get(Procedure, code)


async def list_procedures(session: AsyncSession) -> list[Procedure]:
    return list((await session.scalars(select(Procedure).order_by(Procedure.code))).all())


async def find_procedure(session: AsyncSession, term: str) -> Procedure | None:
    """A treatment by its code ("CROWN") or by its exact name ("Crown").

    Exact on both, ignoring case and spacing, for the same reason as
    find_insurance_plan: fuzzy matching is what turns "crown" into "crown
    lengthening" and quotes the wrong price. The agent is given the list of
    codes in its prompt, so it has no need to guess at one.
    """
    normalized = _normalize_name(term)
    stmt = select(Procedure).where(
        (func.lower(Procedure.code) == normalized) | (func.lower(Procedure.name) == normalized)
    )
    return await session.scalar(stmt.order_by(Procedure.code).limit(1))


async def find_insurance_plan(session: AsyncSession, plan_name: str) -> InsurancePlan | None:
    """Look up a plan by exact name, ignoring case and spacing.

    Returns None when the plan is not on file. That is a different answer from
    a row with accepted=False: "we don't know" versus "we don't take it". The
    agent tool built on this in Phase 4 must say "unknown", never "no".
    Deliberately not fuzzy -- see InsurancePlan.
    """
    stmt = select(InsurancePlan).where(
        func.lower(InsurancePlan.plan_name) == _normalize_name(plan_name)
    )
    return await session.scalar(stmt)


# ---------------------------------------------------------------------------
# Patients
# ---------------------------------------------------------------------------
async def find_patients(
    session: AsyncSession,
    *,
    phone: str | None = None,
    name: str | None = None,
    dob: date | None = None,
) -> list[Patient]:
    """Patients matching every criterion given.

    Zero, one or several. Several is not an error: it is the signal to ask a
    follow-up question ("what's your date of birth?"). An unparseable phone
    number raises ValueError, so the caller can ask for it again.
    """
    if phone is None and name is None and dob is None:
        raise ValueError("find_patients needs at least one criterion")
    stmt = select(Patient)
    if phone is not None:
        stmt = stmt.where(Patient.phone == normalize_phone(phone))
    if name is not None:
        stmt = stmt.where(func.lower(Patient.full_name) == _normalize_name(name))
    if dob is not None:
        stmt = stmt.where(Patient.dob == dob)
    return list((await session.scalars(stmt.order_by(Patient.id))).all())


# ---------------------------------------------------------------------------
# Scheduling inputs -- returned as domain objects
# ---------------------------------------------------------------------------
async def get_provider(session: AsyncSession, provider_id: int) -> Provider | None:
    return await session.get(Provider, provider_id)


async def get_providers(session: AsyncSession, provider_ids: Collection[int]) -> list[Provider]:
    if not provider_ids:
        return []
    stmt = select(Provider).where(Provider.id.in_(provider_ids)).order_by(Provider.id)
    return list((await session.scalars(stmt)).all())


async def providers_for_specialty(session: AsyncSession, specialty: str) -> list[Provider]:
    stmt = select(Provider).where(Provider.specialty == specialty).order_by(Provider.id)
    return list((await session.scalars(stmt)).all())


async def schedules_for(
    session: AsyncSession, provider_ids: Collection[int]
) -> list[ScheduleWindow]:
    stmt = select(ProviderSchedule).where(ProviderSchedule.provider_id.in_(provider_ids))
    return [
        ScheduleWindow(r.provider_id, r.weekday, r.start_time, r.end_time)
        for r in (await session.scalars(stmt)).all()
    ]


async def all_schedules(session: AsyncSession) -> list[ScheduleWindow]:
    """Every provider's working windows -- the clinic's opening hours are their union."""
    stmt = select(ProviderSchedule).order_by(ProviderSchedule.weekday, ProviderSchedule.start_time)
    return [
        ScheduleWindow(r.provider_id, r.weekday, r.start_time, r.end_time)
        for r in (await session.scalars(stmt)).all()
    ]


async def closures_with_reasons(
    session: AsyncSession, first: date, last: date
) -> list[ClinicClosure]:
    """Closures in a window, with why -- closures_between returns only the dates.

    Slot computation needs a set of dates and nothing else; telling a patient
    the clinic is shut needs the reason, or the answer sounds evasive.
    """
    stmt = (
        select(ClinicClosure)
        .where(ClinicClosure.closed_on.between(first, last))
        .order_by(ClinicClosure.closed_on)
    )
    return list((await session.scalars(stmt)).all())


async def closures_between(session: AsyncSession, first: date, last: date) -> set[date]:
    stmt = select(ClinicClosure.closed_on).where(ClinicClosure.closed_on.between(first, last))
    return set((await session.scalars(stmt)).all())


async def booked_intervals(
    session: AsyncSession, provider_ids: Collection[int], start: datetime, end: datetime
) -> list[tuple[int, Interval]]:
    """Live bookings of these providers that overlap [start, end).

    Written with the same `provider_id =` / `tstzrange &&` shape as the
    no_overlap constraint, so the GiST index behind that constraint serves this
    query too.
    """
    period = func.tstzrange(Appointment.start_at, Appointment.end_at)
    stmt = select(Appointment.provider_id, Appointment.start_at, Appointment.end_at).where(
        Appointment.status == "booked",
        Appointment.provider_id.in_(provider_ids),
        period.op("&&")(func.tstzrange(start, end)),
    )
    return [(pid, Interval(s, e)) for pid, s, e in (await session.execute(stmt)).all()]


# ---------------------------------------------------------------------------
# Appointments
# ---------------------------------------------------------------------------
async def get_appointment(
    session: AsyncSession, *, appointment_id: int | None = None, idempotency_key: str | None = None
) -> Appointment | None:
    """The verification read: look a booking up by id or by the key it was written with."""
    if (appointment_id is None) == (idempotency_key is None):
        raise ValueError("pass exactly one of appointment_id or idempotency_key")
    if appointment_id is not None:
        return await session.get(Appointment, appointment_id)
    return await session.scalar(
        select(Appointment).where(Appointment.idempotency_key == idempotency_key)
    )


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------
async def get_transcript(session: AsyncSession, thread_id: uuid.UUID) -> list[Message] | None:
    """Every message of one conversation, oldest first; None if there is none."""
    conversation_id = await session.scalar(
        select(Conversation.id).where(Conversation.thread_id == thread_id)
    )
    if conversation_id is None:
        return None
    stmt = select(Message).where(Message.conversation_id == conversation_id).order_by(Message.id)
    return list((await session.scalars(stmt)).all())
