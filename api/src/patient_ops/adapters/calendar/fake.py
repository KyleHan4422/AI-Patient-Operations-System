"""FakeCalendar: a scheduling backend backed by our own Postgres.

It behaves like an external practice-management system would -- it validates
what it is asked to write, rejects conflicts, hands back its own reference --
but its failures come only from the database. Injected failures are layered on
from outside (faults.py), so this file contains no test hooks at all.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from patient_ops.adapters.calendar.base import Booking, BookingRequest, CalendarProvider, Slot
from patient_ops.db import repo
from patient_ops.db.errors import translate_db_errors
from patient_ops.db.models import Appointment
from patient_ops.domain.availability import (
    Interval,
    SchedulingPolicy,
    compute_slots,
    fits_schedule,
)
from patient_ops.errors import ErrorCode, ToolError

MAX_SEARCH_DAYS = 31  # bounds the work one availability query can ask for


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _to_booking(row: Appointment, *, created: bool) -> Booking:
    return Booking(
        appointment_id=row.id,
        external_ref=row.external_ref,
        patient_id=row.patient_id,
        provider_id=row.provider_id,
        procedure_code=row.procedure_code,
        start_at=row.start_at,
        end_at=row.end_at,
        status=row.status,
        created=created,
    )


def _same_request(row: Appointment, req: BookingRequest) -> bool:
    return (row.patient_id, row.provider_id, row.procedure_code, row.start_at, row.end_at) == (
        req.patient_id,
        req.provider_id,
        req.procedure_code,
        req.start_at,
        req.end_at,
    )


class FakeCalendar(CalendarProvider):
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        tz: ZoneInfo,
        policy: SchedulingPolicy,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._sessions = session_factory
        self._tz = tz
        self._policy = policy
        self._clock = clock  # injected so availability is testable at a fixed "now"

    # -----------------------------------------------------------------------
    async def find_slots(
        self, procedure_code: str, date_from: date, date_to: date, provider_id: int | None = None
    ) -> list[Slot]:
        if date_to < date_from:
            raise ToolError(ErrorCode.INVALID, "date_to is before date_from")
        if (date_to - date_from).days >= MAX_SEARCH_DAYS:
            raise ToolError(ErrorCode.INVALID, f"search at most {MAX_SEARCH_DAYS} days at a time")

        with translate_db_errors():
            async with self._sessions() as session:
                procedure = await repo.get_procedure(session, procedure_code)
                if procedure is None:
                    raise ToolError(ErrorCode.INVALID, f"unknown procedure {procedure_code!r}")

                qualified = await repo.providers_for_specialty(session, procedure.specialty)
                provider_ids = [p.id for p in qualified if provider_id in (None, p.id)]
                if provider_id is not None and not provider_ids:
                    raise ToolError(
                        ErrorCode.INVALID,
                        f"provider {provider_id} does not perform {procedure_code}",
                    )

                schedules = await repo.schedules_for(session, provider_ids)
                closures = await repo.closures_between(session, date_from, date_to)
                booked = await repo.booked_intervals(
                    session,
                    provider_ids,
                    datetime.combine(date_from, time.min, tzinfo=self._tz),
                    datetime.combine(date_to + timedelta(days=1), time.min, tzinfo=self._tz),
                )

        candidates = compute_slots(
            schedules=schedules,
            closures=closures,
            booked=booked,
            duration=timedelta(minutes=procedure.duration_min),
            date_from=date_from,
            date_to=date_to,
            tz=self._tz,
            now=self._clock(),
            policy=self._policy,
        )
        return [Slot(c.provider_id, procedure_code, c.start, c.end) for c in candidates]

    # -----------------------------------------------------------------------
    async def book(self, request: BookingRequest) -> Booking:
        try:
            return await self._insert(request)
        except ToolError as err:
            if err.code is not ErrorCode.CONFLICT:
                raise
            # The row we collided with may be our own: an earlier attempt of
            # this same request that committed while this one was in flight.
            # A conflict with yourself is a replay, not a conflict.
            replay = await self._replay(request)
            if replay is None:
                raise
            return replay

    async def _insert(self, req: BookingRequest) -> Booking:
        interval = Interval(req.start_at, req.end_at)
        local_day = req.start_at.astimezone(self._tz).date()

        with translate_db_errors():
            async with self._sessions() as session, session.begin():
                # Like a real backend, it does not trust the caller: the
                # provider must perform the procedure, and the time must be
                # inside their working hours. (Whether the length matches the
                # procedure's duration is booking policy -- guardrail G5, in
                # guardrails/booking_policy.py -- since staff may legitimately
                # book a longer visit.)
                procedure = await repo.get_procedure(session, req.procedure_code)
                provider = await repo.get_provider(session, req.provider_id)
                if procedure is None or provider is None:
                    raise ToolError(ErrorCode.INVALID, "unknown provider or procedure")
                if provider.specialty != procedure.specialty:
                    raise ToolError(
                        ErrorCode.INVALID, f"{provider.name} does not perform {procedure.code}"
                    )

                schedules = await repo.schedules_for(session, [req.provider_id])
                closures = await repo.closures_between(session, local_day, local_day)
                if not fits_schedule(req.provider_id, interval, schedules, closures, self._tz):
                    raise ToolError(ErrorCode.INVALID, "outside the provider's working hours")

                # ON CONFLICT names its target on purpose. A bare
                # `ON CONFLICT DO NOTHING` would also swallow the no_overlap
                # violation: a double booking would silently insert nothing,
                # and the lookup below would find nothing either.
                stmt = (
                    pg_insert(Appointment)
                    .values(
                        patient_id=req.patient_id,
                        provider_id=req.provider_id,
                        procedure_code=req.procedure_code,
                        start_at=req.start_at,
                        end_at=req.end_at,
                        source=req.source,
                        idempotency_key=req.idempotency_key,
                        external_ref=f"FC-{secrets.token_hex(4).upper()}",
                    )
                    .on_conflict_do_nothing(index_elements=[Appointment.idempotency_key])
                    .returning(Appointment)
                )
                inserted = (await session.scalars(stmt)).one_or_none()
                if inserted is not None:
                    return _to_booking(inserted, created=True)

                # Nothing inserted: this key was already used. A fresh
                # statement under READ COMMITTED sees the committed row.
                existing = await repo.get_appointment(session, idempotency_key=req.idempotency_key)
                if existing is None:
                    raise ToolError(ErrorCode.UNKNOWN, "idempotency conflict but no row found")
                return self._check_replay(existing, req)

    async def _replay(self, req: BookingRequest) -> Booking | None:
        with translate_db_errors():
            async with self._sessions() as session:
                existing = await repo.get_appointment(session, idempotency_key=req.idempotency_key)
        return None if existing is None else self._check_replay(existing, req)

    @staticmethod
    def _check_replay(existing: Appointment, req: BookingRequest) -> Booking:
        # Same key, different request: not a retry but a bug in the caller.
        # Returning the old booking would hide it.
        if not _same_request(existing, req):
            raise ToolError(
                ErrorCode.INVALID, "idempotency key reused with different booking parameters"
            )
        return _to_booking(existing, created=False)

    # -----------------------------------------------------------------------
    async def get_booking(
        self, *, appointment_id: int | None = None, idempotency_key: str | None = None
    ) -> Booking | None:
        with translate_db_errors():
            async with self._sessions() as session:
                row = await repo.get_appointment(
                    session, appointment_id=appointment_id, idempotency_key=idempotency_key
                )
        return None if row is None else _to_booking(row, created=False)
