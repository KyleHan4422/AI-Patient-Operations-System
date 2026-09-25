"""The booking desk: everything the booking path reaches, and the record of it.

The write-side counterpart of tools/registry.py. One object per turn, handed to
the deterministic booking nodes through the graph's runtime context -- never to
an agent: scripts/check_invariants.py fails the build if anything under
agents/ imports this module. Here is where a slot is held, where an appointment
is written and where it is read back, so here is exactly what a language model
must not be able to reach.

Every coordination feature from Phase 5 is used here, and each degrades the
way the README's table says:

    holds     a slot offered to this conversation is held for HOLD_TTL_S
    breaker   the calendar is wrapped in it (adapters/calendar/breaker.py)
    dedup     two identical bookings in flight make one calendar call

The calls the turn made go into `trace`, in the same shape as the read-only
toolset's, so `respond` writes them to tool_calls with everything else.
"""

from __future__ import annotations

import hashlib
import time as clock
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any, TypeVar
from zoneinfo import ZoneInfo

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from patient_ops.adapters.calendar.base import Booking, BookingRequest, CalendarProvider, Slot
from patient_ops.db import repo
from patient_ops.errors import ToolError
from patient_ops.obs.logging import get_logger
from patient_ops.redis_layer.holds import HoldResult, HoldStatus, SlotHolds
from patient_ops.redis_layer.idempotency import InFlightDedup
from patient_ops.tools.registry import ToolInvocation

log = get_logger(__name__)

T = TypeVar("T")

FIND_PATIENT = "find_patient"
CHECK_AVAILABILITY = "check_availability"
BOOK_APPOINTMENT = "book_appointment"
GET_APPOINTMENT = "get_appointment"


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class PatientMatch:
    patient_id: int
    full_name: str


def idempotency_key(thread_id: str, patient_id: int, slot: Slot) -> str:
    """The key for "this conversation books this slot for this patient".

    Derived, not random: a patient who says "yes" twice, or a client that
    retries after a timeout, sends the same key and gets the same appointment.
    A different slot is a different request and gets a different key.
    """
    raw = "|".join(
        (
            thread_id,
            str(patient_id),
            str(slot.provider_id),
            slot.procedure_code,
            slot.start_at.astimezone(UTC).isoformat(),
            slot.end_at.astimezone(UTC).isoformat(),
        )
    )
    return "chat-" + hashlib.sha256(raw.encode()).hexdigest()[:32]


def _masked_phone(phone: str | None) -> str | None:
    # tool_calls rows are read by people on the /ops view; the last four
    # digits are enough to tell two lookups apart.
    if phone is None:
        return None
    digits = "".join(c for c in phone if c.isdigit())
    return f"***{digits[-4:]}" if digits else "***"


@dataclass
class BookingDesk:
    session_factory: async_sessionmaker[AsyncSession]
    calendar: CalendarProvider  # already wrapped in the breaker
    holds: SlotHolds
    dedup: InFlightDedup
    tz: ZoneInfo
    now: Callable[[], datetime] = _utc_now
    trace: list[ToolInvocation] = field(default_factory=list)
    _catalogue: list[tuple[str, str]] | None = field(default=None, init=False, repr=False)

    def today(self) -> date:
        return self.now().astimezone(self.tz).date()

    # -- reads ---------------------------------------------------------------
    async def procedure_catalogue(self) -> list[tuple[str, str]]:
        """(code, name) of every treatment. Read once per turn: the desk is."""
        if self._catalogue is None:
            async with self.session_factory() as session:
                self._catalogue = [(p.code, p.name) for p in await repo.list_procedures(session)]
        return self._catalogue

    async def provider_names(self, provider_ids: set[int]) -> dict[int, str]:
        async with self.session_factory() as session:
            return {p.id: p.name for p in await repo.get_providers(session, provider_ids)}

    async def find_patients(
        self, *, phone: str | None = None, name: str | None = None, dob: date | None = None
    ) -> list[PatientMatch]:
        """Zero, one or several patients. Raises ValueError on an unreadable phone."""

        async def call() -> list[PatientMatch]:
            async with self.session_factory() as session:
                rows = await repo.find_patients(session, phone=phone, name=name, dob=dob)
            return [PatientMatch(r.id, r.full_name) for r in rows]

        args = {
            "phone": _masked_phone(phone),
            "name": name,
            "dob": dob.isoformat() if dob else None,
        }
        return await self._traced(
            FIND_PATIENT,
            {k: v for k, v in args.items() if v is not None},
            call,
            lambda found: f"{len(found)} match(es)",
        )

    async def find_slots(self, procedure_code: str, date_from: date, date_to: date) -> list[Slot]:
        return await self._traced(
            CHECK_AVAILABILITY,
            {"procedure": procedure_code, "from": date_from.isoformat(), "to": date_to.isoformat()},
            lambda: self.calendar.find_slots(procedure_code, date_from, date_to),
            lambda slots: f"{len(slots)} slot(s)",
        )

    async def read_back(self, key: str) -> Booking | None:
        """The appointment this key wrote, from the system of record."""
        return await self._traced(
            GET_APPOINTMENT,
            {"idempotency_key": key},
            lambda: self.calendar.get_booking(idempotency_key=key),
            lambda row: "not found" if row is None else f"appointment {row.appointment_id}",
        )

    # -- holds: advice about who was offered what, never the guarantee --------
    async def hold(self, slot: Slot, owner: str) -> HoldResult:
        return await self.holds.place(slot, owner)

    async def hold_status(self, slot: Slot, owner: str) -> HoldStatus:
        return await self.holds.check(slot, owner)

    async def release_holds(self, owner: str, *, keep: Slot | None = None) -> None:
        released = await self.holds.release_all(owner, keep=keep)
        if released:
            log.info("holds_released", owner=owner, count=released)

    # -- the write ------------------------------------------------------------
    async def book(self, request: BookingRequest) -> Booking:
        """Write the appointment once, however many identical requests arrive.

        The dedup layer only saves a duplicate from making its own calendar
        call; the calendar's UNIQUE idempotency key is what guarantees one row.
        """
        key = request.idempotency_key
        return await self._traced(
            BOOK_APPOINTMENT,
            {
                "idempotency_key": key,
                "provider_id": request.provider_id,
                "procedure": request.procedure_code,
                "start_at": request.start_at.isoformat(),
            },
            lambda: self.dedup.run_once(
                key,
                lambda: self.calendar.book(request),
                read_back=lambda: self.calendar.get_booking(idempotency_key=key),
                ref=lambda booking: booking.appointment_id,
            ),
            lambda booking: (
                f"appointment {booking.appointment_id}"
                + ("" if booking.created else " (already written)")
            ),
        )

    # -- the one place a call is timed, traced and logged ----------------------
    async def _traced(
        self,
        name: str,
        args: dict[str, Any],
        call: Callable[[], Awaitable[T]],
        summarize: Callable[[T], str],
    ) -> T:
        started = clock.perf_counter()
        try:
            result = await call()
        except ToolError as exc:
            # A classified failure is part of the booking path now -- a taken
            # slot is re-offered, an open breaker is explained -- so it gets
            # its row like any other call.
            self._record(name, args, started, f"error: {exc.code}")
            raise
        self._record(name, args, started, summarize(result))
        return result

    def _record(self, name: str, args: dict[str, Any], started: float, summary: str) -> None:
        invocation = ToolInvocation(
            name=name,
            args=args,
            latency_ms=round((clock.perf_counter() - started) * 1000),
            summary=summary,
        )
        self.trace.append(invocation)
        log.info(
            "tool_call",
            tool=name,
            args=args,
            latency_ms=invocation.latency_ms,
            summary=summary,
        )
