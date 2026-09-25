"""Fault injection for any CalendarProvider, as a wrapper.

FaultyCalendar decorates a real provider: it forwards every call unless the
injector says this attempt should fail, and then fails the way a remote system
would. Because it wraps rather than branches, the real implementation carries
no test hooks, and the same wrapper will work unchanged around a live
practice-management adapter.

What each mode means for the calendar:

    timeout              raise TRANSIENT before calling through -- nothing happened
    timeout_after_write  call through (the write commits), then raise TRANSIENT --
                         the caller cannot tell this apart from `timeout`, which is
                         exactly why a timed-out write must be verified by reading
                         it back (Phase 8)
    conflict             raise CONFLICT without calling through
    server_error         raise TRANSIENT, as if the remote returned HTTP 500
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

from patient_ops.adapters.calendar.base import Booking, BookingRequest, CalendarProvider, Slot
from patient_ops.adapters.calendar.breaker import BreakerCalendar
from patient_ops.adapters.calendar.fake import FakeCalendar
from patient_ops.domain.availability import SchedulingPolicy
from patient_ops.errors import ErrorCode, ToolError
from patient_ops.faults import INJECTED, FaultInjector, FaultMode

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from patient_ops.config import Settings
    from patient_ops.degradation import DegradedModes
    from patient_ops.redis_layer.breaker import FailoverBreaker


def _injected(target: str, mode: FaultMode) -> ToolError:
    if mode is FaultMode.CONFLICT:
        return ToolError(ErrorCode.CONFLICT, f"{target}: slot taken (injected)", cause=INJECTED)
    if mode is FaultMode.SERVER_ERROR:
        return ToolError(ErrorCode.TRANSIENT, f"{target}: HTTP 500 (injected)", cause=INJECTED)
    return ToolError(ErrorCode.TRANSIENT, f"{target}: timed out (injected)", cause=INJECTED)


class FaultyCalendar(CalendarProvider):
    def __init__(self, inner: CalendarProvider, injector: FaultInjector) -> None:
        self.inner = inner
        self.injector = injector

    async def find_slots(
        self, procedure_code: str, date_from: date, date_to: date, provider_id: int | None = None
    ) -> list[Slot]:
        if mode := self.injector.next_fault("check_availability"):
            raise _injected("check_availability", mode)
        return await self.inner.find_slots(procedure_code, date_from, date_to, provider_id)

    async def book(self, request: BookingRequest) -> Booking:
        mode = self.injector.next_fault("book_appointment")
        if mode is FaultMode.TIMEOUT_AFTER_WRITE:
            await self.inner.book(request)  # the write really happens...
            raise _injected("book_appointment", mode)  # ...and the answer is lost
        if mode:
            raise _injected("book_appointment", mode)
        return await self.inner.book(request)

    async def get_booking(
        self, *, appointment_id: int | None = None, idempotency_key: str | None = None
    ) -> Booking | None:
        if mode := self.injector.next_fault("get_appointment"):
            raise _injected("get_appointment", mode)
        return await self.inner.get_booking(
            appointment_id=appointment_id, idempotency_key=idempotency_key
        )


def build_calendar(
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
    *,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    breaker: FailoverBreaker | None = None,
    degraded: DegradedModes | None = None,
    injector: FaultInjector | None = None,
) -> CalendarProvider:
    """The calendar the app should use: FakeCalendar, wrapped in the fault
    injector if FAULT_INJECT is set, and in the circuit breaker if one is given
    -- outermost, so injected failures are failures the breaker counts.

    Pass the process's `injector` when building one calendar per turn: its
    attempt counts are the scenario, and "the first booking times out" means
    the first of the process, not the first of every turn."""
    calendar: CalendarProvider = FakeCalendar(
        session_factory,
        tz=settings.clinic_tz,
        policy=SchedulingPolicy(
            step=timedelta(minutes=settings.slot_step_min),
            min_lead=timedelta(minutes=settings.booking_min_lead_min),
            max_horizon=timedelta(days=settings.booking_max_horizon_days),
        ),
        clock=clock,
    )
    injector = injector if injector is not None else FaultInjector(settings.fault_specs)
    if injector:
        calendar = FaultyCalendar(calendar, injector)
    if breaker is not None:
        calendar = BreakerCalendar(calendar, breaker, degraded=degraded)
    return calendar
