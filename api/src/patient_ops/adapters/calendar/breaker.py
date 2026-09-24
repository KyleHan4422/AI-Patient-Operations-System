"""A circuit breaker around any CalendarProvider, as a wrapper.

Same shape as FaultyCalendar: it decorates a provider and changes nothing
about it. Wrap the fault injector, not the other way round --

    BreakerCalendar(FaultyCalendar(FakeCalendar))

-- so injected timeouts are failures the breaker sees, and a test can open it.

What counts as a failure is the decision that matters here. Only TRANSIENT and
UNKNOWN say "the calendar is unwell". CONFLICT (the slot is taken) and INVALID
(the request was wrong) are the calendar working correctly; counting them would
let one popular slot, or one confused patient, switch booking off for everyone.

Per turn: it carries that turn's DegradedModes. The breaker it wraps is
long-lived and shared.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import date
from typing import TYPE_CHECKING, TypeVar

from patient_ops.adapters.calendar.base import Booking, BookingRequest, CalendarProvider, Slot
from patient_ops.errors import ErrorCode, ToolError
from patient_ops.redis_layer.breaker import BreakerEvent

if TYPE_CHECKING:
    from patient_ops.degradation import DegradedModes
    from patient_ops.redis_layer.breaker import FailoverBreaker

T = TypeVar("T")

CIRCUIT_OPEN = "CircuitOpen"  # ToolError.cause when the breaker refused the call
UNWELL = frozenset({ErrorCode.TRANSIENT, ErrorCode.UNKNOWN})


class BreakerCalendar(CalendarProvider):
    def __init__(
        self,
        inner: CalendarProvider,
        breaker: FailoverBreaker,
        *,
        degraded: DegradedModes | None = None,
    ) -> None:
        self.inner = inner
        self.breaker = breaker
        self.degraded = degraded

    async def _guarded(self, op: str, call: Callable[[], Awaitable[T]]) -> T:
        permit = await self.breaker.allow(self.degraded)
        if permit is None:
            # TRANSIENT, not a new code: to everything above, "the calendar is
            # not answering right now" is the same fact whether it timed out or
            # we declined to ask. Only `cause` tells them apart, for the logs.
            raise ToolError(ErrorCode.TRANSIENT, f"{op}: calendar circuit open", cause=CIRCUIT_OPEN)
        try:
            result = await call()
        except ToolError as exc:
            event = BreakerEvent.FAILURE if exc.code in UNWELL else BreakerEvent.SUCCESS
            await self.breaker.record(permit, event, self.degraded)
            raise
        except asyncio.CancelledError:
            # The client went away mid-call: no verdict on the calendar. If
            # this call was the half-open probe, give the probe back now
            # rather than make every worker wait out another cooldown.
            await asyncio.shield(self.breaker.record(permit, BreakerEvent.ABANDON, self.degraded))
            raise
        except Exception:
            # Anything the adapter failed to classify is UNKNOWN, and UNKNOWN
            # counts as unwell -- an unclassified failure must not be free.
            await self.breaker.record(permit, BreakerEvent.FAILURE, self.degraded)
            raise
        await self.breaker.record(permit, BreakerEvent.SUCCESS, self.degraded)
        return result

    async def find_slots(
        self, procedure_code: str, date_from: date, date_to: date, provider_id: int | None = None
    ) -> list[Slot]:
        return await self._guarded(
            "check_availability",
            lambda: self.inner.find_slots(procedure_code, date_from, date_to, provider_id),
        )

    async def book(self, request: BookingRequest) -> Booking:
        return await self._guarded("book_appointment", lambda: self.inner.book(request))

    async def get_booking(
        self, *, appointment_id: int | None = None, idempotency_key: str | None = None
    ) -> Booking | None:
        # Guarded too. After a timed-out write this read is what verification
        # depends on (Phase 8), and against an open breaker it would only time
        # out again -- failing fast is the honest answer.
        return await self._guarded(
            "get_appointment",
            lambda: self.inner.get_booking(
                appointment_id=appointment_id, idempotency_key=idempotency_key
            ),
        )
