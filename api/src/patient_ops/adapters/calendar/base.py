"""The calendar boundary: the interface every scheduling backend implements.

In production this is a practice-management system (Dentrix, Open Dental, ...)
reached over HTTP. This project ships FakeCalendar, backed by Postgres, and a
fault-injecting wrapper around it: reproducible failures are worth more here
than the realism of a live integration.

Failures cross this boundary only as ToolError with an ErrorCode. Nothing above
it ever handles a driver or HTTP exception.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class Slot:
    provider_id: int
    procedure_code: str
    start_at: datetime
    end_at: datetime


@dataclass(frozen=True)
class BookingRequest:
    patient_id: int
    provider_id: int
    procedure_code: str
    start_at: datetime
    end_at: datetime
    # Identifies the request, not the slot. Sending the same request twice
    # (a double click, a retry after a timeout) must produce one appointment.
    idempotency_key: str
    source: str = "web"

    def __post_init__(self) -> None:
        # A naive datetime would be read in the database session's timezone --
        # silently wrong by hours. It is a programming error, so it fails loudly.
        if self.start_at.tzinfo is None or self.end_at.tzinfo is None:
            raise ValueError("booking times must be timezone-aware")


@dataclass(frozen=True)
class Booking:
    appointment_id: int
    external_ref: str | None
    patient_id: int
    provider_id: int
    procedure_code: str
    start_at: datetime
    end_at: datetime
    status: str
    # True only on the book() call that inserted the row. False means an
    # earlier request with the same idempotency key already did -- a replay.
    created: bool = False


class CalendarProvider(ABC):
    @abstractmethod
    async def find_slots(
        self, procedure_code: str, date_from: date, date_to: date, provider_id: int | None = None
    ) -> list[Slot]:
        """Bookable slots, inclusive of both clinic-local dates. Empty is not an error."""

    @abstractmethod
    async def book(self, request: BookingRequest) -> Booking:
        """Write the booking, or return the one this idempotency key already wrote.

        Raises ToolError: CONFLICT if the slot is taken, INVALID if the request
        is impossible, TRANSIENT if the backend is unreachable.
        """

    @abstractmethod
    async def get_booking(
        self, *, appointment_id: int | None = None, idempotency_key: str | None = None
    ) -> Booking | None:
        """Read a booking back. A separate call from book() on purpose: after a
        timeout, this read is the only way to learn whether the write happened."""
