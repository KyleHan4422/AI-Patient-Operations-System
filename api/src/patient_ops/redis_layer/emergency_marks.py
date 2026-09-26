"""R6: "this conversation reported an emergency at T", for the booking path.

An emergency turn (guardrails/emergency.py, graph/turn.py) ends any booking in
progress by writing to the checkpoint. If that write fails, the booking is
still open, and the patient's next "yes" -- said to "call 911", not to "Shall
I book it?" -- would book it. So the emergency turn also leaves this mark, in
a different store, and the booking path will not write a booking whose
read-back was asked before the mark: it asks again.

Advice, like the slot holds: with Redis down no mark is left or read, and the
checkpoint write is the only protection. Both failing at once is the gap that
remains.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from patient_ops import degradation
from patient_ops.degradation import DegradedModes
from patient_ops.errors import ErrorCode, ToolError
from patient_ops.redis_layer.client import Coordinator


def mark_key(thread_id: str) -> str:
    return f"g0:thread:{thread_id}"


class EmergencyMarks:
    def __init__(
        self,
        coordinator: Coordinator,
        *,
        ttl: timedelta = timedelta(hours=1),
        degraded: DegradedModes | None = None,
    ) -> None:
        self._coord = coordinator
        self._ttl_s = int(ttl.total_seconds())
        self.degraded = degraded if degraded is not None else DegradedModes()

    def _fallback(self, exc: ToolError) -> None:
        if exc.code is not ErrorCode.DEGRADED:
            raise exc
        self.degraded.note(degradation.EMERGENCY_MARK, exc.cause)

    async def mark(self, thread_id: str, at: datetime) -> None:
        try:
            await self._coord.run(
                "emergency_mark",
                lambda r: r.set(mark_key(thread_id), at.isoformat(), ex=self._ttl_s),
            )
        except ToolError as exc:
            self._fallback(exc)

    async def since(self, thread_id: str) -> datetime | None:
        """When this conversation last reported an emergency, if within the TTL."""
        try:
            value = await self._coord.run(
                "emergency_mark_get", lambda r: r.get(mark_key(thread_id))
            )
        except ToolError as exc:
            self._fallback(exc)
            return None
        if value is None:
            return None
        return datetime.fromisoformat(value.decode() if isinstance(value, bytes) else value)
