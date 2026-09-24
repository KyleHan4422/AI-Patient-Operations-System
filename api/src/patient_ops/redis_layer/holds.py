"""R1 -- soft holds on offered slots.

Between "here are three times" and "the second one, please" a patient takes
thirty seconds to two minutes, and in that window the slot can go to someone
else. That window cannot be closed -- nothing is booked yet -- but it can be
made rarer: each offered slot gets a hold that says "offered to this
conversation", other conversations are not offered it, and the hold
disappears by itself when its TTL runs out.

That last part is why this is Redis and not a Postgres table. A hold table
needs a sweeper, and an advisory lock does not expire at all.

A hold is advice, never the guarantee. Whether two bookings can overlap is
decided by the no_overlap EXCLUDE constraint, at write time. So when Redis is
not there, `place` answers DEGRADED -- offer the slot anyway, and let the
constraint settle a collision -- and a turn that could not hold says so in
its degraded modes.

Keys are per grid cell, not per slot. A 60-minute slot at 10:00 and a
30-minute slot at 10:30 have different start times but overlap; holding
"provider 3 at 10:00" and "provider 3 at 10:30" as unrelated keys would let
both be offered. So a slot holds every `step`-sized cell it touches, on a grid
anchored to the epoch, and two intervals that overlap always share a cell.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING

from patient_ops import degradation
from patient_ops.degradation import DegradedModes
from patient_ops.errors import ErrorCode, ToolError

if TYPE_CHECKING:
    from patient_ops.adapters.calendar.base import Slot
    from patient_ops.redis_layer.client import Coordinator


class HoldResult(StrEnum):
    HELD = "held"  # every cell is ours, for the next TTL
    TAKEN = "taken"  # another conversation holds part of it -- do not offer it
    DEGRADED = "degraded"  # Redis is not there -- offer it; Postgres decides at booking


class HoldStatus(StrEnum):
    MINE = "mine"  # still held by this conversation
    GONE = "gone"  # expired or released, and nobody else has it (yet)
    OTHERS = "others"  # someone else holds at least part of it now
    DEGRADED = "degraded"  # cannot tell -- Redis is not there


_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def cells(slot: Slot, step: timedelta) -> list[datetime]:
    """The start of every `step`-sized grid cell the slot overlaps, in UTC.

    The grid is anchored to the epoch rather than to the slot, so every slot
    of every length lands on the same grid. [start, end) is half-open, as
    everywhere in this project: a slot ending at 10:30 does not touch the
    10:30 cell.
    """
    step_s = int(step.total_seconds())
    if step_s <= 0:
        raise ValueError("step must be positive")
    start = int((slot.start_at - _EPOCH).total_seconds())
    end = int((slot.end_at - _EPOCH).total_seconds())
    if end <= start:
        raise ValueError("a slot must end after it starts")
    first = start - start % step_s
    return [_EPOCH + timedelta(seconds=s) for s in range(first, end, step_s)]


def hold_key(provider_id: int, cell: datetime) -> str:
    # Readable on purpose: `redis-cli --scan --pattern 'hold:*'` is how a
    # human checks what is being held right now.
    return f"hold:{provider_id}:{cell.astimezone(UTC):%Y-%m-%dT%H:%MZ}"


def owner_key(owner: str) -> str:
    return f"holds:owner:{owner}"


class SlotHolds:
    """Holds for one turn. Cheap to build: the Redis connection is the Coordinator's."""

    def __init__(
        self,
        coordinator: Coordinator,
        *,
        ttl: timedelta,
        step: timedelta,
        degraded: DegradedModes | None = None,
    ) -> None:
        self._coord = coordinator
        self._ttl_ms = int(ttl.total_seconds() * 1000)
        self._step = step
        self.degraded = degraded if degraded is not None else DegradedModes()

    def _keys(self, slot: Slot) -> list[str]:
        return [hold_key(slot.provider_id, c) for c in cells(slot, self._step)]

    def _fallback(self, exc: ToolError) -> None:
        if exc.code is not ErrorCode.DEGRADED:
            raise exc
        self.degraded.note(degradation.HOLDS, exc.cause)

    async def place(self, slot: Slot, owner: str) -> HoldResult:
        script = self._coord.script("holds_place")
        keys = [owner_key(owner), *self._keys(slot)]
        try:
            held = await self._coord.run(
                "hold_place",
                lambda r: script(keys=keys, args=[owner, self._ttl_ms], client=r),
            )
        except ToolError as exc:
            self._fallback(exc)
            return HoldResult.DEGRADED
        return HoldResult.HELD if held == 1 else HoldResult.TAKEN

    async def check(self, slot: Slot, owner: str) -> HoldStatus:
        keys = self._keys(slot)
        try:
            # One MGET is one command, so this reads every cell at one instant.
            values = await self._coord.run("hold_check", lambda r: r.mget(keys))
        except ToolError as exc:
            self._fallback(exc)
            return HoldStatus.DEGRADED
        if any(v is not None and v != owner for v in values):
            return HoldStatus.OTHERS
        if all(v == owner for v in values):
            return HoldStatus.MINE
        return HoldStatus.GONE

    async def release(self, slot: Slot, owner: str) -> int:
        """Release this slot's cells that `owner` still holds. Returns how many."""
        return await self._release_keys(owner, self._keys(slot))

    async def release_all(self, owner: str, *, keep: Slot | None = None) -> int:
        """Release everything `owner` holds, except the cells of `keep`.

        After a booking, `keep` is the slot just booked: its hold is harmless
        and expires by itself, and the other offers go back to everyone else.
        """
        try:
            members = await self._coord.run("hold_list", lambda r: r.smembers(owner_key(owner)))
        except ToolError as exc:
            self._fallback(exc)
            return 0
        kept = set(self._keys(keep)) if keep is not None else set()
        return await self._release_keys(owner, sorted(set(members) - kept))

    async def _release_keys(self, owner: str, keys: list[str]) -> int:
        if not keys:
            return 0
        script = self._coord.script("holds_release")
        try:
            return await self._coord.run(
                "hold_release",
                lambda r: script(keys=[owner_key(owner), *keys], args=[owner], client=r),
            )
        except ToolError as exc:
            # Nothing to undo: whatever was not released expires by itself.
            self._fallback(exc)
            return 0
