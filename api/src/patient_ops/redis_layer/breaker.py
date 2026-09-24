"""R3 -- a circuit breaker whose state every worker shares.

When the calendar is failing, every request that still calls it waits out a
timeout and adds load to a system that is already down. A breaker counts
consecutive failures and, past a threshold, stops calling for a while:

    CLOSED     calls go through; consecutive failures are counted
      | failures >= threshold
    OPEN       calls fail fast, without touching the calendar
      | cooldown elapsed: the next caller becomes the probe
    HALF_OPEN  exactly one probe goes through; everyone else still fails fast
      | probe succeeds -> CLOSED        probe fails -> OPEN again

An in-process breaker stops working as soon as there are two uvicorn workers:
each one learns separately, so the calendar takes N times the failures before
it gets any rest. So the state lives in Redis, and one Lua script per
transition makes "is it my turn to probe?" a question only one worker can
answer yes to.

When Redis is not there, FailoverBreaker falls back to a per-process breaker:
weaker protection, but protection -- and the turn says it degraded.

There are two implementations of one state machine, and a script cannot import
Python. `breaker_step` is the specification; lua/breaker.lua mirrors it rule
for rule; test_breaker.py runs a single transition table against both.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from patient_ops import degradation
from patient_ops.errors import ErrorCode, ToolError
from patient_ops.obs.logging import get_logger

if TYPE_CHECKING:
    from patient_ops.degradation import DegradedModes
    from patient_ops.redis_layer.client import Coordinator

log = get_logger(__name__)


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class BreakerEvent(StrEnum):
    ALLOW = "allow"  # "may I call?"
    SUCCESS = "success"  # the call worked
    FAILURE = "failure"  # the call failed in a way that says the backend is unwell
    # The call ended with no verdict -- cancelled because the client went away.
    # Says nothing about the backend, but a probe that ends this way must hand
    # its turn back, or every worker waits out another cooldown for nothing.
    ABANDON = "abandon"


@dataclass(frozen=True)
class BreakerPolicy:
    failure_threshold: int
    cooldown_ms: int


@dataclass(frozen=True)
class BreakerSnapshot:
    state: BreakerState = BreakerState.CLOSED
    failures: int = 0  # consecutive, while CLOSED
    # OPEN: when it opened. HALF_OPEN: when the current probe was let through.
    since_ms: int = 0


def breaker_step(
    s: BreakerSnapshot, event: BreakerEvent, now_ms: int, policy: BreakerPolicy
) -> tuple[BreakerSnapshot, bool]:
    """One transition. Returns the new state and, for ALLOW, whether the call may go ahead.

    Pure: no clock, no I/O. For SUCCESS and FAILURE the boolean means nothing.
    """
    if event is BreakerEvent.ALLOW:
        if s.state is BreakerState.CLOSED:
            return s, True
        if now_ms - s.since_ms >= policy.cooldown_ms:
            # OPEN: the cooldown is over and this caller is the probe.
            # HALF_OPEN: the last probe never reported back (its worker died);
            # after another cooldown, let a new one through rather than stay
            # half-open for ever.
            return BreakerSnapshot(BreakerState.HALF_OPEN, 0, now_ms), True
        return s, False

    if event is BreakerEvent.ABANDON:
        if s.state is BreakerState.HALF_OPEN:
            # Back-date the probe so the next ALLOW becomes a new one at once.
            return replace(s, since_ms=now_ms - policy.cooldown_ms), True
        return s, True

    if event is BreakerEvent.SUCCESS:
        if s.state is BreakerState.OPEN:
            # A call let through before the breaker opened, finishing late.
            # It says nothing about the calendar *now*; only a probe does.
            return s, True
        return BreakerSnapshot(), True

    # FAILURE
    if s.state is BreakerState.CLOSED:
        failures = s.failures + 1
        if failures >= policy.failure_threshold:
            return BreakerSnapshot(BreakerState.OPEN, 0, now_ms), True
        return replace(s, failures=failures), True
    if s.state is BreakerState.HALF_OPEN:
        return BreakerSnapshot(BreakerState.OPEN, 0, now_ms), True
    return s, True  # already OPEN: the cooldown runs from when it opened


def wall_clock_ms() -> int:
    # Wall time, not monotonic: the state is shared by processes, and a
    # monotonic clock means nothing outside the process that read it.
    return int(time.time() * 1000)


class Breaker(Protocol):
    async def allow(self) -> bool: ...
    async def record_success(self) -> None: ...
    async def record_failure(self) -> None: ...
    async def record_abandon(self) -> None: ...
    async def snapshot(self) -> BreakerSnapshot: ...


class LocalBreaker:
    """In-process. The fallback, and the reference the Redis one is tested against."""

    def __init__(self, policy: BreakerPolicy, *, clock: Callable[[], int] = wall_clock_ms):
        self.policy = policy
        self._clock = clock
        self._s = BreakerSnapshot()

    async def _step(self, event: BreakerEvent) -> bool:
        before = self._s.state
        self._s, allowed = breaker_step(self._s, event, self._clock(), self.policy)
        if self._s.state is not before:
            log.warning("breaker_transition", scope="local", old=before, new=self._s.state)
        return allowed

    async def allow(self) -> bool:
        return await self._step(BreakerEvent.ALLOW)

    async def record_success(self) -> None:
        await self._step(BreakerEvent.SUCCESS)

    async def record_failure(self) -> None:
        await self._step(BreakerEvent.FAILURE)

    async def record_abandon(self) -> None:
        await self._step(BreakerEvent.ABANDON)

    async def snapshot(self) -> BreakerSnapshot:
        return self._s


class RedisBreaker:
    """Shared by every worker. Raises ToolError(DEGRADED) when Redis is not there."""

    def __init__(
        self,
        coordinator: Coordinator,
        name: str,
        policy: BreakerPolicy,
        *,
        clock: Callable[[], int] = wall_clock_ms,
    ) -> None:
        self._coord = coordinator
        self.key = f"breaker:{name}"
        self.policy = policy
        # The caller's clock, passed into the script, rather than Redis TIME:
        # it keeps the script a pure function of its arguments, so tests
        # replay any timeline without sleeping.
        self._clock = clock

    async def _step(self, event: BreakerEvent) -> bool:
        script = self._coord.script("breaker")
        args = [event.value, self._clock(), self.policy.failure_threshold, self.policy.cooldown_ms]
        old, new, allowed = await self._coord.run(
            f"breaker_{event}", lambda r: script(keys=[self.key], args=args, client=r)
        )
        if old != new:
            log.warning("breaker_transition", scope="shared", key=self.key, old=old, new=new)
        return allowed == 1

    async def allow(self) -> bool:
        return await self._step(BreakerEvent.ALLOW)

    async def record_success(self) -> None:
        await self._step(BreakerEvent.SUCCESS)

    async def record_failure(self) -> None:
        await self._step(BreakerEvent.FAILURE)

    async def record_abandon(self) -> None:
        await self._step(BreakerEvent.ABANDON)

    async def snapshot(self) -> BreakerSnapshot:
        raw = await self._coord.run("breaker_read", lambda r: r.hgetall(self.key))
        if not raw:
            return BreakerSnapshot()
        return BreakerSnapshot(
            BreakerState(raw["state"]), int(raw.get("failures", 0)), int(raw.get("since_ms", 0))
        )


@dataclass(frozen=True)
class Permit:
    """Proof that a call was admitted, and by which breaker.

    The outcome of a call goes back to the breaker that admitted it. Without
    this, a Redis blip between "may I?" and "it worked" would send the answer
    to the other breaker: the shared one would stay half-open waiting for a
    probe result that went to the local one, which never admitted anything.
    """

    breaker: RedisBreaker | LocalBreaker


class FailoverBreaker:
    """The shared breaker while Redis answers; this process's own when it does not.

    Long-lived (one per protected dependency, per process), so the local
    breaker keeps counting across turns. Which turn degraded is the caller's
    to record: each method takes that turn's DegradedModes.
    """

    def __init__(self, shared: RedisBreaker, local: LocalBreaker) -> None:
        self.shared = shared
        self.local = local

    def _note(self, exc: ToolError, degraded: DegradedModes | None) -> None:
        if exc.code is not ErrorCode.DEGRADED:
            raise exc
        if degraded is not None:
            degraded.note(degradation.BREAKER, exc.cause)

    async def allow(self, degraded: DegradedModes | None = None) -> Permit | None:
        """A Permit if the call may go ahead, None if it must fail fast."""
        try:
            if await self.shared.allow():
                return Permit(self.shared)
            return None
        except ToolError as exc:
            self._note(exc, degraded)
        return Permit(self.local) if await self.local.allow() else None

    async def record(
        self, permit: Permit, event: BreakerEvent, degraded: DegradedModes | None = None
    ) -> None:
        if event is BreakerEvent.ALLOW:
            raise ValueError("record() takes an outcome, not ALLOW")
        try:
            await permit.breaker._step(event)
        except ToolError as exc:
            # The shared breaker admitted the call and Redis is gone before
            # the outcome could be written. The outcome is lost, not
            # misfiled: a half-open probe that never reports back is let go
            # after one cooldown, by the same rule as a probe whose worker died.
            self._note(exc, degraded)
            log.warning("breaker_outcome_lost", key=self.shared.key, outcome=event)


def build_breaker(
    coordinator: Coordinator,
    name: str,
    *,
    failure_threshold: int,
    cooldown_s: float,
    clock: Callable[[], int] = wall_clock_ms,
) -> FailoverBreaker:
    policy = BreakerPolicy(failure_threshold, int(cooldown_s * 1000))
    return FailoverBreaker(
        RedisBreaker(coordinator, name, policy, clock=clock), LocalBreaker(policy, clock=clock)
    )
