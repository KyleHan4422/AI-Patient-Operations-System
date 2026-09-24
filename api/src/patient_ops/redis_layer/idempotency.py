"""R4 -- in-flight dedup: the same request, twice, at the same moment.

A double click, a voice channel that hears "yes" twice, a client retrying
after a timeout: two identical booking requests reach the write at once.
Postgres already guarantees the outcome -- the UNIQUE idempotency key means
one appointment, whatever happens -- so this layer is not about correctness.
It saves the second request from making its own call to the calendar and
receiving a conflict it has to explain:

    Redis catches duplicates inside the concurrency window.
    Postgres guarantees the final result.

Hence the rule that runs through this module: Redis may say *that* the work is
done, never *what* it produced. The result is always read back from the system
of record, and when Redis is not there the request simply goes ahead and lets
the UNIQUE key decide.
"""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, TypeVar

from patient_ops import degradation
from patient_ops.degradation import DegradedModes
from patient_ops.errors import ErrorCode, ToolError
from patient_ops.obs.logging import get_logger

if TYPE_CHECKING:
    from patient_ops.redis_layer.client import Coordinator

T = TypeVar("T")

log = get_logger(__name__)

INFLIGHT = "inflight:"
DONE = "done:"
DONE_TTL_MS = 10 * 60 * 1000  # long enough to absorb any retry a client makes
POLL_INTERVAL_S = 0.05


class ClaimState(StrEnum):
    CLAIMED = "claimed"  # ours: go ahead and do the work
    IN_FLIGHT = "in_flight"  # an identical request is doing it right now
    DONE = "done"  # an identical request already did it
    DEGRADED = "degraded"  # Redis is not there: go ahead, Postgres decides


@dataclass(frozen=True)
class Claim:
    state: ClaimState
    token: str | None = None  # set when CLAIMED; needed to settle the claim
    ref: str | None = None  # set when DONE; what the first request produced


def idem_key(key: str) -> str:
    return f"idem:{key}"


class InFlightDedup:
    def __init__(
        self,
        coordinator: Coordinator,
        *,
        inflight_ttl_s: float,
        wait_s: float,
        degraded: DegradedModes | None = None,
    ) -> None:
        self._coord = coordinator
        self._ttl_ms = int(inflight_ttl_s * 1000)
        self._wait_s = wait_s
        self.degraded = degraded if degraded is not None else DegradedModes()

    def _fallback(self, exc: ToolError) -> None:
        if exc.code is not ErrorCode.DEGRADED:
            raise exc
        self.degraded.note(degradation.IDEMPOTENCY, exc.cause)

    async def claim(self, key: str) -> Claim:
        token = secrets.token_hex(8)
        try:
            # SET NX GET: one command that either claims the key, or returns
            # what is already there -- no gap between "is it free?" and "take it".
            previous = await self._coord.run(
                "idem_claim",
                lambda r: r.set(
                    idem_key(key), INFLIGHT + token, nx=True, px=self._ttl_ms, get=True
                ),
            )
        except ToolError as exc:
            self._fallback(exc)
            return Claim(ClaimState.DEGRADED)
        return _parse(previous, token)

    async def peek(self, key: str) -> Claim | None:
        """What an identical request left behind, or None if nothing is there."""
        try:
            value = await self._coord.run("idem_peek", lambda r: r.get(idem_key(key)))
        except ToolError as exc:
            self._fallback(exc)
            return Claim(ClaimState.DEGRADED)
        return None if value is None else _parse(value, None)

    async def finish(self, key: str, token: str, ref: str) -> bool:
        return await self._settle(key, token, DONE + ref, DONE_TTL_MS)

    async def abandon(self, key: str, token: str) -> bool:
        """Give the key back, so a retry of this request is not told it is in flight."""
        return await self._settle(key, token, "", 0)

    async def _settle(self, key: str, token: str, value: str, ttl_ms: int) -> bool:
        script = self._coord.script("idem_settle")
        try:
            settled = await self._coord.run(
                "idem_settle",
                lambda r: script(
                    keys=[idem_key(key)], args=[INFLIGHT + token, value, ttl_ms], client=r
                ),
            )
        except ToolError as exc:
            # The claim expires by itself; until then a duplicate waits and
            # then reads the result back from Postgres. Slower, never wrong.
            self._fallback(exc)
            return False
        return settled == 1

    async def run_once(
        self,
        key: str,
        action: Callable[[], Awaitable[T]],
        *,
        read_back: Callable[[], Awaitable[T | None]],
        ref: Callable[[T], object],
    ) -> T:
        """Do `action` once per `key` across concurrent callers.

        `action` must itself be idempotent on `key` (the calendar's book() is:
        same key, same appointment) -- that is what makes every fallback here
        safe. `read_back` fetches the result by the same key from the system of
        record; `ref` names a result for the log ("appointment 42").
        """
        deadline = asyncio.get_running_loop().time() + self._wait_s
        claim = await self.claim(key)
        while True:
            if claim.state is ClaimState.CLAIMED:
                assert claim.token is not None
                return await self._do(key, claim.token, action, ref)
            if claim.state is ClaimState.DEGRADED:
                return await action()
            if claim.state is ClaimState.DONE:
                result = await read_back()
                if result is not None:
                    log.info("idempotency_hit", key=key, ref=claim.ref)
                    return result
                # Redis says done but the system of record has no such row --
                # Redis is the one that is wrong. Go ahead; the UNIQUE key
                # still guarantees a single row.
                return await action()

            # IN_FLIGHT: wait for the other request rather than racing it.
            if asyncio.get_running_loop().time() >= deadline:
                return await self._gave_up_waiting(key, read_back)
            await asyncio.sleep(POLL_INTERVAL_S)
            seen = await self.peek(key)
            # Gone means the other request abandoned its claim (it failed) or
            # the claim expired: try to take it ourselves.
            claim = await self.claim(key) if seen is None else seen

    async def _do(
        self,
        key: str,
        token: str,
        action: Callable[[], Awaitable[T]],
        ref: Callable[[T], object],
    ) -> T:
        try:
            result = await action()
        except BaseException:
            # Including cancellation: a claim left behind would make a retry
            # wait out the full TTL for a request that is no longer running.
            await self._settle_quietly(asyncio.shield(self.abandon(key, token)), key)
            raise  # the action's own error, never one from settling the claim
        await self._settle_quietly(self.finish(key, token, str(ref(result))), key)
        return result

    async def _settle_quietly(self, settling: Awaitable[bool], key: str) -> None:
        """Bookkeeping after the work, which must never replace the work's outcome.

        The booking has already happened, or already failed with its own error
        (a CONFLICT the caller must handle as a CONFLICT). If Redis misbehaves
        now, the claim simply expires by its TTL.
        """
        try:
            await settling
        except Exception:
            log.warning("idempotency_settle_failed", key=key, exc_info=True)

    async def _gave_up_waiting(self, key: str, read_back: Callable[[], Awaitable[T | None]]) -> T:
        result = await read_back()
        if result is not None:
            return result
        # The other request is still running and has written nothing yet.
        # Doing the work now would only race it; whether it lands is exactly
        # the question verification answers (Phase 8).
        raise ToolError(
            ErrorCode.UNKNOWN,
            "an identical request is still in progress",
            cause="InFlightDuplicate",
        )


def _parse(value: str | None, token: str | None) -> Claim:
    if value is None:
        return Claim(ClaimState.CLAIMED, token=token)
    if value.startswith(DONE):
        return Claim(ClaimState.DONE, ref=value.removeprefix(DONE))
    return Claim(ClaimState.IN_FLIGHT)
