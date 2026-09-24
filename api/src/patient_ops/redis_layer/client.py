"""The one door into Redis.

Redis is the coordination layer: slot holds, the circuit breaker, in-flight
dedup, rate limiting. Every one of those has a fallback, and every fallback
starts the same way -- "Redis could not be reached". So there is exactly one
place that decides what that means:

    Coordinator.run(op, fn)   runs one Redis operation. Redis being unable to
                              serve -- see UNAVAILABLE below -- or
                              FAULT_INJECT=redis:unavailable comes out as
                              ToolError(DEGRADED).

"Unable to serve" is wider than "unreachable". A Redis that answers but
refuses writes is just as gone for a hold or a token bucket: a failed RDB
snapshot (MISCONF), maxmemory with no eviction (OOM), a replica left behind by
a failover (READONLY). Those arrive as *replies*, not connection errors, and
missing them would turn a cache outage into a 500 on every chat turn.

What does *not* become DEGRADED: a Lua script error, a wrong-type reply, a bad
argument. Those are bugs in this code, not an outage, and taking the fallback
would hide them -- so they propagate.

No retries, and no waiting twice. Retrying an optional dependency only makes
the request that needs its fallback slower (failure matrix #9: fail fast,
degrade, do not escalate). And once Redis has failed, the next few seconds of
operations take the fallback straight away instead of each waiting out its
own timeout -- a blackholed host costs one timeout per window, not one per call.

Lua scripts live in lua/, one file per atomic operation. A multi-step change
that must not interleave with another client's (check the value, then delete
it) is a script, because Redis runs a script as one command.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from functools import cache
from pathlib import Path
from typing import TypeVar

import redis.asyncio as aioredis
from redis.commands.core import AsyncScript
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import OutOfMemoryError, ReadOnlyError, ResponseError
from redis.exceptions import TimeoutError as RedisTimeoutError

from patient_ops.errors import ErrorCode, ToolError
from patient_ops.faults import INJECTED, FaultInjector

T = TypeVar("T")

LUA_DIR = Path(__file__).parent / "lua"
FAULT_TARGET = "redis"

# Replies that mean "this server cannot serve right now", by class where
# redis-py has one and by error-code prefix where it does not. Inside a Lua
# script Redis 7 keeps the code of the command that failed, so a script that
# hits OOM on its first SET is classified the same way as a plain SET.
_UNAVAILABLE_CLASSES = (RedisConnectionError, RedisTimeoutError, OutOfMemoryError, ReadOnlyError)
_UNAVAILABLE_CODES = frozenset({"MISCONF", "MASTERDOWN", "CLUSTERDOWN", "TRYAGAIN", "NOREPLICAS"})


def is_unavailable(exc: BaseException) -> bool:
    if isinstance(exc, _UNAVAILABLE_CLASSES):
        return True
    return isinstance(exc, ResponseError) and str(exc).split(" ", 1)[0] in _UNAVAILABLE_CODES


@cache
def lua_source(name: str) -> str:
    return (LUA_DIR / f"{name}.lua").read_text(encoding="utf-8")


class Coordinator:
    def __init__(
        self,
        client: aioredis.Redis,
        injector: FaultInjector | None = None,
        *,
        down_backoff_s: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.client = client
        self._injector = injector or FaultInjector()
        self._scripts: dict[str, AsyncScript] = {}
        # After a failure, how long to skip Redis before trying it again.
        # Per process, and monotonic: this is local memory, not shared state.
        self._down_backoff_s = down_backoff_s
        self._clock = clock
        self._down_until = 0.0

    def script(self, name: str) -> AsyncScript:
        """A Lua script from lua/, registered once.

        Calls go by EVALSHA; redis-py reloads the script by itself when the
        server does not know it (after a restart, or a FLUSHALL of the cache).
        """
        if name not in self._scripts:
            self._scripts[name] = self.client.register_script(lua_source(name))
        return self._scripts[name]

    async def run(self, op: str, fn: Callable[[aioredis.Redis], Awaitable[T]]) -> T:
        """Run one Redis operation, or raise ToolError(DEGRADED) if Redis is not there."""
        if self._injector.next_fault(FAULT_TARGET):
            detail = f"{op}: redis unavailable (injected)"
            raise ToolError(ErrorCode.DEGRADED, detail, cause=INJECTED)
        if self._clock() < self._down_until:
            raise ToolError(
                ErrorCode.DEGRADED, f"{op}: redis failed moments ago", cause="RecentlyUnavailable"
            )
        try:
            return await fn(self.client)
        except Exception as exc:
            if not is_unavailable(exc):
                raise
            self._down_until = self._clock() + self._down_backoff_s
            raise ToolError(
                ErrorCode.DEGRADED, f"{op}: redis unavailable", cause=type(exc).__name__
            ) from exc

    @property
    def recently_down(self) -> bool:
        return self._clock() < self._down_until


def build_redis_client(
    url: str, *, connect_timeout_s: float, socket_timeout_s: float
) -> aioredis.Redis:
    # decode_responses: every value this project stores is text (thread ids,
    # "inflight", "done:42", breaker fields), so replies come back as str.
    return aioredis.from_url(
        url,
        socket_connect_timeout=connect_timeout_s,
        socket_timeout=socket_timeout_s,
        decode_responses=True,
    )
