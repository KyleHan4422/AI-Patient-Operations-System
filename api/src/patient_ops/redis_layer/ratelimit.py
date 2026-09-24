"""R5 -- a token bucket per client.

Each client gets `capacity` turns in a burst, then one more every
1/`refill_per_s` seconds. A chat turn costs a model call, so this is what
stands between one script in a loop and the clinic's LLM bill.

Failing open is the fallback: when Redis is not there the request goes ahead,
unmetered, and the turn records that it did. A limiter that fails closed turns
a cache outage into a full outage of the front desk.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from patient_ops import degradation
from patient_ops.errors import ErrorCode, ToolError
from patient_ops.redis_layer.breaker import wall_clock_ms

if TYPE_CHECKING:
    from patient_ops.degradation import DegradedModes
    from patient_ops.redis_layer.client import Coordinator


@dataclass(frozen=True)
class RateDecision:
    allowed: bool
    retry_after_s: int = 0  # whole seconds, as the Retry-After header wants


class RateLimiter:
    def __init__(
        self,
        coordinator: Coordinator,
        *,
        capacity: int,
        refill_per_s: float,
        clock: Callable[[], int] = wall_clock_ms,
    ) -> None:
        self._coord = coordinator
        self.capacity = capacity
        self._rate_per_ms = refill_per_s / 1000
        self._clock = clock

    async def take(self, scope: str, ident: str, degraded: DegradedModes) -> RateDecision:
        script = self._coord.script("token_bucket")
        args = [self.capacity, self._rate_per_ms, self._clock()]
        try:
            allowed, retry_ms = await self._coord.run(
                "rate_limit",
                lambda r: script(keys=[f"rl:{scope}:{ident}"], args=args, client=r),
            )
        except ToolError as exc:
            if exc.code is not ErrorCode.DEGRADED:
                raise
            degraded.note(degradation.RATE_LIMIT, exc.cause)
            return RateDecision(allowed=True)
        if allowed == 1:
            return RateDecision(allowed=True)
        return RateDecision(allowed=False, retry_after_s=max(1, math.ceil(retry_ms / 1000)))
