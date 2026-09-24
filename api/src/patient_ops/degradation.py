"""Which fallbacks one turn took.

A fallback nobody can see is how a degraded system runs for a month before
anyone notices the conflict rate has doubled. So every time a component takes
its "Redis is not there" path, it says so here, and the turn carries the list
out: into the graph state (`degraded_modes`), the SSE `done` event, and a WARN
log line per mode.

One instance per turn, like the read-only toolset: it accumulates that turn's
record and is thrown away with it.
"""

from __future__ import annotations

from typing import Final

from patient_ops.obs.logging import get_logger

log = get_logger(__name__)

# The names a turn reports. Stable strings: logs, evals and (Phase 10) the /ops
# banner group by them.
HOLDS: Final = "holds"  # R1: slots offered without a hold; Postgres EXCLUDE decides
IDEMPOTENCY: Final = "idempotency"  # R4: no in-flight dedup; Postgres UNIQUE decides
BREAKER: Final = "breaker"  # R3: the breaker is per process, not shared
RATE_LIMIT: Final = "rate_limit"  # R5: the request was let through unmetered


class DegradedModes:
    def __init__(self) -> None:
        self._modes: dict[str, str] = {}  # mode -> the first cause seen

    def note(self, mode: str, cause: str | None = None) -> None:
        if mode in self._modes:
            return  # once per turn is enough to say it happened
        self._modes[mode] = cause or ""
        log.warning("degraded_mode", mode=mode, cause=cause)

    @property
    def modes(self) -> list[str]:
        return list(self._modes)

    def __bool__(self) -> bool:
        return bool(self._modes)
