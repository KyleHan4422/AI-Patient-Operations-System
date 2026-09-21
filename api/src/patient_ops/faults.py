"""Fault specifications: the grammar for breaking things on purpose.

Every failure scenario in this project must be reproducible from configuration,
not from luck. A spec names a target (the operation to break), a mode (how to
break it) and optionally which attempt to break, so "the first booking call
times out, the retry succeeds" is one line:

    FAULT_INJECT=book_appointment:timeout@1

This module is only the grammar and the attempt counter. It knows nothing about
calendars or Redis: each adapter decides what a mode means for it (see
adapters/calendar/faults.py). Later phases register more targets here -- redis
(Phase 5), hold and agent (Phase 8).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum


class FaultMode(StrEnum):
    TIMEOUT = "timeout"  # no answer, and nothing happened
    TIMEOUT_AFTER_WRITE = "timeout_after_write"  # the write committed; the answer was lost
    CONFLICT = "conflict"  # someone else got the slot
    SERVER_ERROR = "server_error"  # the remote system answered HTTP 500


_READ_MODES = frozenset({FaultMode.TIMEOUT, FaultMode.SERVER_ERROR})

# Which modes make sense for which target. A typo in FAULT_INJECT must fail at
# boot -- otherwise nothing is injected and a failure test passes for the
# wrong reason.
KNOWN_TARGETS: dict[str, frozenset[FaultMode]] = {
    "check_availability": _READ_MODES,
    "get_appointment": _READ_MODES,
    "book_appointment": frozenset(FaultMode),
}


@dataclass(frozen=True)
class FaultSpec:
    target: str
    mode: FaultMode
    on_attempt: int | None = None  # None = fire on every attempt

    def fires_on(self, target: str, attempt: int) -> bool:
        return self.target == target and self.on_attempt in (None, attempt)


def parse_fault_specs(value: str) -> tuple[FaultSpec, ...]:
    """Parse comma-separated ``target:mode[@attempt]`` entries.

    Raises ValueError on anything it does not fully understand.
    """
    specs: list[FaultSpec] = []
    for raw in value.split(","):
        entry = raw.strip()
        if not entry:
            continue
        target, sep, rest = entry.partition(":")
        if not sep:
            raise ValueError(f"fault spec {entry!r}: expected 'target:mode[@attempt]'")
        mode_str, at, attempt_str = rest.partition("@")

        if target not in KNOWN_TARGETS:
            raise ValueError(f"fault spec {entry!r}: unknown target {target!r}")
        try:
            mode = FaultMode(mode_str)
        except ValueError:
            raise ValueError(f"fault spec {entry!r}: unknown mode {mode_str!r}") from None
        if mode not in KNOWN_TARGETS[target]:
            raise ValueError(f"fault spec {entry!r}: mode {mode} does not apply to {target}")

        on_attempt: int | None = None
        if at:
            if not attempt_str.isdigit() or int(attempt_str) < 1:
                raise ValueError(f"fault spec {entry!r}: attempt must be a positive integer")
            on_attempt = int(attempt_str)

        specs.append(FaultSpec(target, mode, on_attempt))
    return tuple(specs)


class FaultInjector:
    """Counts attempts per target and decides which fault, if any, fires now.

    Attempt counts are scenario state: use one injector per test or eval case.
    """

    def __init__(self, specs: Iterable[FaultSpec] = ()) -> None:
        self._specs = tuple(specs)
        self._attempts: Counter[str] = Counter()

    @classmethod
    def from_string(cls, value: str) -> FaultInjector:
        return cls(parse_fault_specs(value))

    def next_fault(self, target: str) -> FaultMode | None:
        """Record one attempt at `target` and return the fault to inject, if any."""
        self._attempts[target] += 1
        attempt = self._attempts[target]
        return next((s.mode for s in self._specs if s.fires_on(target, attempt)), None)

    def attempts(self, target: str) -> int:
        return self._attempts[target]

    def __bool__(self) -> bool:
        return bool(self._specs)
