"""The failure vocabulary shared by every adapter and tool.

Retry decisions are made from the code, never from an exception's message or
type. That keeps the one question that causes real production bugs -- "is it
safe to try this again?" -- answerable in one place (Phase 8 adds
`retry_policy(code)` on top of this enum).

Lives at the package root, not in tools/: adapters sit below tools and raise
these, so they must not import from the layer above them.
"""

from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    TRANSIENT = "transient"  # timeout, 5xx, connection reset -> back off and retry
    CONFLICT = "conflict"  # slot taken -> never retry; re-check availability
    INVALID = "invalid"  # bad arguments -> never retry; ask again
    PERMANENT = "permanent"  # auth failure, 4xx -> never retry; escalate
    DEGRADED = "degraded"  # an optional dependency is down -> take the fallback path
    UNKNOWN = "unknown"  # unclassified -> retry at most once, and always verify


class ToolError(Exception):
    """A classified failure.

    `cause` records the original exception type (e.g. "ExclusionViolation"),
    so the mapping from raw error to code is auditable after the fact.
    """

    def __init__(self, code: ErrorCode, detail: str, *, cause: str | None = None) -> None:
        super().__init__(f"[{code}] {detail}")
        self.code = code
        self.detail = detail
        self.cause = cause
