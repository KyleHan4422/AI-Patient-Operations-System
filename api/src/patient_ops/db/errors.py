"""Translate database errors into the shared failure vocabulary.

Classification is by SQLSTATE -- the five-character code Postgres attaches to
every error -- never by message text, which varies by version and locale.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from patient_ops.errors import ErrorCode, ToolError

EXCLUSION_VIOLATION = "23P01"  # no_overlap: the slot is taken
FOREIGN_KEY_VIOLATION = "23503"  # unknown patient / provider / procedure
CHECK_VIOLATION = "23514"  # e.g. end_at <= start_at
CONNECTION_EXCEPTION_CLASS = "08"  # 08xxx: connection failures
ADMIN_SHUTDOWN = "57P01"  # server restarting


def classify_db_error(exc: DBAPIError | PoolTimeoutError) -> ToolError:
    if isinstance(exc, PoolTimeoutError):
        return ToolError(ErrorCode.TRANSIENT, "no free database connection", cause="PoolTimeout")

    sqlstate = getattr(exc.orig, "sqlstate", None) or ""
    cause = type(exc.orig).__name__
    if sqlstate == EXCLUSION_VIOLATION:
        return ToolError(ErrorCode.CONFLICT, "the slot is no longer free", cause=cause)
    if sqlstate == FOREIGN_KEY_VIOLATION:
        return ToolError(ErrorCode.INVALID, "unknown patient, provider or procedure", cause=cause)
    if sqlstate == CHECK_VIOLATION:
        return ToolError(ErrorCode.INVALID, "request violates a data rule", cause=cause)
    if (
        sqlstate.startswith(CONNECTION_EXCEPTION_CLASS)
        or sqlstate == ADMIN_SHUTDOWN
        or isinstance(exc, OperationalError | InterfaceError)  # refused, dropped, closed
    ):
        return ToolError(ErrorCode.TRANSIENT, "database unavailable", cause=cause)
    return ToolError(ErrorCode.UNKNOWN, f"unclassified database error {sqlstate}", cause=cause)


@contextmanager
def translate_db_errors() -> Iterator[None]:
    """Nothing above the adapter layer ever sees a driver exception."""
    try:
        yield
    except (DBAPIError, PoolTimeoutError) as exc:
        raise classify_db_error(exc) from exc
