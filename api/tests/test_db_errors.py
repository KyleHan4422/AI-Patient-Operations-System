"""Database error -> ErrorCode. Pure: exceptions are constructed, not provoked.

This mapping decides, from Phase 8 on, whether a failed call may be retried --
so every branch is pinned here, including the ones a healthy test database
never produces (dropped connections, a restarting server, an exhausted pool).
"""

from __future__ import annotations

import pytest
from sqlalchemy.exc import DBAPIError, IntegrityError, InterfaceError, OperationalError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from patient_ops.db.errors import classify_db_error, translate_db_errors
from patient_ops.errors import ErrorCode, ToolError


class DriverError(Exception):
    """Stands in for a psycopg exception: all the classifier reads is .sqlstate."""

    def __init__(self, sqlstate: str | None) -> None:
        super().__init__(sqlstate)
        self.sqlstate = sqlstate


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (IntegrityError("INSERT", {}, DriverError("23P01")), ErrorCode.CONFLICT),
        (IntegrityError("INSERT", {}, DriverError("23503")), ErrorCode.INVALID),
        (IntegrityError("INSERT", {}, DriverError("23514")), ErrorCode.INVALID),
        (OperationalError("SELECT", {}, DriverError(None)), ErrorCode.TRANSIENT),  # refused
        (InterfaceError("SELECT", {}, DriverError(None)), ErrorCode.TRANSIENT),  # dropped
        (DBAPIError("SELECT", {}, DriverError("08006")), ErrorCode.TRANSIENT),  # link failure
        (DBAPIError("SELECT", {}, DriverError("57P01")), ErrorCode.TRANSIENT),  # server restart
        (PoolTimeoutError("pool exhausted"), ErrorCode.TRANSIENT),
        (DBAPIError("SELECT", {}, DriverError("XX000")), ErrorCode.UNKNOWN),  # never guess
    ],
)
def test_classification(exc: Exception, code: ErrorCode):
    assert classify_db_error(exc).code is code


def test_translate_keeps_the_original_as_cause():
    original = IntegrityError("INSERT", {}, DriverError("23P01"))
    with pytest.raises(ToolError) as caught, translate_db_errors():
        raise original
    assert caught.value.__cause__ is original  # full traceback survives for debugging
    assert caught.value.cause == "DriverError"


def test_translate_lets_other_errors_through():
    already_classified = ToolError(ErrorCode.INVALID, "outside working hours")
    with pytest.raises(ToolError) as caught, translate_db_errors():
        raise already_classified
    assert caught.value is already_classified
