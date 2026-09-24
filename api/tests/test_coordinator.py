"""Coordinator: which Redis failures are outages, and how long to remember one.

Pure: a stub stands in for the client, so each redis-py exception can be
raised on demand. The real server's replies are checked in
test_degradation.py (an actual OOM) -- this file pins the classification.
"""

from __future__ import annotations

import pytest
from redis.exceptions import (
    ConnectionError,
    DataError,
    NoScriptError,
    OutOfMemoryError,
    ReadOnlyError,
    ResponseError,
    TimeoutError,
)

from patient_ops.errors import ErrorCode, ToolError
from patient_ops.redis_layer.client import Coordinator, is_unavailable
from tests.conftest import redis_test_url


@pytest.mark.parametrize(
    "exc",
    [
        ConnectionError("Connection refused"),
        TimeoutError("Timeout reading from socket"),
        # Answers, but will not serve -- and redis-py has a class for these:
        OutOfMemoryError("command not allowed when used memory > 'maxmemory'."),
        ReadOnlyError("You can't write against a read only replica."),
        # ...and not for these, so they are recognised by their error code:
        ResponseError("MISCONF Redis is configured to save RDB snapshots, but ..."),
        ResponseError("MASTERDOWN Link with MASTER is down"),
        ResponseError("CLUSTERDOWN The cluster is down"),
    ],
    ids=lambda e: type(e).__name__ + ":" + str(e).split(" ", 1)[0],
)
def test_outages_are_unavailable(exc):
    assert is_unavailable(exc)


@pytest.mark.parametrize(
    "exc",
    [
        ResponseError("ERR Error running script: attempt to compare nil with number"),
        ResponseError("WRONGTYPE Operation against a key holding the wrong kind of value"),
        DataError("Invalid input of type: 'NoneType'"),
        NoScriptError("No matching script."),  # redis-py reloads the script itself
        ValueError("not redis at all"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_bugs_are_not_outages(exc):
    """Taking the fallback here would hide a defect behind 'degraded'."""
    assert not is_unavailable(exc)


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class Calls:
    """The operation handed to run(): counts how often it actually reached Redis."""

    def __init__(self, exc: Exception | None) -> None:
        self.exc = exc
        self.n = 0

    async def __call__(self, _client):
        self.n += 1
        if self.exc:
            raise self.exc
        return "PONG"


async def test_a_readonly_reply_degrades():
    coord = Coordinator(client=None)
    with pytest.raises(ToolError) as exc:
        await coord.run("hold_place", Calls(ReadOnlyError("read only replica")))
    assert exc.value.code is ErrorCode.DEGRADED
    assert exc.value.cause == "ReadOnlyError"


async def test_a_script_bug_propagates_unchanged():
    coord = Coordinator(client=None)
    with pytest.raises(ResponseError):
        await coord.run("breaker_allow", Calls(ResponseError("ERR Error running script")))
    assert not coord.recently_down, "a bug is not an outage, so nothing is skipped afterwards"


async def test_after_a_failure_redis_is_skipped_for_the_backoff_window():
    """A blackholed host costs one timeout per window, not one per call."""
    clock = Clock()
    coord = Coordinator(client=None, down_backoff_s=5, clock=clock)
    failing = Calls(TimeoutError("Timeout connecting to server"))

    with pytest.raises(ToolError):
        await coord.run("rate_limit", failing)
    for _ in range(3):
        with pytest.raises(ToolError) as exc:
            await coord.run("rate_limit", failing)
        assert exc.value.cause == "RecentlyUnavailable"
    assert failing.n == 1, "only the first call waited on Redis"

    clock.now += 5
    working = Calls(None)
    assert await coord.run("rate_limit", working) == "PONG", "tried again after the window"
    assert working.n == 1


# ---------------------------------------------------------------------------
# The test suite's own Redis URL
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("redis://localhost:6379/0", "redis://localhost:6379/15"),
        ("redis://localhost:6379", "redis://localhost:6379/15"),  # no database at all
        # Credentials and query string survive.
        (
            "redis://:pw@127.0.0.1:6380/3?health_check_interval=5",
            "redis://:pw@127.0.0.1:6380/15?health_check_interval=5",
        ),
    ],
)
def test_the_test_database_replaces_only_the_database(given, expected):
    assert redis_test_url(given) == expected


def test_tests_refuse_a_redis_that_is_not_local():
    """FLUSHDB must never reach a shared server, whatever REDIS_URL says."""
    with pytest.raises(ValueError, match="non-local"):
        redis_test_url("redis://staging-cache.internal:6379/0")
