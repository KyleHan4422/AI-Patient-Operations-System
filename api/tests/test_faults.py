"""The fault-spec grammar and its safety rules. Pure: no database."""

from __future__ import annotations

import pytest

from patient_ops.adapters.calendar.fake import FakeCalendar
from patient_ops.adapters.calendar.faults import FaultyCalendar, build_calendar
from patient_ops.config import Settings
from patient_ops.faults import FaultInjector, FaultMode, FaultSpec, parse_fault_specs


def test_parse_specs():
    assert parse_fault_specs("book_appointment:timeout@1, check_availability:server_error") == (
        FaultSpec("book_appointment", FaultMode.TIMEOUT, on_attempt=1),
        FaultSpec("check_availability", FaultMode.SERVER_ERROR, on_attempt=None),
    )
    assert parse_fault_specs("") == ()


@pytest.mark.parametrize(
    "bad",
    [
        "book_apointment:timeout",  # typo in the target
        "book_appointment:explode",  # unknown mode
        "book_appointment",  # no mode
        "check_availability:conflict",  # a read cannot conflict
        "book_appointment:timeout@0",  # attempts count from 1
        "book_appointment:timeout@first",
    ],
)
def test_malformed_specs_are_rejected(bad: str):
    with pytest.raises(ValueError):
        parse_fault_specs(bad)


def test_injector_counts_attempts_per_target():
    injector = FaultInjector.from_string("book_appointment:timeout@2")
    assert injector.next_fault("book_appointment") is None  # attempt 1
    assert injector.next_fault("check_availability") is None  # other targets count separately
    assert injector.next_fault("book_appointment") is FaultMode.TIMEOUT  # attempt 2
    assert injector.next_fault("book_appointment") is None  # attempt 3


def test_a_typo_in_fault_inject_fails_at_boot():
    with pytest.raises(ValueError, match="unknown target"):
        Settings(app_env="dev", fault_inject="book_apointment:timeout")


def test_fault_injection_is_refused_in_production():
    with pytest.raises(ValueError, match="refused when APP_ENV=prod"):
        Settings(app_env="prod", fault_inject="book_appointment:timeout")


def test_build_calendar_wraps_only_when_faults_are_configured():
    # fault_inject passed explicitly: a FAULT_INJECT left in a developer's .env
    # must not change what this test sees.
    no_faults = Settings(app_env="test", fault_inject="")
    assert isinstance(build_calendar(no_faults, session_factory=None), FakeCalendar)
    wrapped = build_calendar(
        Settings(app_env="test", fault_inject="book_appointment:timeout"), session_factory=None
    )
    assert isinstance(wrapped, FaultyCalendar)
