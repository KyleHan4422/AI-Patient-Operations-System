"""The sentences the clinic says about its own records.

Pure functions of a row, so these are assertions rather than hopes: the reply
to a plan that is not on file cannot become a refusal, and a treatment with no
price cannot acquire a number, whatever a language model would have written.
"""

from __future__ import annotations

from datetime import date, time
from decimal import Decimal

import pytest

from patient_ops.graph.replies import (
    Citation,
    facts_block,
    hours_sentence,
    insurance_sentence,
    price_sentence,
    sources_block,
)
from patient_ops.tools.facts import ClosedDay, HoursFact, InsuranceFact, OpeningWindow, PriceFact


def plan(status: str, **kwargs) -> InsuranceFact:
    fields = {"asked_for": "Aetna", "plan_name": "Aetna", "status": status} | kwargs
    return InsuranceFact(**fields)


def procedure(status: str, **kwargs) -> PriceFact:
    fields = {
        "asked_for": "crown",
        "code": "CROWN",
        "name": "Crown",
        "price_min": None,
        "price_max": None,
        "status": status,
    } | kwargs
    return PriceFact(**fields)


# ---------------------------------------------------------------------------
# Insurance: the distinction the whole phase turns on
# ---------------------------------------------------------------------------
def test_an_unknown_plan_is_never_reported_as_a_refusal():
    """ "Not on file" and "not accepted" are different answers to a patient.

    Aetna is absent from the demo data for this case. A model asked to word it
    reaches for "no, we don't take Aetna" -- which the clinic never said.
    """
    said = insurance_sentence(plan("unknown", plan_name=None)).lower()
    assert "on file" in said
    assert "don't accept" not in said and "do not accept" not in said
    assert "isn't a no" in said or "is not a no" in said


def test_an_accepted_plan_says_which_kind_of_accepted():
    in_network = insurance_sentence(plan("in_network", plan_name="Delta Dental PPO"))
    out = insurance_sentence(plan("out_of_network", plan_name="Delta Dental PPO"))
    assert "in network" in in_network and "Delta Dental PPO" in in_network
    assert "out of network" in out, "the difference is what the patient pays"


def test_a_refused_plan_says_so_plainly():
    assert insurance_sentence(plan("not_accepted", plan_name="Cigna DPPO")) == (
        "We don't accept Cigna DPPO."
    )


def test_notes_on_file_are_passed_on():
    said = insurance_sentence(plan("in_network", notes="Orthodontics needs pre-approval."))
    assert said.endswith("Orthodontics needs pre-approval.")


# ---------------------------------------------------------------------------
# Prices: a missing price never becomes a number
# ---------------------------------------------------------------------------
def test_a_treatment_with_no_price_on_file_quotes_no_figure():
    said = price_sentence(procedure("no_price_on_file", asked_for="whitening", name="Whitening"))
    assert not any(character.isdigit() for character in said)
    assert "front desk" in said


def test_an_unknown_treatment_is_not_a_missing_price():
    said = price_sentence(procedure("unknown_procedure", asked_for="tooth tattoo", code=None))
    assert "tooth tattoo" in said and "list" in said


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [
        (Decimal("180"), Decimal("180"), "$180"),
        (Decimal("900"), Decimal("1400"), "$900-$1,400"),
        (Decimal("99.50"), Decimal("99.50"), "$99.50"),
    ],
)
def test_a_price_on_file_is_quoted_as_the_row_holds_it(low, high, expected):
    said = price_sentence(procedure("priced", price_min=low, price_max=high))
    assert expected in said


# ---------------------------------------------------------------------------
# Opening hours
# ---------------------------------------------------------------------------
def weekdays(start: int, end: int, opens: time, closes: time) -> list[OpeningWindow]:
    return [OpeningWindow(day, opens, closes) for day in range(start, end + 1)]


def test_identical_consecutive_days_collapse_into_one_line():
    said = hours_sentence(
        HoursFact(
            windows=tuple(
                [*weekdays(0, 4, time(9), time(17)), OpeningWindow(5, time(9), time(12))]
            ),
            closures=(),
            horizon_days=45,
        )
    )
    assert "Mon-Fri 09:00-17:00" in said
    assert "Sat 09:00-12:00" in said


def test_a_lunch_break_is_two_windows_on_the_same_day():
    said = hours_sentence(
        HoursFact(
            windows=(OpeningWindow(0, time(9), time(12)), OpeningWindow(0, time(13), time(17))),
            closures=(),
            horizon_days=45,
        )
    )
    assert "Mon 09:00-12:00, 13:00-17:00" in said


def test_closures_and_the_limit_of_the_answer_are_both_stated():
    said = hours_sentence(
        HoursFact(
            windows=weekdays(0, 0, time(9), time(17)),
            closures=(ClosedDay(date(2026, 11, 26), "Thanksgiving"),),
            horizon_days=45,
        )
    )
    assert "26 Nov (Thanksgiving)" in said
    assert "free appointment times" in said, "open hours are not availability"


def test_no_schedule_on_file_invents_none():
    said = hours_sentence(HoursFact(windows=(), closures=(), horizon_days=45))
    assert not any(character.isdigit() for character in said)


# ---------------------------------------------------------------------------
# Several facts, and provenance
# ---------------------------------------------------------------------------
def test_facts_are_answered_in_the_order_they_were_looked_up():
    block = facts_block(
        [plan("in_network"), procedure("priced", price_min=Decimal("1"), price_max=Decimal("1"))]
    )
    assert block.index("Aetna") < block.index("Crown")


def test_a_single_source_is_named_with_its_date():
    line = sources_block([Citation(7, "Aftercare > Dry Socket", "aftercare.md", date(2026, 1, 15))])
    assert line == "Source: Aftercare > Dry Socket (as of 2026-01-15)"


def test_several_sources_are_listed():
    listed = sources_block(
        [
            Citation(7, "Aftercare > Dry Socket", "aftercare.md", date(2026, 1, 15)),
            Citation(9, "Visiting > Parking", "visiting.md", date(2026, 2, 1)),
        ]
    )
    assert listed.startswith("Sources:\n- ")
    assert listed.count("\n- ") == 2
