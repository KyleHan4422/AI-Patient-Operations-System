"""The exact lookups, against a real database.

What is tested here is the shape of the answer, not the row: that a plan the
clinic has never heard of comes back as `unknown` and not as `not_accepted`,
that a treatment with no price comes back as itself rather than as a zero, and
that the clinic's opening hours are the union of its providers' schedules.
"""

from __future__ import annotations

from datetime import date, time
from decimal import Decimal

import pytest

from patient_ops.db.models import InsurancePlan, Procedure, Provider, ProviderSchedule
from patient_ops.tools.clinic import lookup_insurance_plan, lookup_price, opening_hours
from tests.factories import CLOSED_MONDAY

TODAY = date(2026, 10, 1)  # a Thursday, the week before CLOSED_MONDAY


@pytest.fixture
async def plans(session_factory):
    """Three plans, mirroring the demo data's deliberate near-miss: one accepted
    in network, one accepted out of network, one refused -- and Aetna absent."""
    async with session_factory() as session, session.begin():
        session.add_all(
            [
                InsurancePlan(plan_name="Delta Dental PPO", accepted=True, in_network=True),
                InsurancePlan(
                    plan_name="MetLife PDP",
                    accepted=True,
                    in_network=False,
                    notes="Claims are submitted for you.",
                ),
                InsurancePlan(plan_name="Cigna DPPO", accepted=False, in_network=False),
            ]
        )
    return session_factory


async def look_up_plan(session_factory, name: str):
    async with session_factory() as session:
        return await lookup_insurance_plan(session, name)


async def look_up_price(session_factory, term: str):
    async with session_factory() as session:
        return await lookup_price(session, term)


# ---------------------------------------------------------------------------
# Insurance
# ---------------------------------------------------------------------------
async def test_a_plan_that_is_not_on_file_is_unknown_not_refused(plans):
    """The failure this phase is built against, at the layer it starts in."""
    fact = await look_up_plan(plans, "Aetna")
    assert fact.status == "unknown"
    assert fact.definite is False
    assert fact.plan_name is None
    assert fact.asked_for == "Aetna", "what the patient called it survives for the reply"


async def test_a_refused_plan_is_definite(plans):
    fact = await look_up_plan(plans, "Cigna DPPO")
    assert fact.status == "not_accepted" and fact.definite


@pytest.mark.parametrize("written", ["Delta Dental PPO", "delta dental ppo", "  Delta  Dental PPO"])
async def test_a_plan_is_found_however_it_was_typed(plans, written: str):
    assert (await look_up_plan(plans, written)).status == "in_network"


async def test_a_near_neighbour_is_not_a_match(plans):
    """ "Delta Dental" is not "Delta Dental PPO". Exact, deliberately: a fuzzy
    match here quotes the wrong plan's answer with total confidence."""
    assert (await look_up_plan(plans, "Delta Dental")).status == "unknown"


async def test_accepted_out_of_network_keeps_its_notes(plans):
    fact = await look_up_plan(plans, "MetLife PDP")
    assert fact.status == "out_of_network"
    assert fact.notes == "Claims are submitted for you."


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------
@pytest.fixture
async def priced(session_factory, clinic):
    async with session_factory() as session, session.begin():
        session.add(
            Procedure(
                code="FILLING",
                name="Filling",
                duration_min=45,
                specialty="general",
                price_min=Decimal("180"),
                price_max=Decimal("320"),
            )
        )
    return session_factory


async def test_a_price_on_file_comes_back_whole(priced):
    fact = await look_up_price(priced, "FILLING")
    assert fact.status == "priced" and fact.definite
    assert (fact.price_min, fact.price_max) == (Decimal("180.00"), Decimal("320.00"))


async def test_a_treatment_with_no_price_is_not_a_missing_treatment(priced):
    """CROWN exists with no price: the two ways of not knowing are different
    answers, and the reply says which one it is."""
    fact = await look_up_price(priced, "CROWN")
    assert fact.status == "no_price_on_file"
    assert fact.name == "Crown" and fact.price_min is None


async def test_a_treatment_that_is_not_offered_says_so(priced):
    fact = await look_up_price(priced, "IMPLANT")
    assert fact.status == "unknown_procedure"
    assert fact.code is None and fact.asked_for == "IMPLANT"


async def test_a_treatment_is_found_by_code_or_by_name(priced):
    by_code = await look_up_price(priced, "FILLING")
    by_name = await look_up_price(priced, "filling")
    assert by_code.code == by_name.code == "FILLING"


# ---------------------------------------------------------------------------
# Opening hours
# ---------------------------------------------------------------------------
async def test_opening_hours_are_the_union_of_every_provider(session_factory, clinic):
    """The dentist works 09-12 and 13-17; the hygienist works 09-17. The clinic
    is open 09-17 -- not "closed for lunch", which is a fact about one person."""
    async with session_factory() as session:
        fact = await opening_hours(session, today=TODAY)

    weekdays = {w.weekday for w in fact.windows}
    assert weekdays == {0, 1, 2, 3, 4}
    assert all((w.start, w.end) == (time(9), time(17)) for w in fact.windows)
    assert fact.definite


async def test_a_day_only_one_provider_works_is_still_an_open_day(session_factory, clinic):
    async with session_factory() as session, session.begin():
        session.add(
            ProviderSchedule(
                provider_id=clinic.dentist_id, weekday=5, start_time=time(9), end_time=time(12)
            )
        )
    async with session_factory() as session:
        fact = await opening_hours(session, today=TODAY)

    saturday = [w for w in fact.windows if w.weekday == 5]
    assert saturday == [type(saturday[0])(5, time(9), time(12))]


async def test_closures_inside_the_horizon_are_reported(session_factory, clinic):
    async with session_factory() as session:
        near = await opening_hours(session, today=TODAY, horizon_days=45)
        far = await opening_hours(session, today=TODAY, horizon_days=1)

    assert [c.day for c in near.closures] == [CLOSED_MONDAY]
    assert near.closures[0].reason == "Staff training"
    assert far.closures == (), "a closure two months out is noise, not an answer"


async def test_a_clinic_with_no_schedule_on_file_is_not_definite(session_factory):
    async with session_factory() as session, session.begin():
        session.add(Provider(name="Dr. Nobody", specialty="general"))
    async with session_factory() as session:
        fact = await opening_hours(session, today=TODAY)
    assert fact.windows == () and fact.definite is False
