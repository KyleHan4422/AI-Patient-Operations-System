"""The demo seed: idempotent, and shaped for the evaluation cases it exists for."""

from __future__ import annotations

from patient_ops.db import repo
from tests.factories import load_script

seed_script = load_script("seed")


async def test_seed_is_idempotent(session_factory):
    first = await seed_script.seed(session_factory)
    second = await seed_script.seed(session_factory)
    assert first == second
    assert first["providers"] == 3 and first["procedures"] == 7


async def test_unknown_plan_is_not_the_same_as_a_rejected_plan(session_factory):
    """The repo already distinguishes "we don't know" from "we don't take it"."""
    await seed_script.seed(session_factory)
    async with session_factory() as session:
        assert await repo.find_insurance_plan(session, "Aetna") is None  # not on file
        cigna = await repo.find_insurance_plan(session, "cigna dppo")
        assert cigna is not None and cigna.accepted is False  # on file, rejected
        delta = await repo.find_insurance_plan(session, "  delta   dental ppo ")
        assert delta is not None and delta.accepted and delta.in_network


async def test_plan_lookup_is_exact_not_fuzzy(session_factory):
    await seed_script.seed(session_factory)
    async with session_factory() as session:
        assert await repo.find_insurance_plan(session, "Delta Dental") is None
        assert await repo.find_insurance_plan(session, "Cigna PPO") is None


async def test_patient_search_can_return_several_matches(session_factory):
    await seed_script.seed(session_factory)
    async with session_factory() as session:
        same_name = await repo.find_patients(session, name="maria garcia")
        assert len(same_name) == 2  # ambiguous: the agent must ask a follow-up question
        by_phone = await repo.find_patients(session, phone="212.555.0103")
        assert [p.full_name for p in by_phone] == ["James Wilson"]


async def test_a_procedure_without_a_price_on_file(session_factory):
    await seed_script.seed(session_factory)
    async with session_factory() as session:
        whitening = await repo.get_procedure(session, "WHITENING")
        assert whitening is not None
        assert whitening.price_min is None and whitening.price_max is None
