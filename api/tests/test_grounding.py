"""The rules that decide whether an answer may be said. No model, no database.

Phase 3 measured the residual these rules exist to absorb: seventeen questions
in the labelled set are on a topic the corpus covers and are not answered by
it, and no retrieval threshold can separate them. Each rule below turns one
way of getting that wrong into an abstention.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from patient_ops.agents.knowledge import FinalAnswer
from patient_ops.db.models import KB_GAP_REASONS
from patient_ops.graph.grounding import (
    GAP_REASONS,
    Abstained,
    Grounded,
    decide,
    unsupported_figures,
)
from patient_ops.rag.retrieve import RetrievedChunk
from patient_ops.tools.facts import InsuranceFact, PriceFact

FEE = (
    "We ask for 24 hours' notice. A visit missed without notice is charged at $50, "
    "and we waive it the first time."
)


def passage(chunk_id: int, content: str, heading: str = "Appointments > Missed") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_title="Appointments",
        source_path="appointments-and-cancellations.md",
        heading_path=heading,
        content=content,
        effective_date=date(2026, 1, 15),
        score=0.61,
    )


def retrieved(*chunks: RetrievedChunk) -> dict[int, RetrievedChunk]:
    return {chunk.chunk_id: chunk for chunk in chunks}


def answer(text: str, citations: list[int], *, sufficient: bool = True) -> FinalAnswer:
    return FinalAnswer(sufficient=sufficient, answer=text, citations=citations)


# ---------------------------------------------------------------------------
# The rule that makes "never quote a figure you were not given" mechanical
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("written", "supported"),
    [
        ("We need 24 hours' notice.", True),
        ("The fee is $50.", True),
        ("The fee is $1,200.", False),
        ("We need 48 hours' notice.", False),
        ("We need a day's notice.", True),  # figures in words are not checked
        ("It is about 25 hours.", False),
        ("The fee is $50.00.", True),  # 50.00 and 50 are the same number
    ],
)
def test_a_figure_must_appear_in_a_cited_passage(written: str, supported: bool):
    assert bool(unsupported_figures(written, [passage(1, FEE)])) is not supported


def test_the_effective_date_counts_as_evidence():
    """The agent was shown "as of 2026-01-15" with the passage, so quoting it
    back is not an invention."""
    assert not unsupported_figures("This is our policy as of 2026-01-15.", [passage(1, FEE)])


def test_an_invented_figure_never_reaches_the_patient():
    decision = decide(
        answer("A missed visit is charged at $250.", [1]),
        facts=[],
        passages=retrieved(passage(1, FEE)),
        searched=True,
    )
    assert isinstance(decision, Abstained)
    assert decision.reason == "unsupported_figure"
    assert "250" not in decision.text


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------
def test_a_grounded_answer_is_said_with_its_source():
    decision = decide(
        answer("We ask for 24 hours' notice.", [1]),
        facts=[],
        passages=retrieved(passage(1, FEE)),
        searched=True,
    )
    assert isinstance(decision, Grounded)
    assert decision.kind == "passages"
    assert decision.text.startswith("We ask for 24 hours' notice.")
    assert "Source: Appointments > Missed (as of 2026-01-15)" in decision.text
    assert [c.chunk_id for c in decision.citations] == [1]


def test_a_citation_that_was_never_retrieved_is_refused():
    """The agent naming a passage it was not shown is not evidence of anything."""
    decision = decide(
        answer("We ask for 24 hours' notice.", [1, 99]),
        facts=[],
        passages=retrieved(passage(1, FEE)),
        searched=True,
    )
    assert isinstance(decision, Abstained) and decision.reason == "fabricated_citation"


def test_an_answer_with_no_citation_at_all_is_refused():
    decision = decide(
        answer("We ask for 24 hours' notice.", []),
        facts=[],
        passages=retrieved(passage(1, FEE)),
        searched=True,
    )
    assert isinstance(decision, Abstained) and decision.reason == "answer_without_citation"


def test_a_repeated_citation_is_named_once():
    decision = decide(
        answer("We ask for 24 hours' notice.", [1, 1]),
        facts=[],
        passages=retrieved(passage(1, FEE)),
        searched=True,
    )
    assert isinstance(decision, Grounded) and len(decision.citations) == 1


# ---------------------------------------------------------------------------
# Saying so when the passages do not answer the question
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("passages", "searched", "expected"),
    [
        ({1: passage(1, FEE)}, True, "passages_did_not_answer"),
        ({}, True, "no_passage_above_threshold"),
        ({}, False, "no_answer"),
    ],
)
def test_an_insufficient_verdict_says_which_kind_of_gap_it_was(passages, searched, expected):
    """The three are different jobs: write the missing document, reword the one
    that exists, or work out why the agent never looked."""
    decision = decide(
        answer("", [], sufficient=False), facts=[], passages=passages, searched=searched
    )
    assert isinstance(decision, Abstained) and decision.reason == expected


def test_an_agent_that_never_answered_abstains():
    decision = decide(None, facts=[], passages={}, searched=True)
    assert isinstance(decision, Abstained) and decision.reason == "no_answer"


# ---------------------------------------------------------------------------
# Facts are not checked here: they were never written by a model
# ---------------------------------------------------------------------------
def test_a_record_lookup_is_answered_from_the_record():
    fact = InsuranceFact(
        asked_for="Delta Dental PPO", plan_name="Delta Dental PPO", status="in_network"
    )
    decision = decide(answer("", []), facts=[fact], passages={}, searched=False)
    assert isinstance(decision, Grounded)
    assert decision.kind == "facts"
    assert "in network" in decision.text


def test_a_fact_the_records_could_not_settle_silences_the_prose():
    """The dangerous case, and why the prose is dropped even though it passed.

    "I have no plan by that name on file" followed by a fluent paragraph about
    insurance is how a patient ends up hearing a refusal the clinic never made.
    """
    unknown = InsuranceFact(asked_for="Aetna", plan_name=None, status="unknown")
    decision = decide(
        answer("We ask for 24 hours' notice.", [1]),
        facts=[unknown],
        passages=retrieved(passage(1, FEE)),
        searched=True,
    )
    assert isinstance(decision, Grounded)
    assert decision.kind == "facts"
    assert "24 hours" not in decision.text
    assert decision.citations == ()


def test_a_definite_fact_and_a_grounded_answer_are_both_told():
    priced = PriceFact(
        asked_for="CROWN",
        code="CROWN",
        name="Crown",
        price_min=Decimal("900"),
        price_max=Decimal("1400"),
        status="priced",
    )
    decision = decide(
        answer("We ask for 24 hours' notice.", [1]),
        facts=[priced],
        passages=retrieved(passage(1, FEE)),
        searched=True,
    )
    assert isinstance(decision, Grounded)
    assert decision.kind == "facts_and_passages"
    assert "$900-$1,400" in decision.text and "24 hours" in decision.text


def test_an_unsupported_answer_is_still_reported_when_the_records_answered():
    """The patient gets the record either way. What must not happen is the
    invention going unnoticed because the turn happened to be answerable."""
    priced = PriceFact(
        asked_for="CROWN",
        code="CROWN",
        name="Crown",
        price_min=Decimal("900"),
        price_max=Decimal("1400"),
        status="priced",
    )
    decision = decide(
        answer("A missed visit is charged at $250.", [1]),
        facts=[priced],
        passages=retrieved(passage(1, FEE)),
        searched=True,
    )
    assert isinstance(decision, Grounded)
    assert decision.kind == "facts"
    assert decision.rejected == "unsupported_figure"
    assert "250" not in decision.text


def test_a_grounded_answer_reports_no_rejection():
    decision = decide(
        answer("We ask for 24 hours' notice.", [1]),
        facts=[],
        passages=retrieved(passage(1, FEE)),
        searched=True,
    )
    assert isinstance(decision, Grounded) and decision.rejected is None


def test_an_abandoned_probe_does_not_silence_a_grounded_answer():
    """The agent looked up CROWNS, got nothing, asked again for CROWN and got
    the price. The price is settled; throwing away a checked answer because of
    the first attempt would be a loss for nobody's benefit."""
    probe = PriceFact(
        asked_for="CROWNS",
        code=None,
        name=None,
        price_min=None,
        price_max=None,
        status="unknown_procedure",
    )
    priced = PriceFact(
        asked_for="CROWN",
        code="CROWN",
        name="Crown",
        price_min=Decimal("900"),
        price_max=Decimal("1400"),
        status="priced",
    )
    decision = decide(
        answer("We ask for 24 hours' notice.", [1]),
        facts=[probe, priced],
        passages=retrieved(passage(1, FEE)),
        searched=True,
    )
    assert isinstance(decision, Grounded)
    assert decision.kind == "facts_and_passages"
    assert "24 hours" in decision.text


def test_a_lookup_repeated_verbatim_is_answered_once():
    fact = InsuranceFact(asked_for="Aetna", plan_name=None, status="unknown")
    decision = decide(answer("", []), facts=[fact, fact], passages={}, searched=False)
    assert isinstance(decision, Grounded)
    assert decision.text.count("Aetna") == 1


def test_nothing_at_all_abstains_rather_than_saying_nothing():
    decision = decide(answer("", []), facts=[], passages={}, searched=False)
    assert isinstance(decision, Abstained)
    assert decision.text.strip(), "an abstention is still an answer to give"


# ---------------------------------------------------------------------------
def test_every_reason_can_be_written_to_the_database():
    """The CHECK constraint on kb_gaps.reason and the reasons this module can
    produce are the same set, or an abstention fails on the INSERT."""
    assert set(GAP_REASONS) == set(KB_GAP_REASONS)
