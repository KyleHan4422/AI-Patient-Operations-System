"""The booking reader: a message in, a typed proposal out, and nothing else.

It has no tools to call, so there is no loop to test -- only what it does with
what a model returns, and what it is shown.
"""

from __future__ import annotations

from datetime import date

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from patient_ops.adapters.llm.fake import EchoChatModel
from patient_ops.agents.booking import BookingProposal, read_booking_request
from tests.fakes import ScriptedModel

TODAY = date(2026, 10, 1)
CATALOGUE = [("CLEANING", "Adult cleaning"), ("EXAM", "Comprehensive exam")]


def proposal_call(args: dict) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": "BookingProposal", "args": args, "id": "c1"}])


async def test_a_proposal_that_will_not_parse_is_an_empty_one():
    model = ScriptedModel(script=[proposal_call({"choice": "the blue one"})])
    proposal = await read_booking_request(
        [HumanMessage("the blue one")], model=model, today=TODAY, catalogue=CATALOGUE
    )
    assert proposal == BookingProposal(action="unclear")


async def test_the_reader_is_shown_today_the_treatments_and_the_numbered_offers():
    model = ScriptedModel(script=[proposal_call({"choice": 2})])
    proposal = await read_booking_request(
        [HumanMessage("the second")],
        model=model,
        today=TODAY,
        catalogue=CATALOGUE,
        offered=[
            "Thursday 2026-10-01 10:00 with Dr. Chen",
            "Friday 2026-10-02 09:00 with Dr. Chen",
        ],
    )
    assert proposal.choice == 2
    [system, *_] = model.seen[0]
    assert isinstance(system, SystemMessage)
    assert "Today is Thursday 2026-10-01." in system.text
    assert "CLEANING (Adult cleaning)" in system.text
    assert "2) Friday 2026-10-02 09:00 with Dr. Chen" in system.text
    assert model.bound == [["BookingProposal"]], "a schema to fill, and no tool to call"


async def test_the_offline_reader_takes_a_yes_only_as_an_answer_to_the_question():
    model = EchoChatModel()
    asked = [HumanMessage("book me a cleaning"), AIMessage("Shall I book it?"), HumanMessage("yes")]
    unasked = [HumanMessage("book me a cleaning"), AIMessage("Which one?"), HumanMessage("yes")]
    kwargs = {"model": model, "today": TODAY, "catalogue": CATALOGUE}
    assert (await read_booking_request(asked, **kwargs)).confirmed == "yes"
    assert (await read_booking_request(unasked, **kwargs)).confirmed is None
