"""The knowledge agent's loop, driven by a scripted model. No network, no database.

The loop is a state machine over whatever a model happens to return, so the
branches worth testing are the ones a well-behaved model never takes: a verdict
that will not parse, an answer written before anything was looked up, a model
that keeps searching and never concludes. All three end the same way, which is
the property that matters: the turn abstains.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from patient_ops.agents.knowledge import FINAL_ANSWER, answer_question, system_prompt
from patient_ops.tools.registry import (
    GET_OPENING_HOURS,
    LOOKUP_INSURANCE_PLAN,
    LOOKUP_PRICE,
    SEARCH_DOCUMENTS,
)
from tests.fakes import ScriptedModel, StubToolset

PASSAGE = "[chunk 3] Appointments > Missed\n(appointments.md, as of 2026-01-15)\nWe ask 24 hours."


def calling(*calls: tuple[str, dict]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": f"call_{n}"} for n, (name, args) in enumerate(calls)
        ],
    )


def final(**args) -> AIMessage:
    return calling((FINAL_ANSWER, {"sufficient": True, "answer": "", "citations": []} | args))


async def run(script: list[AIMessage], *, max_rounds: int = 3, question: str = "how do I cancel?"):
    model = ScriptedModel(script=script)
    toolset = StubToolset(**{SEARCH_DOCUMENTS: PASSAGE, LOOKUP_INSURANCE_PLAN: "not on file"})
    outcome = await answer_question(
        [HumanMessage(question)], model=model, toolset=toolset, max_rounds=max_rounds
    )
    return outcome, model, toolset


# ---------------------------------------------------------------------------
# The ordinary path
# ---------------------------------------------------------------------------
async def test_a_lookup_then_an_answer():
    outcome, _, toolset = await run(
        [
            calling((SEARCH_DOCUMENTS, {"query": "cancelling"})),
            final(answer="We ask 24 hours.", citations=[3]),
        ]
    )
    assert outcome.rounds == 2
    assert outcome.verdict is not None
    assert outcome.verdict.answer == "We ask 24 hours."
    assert outcome.verdict.citations == [3]
    assert toolset.asked == [(SEARCH_DOCUMENTS, {"query": "cancelling"})]


async def test_what_a_tool_returned_is_shown_to_the_model():
    _, model, _ = await run(
        [calling((SEARCH_DOCUMENTS, {"query": "cancelling"})), final(citations=[3])]
    )
    assert PASSAGE in model.seen[-1][-1].text, "the second round sees the passage"


async def test_the_agent_may_look_more_than_once():
    outcome, _, toolset = await run(
        [
            calling((SEARCH_DOCUMENTS, {"query": "first try"})),
            calling((SEARCH_DOCUMENTS, {"query": "other words"})),
            final(citations=[3]),
        ]
    )
    assert outcome.rounds == 3
    assert [args["query"] for _, args in toolset.asked] == ["first try", "other words"]


async def test_two_lookups_in_one_round_both_run():
    _, _, toolset = await run(
        [
            calling((LOOKUP_INSURANCE_PLAN, {"plan_name": "Aetna"}), (GET_OPENING_HOURS, {})),
            final(),
        ]
    )
    assert [name for name, _ in toolset.asked] == [LOOKUP_INSURANCE_PLAN, GET_OPENING_HOURS]


# ---------------------------------------------------------------------------
# The branches a good model never takes
# ---------------------------------------------------------------------------
async def test_an_answer_sent_before_the_evidence_waits_for_it():
    """A model that asks and answers in the same breath has not read the reply.

    The lookup runs and the verdict is discarded; one more round with the
    passage in hand is strictly better than a verdict citing what it never saw.
    """
    outcome, _, toolset = await run(
        [
            calling(
                (SEARCH_DOCUMENTS, {"query": "cancelling"}),
                (FINAL_ANSWER, {"sufficient": True, "answer": "Guessed.", "citations": [3]}),
            ),
            final(answer="Read it this time.", citations=[3]),
        ]
    )
    assert toolset.asked, "the lookup still ran"
    assert outcome.verdict is not None and outcome.verdict.answer == "Read it this time."


async def test_arguments_the_tool_rejects_are_handed_back_not_raised():
    """A typo in a tool call costs a round, not the turn.

    The verdict path already recovers from an unparseable answer; this is the
    same failure one step earlier, and it used to kill the turn instead.
    """
    outcome, model, toolset = await run(
        [
            calling((LOOKUP_PRICE, {"wrong_argument": "CROWN"})),
            calling((LOOKUP_PRICE, {"procedure": "CROWN"})),
            final(),
        ]
    )
    assert outcome.verdict is not None, "the model got another round"
    assert toolset.asked == [(LOOKUP_PRICE, {"procedure": "CROWN"})]
    assert "rejected" in model.seen[1][-1].text, "it was told what went wrong"


async def test_a_verdict_that_will_not_parse_is_an_abstention():
    outcome, _, _ = await run([calling((FINAL_ANSWER, {"sufficient": "maybe"}))])
    assert outcome.verdict is None


async def test_free_text_instead_of_a_verdict_is_an_abstention():
    """The model is bound with tool_choice="any", so this should be
    unreachable with a real provider -- and is still not treated as an answer."""
    outcome, _, _ = await run([AIMessage("Sure, we ask for 24 hours' notice!")])
    assert outcome.verdict is None


async def test_a_model_that_never_concludes_runs_out_of_rounds():
    searching = calling((SEARCH_DOCUMENTS, {"query": "again"}))
    outcome, _, toolset = await run([searching, searching], max_rounds=2)
    assert outcome.verdict is None and outcome.rounds == 2
    assert len(toolset.asked) == 2, "the budget is the number of rounds, not of tools"


# ---------------------------------------------------------------------------
# What the agent is given
# ---------------------------------------------------------------------------
async def test_the_only_tools_bound_are_read_only_ones_and_the_answer_schema():
    """The action space, asserted. Nothing that writes is reachable, so an
    agent that decides to book an appointment has no way to act on it."""
    _, model, _ = await run([final()])
    assert set(model.bound[0]) == {
        SEARCH_DOCUMENTS,
        LOOKUP_INSURANCE_PLAN,
        LOOKUP_PRICE,
        GET_OPENING_HOURS,
        FINAL_ANSWER,
    }


async def test_the_treatments_on_file_are_named_in_the_prompt():
    """So a price lookup is called with a code that exists, rather than one the
    model invented -- a lookup left to guess its argument is a fuzzy match."""
    _, model, _ = await run([final()])
    assert "CROWN (Crown)" in model.seen[0][0].text


async def test_the_prompt_survives_an_empty_catalogue():
    assert system_prompt([]).endswith("figure.")


async def test_the_agent_sees_the_conversation_it_was_given():
    model = ScriptedModel(script=[final()])
    toolset = StubToolset()
    history = [HumanMessage("do you take Aetna?"), AIMessage("I don't have that plan on file.")]
    await answer_question(
        [*history, HumanMessage("what about Delta?")], model=model, toolset=toolset
    )
    shown = [m.text for m in model.seen[0]]
    assert "do you take Aetna?" in shown and "what about Delta?" in shown
