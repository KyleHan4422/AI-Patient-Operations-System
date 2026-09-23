"""The knowledge agent: works out what the clinic's records say, and proposes an answer.

The first autonomous reasoner in the system, and the rule it is built under is
the project's central one:

    Agent autonomy is scoped by the reversibility of the action space.

It chooses what to look up, in what order, and whether to look again. It cannot
change anything: the only data it can reach is the read-only toolset it is
handed (tools/registry.py), and scripts/check_invariants.py fails the build if
anything under agents/ acquires a wider reach. Its output is a *proposal* --
graph/grounding.py decides whether the patient ever sees it.

The loop is a plain one, on purpose:

    bind the tools and FinalAnswer, forcing a tool call every round
    -> the model calls tools, or calls FinalAnswer to stop
    -> at most `max_rounds` rounds, then the turn abstains

FinalAnswer is a schema, not a tool: it is never executed, it is how the model
hands back something typed. Forcing a tool call (`tool_choice="any"`) means
there is no path where the model simply talks to the patient -- every answer
arrives as structured data that can be checked.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field, ValidationError

from patient_ops.obs.logging import get_logger
from patient_ops.tools.registry import (
    GET_OPENING_HOURS,
    LOOKUP_INSURANCE_PLAN,
    LOOKUP_PRICE,
    SEARCH_DOCUMENTS,
    ReadOnlyToolset,
)

log = get_logger(__name__)

FINAL_ANSWER = "FinalAnswer"


class FinalAnswer(BaseModel):
    """End your turn and hand back what you found. Always finish with this."""

    sufficient: bool = Field(
        description=(
            "True only if the material in front of you answers the question that was "
            "actually asked. A passage on the right topic that does not contain the "
            "answer is not sufficient."
        )
    )
    answer: str = Field(
        default="",
        description=(
            "Two or three plain sentences for the patient, containing only what the "
            "cited passages say. Empty when the answer came from a records lookup."
        ),
    )
    citations: list[int] = Field(
        default_factory=list,
        description="The N of every [chunk N] passage used in the answer.",
    )


@dataclass(frozen=True)
class AgentOutcome:
    """What the agent came back with. `verdict` is None when it never finished --
    it ran out of rounds, or produced something that would not parse. Both are
    abstentions, which is the safe way to fail."""

    verdict: FinalAnswer | None
    rounds: int


SYSTEM_PROMPT = f"""\
You are the front-desk assistant of a dental clinic, working out the answer to a
patient's question.

Everything you say comes from the tools. You have no other knowledge of this
clinic -- not its hours, prices, insurance, policies, staff or availability --
and general dental knowledge is not an answer about this clinic.

Which tool:
- Insurance plans, prices and opening hours are records, not prose: use
  {LOOKUP_INSURANCE_PLAN}, {LOOKUP_PRICE} and {GET_OPENING_HOURS}. The
  documents do not contain them.
- Everything else the clinic has written down -- policies, cancellations,
  aftercare, what to bring, what to expect -- is in {SEARCH_DOCUMENTS}.
- You may use several tools, and you may search again in different words if the
  first passages miss the point.

Then call {FINAL_ANSWER} to end your turn:
- sufficient: true only if what you have answers the question that was actually
  asked. Reporting that the clinic's documents do not cover something is useful
  -- it tells the clinic which document to write. Guessing is not.
- answer: two or three plain, warm sentences, saying only what the cited
  passages say. Every figure -- a price, a fee, a number of days or hours, a
  percentage -- must appear in a passage you cite. Never round one, convert one,
  average two, or bring one in from anywhere else.
- citations: the N of every [chunk N] passage you used.
- If the answer came from {LOOKUP_INSURANCE_PLAN}, {LOOKUP_PRICE} or
  {GET_OPENING_HOURS}, leave answer and citations empty. The patient is told
  what the record says, in the clinic's own words; anything you write there is
  discarded.

You cannot book, change or cancel appointments and you cannot see free slots.
Never give medical or dental advice, and never offer a typical or approximate
figure."""


def system_prompt(catalogue: Sequence[tuple[str, str]]) -> str:
    """The prompt, with the clinic's treatment codes appended.

    The codes are in the prompt so that {LOOKUP_PRICE} is called with one that
    exists. A lookup tool left to guess its own argument is a fuzzy match in
    disguise, which is exactly what the exact tables are here to avoid.
    """
    if not catalogue:
        return SYSTEM_PROMPT
    listed = ", ".join(f"{code} ({name})" for code, name in catalogue)
    return f"{SYSTEM_PROMPT}\n\nTreatments on file: {listed}."


async def answer_question(
    messages: Sequence[AnyMessage],
    *,
    model: BaseChatModel,
    toolset: ReadOnlyToolset,
    max_rounds: int = 3,
) -> AgentOutcome:
    tools = toolset.tools()
    by_name = {tool.name: tool for tool in tools}
    bound = model.bind_tools([*tools, FinalAnswer], tool_choice="any")

    catalogue = await toolset.procedure_catalogue()
    scratch: list[AnyMessage] = [SystemMessage(system_prompt(catalogue)), *messages]

    for round_no in range(1, max_rounds + 1):
        reply = await bound.ainvoke(scratch)
        scratch.append(reply)
        calls = list(reply.tool_calls)
        lookups = [c for c in calls if c["name"] in by_name]

        if lookups:
            # Data tools win over a FinalAnswer sent in the same message: the
            # model is answering before it has seen what it asked for, and one
            # more round with the evidence in hand is strictly better than a
            # verdict citing passages it has not read.
            for call in lookups:
                try:
                    result = await by_name[call["name"]].ainvoke(call["args"])
                except ValidationError as exc:
                    # Arguments that do not fit the tool's schema. Handed back
                    # as the tool's answer rather than raised: the model has a
                    # round left to correct itself, and the alternative is a
                    # failed turn over a typo. A ToolError -- the corpus is
                    # missing, the database is down -- still propagates.
                    log.warning("tool_args_rejected", tool=call["name"], error=str(exc))
                    result = f"That call was rejected: {exc}. Call it again with valid arguments."
                scratch.append(ToolMessage(result, tool_call_id=call["id"]))
            continue

        final = next((c for c in calls if c["name"] == FINAL_ANSWER), None)
        if final is None:
            # tool_choice="any" should make this unreachable with a real
            # provider. Abstain rather than treat free text as an answer.
            log.warning("agent_said_nothing", round=round_no, content=reply.text[:200])
            return AgentOutcome(verdict=None, rounds=round_no)
        try:
            verdict = FinalAnswer.model_validate(final["args"])
        except ValidationError as exc:
            log.warning("agent_verdict_unparseable", round=round_no, error=str(exc))
            return AgentOutcome(verdict=None, rounds=round_no)
        log.info(
            "agent_verdict",
            rounds=round_no,
            sufficient=verdict.sufficient,
            citations=verdict.citations,
            tools_used=[i.name for i in toolset.trace],
        )
        return AgentOutcome(verdict=verdict, rounds=round_no)

    # Out of rounds: the model kept looking things up and never concluded.
    log.warning("agent_out_of_rounds", rounds=max_rounds)
    return AgentOutcome(verdict=None, rounds=max_rounds)
