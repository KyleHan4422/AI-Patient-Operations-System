"""A question about the clinic, from the message to the row it leaves behind.

A real Postgres, the real corpus, the real tools -- and the offline model,
which answers by keyword. What is asserted here is the part that does not
depend on the model's judgement: which tool a question reaches, that an exact
fact is worded by the clinic and not by the model, that an invented figure is
stopped before the patient sees it, and that an abstention leaves a record of
what the clinic could not answer.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import select

from patient_ops.adapters.llm.embeddings import (
    FAKE_EMBEDDING_MODEL,
    HashingEmbeddings,
    min_score_for,
)
from patient_ops.agents.knowledge import FINAL_ANSWER
from patient_ops.config import get_settings
from patient_ops.db.models import InsurancePlan, KbGap, Message, Procedure, ToolCall
from patient_ops.db.transcript import SqlTranscript
from patient_ops.errors import ErrorCode, ToolError
from patient_ops.graph.build import build_graph
from patient_ops.graph.context import GraphContext
from patient_ops.graph.replies import ABSTENTION
from patient_ops.rag.ingest import ingest
from patient_ops.rag.retrieve import search_knowledge_base
from patient_ops.tools.registry import (
    GET_OPENING_HOURS,
    LOOKUP_INSURANCE_PLAN,
    LOOKUP_PRICE,
    SEARCH_DOCUMENTS,
    ReadOnlyToolset,
)
from tests.factories import TZ
from tests.fakes import ScriptedModel

MODEL = FAKE_EMBEDDING_MODEL
MIN_SCORE = min_score_for(MODEL)
embeddings = HashingEmbeddings()
NOW = datetime(2026, 10, 1, 16, 0, tzinfo=UTC)


@pytest.fixture
async def demo(session_factory, clinic):
    """The clinic, plus the three rows Phase 1 seeded deliberate gaps into:
    a plan that is accepted, one that is refused, and a treatment with no price.
    Aetna is absent, which is the point of it."""
    async with session_factory() as session, session.begin():
        session.add_all(
            [
                InsurancePlan(plan_name="Delta Dental PPO", accepted=True, in_network=True),
                InsurancePlan(plan_name="Cigna DPPO", accepted=False, in_network=False),
                Procedure(
                    code="WHITENING",
                    name="Whitening",
                    duration_min=60,
                    specialty="general",
                    price_min=None,
                    price_max=None,
                ),
            ]
        )
    await ingest(session_factory, embeddings, kb_dir=get_settings().kb_dir, model_name=MODEL)
    return session_factory


class Turn:
    """One conversation over the real graph, with the real toolset."""

    def __init__(self, session_factory, model=None) -> None:
        self.session_factory = session_factory
        self.graph = build_graph(InMemorySaver())
        self.thread_id = uuid.uuid4()
        self.model = model
        self.toolset = ReadOnlyToolset(
            session_factory=session_factory,
            embeddings=embeddings,
            embedding_model=MODEL,
            min_score=MIN_SCORE,
            top_k=4,
            tz=TZ,
            now=lambda: NOW,
        )

    async def ask(self, text: str) -> str:
        from patient_ops.adapters.llm.fake import EchoChatModel
        from patient_ops.graph.turn import Final, run_turn

        context = GraphContext(
            thread_id=self.thread_id,
            request_id="req-test",
            chat_model=self.model or EchoChatModel(),
            recorder=SqlTranscript(self.session_factory),
            toolset=self.toolset,
        )
        reply = None
        async for event in run_turn(self.graph, text=text, context=context):
            if isinstance(event, Final):
                reply = event.text
        assert reply is not None
        return reply

    @property
    def tools_used(self) -> list[str]:
        return [call.name for call in self.toolset.trace]

    async def rows(self, model):
        async with self.session_factory() as session:
            return list((await session.scalars(select(model).order_by(model.id))).all())


def intent(label: str) -> AIMessage:
    """The classifier's reply, as a scripted model must produce it first."""
    call = {"name": "IntentDecision", "args": {"intent": label}, "id": "call_intent"}
    return AIMessage("", tool_calls=[call])


def calling(name: str, args: dict) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": "call_1"}])


# ---------------------------------------------------------------------------
# Exact facts are answered from the record, in the clinic's words
# ---------------------------------------------------------------------------
async def test_a_plan_not_on_file_is_never_reported_as_a_refusal(demo):
    turn = Turn(demo)
    reply = await turn.ask("Do you take Aetna?")

    assert turn.tools_used == [LOOKUP_INSURANCE_PLAN]
    assert "on file" in reply
    assert "don't accept" not in reply.lower()


async def test_a_plan_the_clinic_refuses_is_said_plainly(demo):
    turn = Turn(demo)
    reply = await turn.ask("Do you take Cigna DPPO?")
    assert reply.startswith("We don't accept Cigna DPPO.")


async def test_an_accepted_plan_is_answered_from_the_row(demo):
    reply = await Turn(demo).ask("Do you take Delta Dental PPO?")
    assert "in network" in reply


async def test_a_treatment_with_no_price_quotes_no_figure(demo):
    turn = Turn(demo)
    reply = await turn.ask("How much is whitening?")

    assert turn.tools_used == [LOOKUP_PRICE]
    assert not any(character.isdigit() for character in reply)
    assert "front desk" in reply


async def test_opening_hours_come_from_the_schedules(demo):
    turn = Turn(demo)
    reply = await turn.ask("Are you open on Saturday?")

    assert turn.tools_used == [GET_OPENING_HOURS]
    assert "Mon-Fri 09:00-17:00" in reply
    assert "free appointment times" in reply


async def test_an_insurance_word_alone_is_not_a_plan_name(demo):
    """ "Does my insurance cover cleanings" names no plan. Looking one up would
    answer it with "I have no plan called cleanings on file"."""
    turn = Turn(demo)
    await turn.ask("Does my insurance cover cleanings?")
    assert turn.tools_used == [SEARCH_DOCUMENTS]


async def test_a_fee_question_is_a_document_question_not_a_price_lookup(demo):
    """ "How much is the missed appointment fee" names no treatment on file. It
    is a policy question, and the price table is not where policies live."""
    turn = Turn(demo)
    await turn.ask("How much is the missed appointment fee?")
    assert turn.tools_used == [SEARCH_DOCUMENTS]


# ---------------------------------------------------------------------------
# Prose is checked against what was retrieved
# ---------------------------------------------------------------------------
async def test_a_grounded_answer_carries_its_source(demo):
    reply = await Turn(demo).ask("How do I get my records sent to another dentist?")
    assert "Source:" in reply or "Sources:" in reply
    assert reply != ABSTENTION


async def test_an_invented_figure_never_reaches_the_patient(demo):
    """The rule that makes "never quote an approximate figure" mechanical.

    The model is scripted to cite a passage it really was shown and then quote
    a number that is not in it. What the patient gets is the abstention.
    """
    question = "how much notice do I need to give to cancel?"
    async with demo() as session:
        found = await search_knowledge_base(
            session, embeddings, question, model=MODEL, k=4, min_score=MIN_SCORE
        )
    cited = found.chunks[0].chunk_id

    lying = ScriptedModel(
        script=[
            intent("knowledge"),
            calling(SEARCH_DOCUMENTS, {"query": question}),
            calling(
                FINAL_ANSWER,
                {
                    "sufficient": True,
                    "answer": "Cancel any time; the fee is $9,999.",
                    "citations": [cited],
                },
            ),
        ]
    )
    turn = Turn(demo, model=lying)
    reply = await turn.ask(question)

    assert reply == ABSTENTION
    assert "9,999" not in reply
    [gap] = await turn.rows(KbGap)
    assert gap.reason == "unsupported_figure"


async def test_a_citation_the_agent_was_never_shown_is_refused(demo):
    question = "how do I cancel?"
    inventing = ScriptedModel(
        script=[
            intent("knowledge"),
            calling(SEARCH_DOCUMENTS, {"query": question}),
            calling(
                FINAL_ANSWER,
                {"sufficient": True, "answer": "We ask for notice.", "citations": [999_999]},
            ),
        ]
    )
    turn = Turn(demo, model=inventing)
    assert await turn.ask(question) == ABSTENTION
    [gap] = await turn.rows(KbGap)
    assert gap.reason == "fabricated_citation"


# ---------------------------------------------------------------------------
# What the turn leaves behind
# ---------------------------------------------------------------------------
async def test_an_abstention_records_what_the_clinic_could_not_answer(demo):
    """Off topic: nothing comes close, and the gap says how close, against
    which model, and which section was nearest. That is the list of documents
    the clinic has not written."""
    turn = Turn(demo)
    reply = await turn.ask("What is the weather in Paris tomorrow?")

    assert reply == ABSTENTION
    [gap] = await turn.rows(KbGap)
    assert gap.reason == "no_passage_above_threshold"
    assert gap.question == "What is the weather in Paris tomorrow?"
    assert gap.embedding_model == MODEL
    assert gap.best_score < gap.threshold == MIN_SCORE
    assert gap.nearest_heading, "the nearest section is recorded even below the threshold"


async def test_every_lookup_is_recorded_with_the_reply_it_produced(demo):
    turn = Turn(demo)
    await turn.ask("Do you take Delta Dental PPO?")

    calls = await turn.rows(ToolCall)
    messages = await turn.rows(Message)
    assert [c.name for c in calls] == [LOOKUP_INSURANCE_PLAN]
    assert calls[0].args == {"plan_name": "Delta Dental PPO"}
    assert calls[0].summary == "in_network"
    assert calls[0].latency_ms >= 0
    assert calls[0].message_id == messages[-1].id, "attached to the assistant's reply"


async def test_an_answered_turn_records_no_gap(demo):
    turn = Turn(demo)
    await turn.ask("Do you take Delta Dental PPO?")
    assert await turn.rows(KbGap) == []


async def test_the_transcript_says_how_the_reply_was_decided(demo):
    turn = Turn(demo)
    await turn.ask("Do you take Aetna?")
    assistant = (await turn.rows(Message))[-1]
    assert assistant.meta["intent"] == "knowledge"
    assert assistant.meta["answer_kind"] == "facts"


# ---------------------------------------------------------------------------
# ★ Phase 3's contract, at the layer that depends on it
# ---------------------------------------------------------------------------
async def test_a_corpus_that_was_never_ingested_fails_the_turn(session_factory, clinic):
    """Not an abstention. An empty knowledge base means exactly one thing --
    "nothing relevant" -- and forgetting `make ingest` is not that. If this
    ever abstained, the assistant would say "I don't have that on file" to
    every question, correctly worded and completely wrong.
    """
    turn = Turn(session_factory)
    with pytest.raises(ToolError) as raised:
        await turn.ask("How do I cancel an appointment?")
    assert raised.value.code == ErrorCode.PERMANENT
    assert "make ingest" in raised.value.detail
