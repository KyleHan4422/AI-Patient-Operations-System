"""The transcript tables, written the way the graph's respond node writes them."""

from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from patient_ops.db import repo
from patient_ops.db.models import Conversation, KbGap, ToolCall
from patient_ops.db.transcript import (
    KbGapRecord,
    SqlTranscript,
    ToolCallRecord,
    TurnRecord,
)


def turn(thread_id: uuid.UUID, n: int, channel: str = "web") -> TurnRecord:
    return TurnRecord(
        thread_id=thread_id,
        channel=channel,  # type: ignore[arg-type]
        user_text=f"question {n}",
        assistant_text=f"answer {n}",
        meta={"request_id": f"req-{n}"},
    )


async def conversation_count(session_factory) -> int:
    async with session_factory() as session:
        return await session.scalar(select(func.count()).select_from(Conversation))


async def test_turns_append_to_one_conversation_in_order(session_factory):
    transcript = SqlTranscript(session_factory)
    thread = uuid.uuid4()
    await transcript.record_turn(turn(thread, 1))
    await transcript.record_turn(turn(thread, 2))

    async with session_factory() as session:
        messages = await repo.get_transcript(session, thread)
    assert [(m.role, m.content) for m in messages] == [
        ("user", "question 1"),
        ("assistant", "answer 1"),
        ("user", "question 2"),
        ("assistant", "answer 2"),
    ]
    assert messages[0].meta == {"request_id": "req-1"}
    assert await conversation_count(session_factory) == 1


async def test_racing_first_turns_create_one_conversation(session_factory):
    """Two first turns on the same new thread: ON CONFLICT settles the race."""
    transcript = SqlTranscript(session_factory)
    thread = uuid.uuid4()
    await asyncio.gather(*(transcript.record_turn(turn(thread, n)) for n in range(5)))

    assert await conversation_count(session_factory) == 1
    async with session_factory() as session:
        assert len(await repo.get_transcript(session, thread)) == 10


async def test_threads_are_isolated(session_factory):
    transcript = SqlTranscript(session_factory)
    a, b = uuid.uuid4(), uuid.uuid4()
    await transcript.record_turn(turn(a, 1))
    await transcript.record_turn(turn(b, 2))
    async with session_factory() as session:
        assert [m.content for m in await repo.get_transcript(session, a)] == [
            "question 1",
            "answer 1",
        ]


async def test_unknown_thread_is_none_not_empty(session_factory):
    async with session_factory() as session:
        assert await repo.get_transcript(session, uuid.uuid4()) is None


async def test_unknown_channel_is_rejected_by_the_database(session_factory):
    with pytest.raises(IntegrityError, match="ck_conversations_channel"):
        await SqlTranscript(session_factory).record_turn(turn(uuid.uuid4(), 1, channel="sms"))


# ---------------------------------------------------------------------------
# What a turn records besides what was said (Phase 4)
# ---------------------------------------------------------------------------
async def test_a_slow_lookup_is_recorded_rather_than_overflowing(session_factory):
    """A search embeds its query through the provider: 30s of timeout plus the
    SDK's retries is past what a SMALLINT holds. Overflowing would fail the
    insert of a turn that had already been answered."""
    transcript = SqlTranscript(session_factory)
    thread = uuid.uuid4()
    slow = ToolCallRecord(
        name="search_documents",
        args={"query": "anything"},
        summary="4 passage(s)",
        latency_ms=95_000,
    )
    await transcript.record_turn(
        TurnRecord(
            thread_id=thread,
            channel="web",
            user_text="q",
            assistant_text="a",
            tool_calls=(slow,),
        )
    )
    async with session_factory() as session:
        [recorded] = (await session.scalars(select(ToolCall))).all()
    assert recorded.latency_ms == 95_000


async def test_a_turn_writes_its_lookups_and_its_gap_together(session_factory):
    """One transaction: a reply whose evidence went missing is not a record of
    anything."""
    transcript = SqlTranscript(session_factory)
    thread = uuid.uuid4()
    await transcript.record_turn(
        TurnRecord(
            thread_id=thread,
            channel="web",
            user_text="do you do braces?",
            assistant_text="I don't have that on file.",
            tool_calls=(
                ToolCallRecord(
                    name="search_documents", args={"query": "braces"}, summary="0", latency_ms=12
                ),
            ),
            kb_gap=KbGapRecord(
                question="do you do braces?",
                reason="no_passage_above_threshold",
                best_score=0.12,
                threshold=0.345,
                embedding_model="text-embedding-3-small",
                nearest_heading="Procedures Explained > Fillings",
                nearest_source="procedures-explained.md",
            ),
        )
    )

    async with session_factory() as session:
        messages = await repo.get_transcript(session, thread)
        [call] = (await session.scalars(select(ToolCall))).all()
        [gap] = (await session.scalars(select(KbGap))).all()

    assistant = messages[-1]
    assert call.message_id == assistant.id, "the lookups hang off the reply they produced"
    assert gap.message_id == assistant.id
    assert gap.reason == "no_passage_above_threshold"


async def test_an_unknown_gap_reason_is_rejected_by_the_database(session_factory):
    """The CHECK is the backstop for graph/grounding.GAP_REASONS drifting."""
    transcript = SqlTranscript(session_factory)
    with pytest.raises(IntegrityError):
        await transcript.record_turn(
            TurnRecord(
                thread_id=uuid.uuid4(),
                channel="web",
                user_text="q",
                assistant_text="a",
                kb_gap=KbGapRecord(
                    question="q",
                    reason="because-i-said-so",
                    best_score=None,
                    threshold=None,
                    embedding_model="m",
                    nearest_heading=None,
                    nearest_source=None,
                ),
            )
        )
