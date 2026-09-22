"""The transcript tables, written the way the graph's respond node writes them."""

from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from patient_ops.db import repo
from patient_ops.db.models import Conversation
from patient_ops.db.transcript import SqlTranscript, TurnRecord


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
