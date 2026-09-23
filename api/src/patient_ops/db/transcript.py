"""Writes down a completed turn: what was said, what was looked up, what was missing.

A write path, so it lives outside repo.py (which only reads). Its one caller is
the graph's `respond` node -- the single exit every turn passes through.

One transaction per turn, now covering four tables: the conversation row is
upserted, both messages are inserted, and the turn's tool calls and its KB_GAP
(if it abstained) go in with them. So the transcript never shows a question
without its answer, and never shows an answer whose evidence went missing.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from sqlalchemy import func, insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from patient_ops.db.models import Conversation, KbGap, Message, ToolCall


@dataclass(frozen=True)
class ToolCallRecord:
    name: str
    args: dict[str, Any]
    summary: str
    latency_ms: int


@dataclass(frozen=True)
class KbGapRecord:
    """A question the records could not answer. Field for field, kb_gaps."""

    question: str
    reason: str
    best_score: float | None
    threshold: float | None
    embedding_model: str
    nearest_heading: str | None
    nearest_source: str | None


@dataclass(frozen=True)
class TurnRecord:
    thread_id: uuid.UUID
    channel: Literal["web", "voice"]
    user_text: str
    assistant_text: str
    meta: dict[str, Any] = field(default_factory=dict)
    tool_calls: tuple[ToolCallRecord, ...] = ()
    kb_gap: KbGapRecord | None = None


class SqlTranscript:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def record_turn(self, turn: TurnRecord) -> None:
        # Upsert, not "look it up, insert if missing": two first turns racing
        # on one thread would both find nothing and both insert. ON CONFLICT
        # makes the database settle it -- Phase 1's lesson, in miniature.
        upsert = (
            pg_insert(Conversation)
            .values(thread_id=turn.thread_id, channel=turn.channel)
            .on_conflict_do_update(
                index_elements=[Conversation.thread_id],
                set_={"last_message_at": func.now()},
            )
            .returning(Conversation.id)
        )
        async with self._session_factory.begin() as session:
            conversation_id = await session.scalar(upsert)
            # Two statements rather than one executemany, because what follows
            # hangs off the assistant message and needs its id back.
            await session.execute(
                insert(Message).values(
                    conversation_id=conversation_id,
                    role="user",
                    content=turn.user_text,
                    meta=turn.meta,
                )
            )
            assistant_id = await session.scalar(
                insert(Message)
                .values(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=turn.assistant_text,
                    meta=turn.meta,
                )
                .returning(Message.id)
            )
            if turn.tool_calls:
                await session.execute(
                    insert(ToolCall),
                    [{"message_id": assistant_id, **asdict(call)} for call in turn.tool_calls],
                )
            if turn.kb_gap is not None:
                await session.execute(
                    insert(KbGap).values(message_id=assistant_id, **asdict(turn.kb_gap))
                )
