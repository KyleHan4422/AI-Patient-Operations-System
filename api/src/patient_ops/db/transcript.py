"""Writes the human-readable transcript: one call per completed turn.

A write path, so it lives outside repo.py (which only reads). Its one caller is
the graph's `respond` node -- the single exit every turn passes through.

One transaction per turn: the conversation row is upserted and both messages
are inserted together, so the transcript never shows a question without its
answer.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from sqlalchemy import func, insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from patient_ops.db.models import Conversation, Message


@dataclass(frozen=True)
class TurnRecord:
    thread_id: uuid.UUID
    channel: Literal["web", "voice"]
    user_text: str
    assistant_text: str
    meta: dict[str, Any] = field(default_factory=dict)


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
            await session.execute(
                insert(Message),
                [
                    {
                        "conversation_id": conversation_id,
                        "role": role,
                        "content": content,
                        "meta": turn.meta,
                    }
                    for role, content in (
                        ("user", turn.user_text),
                        ("assistant", turn.assistant_text),
                    )
                ],
            )
