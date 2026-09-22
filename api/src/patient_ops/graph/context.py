"""What each turn hands the graph from outside: identity and dependencies.

Nodes never import a model provider or the database layer; they receive them
here. So a test can run the whole graph on a fake model and an in-memory
recorder, and the agents of Phase 4 will receive their model the same way --
without an import that the architectural check would have to allow.

Unlike AgentState, none of this is checkpointed: it is supplied fresh on
every turn.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Literal, Protocol

from langchain_core.language_models import BaseChatModel

from patient_ops.db.transcript import TurnRecord


class TurnRecorder(Protocol):
    async def record_turn(self, turn: TurnRecord) -> None: ...


@dataclass(frozen=True)
class GraphContext:
    thread_id: uuid.UUID
    request_id: str
    chat_model: BaseChatModel
    recorder: TurnRecorder
    channel: Literal["web", "voice"] = "web"
    history_limit: int = 20  # messages shown to the model per turn
