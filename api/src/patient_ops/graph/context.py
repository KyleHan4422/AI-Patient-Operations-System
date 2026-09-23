"""What each turn hands the graph from outside: identity and dependencies.

Nodes never import a model provider or the database layer; they receive them
here. So a test can run the whole graph on a fake model and an in-memory
recorder, and the knowledge agent receives its model and its read-only tools
the same way -- without an import that the architectural check would have to
allow.

Unlike AgentState, none of this is checkpointed: it is supplied fresh on
every turn.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Literal, Protocol

from langchain_core.language_models import BaseChatModel

from patient_ops.db.transcript import TurnRecord
from patient_ops.tools.registry import ReadOnlyToolset


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
    # The read-only tools this turn may use, and the record of what it used.
    # Built per turn (it accumulates that turn's evidence), so it is supplied
    # here rather than compiled into the graph. None on a turn that reaches no
    # agent -- the nodes that need it say so.
    toolset: ReadOnlyToolset | None = None
    # How many times the agent may call tools before the turn gives up and
    # abstains. A cost ceiling and a loop guard in one number.
    max_tool_rounds: int = 3
