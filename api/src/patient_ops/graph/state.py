"""What the graph carries from step to step -- and, through the checkpointer,
from turn to turn.

Only the fields Phase 2 uses. Later phases add theirs (intent, patient,
offered_slots, ...). Adding a field is safe for conversations already saved;
renaming or removing one is a migration of every stored checkpoint.

Everything here is persisted after every step. So any field that belongs to
one turn only must be reset in that turn's input (graph/turn.py), or the
previous turn's value leaks into this one.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # The conversation as the model sees it. add_messages appends instead of
    # replacing, so a node returns only the messages it adds.
    messages: Annotated[list[AnyMessage], add_messages]
    # What an LLM node proposes to say this turn.
    draft: str | None
    # What the user is actually told this turn. Written only by `respond`.
    final_response: str | None
