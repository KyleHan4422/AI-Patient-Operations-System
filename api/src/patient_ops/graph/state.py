"""What the graph carries from step to step -- and, through the checkpointer,
from turn to turn.

Only the fields the graph needs to carry between its own steps. Adding a field
is safe for conversations already saved; renaming or removing one is a
migration of every stored checkpoint.

What is deliberately *not* here: the passages retrieved this turn and the
agent's tool calls. They live on the per-turn toolset in the runtime context,
because every field below is written into the checkpoint after every step, and
a conversation's long-term memory should not grow by a few kilobytes of quoted
documents per question.

Everything here is persisted after every step. So any field that belongs to
one turn only must be reset in that turn's input (graph/turn.py), or the
previous turn's value leaks into this one.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

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
    # Which branch this turn took: knowledge, booking or smalltalk.
    intent: str | None
    # The knowledge agent's proposal, as a plain dict (agents/knowledge.py's
    # FinalAnswer). A dict, not the model: a schema that gains a field later
    # must still load every conversation already checkpointed.
    verdict: dict[str, Any] | None
    # How the reply was decided -- "facts", "passages", "abstained", ... Not
    # cosmetic: it is what an evaluation run and the /ops view group by.
    answer_kind: str | None
    # kb_chunks.id of every passage the reply rests on.
    citations: list[int] | None
    # Set when the turn abstained: what the clinic could not answer, and how
    # close it got. `respond` persists it with the rest of the turn.
    kb_gap: dict[str, Any] | None
    # The fallbacks this turn took because an optional dependency was down --
    # "rate_limit", "holds", ... Empty when nothing degraded. Written only by
    # `respond`, from the turn's context, so a degraded turn is visible in the
    # state, the transcript's meta and the `done` event alike.
    degraded_modes: list[str] | None
