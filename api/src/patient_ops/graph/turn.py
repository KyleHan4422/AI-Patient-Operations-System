"""One turn of conversation, independent of how it arrived.

The HTTP route turns these events into SSE; the Phase 13 voice channel will
turn them into speech. Neither re-implements what a turn is.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from patient_ops.graph.build import USER_FACING_NODES
from patient_ops.graph.context import GraphContext


@dataclass(frozen=True)
class Stage:
    """What the turn is about to do. Not part of the reply.

    The knowledge branch does not stream -- an answer is checked before it is
    said -- so a patient who asks about a price watches nothing happen for a
    second or two. This says what is happening instead, which is the honest
    version of a typing indicator: it is emitted from the classifier's own
    output, not guessed at by the client.
    """

    intent: str


@dataclass(frozen=True)
class Token:
    """A piece of the reply, as the model writes it. Provisional."""

    text: str


@dataclass(frozen=True)
class Final:
    """The authoritative reply. Emitted only after the turn is saved."""

    text: str
    # Fallbacks the turn took. Not an error: the reply is still the reply.
    degraded: tuple[str, ...] = ()


def turn_input(text: str) -> dict[str, Any]:
    # The per-turn fields are reset explicitly. The checkpoint carries every
    # field into the next turn, so without this a turn that fails before
    # `respond` would still hold the previous turn's reply.
    return {
        "messages": [HumanMessage(text)],
        "draft": None,
        "final_response": None,
        "intent": None,
        "verdict": None,
        "answer_kind": None,
        "citations": None,
        "kb_gap": None,
        "degraded_modes": None,
    }


def turn_config(thread_id: uuid.UUID) -> RunnableConfig:
    return {"configurable": {"thread_id": str(thread_id)}}


async def run_turn(
    graph: CompiledStateGraph, *, text: str, context: GraphContext
) -> AsyncIterator[Stage | Token | Final]:
    final: str | None = None
    degraded: list[str] = []
    async for mode, chunk in graph.astream(
        turn_input(text),
        turn_config(context.thread_id),
        context=context,
        stream_mode=["messages", "updates"],
        # Save each step before starting the next, so that when the loop ends
        # the whole turn is in Postgres.
        durability="sync",
    ):
        if mode == "messages":
            message, metadata = chunk
            if metadata.get("langgraph_node") in USER_FACING_NODES and message.text:
                yield Token(message.text)
        elif mode == "updates":
            if "classify_intent" in chunk:
                yield Stage(chunk["classify_intent"]["intent"])
            if "respond" in chunk:
                final = chunk["respond"]["final_response"]
                degraded = chunk["respond"].get("degraded_modes") or []

    if final is None:  # every path ends in respond; reaching here is a graph bug
        raise RuntimeError("turn finished without passing through respond")
    # After the loop, not when respond's update arrives: an acknowledgement
    # sent before the save would be a promise the system might not keep.
    yield Final(final, tuple(degraded))
