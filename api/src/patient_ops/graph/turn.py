"""One turn of conversation, independent of how it arrived.

The HTTP route turns these events into SSE; the Phase 13 voice channel will
turn them into speech. Neither re-implements what a turn is.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from patient_ops.db.transcript import TurnRecord
from patient_ops.graph.build import USER_FACING_NODES
from patient_ops.graph.context import GraphContext, TurnRecorder
from patient_ops.guardrails.emergency import EmergencyMatch
from patient_ops.guardrails.emergency import reply as emergency_reply
from patient_ops.obs.logging import get_logger

log = get_logger(__name__)

# An emergency reply waits this long, at most, for each of its two writes. The
# writes are how staff see what happened; the reply is what the patient needs.
# With Postgres down, the patient is told to call 911 two seconds late rather
# than not at all.
EMERGENCY_WRITE_TIMEOUT_S = 2.0
# The degraded mode reported when an emergency reply could not be written down.
UNRECORDED = "transcript"


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
    # Set when a guardrail answered instead of the graph: {"id": "G0",
    # "category": "er"}. A client shows such a reply differently.
    guardrail: dict[str, str] | None = None


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
        "booking_proposal": None,
        "appointment_id": None,
        # Not `booking`: a booking in progress is carried from turn to turn
        # on purpose, and only the booking path clears it.
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


async def emergency_turn(
    graph: CompiledStateGraph,
    recorder: TurnRecorder,
    *,
    match: EmergencyMatch,
    text: str,
    thread_id: uuid.UUID,
    channel: Literal["web", "voice"],
    request_id: str,
    clinic_phone: str,
) -> AsyncIterator[Stage | Final]:
    """A turn G0 answered. No model is called and the graph does not run.

    Written down twice, as a graph turn would be: into the transcript, so
    staff see it, and into the checkpoint, so the conversation's next turn
    knows the patient reported an emergency and was told where to go. Each
    write is bounded and best effort -- a failure is logged and reported as
    degraded, and the reply goes out regardless.

    The checkpoint write also ends any booking in progress. The last thing
    the patient was told is to go to the emergency room, not "Shall I book
    it?", so a "yes" on the next turn must not write an appointment. Its
    holds expire by themselves.
    """
    yield Stage("emergency")
    final = emergency_reply(match, clinic_phone)
    guardrail = {"id": "G0", "category": match.category}
    log.warning("emergency_triggered", category=match.category, rule=match.rule_id)

    degraded: list[str] = []
    record = TurnRecord(
        thread_id=thread_id,
        channel=channel,
        user_text=text,
        assistant_text=final,
        meta={
            "request_id": request_id,
            "intent": "emergency",
            "answer_kind": "emergency",
            "guardrail": "G0",
            "category": match.category,
            "rule": match.rule_id,
        },
    )
    update = turn_input(text) | {
        "messages": [HumanMessage(text), AIMessage(final)],
        "intent": "emergency",
        "answer_kind": "emergency",
        "final_response": final,
        "degraded_modes": [],
        "booking": None,
    }
    for name, write in (
        ("transcript", lambda: recorder.record_turn(record)),
        # As `respond`: the turn is over, and the next one starts from START.
        (
            "checkpoint",
            lambda: graph.aupdate_state(turn_config(thread_id), update, as_node="respond"),
        ),
    ):
        try:
            await asyncio.wait_for(write(), EMERGENCY_WRITE_TIMEOUT_S)
        except Exception as exc:
            # Not BaseException: a client disconnect must still cancel.
            log.error("emergency_unrecorded", write=name, error_type=type(exc).__name__)
            if UNRECORDED not in degraded:
                degraded.append(UNRECORDED)
    yield Final(final, tuple(degraded), guardrail)
