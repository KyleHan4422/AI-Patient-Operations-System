"""The single exit of every turn. Deterministic: it never calls a model.

It decides what the user is told, adds that to the conversation, and records
the turn: the two transcript messages, every read-only lookup that was made on
the way, and -- when the turn abstained -- the KB_GAP that says what the clinic
could not answer. All of it in one transaction. Nothing else in the graph
writes an assistant message.

Three branches reach it and each has already been settled elsewhere: an answer
checked against its evidence (verify_answer), a fixed sentence about bookings
(booking_deferred, until Phase 6), or small talk from a model that was shown no
clinic data at all. So this node passes the draft through and falls back to a
fixed line when there is nothing -- which is the same shape it will keep when
Phase 6's booking confirmation is filled from the verified appointment row.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from patient_ops.db.transcript import KbGapRecord, ToolCallRecord, TurnRecord
from patient_ops.graph.context import GraphContext
from patient_ops.graph.state import AgentState

EMPTY_DRAFT_REPLY = "Sorry, I don't have a good answer to that. Could you say it another way?"


async def respond(state: AgentState, runtime: Runtime[GraphContext]) -> dict[str, Any]:
    ctx = runtime.context
    final = (state.get("draft") or "").strip() or EMPTY_DRAFT_REPLY
    user_text = next(m.text for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
    gap = state.get("kb_gap")
    # The trace is read from the context, not from state: it is per-turn
    # evidence, and the checkpoint is the conversation's long-term memory.
    trace = ctx.toolset.trace if ctx.toolset is not None else []
    degraded = ctx.degraded.modes

    await ctx.recorder.record_turn(
        TurnRecord(
            thread_id=ctx.thread_id,
            channel=ctx.channel,
            user_text=user_text,
            assistant_text=final,
            meta={
                "request_id": ctx.request_id,
                "intent": state.get("intent"),
                "answer_kind": state.get("answer_kind"),
                "citations": state.get("citations") or [],
                "degraded_modes": degraded,
            },
            tool_calls=tuple(
                ToolCallRecord(
                    name=call.name,
                    args=call.args,
                    summary=call.summary,
                    latency_ms=call.latency_ms,
                )
                for call in trace
            ),
            kb_gap=KbGapRecord(**gap) if gap else None,
        )
    )
    return {
        "messages": [AIMessage(final)],
        "final_response": final,
        "degraded_modes": degraded,
    }
