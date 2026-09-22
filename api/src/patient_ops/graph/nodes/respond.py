"""The single exit of every turn. Deterministic: it never calls a model.

It decides what the user is told, adds that to the conversation, and records
the turn in the human-readable transcript. Nothing else in the graph writes
an assistant message.

Phase 2 passes the draft through, falling back to a fixed line if the model
produced nothing. Later the text comes from a grounded answer (Phase 4) or,
for a booking, from a template filled from the verified database row
(Phase 6) -- the reason this node exists before it has much to decide.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from patient_ops.db.transcript import TurnRecord
from patient_ops.graph.context import GraphContext
from patient_ops.graph.state import AgentState

EMPTY_DRAFT_REPLY = "Sorry, I don't have a good answer to that. Could you say it another way?"


async def respond(state: AgentState, runtime: Runtime[GraphContext]) -> dict[str, Any]:
    ctx = runtime.context
    final = (state.get("draft") or "").strip() or EMPTY_DRAFT_REPLY
    user_text = next(m.text for m in reversed(state["messages"]) if isinstance(m, HumanMessage))

    await ctx.recorder.record_turn(
        TurnRecord(
            thread_id=ctx.thread_id,
            channel=ctx.channel,
            user_text=user_text,
            assistant_text=final,
            meta={"request_id": ctx.request_id},
        )
    )
    return {"messages": [AIMessage(final)], "final_response": final}
