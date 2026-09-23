"""A reply to a message that asks the clinic nothing: hello, thanks, goodbye.

The one node in the graph whose words reach the patient as the model wrote
them, and the only node that may stream, because it is the only one with no
clinic fact in it. Everything factual goes through the knowledge agent and the
grounding check, and arrives in one piece.

Phase 2's draft_reply, narrowed. Its prompt had to carry the whole "you can
look nothing up" warning; now that is true of this branch only, and the
routing -- not the prompt -- is what keeps a price question away from a model
with no tools.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import SystemMessage, trim_messages
from langgraph.runtime import Runtime

from patient_ops.graph.context import GraphContext
from patient_ops.graph.state import AgentState
from patient_ops.obs.logging import get_logger

log = get_logger(__name__)

# Prepended at call time, never stored in the checkpoint: it is not something
# the user said, and changing it must not require rewriting saved
# conversations.
SYSTEM_PROMPT = """\
You are the front-desk assistant of a dental clinic. The patient's message is
small talk -- a greeting, a thank-you, a goodbye, or a remark with no question
in it.

Answer in one or two warm, plain sentences. State nothing about this clinic:
not its opening hours, prices, insurance, policies, staff or availability. You
have none of that in front of you here. If the patient seems to be working up
to such a question, simply invite them to ask it.

Do not give medical or dental advice."""


async def smalltalk_reply(state: AgentState, runtime: Runtime[GraphContext]) -> dict[str, Any]:
    ctx = runtime.context
    # The context window is a budget: show the model the last N messages,
    # starting on a user turn. The checkpoint still keeps every message.
    history = trim_messages(
        state["messages"],
        strategy="last",
        max_tokens=ctx.history_limit,
        token_counter=len,  # count messages, not tokens
        start_on="human",
    )
    started = time.perf_counter()
    reply = await ctx.chat_model.ainvoke([SystemMessage(SYSTEM_PROMPT), *history])
    log.info(
        "llm_call",
        node="smalltalk_reply",
        latency_ms=round((time.perf_counter() - started) * 1000),
        messages_shown=len(history),
        usage=getattr(reply, "usage_metadata", None),
    )
    return {"draft": reply.text, "answer_kind": "smalltalk"}
