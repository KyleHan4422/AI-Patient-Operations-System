"""Ask the model what to say. A placeholder: the only LLM node of Phase 2.

Phase 4 replaces it with classify_intent -> knowledge_agent, Phase 6 adds the
booking agent. `respond`, after it, does not change.

It proposes; it does not speak. Its output is a draft that `respond` decides
on -- the first appearance of "what the user is told is decided in one place,
and not by a model".
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
#
# This phase has no tools and no clinic data, so the prompt's main job is to
# stop the model inventing any. Invented clinic facts are the failure this
# whole project is built against.
SYSTEM_PROMPT = """\
You are the front-desk assistant of a dental clinic, chatting with a patient.

You currently have no access to any clinic information: not opening hours,
prices, accepted insurance, providers or appointment availability, and you
cannot book, change or cancel appointments. If asked about any of these, say
plainly that you can't look that up yet and that the clinic's staff can help.
Never guess, and never give typical or approximate figures.

Keep replies short, warm and plain. Do not give medical advice."""


async def draft_reply(state: AgentState, runtime: Runtime[GraphContext]) -> dict[str, Any]:
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
        node="draft_reply",
        latency_ms=round((time.perf_counter() - started) * 1000),
        messages_shown=len(history),
        usage=getattr(reply, "usage_metadata", None),
    )
    return {"draft": reply.text}
