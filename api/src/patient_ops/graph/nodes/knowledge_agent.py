"""The node that runs the knowledge agent. Thin on purpose.

Everything that reasons lives in agents/knowledge.py, which knows nothing about
LangGraph and can be tested by calling it. This node is the seam between the
two: it takes the window of conversation the agent is shown, hands over the
model and the read-only toolset from the runtime context, and stores the
agent's *proposal* in state.

Nothing here decides what the patient is told. `verify_answer` does that next,
from the same evidence, without a model.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import trim_messages
from langgraph.runtime import Runtime

from patient_ops.agents.knowledge import answer_question
from patient_ops.graph.context import GraphContext
from patient_ops.graph.state import AgentState
from patient_ops.obs.logging import get_logger

log = get_logger(__name__)


async def knowledge_agent(state: AgentState, runtime: Runtime[GraphContext]) -> dict[str, Any]:
    ctx = runtime.context
    if ctx.toolset is None:  # a wiring mistake, not a runtime condition
        raise RuntimeError("the knowledge agent needs a toolset on the graph context")

    history = trim_messages(
        state["messages"],
        strategy="last",
        max_tokens=ctx.history_limit,
        token_counter=len,
        start_on="human",
    )
    started = time.perf_counter()
    outcome = await answer_question(
        history,
        model=ctx.chat_model,
        toolset=ctx.toolset,
        max_rounds=ctx.max_tool_rounds,
    )
    log.info(
        "agent_finished",
        rounds=outcome.rounds,
        tool_calls=len(ctx.toolset.trace),
        latency_ms=round((time.perf_counter() - started) * 1000),
    )
    # A dict, not the model: state is checkpointed, and a schema that later
    # gains a field must still deserialise every conversation already saved.
    return {"verdict": outcome.verdict.model_dump() if outcome.verdict else None}
