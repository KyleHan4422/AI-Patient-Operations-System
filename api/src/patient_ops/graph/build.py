"""The conversation graph, compiled once per process.

                            +-- booking  ----> booking_deferred --+
    START -> classify_intent|                                     |-> respond -> END
                            +-- smalltalk --> smalltalk_reply ----+
                            |                                     |
                            +-- knowledge -> knowledge_agent ->   |
                                             verify_answer -------+

Every branch converges on `respond`, which is the only node that writes an
assistant message. The knowledge branch is the only one that reaches clinic
data, and it cannot speak directly: `verify_answer` sits between the agent and
the exit, deciding without a model whether what the agent proposed is
supported by what it actually retrieved.

Phase 6 replaces booking_deferred with the deterministic validate -> execute ->
verify path and adds the second read-only agent; nothing else here moves.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from patient_ops.graph.context import GraphContext
from patient_ops.graph.nodes.booking_deferred import booking_deferred
from patient_ops.graph.nodes.classify_intent import classify_intent
from patient_ops.graph.nodes.knowledge_agent import knowledge_agent
from patient_ops.graph.nodes.respond import respond
from patient_ops.graph.nodes.smalltalk_reply import smalltalk_reply
from patient_ops.graph.nodes.verify_answer import verify_answer
from patient_ops.graph.state import AgentState

# Nodes whose model tokens may be streamed to the user. Stream mode "messages"
# emits tokens from EVERY model call in the graph -- the intent classifier's
# JSON, the agent's tool calls, its draft answer -- so streaming is opt-in per
# node, and only one node qualifies.
#
# An answer about the clinic is not streamed at all. It is checked against its
# evidence first, and a sentence shown while it is still provisional has
# already been said: a patient who reads a price that is then withdrawn has
# been told the price. Small talk carries no clinic fact, so it streams.
USER_FACING_NODES: frozenset[str] = frozenset({"smalltalk_reply"})

INTENT_ROUTES: dict[str, str] = {
    "knowledge": "knowledge_agent",
    "booking": "booking_deferred",
    "smalltalk": "smalltalk_reply",
}


def route_by_intent(state: AgentState) -> str:
    """Where the turn goes. An unrecognised intent goes to the knowledge agent:
    it is the branch that can say "I don't have that on file"."""
    return INTENT_ROUTES.get(state.get("intent") or "", "knowledge_agent")


def build_graph(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    graph = StateGraph(AgentState, context_schema=GraphContext)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("knowledge_agent", knowledge_agent)
    graph.add_node("verify_answer", verify_answer)
    graph.add_node("booking_deferred", booking_deferred)
    graph.add_node("smalltalk_reply", smalltalk_reply)
    graph.add_node("respond", respond)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges(
        "classify_intent", route_by_intent, sorted(set(INTENT_ROUTES.values()))
    )
    graph.add_edge("knowledge_agent", "verify_answer")
    graph.add_edge("verify_answer", "respond")
    graph.add_edge("booking_deferred", "respond")
    graph.add_edge("smalltalk_reply", "respond")
    graph.add_edge("respond", END)
    return graph.compile(checkpointer=checkpointer)
