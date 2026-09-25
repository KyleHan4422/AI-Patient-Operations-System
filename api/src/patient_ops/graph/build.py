"""The conversation graph, compiled once per process.

                            +-- booking --> booking_agent -> plan_booking --+
                            |                    (execute_booking ->        |
                            |                     verify_booking) ----------+
    START -> classify_intent|                                               |-> respond -> END
                            +-- smalltalk --> smalltalk_reply --------------+
                            |                                               |
                            +-- knowledge -> knowledge_agent ->             |
                                             verify_answer -----------------+

Every branch converges on `respond`, which is the only node that writes an
assistant message. Neither model-driven branch can speak for itself:
`verify_answer` decides, without a model, whether the knowledge agent's answer
is supported by what it retrieved; and on the booking branch the model only
reads the patient's message -- plan_booking decides the next step, and only
the path through execute_booking and verify_booking can write an appointment
and say it was written.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from patient_ops.graph.context import GraphContext
from patient_ops.graph.nodes.booking import (
    booking_agent,
    execute_booking,
    plan_booking,
    route_after_plan,
    verify_booking,
)
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
    "booking": "booking_agent",
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
    graph.add_node("booking_agent", booking_agent)
    graph.add_node("plan_booking", plan_booking)
    graph.add_node("execute_booking", execute_booking)
    graph.add_node("verify_booking", verify_booking)
    graph.add_node("smalltalk_reply", smalltalk_reply)
    graph.add_node("respond", respond)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges(
        "classify_intent", route_by_intent, sorted(set(INTENT_ROUTES.values()))
    )
    graph.add_edge("knowledge_agent", "verify_answer")
    graph.add_edge("verify_answer", "respond")
    graph.add_edge("booking_agent", "plan_booking")
    # Only a "yes" to a read-back goes on to the write; every other step of a
    # booking is a question or an offer, and goes straight out.
    graph.add_conditional_edges("plan_booking", route_after_plan, ["execute_booking", "respond"])
    # Unconditional: whatever the write did -- succeeded, failed, timed out --
    # the next step is to find out what is actually in the calendar.
    graph.add_edge("execute_booking", "verify_booking")
    graph.add_edge("verify_booking", "respond")
    graph.add_edge("smalltalk_reply", "respond")
    graph.add_edge("respond", END)
    return graph.compile(checkpointer=checkpointer)
