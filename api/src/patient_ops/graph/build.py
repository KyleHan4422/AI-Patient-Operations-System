"""The conversation graph, compiled once per process.

Phase 2:   START -> draft_reply -> respond -> END

The shape later phases grow into is in the master plan (section C): intent
classification, two read-only agents, and the deterministic booking path, all
converging on `respond`.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from patient_ops.graph.context import GraphContext
from patient_ops.graph.nodes.draft_reply import draft_reply
from patient_ops.graph.nodes.respond import respond
from patient_ops.graph.state import AgentState

# Nodes whose model tokens may be streamed to the user. Stream mode "messages"
# emits tokens from EVERY model call in the graph -- including, from Phase 4,
# the intent classifier's JSON -- so streaming is opt-in per node.
USER_FACING_NODES: frozenset[str] = frozenset({"draft_reply"})


def build_graph(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    graph = StateGraph(AgentState, context_schema=GraphContext)
    graph.add_node("draft_reply", draft_reply)
    graph.add_node("respond", respond)
    graph.add_edge(START, "draft_reply")
    graph.add_edge("draft_reply", "respond")
    graph.add_edge("respond", END)
    return graph.compile(checkpointer=checkpointer)
