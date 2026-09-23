"""Bookings, until Phase 6 builds them: one fixed sentence, no model.

The node exists now so the routing it belongs to exists now. When the real
path arrives -- validate, execute, verify, then a confirmation filled from the
row the database actually holds -- it replaces this node and nothing else in
the graph moves.
"""

from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime

from patient_ops.graph.context import GraphContext
from patient_ops.graph.replies import BOOKING_NOT_YET
from patient_ops.graph.state import AgentState


async def booking_deferred(state: AgentState, runtime: Runtime[GraphContext]) -> dict[str, Any]:
    return {"draft": BOOKING_NOT_YET, "answer_kind": "booking_deferred"}
