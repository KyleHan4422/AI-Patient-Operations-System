"""The gate between what the agent proposed and what the patient hears.

It calls no model. Given the agent's verdict and the evidence the toolset
recorded, graph/grounding.py returns either an answer that may be said or an
abstention with a reason, and this node turns that into the turn's draft.

When it abstains it also writes down why, as a KB_GAP: the question, how close
retrieval got, and which section came nearest. That record is the byproduct
the clinic actually wants -- the list of documents it has not written yet --
and the reason Phase 3 measured the residual instead of averaging it away.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from patient_ops.agents.knowledge import FinalAnswer
from patient_ops.graph.context import GraphContext
from patient_ops.graph.grounding import Abstained, decide
from patient_ops.graph.state import AgentState
from patient_ops.obs.logging import get_logger
from patient_ops.tools.registry import ReadOnlyToolset

log = get_logger(__name__)


def _gap(reason: str, question: str, toolset: ReadOnlyToolset) -> dict[str, Any]:
    """What was missing, in the form the kb_gaps table stores.

    The nearest passage is recorded even when it fell below the threshold:
    "0.31 against Insurance & payment > What we accept" says which document is
    thin far better than the question alone does.
    """
    best = max(toolset.searches, key=lambda r: r.best_score, default=None)
    nearest = best.nearest if best is not None else None
    return {
        "question": question,
        "reason": reason,
        "best_score": best.best_score if best is not None else None,
        "threshold": best.threshold if best is not None else None,
        "embedding_model": toolset.embedding_model,
        "nearest_heading": nearest.heading_path if nearest is not None else None,
        "nearest_source": nearest.source_path if nearest is not None else None,
    }


async def verify_answer(state: AgentState, runtime: Runtime[GraphContext]) -> dict[str, Any]:
    ctx = runtime.context
    if ctx.toolset is None:  # a wiring mistake, not a runtime condition
        raise RuntimeError("verify_answer needs the same toolset the agent used")
    toolset = ctx.toolset

    raw = state.get("verdict")
    verdict = FinalAnswer.model_validate(raw) if raw else None
    question = next(m.text for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
    decision = decide(
        verdict,
        facts=toolset.facts,
        passages=toolset.passages,
        searched=bool(toolset.searches),
    )

    if isinstance(decision, Abstained):
        question = next(m.text for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
        gap = _gap(decision.reason, question, toolset)
        # A warning, not an error: abstaining is the system working. It is
        # logged loudly because a rise in one reason is a real signal -- a
        # corpus going stale, or a model starting to invent citations.
        log.warning("abstained", **gap)
        return {"draft": decision.text, "answer_kind": "abstained", "kb_gap": gap}

    if decision.rejected is not None:
        # The turn was answerable from the clinic's records, so the patient is
        # not affected -- but the model wrote prose it could not support, and
        # that is the signal this layer exists to raise. It gets no kb_gaps row
        # (nothing was missing from the corpus) and a warning instead.
        log.warning("prose_refused", reason=decision.rejected, question=question)
    log.info(
        "answer_grounded", kind=decision.kind, citations=[c.chunk_id for c in decision.citations]
    )
    return {
        "draft": decision.text,
        "answer_kind": decision.kind,
        "citations": [c.chunk_id for c in decision.citations],
    }
