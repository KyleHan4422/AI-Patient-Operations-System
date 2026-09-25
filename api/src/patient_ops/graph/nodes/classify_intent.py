"""What the patient wants, in one word. The first node of every turn.

Three destinations, and the routing decides which part of the system is even
allowed to run: the knowledge agent and its read-only tools, the booking path,
or a reply that touches no clinic data at all.

A booking takes several turns, and most of them do not look like a booking on
their own: "yes", "the second one", "(212) 555-0101". The prompt says so, and
while a booking is in progress a message labelled small talk goes to the
booking path anyway -- a "yes" to "Shall I book it?" answered with "You're
welcome!" would leave the patient believing they were booked.

Two things this node is deliberately not:

  a safety device   The emergency filter is keyword-based, deterministic, and
                    runs in the HTTP layer before any model sees the message
                    (Phase 7). A classifier that is right 98% of the time is
                    not where "my face is swelling" gets handled.
  a guess           When it cannot tell, it says `knowledge`. That path can
                    abstain; the booking path cannot ask the documents.

Its JSON never reaches the patient: USER_FACING_NODES in graph/build.py lists
which nodes may stream, and this is not one of them.
"""

from __future__ import annotations

import time
from typing import Any, Literal, get_args

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import SystemMessage, trim_messages
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field, ValidationError

from patient_ops.graph.context import GraphContext
from patient_ops.graph.nodes.booking import active_draft
from patient_ops.graph.state import AgentState
from patient_ops.obs.logging import get_logger

log = get_logger(__name__)

Intent = Literal["knowledge", "booking", "smalltalk"]
INTENTS: tuple[str, ...] = get_args(Intent)
# Where an unreadable answer goes. The one path that is able to say "I don't
# know", so a misroute costs a needless abstention rather than a wrong answer.
FALLBACK_INTENT: Intent = "knowledge"


class IntentDecision(BaseModel):
    """The label for the patient's latest message."""

    intent: Intent = Field(description="One of: knowledge, booking, smalltalk.")


SYSTEM_PROMPT = """\
You label the patient's latest message with what they want from a dental
clinic's front desk. The earlier messages are context; label the last one.

knowledge   A question about this clinic: opening hours, prices, which
            insurance is taken, policies, cancellations, aftercare, what to
            bring, what to expect. Also anything you are not sure about.
booking     They want to make, move, confirm or cancel an appointment, or are
            asking which times are free. Also a reply to the assistant's
            booking questions: a phone number, a name or date of birth, a
            treatment, a day, picking one of the offered times, or yes/no to
            "Shall I book it?".
smalltalk   A greeting, a thank-you, a goodbye, or a remark with no question
            in it."""


async def classify_intent(state: AgentState, runtime: Runtime[GraphContext]) -> dict[str, Any]:
    ctx = runtime.context
    history = trim_messages(
        state["messages"],
        strategy="last",
        max_tokens=min(ctx.history_limit, 6),  # enough for "yes, book that one"
        token_counter=len,
        start_on="human",
    )
    started = time.perf_counter()
    try:
        decision = await ctx.chat_model.with_structured_output(IntentDecision).ainvoke(
            [SystemMessage(SYSTEM_PROMPT), *history]
        )
        intent = IntentDecision.model_validate(decision).intent
    except (OutputParserException, ValidationError, ValueError) as exc:
        # A malformed label, not a provider failure: those keep propagating, so
        # a model that is down fails the turn loudly instead of quietly
        # answering every question from the wrong branch.
        log.warning("intent_unparseable", error=str(exc), fallback=FALLBACK_INTENT)
        intent = FALLBACK_INTENT
        unreadable = True
    else:
        unreadable = False

    # While a booking is open, a reply that is not clearly a question about the
    # clinic continues it. A real "knowledge" label still goes to the agent --
    # "how much is a cleaning?" mid-booking deserves an answer -- but small
    # talk ("ok", "thanks") and an unreadable label do not abandon the booking.
    if intent != "booking" and (intent == "smalltalk" or unreadable) and ctx.booking is not None:
        if active_draft(state, ctx.booking.now()) is not None:
            log.info("intent_continues_booking", classified=intent, unreadable=unreadable)
            intent = "booking"

    log.info(
        "intent_classified",
        intent=intent,
        latency_ms=round((time.perf_counter() - started) * 1000),
    )
    return {"intent": intent}
